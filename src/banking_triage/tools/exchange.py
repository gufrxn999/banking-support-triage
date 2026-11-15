"""Frankfurter exchange-rate client.

Frankfurter serves European Central Bank reference rates: business days only, no
API key, no intraday data. Every response therefore carries the date the rate is
*from*, and callers must be able to say "as of", never "today".

Failure policy: unknown currencies are a user-facing error the graph can recover
from, transient failures are retried, and anything else surfaces as a typed
exception rather than a 500 from the API layer.
"""

from __future__ import annotations

import time
from datetime import date

import httpx

from ..config import settings as default_settings
from ..schemas import ExchangeRate

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ExchangeError(RuntimeError):
    """Base class for anything that stops us returning a rate."""


class UnsupportedCurrencyError(ExchangeError):
    """The currency pair is not served. Recoverable: ask the user for another."""

    def __init__(self, base: str, quote: str, detail: str | None = None) -> None:
        super().__init__(detail or f"{base}->{quote} is not available")
        self.base = base
        self.quote = quote


class ExchangeRateUnavailable(ExchangeError):
    """Transient or upstream failure. Recoverable: fall back to a human."""


def _iso(code: str) -> str:
    cleaned = (code or "").strip().upper()
    if len(cleaned) != 3 or not cleaned.isalpha():
        raise ValueError(f"not an ISO-4217 code: {code!r}")
    return cleaned


class FrankfurterClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        settings = default_settings
        self._max_retries = settings.exchange_max_retries if max_retries is None else max_retries
        self._timeout = settings.exchange_timeout_s if timeout is None else timeout
        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            self._client = httpx.Client(
                base_url=base_url or settings.exchange_api_base,
                timeout=self._timeout,
                headers={"accept": "application/json"},
                transport=transport,
            )

    # --- lifecycle -----------------------------------------------------
    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> FrankfurterClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- calls ---------------------------------------------------------
    def latest(self, base: str, quote: str) -> ExchangeRate:
        base_code, quote_code = _iso(base), _iso(quote)
        if base_code == quote_code:
            raise ValueError("base and quote must differ")

        response = self._request("/latest", params={"base": base_code, "symbols": quote_code})
        payload = response.json()
        rates = payload.get("rates") or {}
        rate = rates.get(quote_code)
        if rate is None:
            raise UnsupportedCurrencyError(
                base_code, quote_code, f"no rate returned for {quote_code} against {base_code}"
            )
        try:
            as_of = date.fromisoformat(payload["date"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExchangeRateUnavailable("upstream response missing a usable date") from exc

        return ExchangeRate(
            base=base_code,
            quote=quote_code,
            rate=float(rate),
            as_of=as_of,
            source_url=str(response.url),
        )

    def currencies(self) -> dict[str, str]:
        response = self._request("/currencies")
        payload = response.json()
        if not isinstance(payload, dict) or not payload:
            raise ExchangeRateUnavailable("currency list was empty or malformed")
        return {str(code).upper(): str(name) for code, name in payload.items()}

    # --- plumbing ------------------------------------------------------
    def _request(self, path: str, params: dict[str, str] | None = None) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.get(path, params=params)
            except httpx.TimeoutException as exc:
                last_error = exc
            except httpx.HTTPError as exc:
                raise ExchangeRateUnavailable(f"transport error: {exc}") from exc
            else:
                if response.status_code in RETRYABLE_STATUS and attempt < self._max_retries:
                    last_error = ExchangeRateUnavailable(
                        f"upstream returned {response.status_code}"
                    )
                else:
                    return self._check(response)

            # backoff: 0.2s, 0.4s, ... capped so a request cannot hang the graph
            time.sleep(min(0.2 * (2**attempt), 1.0))

        raise ExchangeRateUnavailable(f"exhausted retries: {last_error}")

    @staticmethod
    def _check(response: httpx.Response) -> httpx.Response:
        if response.status_code in (400, 404, 422):
            detail = None
            try:
                body = response.json()
                detail = body.get("message") if isinstance(body, dict) else None
            except ValueError:
                pass
            raise UnsupportedCurrencyError("?", "?", detail or f"HTTP {response.status_code}")
        if response.status_code >= 500:
            raise ExchangeRateUnavailable(f"upstream returned {response.status_code}")
        if response.status_code >= 400:
            raise ExchangeRateUnavailable(f"unexpected upstream status {response.status_code}")
        return response

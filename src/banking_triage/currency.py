"""Currency and amount extraction from free text.

Deliberately rule-based: it is auditable, deterministic, and testable, and its
failure modes are visible in the evaluation harness instead of hidden inside a
model. The hard part is not the regex, it is refusing to guess -- a wrong
currency pair is a confidently wrong answer.

`extract_mentions` finds everything; `build_query` only commits when the text
names two distinct currencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import CurrencyQuery

# --- alias tables ---------------------------------------------------------

SYMBOL_ALIASES: dict[str, str] = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
}

# Lowercase words that unambiguously name a currency. Kept to the ECB reference
# set that Frankfurter serves, plus common spoken forms.
_WORD_ALIASES: dict[str, str] = {
    "dollar": "USD",
    "dollars": "USD",
    "buck": "USD",
    "bucks": "USD",
    "usd": "USD",
    "euro": "EUR",
    "euros": "EUR",
    "eur": "EUR",
    "pound": "GBP",
    "pounds": "GBP",
    "quid": "GBP",
    "sterling": "GBP",
    "gbp": "GBP",
    "yen": "JPY",
    "jpy": "JPY",
    "franc": "CHF",
    "francs": "CHF",
    "swiss franc": "CHF",
    "swiss francs": "CHF",
    "chf": "CHF",
    "rupee": "INR",
    "rupees": "INR",
    "inr": "INR",
    "won": "KRW",
    "krw": "KRW",
    "yuan": "CNY",
    "rmb": "CNY",
    "cny": "CNY",
    "canadian dollars": "CAD",
    "canadian dollar": "CAD",
    "cad": "CAD",
    "aud": "AUD",
    "nzd": "NZD",
    "sek": "SEK",
    "nok": "NOK",
    "dkk": "DKK",
    "pln": "PLN",
    "czk": "CZK",
    "huf": "HUF",
    "zar": "ZAR",
    "brl": "BRL",
    "mxn": "MXN",
    "sgd": "SGD",
    "hkd": "HKD",
    "ils": "ILS",
}

# Currencies recognised only when the user writes the ISO code in uppercase.
# Everything here collides with an ordinary English word ("try", "won"), so
# matching them case-insensitively would manufacture currency pairs from prose.
UPPERCASE_ONLY_ISO: frozenset[str] = frozenset({"TRY", "KRW", "RON", "THB", "MYR", "PHP", "ISK"})

# Codes the ECB reference set covers; used to decide whether an uppercase
# three-letter token is plausibly a currency at all.
KNOWN_ISO: frozenset[str] = frozenset(
    {
        "AUD",
        "BGN",
        "BRL",
        "CAD",
        "CHF",
        "CNY",
        "CZK",
        "DKK",
        "EUR",
        "GBP",
        "HKD",
        "HUF",
        "IDR",
        "ILS",
        "INR",
        "ISK",
        "JPY",
        "KRW",
        "MXN",
        "MYR",
        "NOK",
        "NZD",
        "PHP",
        "PLN",
        "RON",
        "SEK",
        "SGD",
        "THB",
        "TRY",
        "USD",
        "ZAR",
    }
)

# Bare three-letter tokens are only trusted when uppercase. Lowercasing them
# would make the English word "try" mean Turkish lira.
_ISO_UPPER = re.compile(r"\b([A-Z]{3})\b")
# Multi-word aliases must be tried before single words ("swiss francs" > "francs").
_MULTIWORD = sorted((k for k in _WORD_ALIASES if " " in k), key=len, reverse=True)
_SINGLEWORD = {k: v for k, v in _WORD_ALIASES.items() if " " not in k}
_WORD_RE = re.compile(r"\b([a-z]+)\b")

_AMOUNT_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:[,\u00a0 ]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\w.])")

# A number belongs to a currency token only if nothing but whitespace or
# punctuation separates them: "$500", "500 USD", "$ 500". This keeps
# "how many quid is 200 bucks" from attaching 200 to both currencies.
MAX_AMOUNT_GAP = 3


@dataclass(frozen=True)
class Mention:
    currency: str
    amount: float | None
    position: int


def _amounts(text: str) -> list[tuple[int, int, float]]:
    found: list[tuple[int, int, float]] = []
    for match in _AMOUNT_RE.finditer(text):
        raw = match.group(1).replace(",", "").replace("\u00a0", "").replace(" ", "")
        try:
            value = float(raw)
        except ValueError:  # pragma: no cover - regex guarantees numeric
            continue
        if value > 0:
            found.append((match.start(), match.end(), value))
    return found


def _nearest_amount(
    text: str, amounts: list[tuple[int, int, float]], start: int, end: int
) -> float | None:
    """Pick the amount directly adjacent to a currency token.

    Adjacent means the intervening characters are only whitespace or
    punctuation. Anything containing a letter or digit means we have walked into
    a different phrase and must not claim the number.
    """
    best: tuple[int, float] | None = None
    for amount_start, amount_end, value in amounts:
        if amount_end <= start:
            gap = text[amount_end:start]
        elif end <= amount_start:
            gap = text[end:amount_start]
        else:
            continue
        if len(gap) > MAX_AMOUNT_GAP or any(char.isalnum() for char in gap):
            continue
        distance = len(gap)
        if best is None or distance < best[0]:
            best = (distance, value)
    return None if best is None else best[1]


def extract_mentions(text: str) -> list[Mention]:
    """Find currency tokens in order of appearance, with any adjacent amount."""
    if not text:
        return []
    amounts = _amounts(text)
    hits: list[tuple[int, int, str]] = []  # (start, end, currency)

    for symbol, iso in SYMBOL_ALIASES.items():
        for match in re.finditer(re.escape(symbol), text):
            hits.append((match.start(), match.end(), iso))

    lowered = text.lower()
    for phrase in _MULTIWORD:
        for match in re.finditer(r"\b" + re.escape(phrase) + r"\b", lowered):
            hits.append((match.start(), match.end(), _WORD_ALIASES[phrase]))

    for match in _WORD_RE.finditer(lowered):
        word = match.group(1)
        iso = _SINGLEWORD.get(word)
        if iso is None:
            continue
        # skip words already consumed by a multiword alias ("swiss francs")
        if match.group(0) in _SINGLEWORD and any(
            start <= match.start() < end for start, end, _ in hits
        ):
            continue
        hits.append((match.start(), match.end(), iso))

    for match in _ISO_UPPER.finditer(text):
        iso = match.group(1)
        if iso in KNOWN_ISO:
            hits.append((match.start(), match.end(), iso))

    hits.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    mentions: list[Mention] = []
    seen_positions: set[int] = set()
    for start, end, iso in hits:
        if start in seen_positions:
            continue
        seen_positions.add(start)
        mentions.append(
            Mention(currency=iso, amount=_nearest_amount(text, amounts, start, end), position=start)
        )
    return mentions


def build_query(text: str) -> CurrencyQuery | None:
    """Return a usable query only if the text names two distinct currencies."""
    mentions = extract_mentions(text)
    distinct: list[Mention] = []
    for mention in mentions:
        if all(mention.currency != seen.currency for seen in distinct):
            distinct.append(mention)
    if len(distinct) < 2:
        return None

    base, quote = distinct[0], distinct[1]
    # Direction follows the amount, not word order: "how many quid is 200 bucks"
    # asks to convert the 200 USD, not the 1 GBP.
    if base.amount is None and quote.amount is not None:
        base, quote = quote, base

    amount = base.amount if base.amount is not None else 1.0
    return CurrencyQuery(base=base.currency, quote=quote.currency, amount=amount)


def describe_mentions(text: str) -> list[str]:
    """Human-readable currency codes found, for clarification prompts."""
    return [mention.currency for mention in extract_mentions(text)]


# An explicit conversion request is not something the intent classifier can
# reliably recognise: BANKING77's `exchange_rate` examples are all "what are
# your rates?", with no pair or amount. "convert 250 EUR to CHF" is instead
# classified as `fiat_currency_support` with high confidence. So the presence of
# a complete pair plus a request cue is treated as stronger evidence than the
# classifier's top-1, and routed to the tool deterministically.
#
# Cues are deliberately conservative: a complaint about a rate, a fee question,
# or a "how do I do this in the app" question must keep its FAQ/handoff route.
_RATE_REQUEST_CUES = frozenset(
    {"rate", "rates", "convert", "conversion", "how much", "worth", "exchange"}
)
_NOT_A_QUOTE_CUES = frozenset(
    {
        "app",
        "charged",
        "charge",
        "fee",
        "fees",
        "wrong",
        "refund",
        "dispute",
        "complaint",
        "support",
        "available",
        "hold",
        "possible",
        "can i",
        "how do i",
        "how can i",
        "where do i",
    }
)


def _cue_pattern(phrases: frozenset[str]) -> re.Pattern[str]:
    ordered = "|".join(re.escape(phrase) for phrase in sorted(phrases, key=len, reverse=True))
    return re.compile(rf"\b(?:{ordered})\b")


_RATE_CUE_RE = _cue_pattern(_RATE_REQUEST_CUES)
_NOT_QUOTE_RE = _cue_pattern(_NOT_A_QUOTE_CUES)


def explicit_conversion_request(text: str) -> CurrencyQuery | None:
    """Return a query only for an explicit "convert this pair" request.

    This is a routing signal, not a parser change: `build_query` still decides
    whether the pair is usable. The cues decide whether a quote is the right
    *kind* of answer at all.
    """
    lowered = (text or "").lower()
    if not _RATE_CUE_RE.search(lowered) or _NOT_QUOTE_RE.search(lowered):
        return None
    return build_query(text)

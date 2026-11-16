# Banking Support Triage

Turns a free-text banking support message into one of **77 canonical intents**, then
either answers it from a curated FAQ, answers an exchange-rate question with a live
ECB reference rate, or escalates to a human — and measures how often each of those
outcomes is correct.

The classifier is a routing signal, not the product. It never writes prose, never
invents policy, and never claims an intent it is not confident about.

**Status: in progress.** M0–M5 are done: measured classifier, routing policy,
tools, multi-turn workflow, and a documented HTTP API. Evaluation harness, CI,
and the demo are next.

---

## Measured results

Trained on 8,002 BANKING77 messages, scored once on the official 3,080-message test
split. Reproduce with the commands further down; full evidence in
[`reports/m1_classifier.json`](reports/m1_classifier.json).

| Metric | Validation (2,000) | Official test (3,080) |
|---|---|---|
| Accuracy | 0.8770 | **0.8925** |
| Macro-F1 | 0.8731 | **0.8926** |
| Weighted-F1 | 0.8773 | 0.8926 |
| Top-3 accuracy | 0.9635 | **0.9685** |

Model: TF-IDF word unigrams (`min_df=1`, `sublinear_tf`, accent-stripped) →
multinomial logistic regression, `C=16`. 2,156 features, 2.0 s to fit, 1.23 MB
artifact. `scikit-learn` 1.9.1 on Python 3.11.9.

**Model selection** used 3-fold stratified CV *inside* the train split only:

| Config | C=1 | C=2 | C=4 | C=8 | C=16 | C=32 |
|---|---|---|---|---|---|---|
| word unigram — macro-F1 | 0.8345 | 0.8555 | 0.8678 | 0.8697 | **0.8716** | 0.8704 |
| word 1–2 gram — macro-F1 | 0.8096 | 0.8370 | 0.8455 | 0.8501 | 0.8504 | 0.8517 |

Two findings worth stating plainly:

- **Bigrams hurt here.** Support messages are short and lexically dense, so bigrams
  mostly add sparse noise. Unigram wins by ~2 macro-F1 points at every `C`.
- **The `C` grid was extended** because the first pass selected `C=8`, the edge of the
  grid. Re-running with `C ∈ {8, 16, 32}` found an interior optimum at 16. A boundary
  pick is not a tuned hyperparameter.

## Decision policy (M2)

Accuracy is not the product; the routing decision is. M2 refits the selected model on
train+val (9,002 messages), freezes thresholds chosen on val, and scores test once.
Evidence: [`reports/m2_calibration.json`](reports/m2_calibration.json).

| Serving model (v2, fit on train+val) | Value |
|---|---|
| Test accuracy | 0.8990 |
| Test macro-F1 | 0.8992 |
| Test ECE (10-bin) | 0.0817 |
| **Auto-routed at ≥0.95 precision** | **2,220 / 3,080 = 72.1%** |
| **Precision on auto-routed traffic** | **0.9622** |

Thresholds: `confidence ≥ 0.54` **and** `top1 − top2 ≥ 0.10`, with fraud and dispute
intents excluded regardless of confidence. That is the headline claim:

> auto-resolves 72.1% of test traffic at 96.2% precision, and clarifies or escalates
> the remaining 27.9%

The 27.9% is not all handed to a human: low-confidence and thin-margin messages get a
clarifying question first (M4), which is cheaper than a ticket and safer than a guess.
Clarifications are not counted as auto-answered, so the 72.1% claim stays conservative.
These figures come from the M2 classifier policy; M6 re-measures the full workflow,
including the explicit-pair rule and tool calls.

**Calibration was tested and rejected.** Sigmoid calibration (per-class OvR, as
`CalibratedClassifierCV` implements it for multiclass) made things substantially worse
on val — ECE 0.2194 vs 0.0825 raw, NLL 0.6466 vs 0.4925:

| Val | ECE | Brier | NLL | Mean confidence |
|---|---|---|---|---|
| Raw multinomial | **0.0825** | **0.1938** | **0.4925** | 0.794 |
| Sigmoid-calibrated | 0.2194 | 0.2561 | 0.6466 | 0.653 |

A 77-class multinomial logistic regression is already reasonably calibrated; wrapping
it in per-class one-vs-rest sigmoid calibration destroys the multinomial structure.
The measured winner ships, and the artifact records `calibrated: false` so nobody has
to guess.

**The safety rules have a measurable cost**, and it is worth stating: at the same 0.95
precision target, ignoring `NEVER_AUTO_ANSWER_INTENTS` would raise val coverage from
67.5% to 83.3%. We give up ~16 points of automation so that a fraud report never gets
an automated answer.

## Tools and content (M3)

**Currency extraction is rule-based, on purpose.** A regex failure is visible and
testable; an LLM hallucinated currency pair is not. Two rules carry most of the weight:

- An amount belongs to a currency only if nothing but whitespace or punctuation separates
  them, so "how many quid **is** 200 bucks" converts 200 USD rather than attaching `200`
  to both tokens.
- Direction follows the amount, not word order, so the same sentence yields
  `base=USD, quote=GBP`.

It refuses to answer on one currency (`None` → clarify), and it does not mistake the
English word "try" for Turkish lira — `TRY`, `KRW`, `RON`, `THB`, `MYR`, `PHP` and `ISK`
are only recognised as uppercase ISO codes.

Verified end to end against the live API:

```
"how much is 500 dollars in euros?"
→ base=USD quote=EUR amount=500
→ 500.00 USD = 452.41 EUR, as of 2026-11-13
→ https://api.frankfurter.dev/v1/latest?base=USD&symbols=EUR
```

**The client degrades instead of exploding.** Typed errors separate "this pair is not
served" (recoverable: ask for another currency) from "upstream is down" (recoverable:
hand off to a human). Transient 429/5xx responses and timeouts retry with backoff; the
whole suite runs offline through `httpx.MockTransport`.

**The FAQ corpus is 42 intent-keyed entries**, each with answer variants and a
clarifying question. Variant choice is lexical overlap with a deterministic tie-break, so
the same input always yields the same answer. The store validates itself against
`taxonomy.py` at load time and refuses to start if the two disagree.

All 42 entries ship with `review_status: "draft"`. They explain what to check and what
happens next, and deliberately assert no bank-specific timelines, fees, or limits —
invented policy is worse than a handoff. Swapping in institution-approved wording is a
content change, not a code change.

## Workflow (M4)

The LangGraph workflow is deliberately boring: one node per job, one conditional edge,
and a final node that refuses to return an empty response.

```
                        ┌─ faq ────────────┐
triage ─────────────────┼─ exchange ───────┤
  (classify + decide)   ├─ clarify ────────┼─→ compose → END
                        └─ escalate ───────┘
```

| Route | Condition | What the user gets |
|---|---|---|
| `faq` | confident, curated copy exists | intent-keyed answer + `source: faq` |
| `exchange_rate` | explicit pair, or confident `exchange_rate` with a pair | dated ECB rate + source URL |
| `clarify` | low confidence, thin margin, or missing currencies | a question that asserts nothing |
| `escalate` | fraud/dispute intent, no curated copy, or rate service down | guidance + human handoff |

**A clarification is a real turn, not a dead end.** Passing the same `session_id`
restores the checkpointed state:

```
turn 1   "what's the exchange rate?"   → clarify, pending_intent=exchange_rate
turn 2   "500 USD to EUR"              → exchange_rate, classifier not re-run
```

The classifier would read `500 USD to EUR` as some unrelated intent; the pending flag is
what makes the second turn work. Sessions are isolated, anonymous requests delete their
checkpoint afterwards, and a terse reply (≤6 words) is classified together with the
previous message so "the virtual one" has context.

**An explicit pair can outrank the classifier, and the evidence says it must.** With the
real artifact:

| Message | Classifier top-1 | Confidence |
|---|---|---|
| `convert 250 EUR to CHF` | `fiat_currency_support` | 0.911 |
| `how much is 500 dollars in euros?` | `fiat_currency_support` | 0.067 |
| `what's the rate for GBP to JPY` | `exchange_rate` | 0.384 (below 0.40) |

BANKING77's `exchange_rate` examples are all "what are your rates?" with no pair or
amount, so slot-filled conversion requests are out of distribution. The router therefore
treats a complete pair plus a request cue (`convert`, `how much`, `rate`, …) as stronger
evidence than the classifier's top-1 — but only when no complaint or how-to cue is
present. `I was charged the wrong rate on a 500 USD to EUR payment` stays a dispute, and
`I want to exchange USD to EUR in the app` stays an `exchange_via_app` FAQ answer. Fraud
and dispute intents always win. This is a routing rule, not a model change; M6 measures
it on a probe set.

**Failures are answers, not exceptions.** The graph never 500s because a tool failed:

| Failure | Behaviour |
|---|---|
| Pair not served by Frankfurter | ask for another pair, keep the intent pending, record error |
| Rate service down | escalate and say the rate was not guessed, record error |
| FAQ corpus drifts from taxonomy | escalate instead of inventing copy, record error |
| Whitespace-only input | clarify instead of raising from the vectorizer |

Every degradation lands in `state.errors`, and `route_reason` records *why* a route was
chosen (`explicit_currency_pair`, `protected_intent`, `ambiguous`, `tool_unavailable`, …)
so a rule-based override never looks like a confident model decision in an audit.

Live run against the real artifact and the live API (2026-11-16):

```
how much is 500 dollars in euros?   → 500.00 USD = 453.97 EUR  (explicit pair)
convert 250 EUR to CHF              → 250.00 EUR = 231.10 CHF  (explicit pair)
I was charged the wrong exchange…   → escalate, no quote        (protected intent)
I want to exchange USD to EUR in…   → faq                        (exchange_via_app)
what's the exchange rate?           → clarify → "500 USD to EUR" → rate
```

The 111 graph, tool, and schema tests all run offline, including every route, every
degradation path, and the multi-turn workflow.

## HTTP API (M5)

| Endpoint | Purpose |
|---|---|
| `POST /triage` | one turn: `{text, session_id?}` → decision, answer, sources, `route_reason`, latency |
| `GET /healthz` | 200 `ok`, or 503 `degraded` when the model or FAQ failed to load |
| `GET /model` | artifact metadata: version, metrics, thresholds, calibration flag |
| `GET /intents` | all 77 intents with `faq_covered` and `auto_answerable` flags |

```powershell
uvicorn banking_triage.api:app --port 8000

Invoke-RestMethod -Uri http://127.0.0.1:8000/triage -Method Post `
  -ContentType application/json -Body '{"text":"convert 250 EUR to CHF"}'
```

Response (abridged):

```json
{
  "intent": "fiat_currency_support",
  "confidence": 0.911,
  "route": "exchange_rate",
  "route_reason": "explicit_currency_pair",
  "answer": "250.00 EUR = 231.10 CHF at the ECB reference rate for 2026-11-16 ...",
  "sources": [{"kind": "exchange_rate", "ref": "https://api.frankfurter.dev/v1/latest?base=EUR&symbols=CHF"}],
  "needs_human": false,
  "latency_ms": 278.2
}
```

That response is the M4 rule made visible: the classifier's top-1 and the router's
decision disagree, and `route_reason` says why.

Operational behaviour, all covered by tests:

- **A missing artifact does not crash the process.** Startup records the error,
  `/healthz` reports `degraded` with 503, and `/triage` returns 503 instead of a stack
  trace. A deploy probe can distinguish "starting" from "broken".
- **Every non-2xx response is an RFC 9457 problem document** (`application/problem+json`)
  with a `request_id`, including FastAPI's own validation errors. Unhandled errors are
  logged and returned as a generic 500 with no internals.
- **`X-Request-ID` is echoed** when supplied, generated otherwise, and appears in the body.
- **`/triage` is a sync endpoint on purpose.** The classifier is CPU-bound and the
  exchange client is synchronous, so FastAPI's threadpool is the right execution model;
  an async endpoint would block the event loop.
- **Services are injectable** (`create_app(services=...)`), so the 12 API tests run with
  no artifact, no FAQ file, and no network. Suite total: 123 tests.

Measured locally: an FAQ answer took ~23 ms and a protected-intent escalation ~22 ms; a
live rate lookup took 278 ms including TLS on a cold connection. M6 will report
p50/p95/p99 over the evaluation set instead of anecdotes.

One deployment limit is worth stating: sessions use `InMemorySaver`, so clarification
state lives in a single process. Multiple workers would need a shared checkpointer
(Redis/Postgres) before this could run behind a load balancer. That is deliberately out
of scope here.

## What it does

```
POST /triage {text, session_id?}
   │
 triage ── classifier → {intent, confidence, top-3}
   │
   ├── faq        confident + curated copy
   ├── exchange   explicit pair, or confident exchange intent with a pair
   ├── clarify    low confidence / thin margin / missing currencies
   └── escalate   fraud, dispute, no curated copy, or tool failure
                       │
                   compose (invariant check) → response
```

Routing policy lives in one auditable file, [`taxonomy.py`](src/banking_triage/taxonomy.py):

- `exchange_rate` triggers a live rate lookup; so does an explicit conversion request
  that names a pair (the M4 rule above).
- `exchange_charge`, `card_payment_wrong_exchange_rate`,
  `wrong_exchange_rate_for_cash_withdrawal` and friends are *complaints about rates*.
  They never go to the tool: they get curated guidance and a human handoff.
- Fraud and dispute intents (`compromised_card`, `lost_or_stolen_card`,
  `transaction_charged_twice`, …) are never auto-answered, regardless of confidence.
- FAQ coverage is 42 of 77 intents. The rest get a generic acknowledgement plus a
  handoff. A confident wrong answer is worse than a handoff.

## Data protocol

| Split | Rows | Role |
|---|---|---|
| train | 8,002 | model fitting + CV |
| val | 2,000 | 20% of the official train split; thresholds, calibration |
| test | 3,080 | official BANKING77 test, scored once |

Three things the data actually does, contrary to what you might assume:

1. **BANKING77's train split is imbalanced** (28–149 messages per intent), while its
   test split has exactly 40 per intent. So macro-F1 is the honest headline metric,
   and val class frequencies track train proportionally rather than equally.
2. **5 test messages also appear in train** (0.16%), all with the same label. These
   are upstream duplicates. The official split is kept verbatim so numbers stay
   comparable with published baselines; the overlap is recorded in
   `data/processed/summary.json` and the training run refuses to proceed if it grows.
3. **1 val message duplicated a train message** and was dropped, because val is ours
   to construct and it must be genuinely held out.

## Reproduce

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

python -m banking_triage.data       # official split → data/processed/*.parquet
python -m banking_triage.train      # CV + fit → v1 artifact
python -m banking_triage.calibrate  # thresholds + refit on train+val → v2 artifact
pytest                              # 123 tests, all offline
ruff check src tests
uvicorn banking_triage.api:app --reload   # docs at http://127.0.0.1:8000/docs
```

`python -m banking_triage.data --force` rebuilds from the HuggingFace hub. The model
artifact is gitignored — it is reproducible from the two commands above.

## Design decisions

- **No LLM in the loop.** A generative model would make every claim about the system
  unfalsifiable. Curated FAQ text plus a deterministic classifier means the same input
  always produces the same answer, and evaluation means something.
- **Calibration before thresholds.** Raw logistic-regression probabilities are usable
  but not trustworthy on out-of-vocabulary text ("hi", "help"). M2 measures ECE and
  picks the auto-route threshold from the validation split, targeting ≥0.95 precision
  on auto-answered traffic.
- **Confidence is not enough on its own.** The top-1/top-2 margin matters: `top_up_*`
  and `exchange_*` siblings produce near-ties that a confidence threshold alone lets
  through.
- **Answer provenance.** Every response carries `sources: [{kind, ref}]` pointing at an
  FAQ id or an API URL, so a reviewer can check the answer was retrieved, not invented.
  The graph also records `route_reason`, so a rule-based route and a model-based route
  are distinguishable.
- **Rules may outrank the classifier when the evidence is structural.** Two currencies
  and an amount are not a lexical hint; they are the request. The override is narrow,
  cue-guarded, and impossible to trigger for a fraud or dispute intent.
- **Exchange rates are dated, never "today".** Frankfurter serves ECB reference rates
  on business days only, so answers say "as of 2026-11-13". Implied live quotes would
  be wrong in a way that matters to a bank.

## Roadmap

| Milestone | Status |
|---|---|
| M0 — scaffold, contracts, licensing | done |
| M1 — data protocol, baseline classifier | done — 0.8926 test macro-F1 (fit on train) |
| M2 — calibration, auto-route thresholds | done — 72.1% routed at 96.2% precision |
| M3 — curated FAQ store, Frankfurter client | done — 42 FAQ entries, live rate verified |
| M4 — LangGraph workflow + clarification turns | done — 5 routes, session resume |
| M5 — FastAPI surface (`/triage`, `/healthz`, `/model`, `/intents`) | done — problem details, 123 tests |
| M6 — evaluation harness (routing, tool calls, latency) | next |
| M7 — CI, Docker, Streamlit demo | planned |

Headline number the project exists to produce is now measured: **deflection rate at a
stated precision**. Everything else in the metrics table exists to prove that number is
trustworthy.

## Scope

Out of scope by design: authentication, PII handling, real account lookups, ticket
creation, and money movement. This is a triage and routing layer over public data.

## Attribution

- [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI), CC BY 4.0.
  Not redistributed here; fetched from the HuggingFace hub.
- [Frankfurter](https://frankfurter.dev) — European Central Bank reference rates.
- Code: MIT, see [LICENSE](LICENSE).

# Banking Support Triage

[![CI](https://github.com/gufrxn999/banking-support-triage/actions/workflows/ci.yml/badge.svg)](https://github.com/gufrxn999/banking-support-triage/actions/workflows/ci.yml)

Turns a free-text banking support message into one of **77 canonical intents**, then
either answers it from a curated FAQ, answers an exchange-rate question with a live
ECB reference rate, asks a clarifying question, or escalates to a human — and measures
how often each of those outcomes is correct.

The classifier is a routing signal, not the product. It never writes prose, never
invents policy, and never claims an intent it is not confident about.

**Status: complete.** M0–M8 are done: measured classifier, routing policy, tools,
multi-turn workflow, HTTP API, evaluation harness, CI, container image, a Streamlit
demo, and a deployable showcase site served by the same FastAPI app. Every number
below is reproducible from the commands in [Reproduce](#reproduce); the badge reports
the latest CI run.

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
  pick is not a tuned hyperparameter. `python -m banking_triage.train` now defaults to
  the full grid above, so a clean rebuild selects `C=16`; CI compares the rebuilt
  reports with the committed ones and fails on any drift.

## Decision policy (M2)

Accuracy is not the product; the routing decision is. M2 refits the selected model on
train+val (9,002 messages), freezes thresholds chosen on val, and scores test once.
Evidence: [`reports/m2_calibration.json`](reports/m2_calibration.json).

| Serving model (v2, fit on train+val) | Value |
|---|---|
| Test accuracy | 0.8990 |
| Test macro-F1 | 0.8992 |
| Test ECE (10-bin) | 0.0817 |
| **Classifier policy: routable at ≥0.95 precision** | **2,220 / 3,080 = 72.1%** |
| **Precision on that routable subset** | **0.9622** |

Thresholds: `confidence ≥ 0.54` **and** `top1 − top2 ≥ 0.10`, with fraud and dispute
intents excluded regardless of confidence. This measures the **classifier policy**: any
confident message counts as routable, whether or not reviewed copy exists for its intent.

> the classifier policy can auto-route 72.1% of test traffic at 96.2% precision

That is a ceiling, not the shipped deflection rate. M6 routes the same test split through
the full workflow and measures **37.2% auto-answered at 95.1% precision**; the difference
is traffic whose confident intent has no reviewed answer yet. Both numbers are reported,
because collapsing them into one would overstate what the product does.

The 27.9% the classifier does not route is not all handed to a human either: low-confidence
and thin-margin messages get a clarifying question first (M4), which is cheaper than a
ticket and safer than a guess. Clarifications are not counted as auto-answered, so neither
coverage figure includes them.

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

The 158 tests all run offline (the model-backed ones skip cleanly when the artifact is
absent), including every route, every degradation path, the multi-turn workflow, the HTTP
contract, the showcase page, and the evaluation arithmetic.

## HTTP API (M5)

| Endpoint | Purpose |
|---|---|
| `GET /` | showcase page: live demo, workflow, measured results, intent explorer |
| `POST /triage` | one turn: `{text, session_id?}` → decision, answer, sources, `route_reason`, trace, latency |
| `GET /healthz` | 200 `ok`, or 503 `degraded` when the model or FAQ failed to load |
| `GET /model` | artifact metadata: version, metrics, thresholds, calibration flag |
| `GET /intents` | all 77 intents with `faq_covered` and `auto_answerable` flags |
| `GET /static/*` | page assets: `styles.css`, `app.js`, `metrics.json` |
| `GET /docs` | OpenAPI schema and interactive try-out |

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
  "currency_query": {"base": "EUR", "quote": "CHF", "amount": 250.0},
  "steps": ["triage", "exchange", "compose"],
  "errors": [],
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
- **Services are injectable** (`create_app(services=...)`), so the 14 API tests run with
  no artifact, no FAQ file, and no network. Suite total: 158 tests.

A single request measured ~23 ms for an FAQ answer and ~278 ms for a live rate lookup
including TLS on a cold connection. M6 measures the distribution over 3,080 turns, with
the network excluded so the number is about the workflow rather than the internet.

One deployment limit is worth stating: sessions use `InMemorySaver`, so clarification
state lives in a single process. Multiple workers would need a shared checkpointer
(Redis/Postgres) before this could run behind a load balancer. That is deliberately out
of scope here.

## Evaluation (M6)

Two measurements, kept separate because they answer different questions. Evidence:
[`reports/m6_evaluation.json`](reports/m6_evaluation.json), reproduced by
`python -m banking_triage.evaluate`.

**1. Probe set — does the policy do what it claims?** 24 hand-written cases over 27 turns,
covering what BANKING77 cannot express: slot-filled conversions, spoken currency aliases,
a fee question that mentions a pair, a rate complaint that mentions both a pair and an
amount, an unsupported currency, and three clarification sessions. The exchange tool is
stubbed with a fixed rate, so the run is reproducible and offline.

| Probe result | Value |
|---|---|
| Probes / turns | 24 / 27 |
| Passed | **24 / 24** |
| Route accuracy | 100% |
| Tool calls expected / made / correct | 8 / 8 / 8 |
| Tool precision / recall | **1.00 / 1.00** |

The tool metrics are the useful part: precision punishes quoting when it should not,
recall punishes asking when it should quote. Both are 1.00. The eight expected calls
include one deliberate failure: the unsupported-pair probe expects the tool to be
attempted, then degraded into a clarification. The two hardest suppression cases —
`I was charged the wrong rate on a 500 USD to EUR payment` and
`what fee applies if I convert 500 USD to EUR?` — both name a pair and neither calls the
tool. The probe file states its expectation for every case, including
`"tool": null`, so "did not call" is asserted rather than inferred from silence.

**2. Official test split — what does the shipped workflow actually resolve?** All 3,080
test messages are routed through the same graph the API serves, with the exchange tool
stubbed.

| Outcome | Messages | Share |
|---|---|---|
| **Auto-answered** | **1,147** | **37.2%** |
| — of which correct | 1,091 | 35.4% |
| Clarified (asks a question, no human yet) | 464 | 15.1% |
| Escalated — confident intent, no reviewed copy | 1,033 | 33.5% |
| Escalated — fraud or dispute intent | 436 | 14.2% |

Precision on auto-answered traffic is **95.1%** (1,091/1,147), against a 95% target.
Latency over the same 3,080 turns: **p50 5.8 ms, p95 6.9 ms, p99 7.9 ms** (in-process,
single thread, network excluded).

**Why 37.2% and not 72.1%?** Because they measure different things, and the harness shows
the difference is content, not model quality:

- Recomputing the M2 policy on the same test rows reproduces it exactly: **2,220 / 3,080
  = 72.1% at 96.2% precision**. The classifier really can support 72%.
- The shipped workflow only auto-answers intents with reviewed copy. 1,033 confident
  messages land on intents that have none, so they escalate instead of improvising.
- The other 436 are protected fraud/dispute intents, which are never auto-answered.

So the binding constraint on deflection is the 42-intent FAQ corpus, not classification
accuracy. That gap is a content roadmap with a measurable ceiling, and the report states
it rather than hiding it behind the higher classifier number.

**Safety is measured two ways, because one of them would overstate it.** No message whose
*predicted* intent is protected was auto-answered (0/3,080, enforced by construction).
But 2 messages whose *gold* label is protected were auto-answered because the classifier
chose a safe neighbouring intent: `I can't transfer money from my account.`
(`declined_transfer` → `transfer_into_account`) and `Is there a fee for exchanging cash?`
(`wrong_exchange_rate_for_cash_withdrawal` → `exchange_charge`). Both got adjacent,
on-topic copy rather than a resolution, and both are listed in the report.

**The 56 wrong auto-answers are mostly neighbour confusions, not topic changes.** The most
common pairs are `fiat_currency_support → exchange_via_app`,
`top_up_failed → top_up_reverted`, `verify_my_identity → why_verify_identity`, and
`card_arrival → card_delivery_estimate`. This is exactly why M3 added the most-confused
neighbour of each risky intent to the corpus: a wrong answer that is still on-topic is a
smaller failure than a generic handoff.

**The explicit-pair rule barely fires on BANKING77** — exactly 1 of 3,080 test messages
reached the tool through that rule. That is the point of the probe set: the dataset cannot
measure the rule, and a green test-split number alone would have hidden that. The probes
carry the evidence, so the rule is tested independently of the corpus.

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
python -m banking_triage.evaluate   # probes + test split → reports/m6_evaluation.json
python -m banking_triage.web_metrics  # reports → web/metrics.json (--check in CI)
pytest                              # 158 tests, all offline
ruff check src tests
uvicorn banking_triage.api:app --reload   # site at /, docs at /docs

pip install -e ".[demo]"                  # optional: the Streamlit page
streamlit run streamlit_app.py            # http://localhost:8501
```

`python -m banking_triage.data --force` rebuilds from the HuggingFace hub. The model
artifact is gitignored — it is reproducible from the two commands above.

## Serving it (M7)

**Container.** The image trains its own model at build time, so it is self-contained: the
artifact cannot drift from the code that serves it, and `/healthz` is never degraded
because someone forgot a build step. It serves both the API and the showcase page at `/`.

```bash
docker build -t banking-triage .
docker run --rm -p 8000:8000 banking-triage
```

It runs as a non-root user, declares a `HEALTHCHECK` against `/healthz`, installs
dependencies before copying source so the expensive layer is cached, and excludes
`.venv`, git, artifacts, and processed data from the build context.

**Compose** runs the API and the demo from the same image:

```bash
docker compose up --build
# API   http://localhost:8000/docs
# demo  http://localhost:8501
```

**Demo.** `pip install -e ".[demo]"` then `streamlit run streamlit_app.py`. The page is a
thin shell over the same `TriageGraph` — same artifact, FAQ corpus, routing policy, and
session store as the API. Each turn shows the answer, the route, the predicted intent,
its confidence, and an expandable trace with `route_reason` and `sources`, so the demo
makes the behaviour auditable rather than just presentable. Sidebar buttons submit
representative prompts; typing continues the same `session_id`, so a clarification can be
answered on the next turn.

Verified end to end in the browser: an FAQ turn (`change_pin` at 0.97), a live quote
(250 EUR = 231.10 CHF as of 2026-11-16), and a clarification followed by a bare
`500 USD to EUR` that resolved without re-classifying. The demo's `main()` is
Streamlit-only; the state-to-view mapping and route labels are pure functions with their
own tests, so the suite still runs without Streamlit installed.

**CI** (`.github/workflows/ci.yml`) has three jobs:

| Job | What it does |
|---|---|
| `quality` | ruff check/format, stale-metrics check, full test suite on Python 3.11 **and** 3.12 |
| `reproduce` | installs the lock, rebuilds data/model, evaluates, and fails if the reports differ from the committed ones |
| `image` | builds the image, asserts `C=16` and `0.54` in `/model`, smoke-tests `/healthz`, `/triage`, `/`, `/static/metrics.json`, and the Streamlit health endpoint |

The image job is the container's only validation: Docker is not installed on the
development machine, so the build either passes in CI or it does not pass at all.

## Showcase website (M8)

The site is part of the application, not a separate frontend deployment. FastAPI serves
it at `/` from [`src/banking_triage/web/`](src/banking_triage/web/), and the page calls
the public `/triage` endpoint documented above. One container, one URL, no CDN, no build
step, no API key.

The demo is interactive rather than a recording:

- typing a message (or picking an example) sends it to the live workflow;
- every answer carries the route, intent, confidence, and measured latency;
- the trace panel reconstructs the turn from the response: the top-3 classifier scores
  against the real threshold, the decision and `route_reason` in plain language, the
  tool call (`currency_query`), sources, degraded-tool errors, and executed `steps`;
- the session supports clarification: `what's the exchange rate?` followed by
  `500 USD to EUR` resolves the follow-up without re-classifying it;
- all 77 intents are browsable, with filters for curated copy, human-only, and no copy;
- health, model version, and serving thresholds come from `/healthz` and `/model`;
- the headline figures load from `/static/metrics.json`, which is generated from the
  committed reports by `python -m banking_triage.web_metrics`; CI fails if it is stale.

Browser-verified against the local API: an FAQ turn (`change_pin` at 0.97), a dated
quote (250 EUR = 231.10 CHF as of 2026-11-16), a missing-pair clarification followed by
a bare `500 USD to EUR`, and the intent search/filter counts (77 / 42 / 12 / 35). The
client inserts API text with `textContent`, and the app serves only its own static
assets.

### Deploy it

The repository includes a Render blueprint ([`render.yaml`](render.yaml)):

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/gufrxn999/banking-support-triage)

Render builds the Dockerfile, trains the model during the build, waits for `/healthz`,
and serves the site at `/`. The free plan is enough for a demo; first builds take a few
minutes, sleeping instances take a few seconds to wake, and an in-memory clarification
session does not survive a restart (the deployment limit noted above).

Any Docker host works the same way. The image honors an injected `$PORT`, so Render,
Railway, Fly, or a plain VM can assign the port:

```bash
docker build -t banking-triage .
docker run --rm -e PORT=8000 -p 8000:8000 banking-triage
# site http://localhost:8000/ · docs http://localhost:8000/docs
```

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
  and an amount are not a lexical hint; they are the request. The override is narrow and
  cue-guarded, and a *confident* fraud or dispute prediction still suppresses it.
- **The reported number is the shipped one.** The 72.1% classifier policy is a ceiling;
  M6 measures what the workflow actually answers (37.2%) and decomposes the rest. Quoting
  the higher number as the product's deflection rate would be the easiest lie to tell
  here, and the probe set exists to make the difference visible.
- **Exchange rates are dated, never "today".** Frankfurter serves ECB reference rates
  on business days only, so answers say "as of 2026-11-13". Implied live quotes would
  be wrong in a way that matters to a bank.
- **The demo, the website, and the image do not re-implement the workflow.** The
  Streamlit page and the showcase client call the same `TriageGraph`/API, and the image
  trains the artifact it serves. There is one code path, and CI exercises it through the
  container.
- **The website is a client, not a mock.** Every answer on the page is a real API
  response with its executed `steps`; there is no canned response path that can drift
  from the backend. Its evidence block is generated from the reports, and CI rejects a
  stale copy.

## Roadmap

| Milestone | Status |
|---|---|
| M0 — scaffold, contracts, licensing | done |
| M1 — data protocol, baseline classifier | done — 0.8926 test macro-F1 (fit on train) |
| M2 — calibration, auto-route thresholds | done — classifier policy 72.1% at 96.2% |
| M3 — curated FAQ store, Frankfurter client | done — 42 FAQ entries, live rate verified |
| M4 — LangGraph workflow + clarification turns | done — 5 routes, session resume |
| M5 — FastAPI surface (`/triage`, `/healthz`, `/model`, `/intents`) | done — problem details, injectable services |
| M6 — evaluation harness (routing, tool calls, latency) | done — 24/24 probes, 37.2% at 95.1% |
| M7 — CI, Docker, Streamlit demo | done — CI matrix, image smoke tests, browser-verified demo |
| M8 — showcase website, deployment config | done — FastAPI-served interactive site, Render blueprint, container smoke tests |

The headline number the project exists to produce is now measured: **deflection rate at a
stated precision, and the gap between what the classifier can support (72.1%) and what
reviewed content actually resolves (37.2%)**. Everything else in the metrics tables exists
to prove those numbers are trustworthy.

## Scope

Out of scope by design: authentication, PII handling, real account lookups, ticket
creation, and money movement. This is a triage and routing layer over public data.

## Attribution

- [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI), CC BY 4.0.
  Not redistributed here; fetched from the HuggingFace hub.
- [Frankfurter](https://frankfurter.dev) — European Central Bank reference rates.
- Code: MIT, see [LICENSE](LICENSE).

# Banking Support Triage

Turns a free-text banking support message into one of **77 canonical intents**, then
either answers it from a curated FAQ, answers an exchange-rate question with a live
ECB reference rate, or escalates to a human — and measures how often each of those
outcomes is correct.

The classifier is a routing signal, not the product. It never writes prose, never
invents policy, and never claims an intent it is not confident about.

**Status: in progress.** M0–M2 are done and measured (below). The FAQ store,
Frankfurter client, LangGraph workflow, and HTTP API are next.

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

> auto-resolves 72.1% of test traffic at 96.2% precision, escalating the remaining 27.9%

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

## What it does

```
POST /triage {text}
   │
 triage ── classifier → {intent, confidence, top-3, decision}
   │
   ├── confidence < τ_escalate ──→ clarify (offer top-3 as a question, hand off)
   ├── intent = exchange_rate ───→ extract base/quote → Frankfurter → compose
   └── otherwise ───────────────→ curated FAQ for that intent → compose
```

Routing policy lives in one auditable file, [`taxonomy.py`](src/banking_triage/taxonomy.py):

- Only `exchange_rate` triggers a live rate lookup.
- `exchange_charge`, `card_payment_wrong_exchange_rate`,
  `wrong_exchange_rate_for_cash_withdrawal` and friends are *complaints about rates*.
  Answering them with a quote would be wrong, so they go to the FAQ path.
- Fraud and dispute intents (`compromised_card`, `lost_or_stolen_card`,
  `transaction_charged_twice`, …) are never auto-answered, regardless of confidence.
- FAQ coverage is 36 of 77 intents. The rest get a generic acknowledgement plus a
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

python -m banking_triage.data      # ~5 k rows → data/processed/*.parquet
python -m banking_triage.train     # CV + final fit → artifacts/model.joblib
pytest
ruff check src tests
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
- **Exchange rates are dated, never "today".** Frankfurter serves ECB reference rates
  on business days only, so answers say "as of 2026-11-13". Implied live quotes would
  be wrong in a way that matters to a bank.

## Roadmap

| Milestone | Status |
|---|---|
| M0 — scaffold, contracts, licensing | done |
| M1 — data protocol, baseline classifier | done — 0.8926 test macro-F1 (fit on train) |
| M2 — calibration, auto-route thresholds | done — 72.1% routed at 96.2% precision |
| M3 — curated FAQ store, Frankfurter client | next |
| M4 — LangGraph workflow + clarification turns | planned |
| M5 — FastAPI surface (`/triage`, `/healthz`, `/model`) | planned |
| M6 — evaluation harness (routing, tool calls, latency) | planned |
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

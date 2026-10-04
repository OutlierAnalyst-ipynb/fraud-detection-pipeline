# Card Transaction Fraud Detection Pipeline

An end-to-end ML system modelled on how a large card issuer (think JPMC / Wells Fargo
card fraud teams) scores transactions: raw data → data contracts → point-in-time
features → model selection → cost-based thresholding → registry → batch scoring → drift monitoring.

Data is synthetic (bank data is never public) but simulates real fraud patterns:
account-takeover bursts, new devices, high-risk merchants, foreign/night activity, plus
legit noise (travel, new phones) so the problem is not trivial.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install
make all        # generate -> train -> score -> monitor  (~30s)
make test
```

## Architecture

```
data/raw/*.csv
   │  data/validate.py      schema, nulls, ranges, duplicate IDs  (fail fast)
   ▼
features/build.py           point-in-time velocity features (no leakage)
   │                        txn_count_1h/24h, amount_sum_24h, secs_since_last,
   │                        hist_amount_mean, is_new_device, is_foreign, ...
   ▼
models/train.py             out-of-time split: train | valid | test
   │                        candidates: LogisticRegression vs HistGradientBoosting
   │                        select on PR-AUC (not ROC-AUC: 1% prevalence)
   │                        threshold = max $ savings within analyst capacity
   ▼
models/registry.py          versioned model + metadata + reference snapshot
   │                        champion/challenger promotion -> PRODUCTION pointer
   ▼
inference/batch_score.py    scores new txns (with 30-day lookback for velocity features)
   ▼
monitoring/drift.py         PSI per feature vs. training reference -> OK / WARN / ALERT
```

## Project layout

```
configs/config.yaml        every knob: paths, split dates, costs, alert capacity, PSI limits
src/fraud/
  cli.py                   entrypoint: generate | train | score | monitor | all
  data/generate.py         synthetic transactions + customer master
  data/validate.py         data contracts
  features/build.py        feature engineering
  models/train.py          training + selection + registration
  models/evaluate.py       PR-AUC, recall@1% alerts, $ recall, net savings, threshold search
  models/registry.py       file-based registry (swap for MLflow later)
  inference/batch_score.py batch scoring job
  monitoring/drift.py      PSI drift
tests/                     unit tests (incl. leakage tests) + end-to-end smoke test
```

## Concepts this project teaches

1. **Leakage-safe features.** `tests/test_features.py` proves velocity counts exclude the
   current transaction and history means only use the past. This is the bug that most often
   makes a fraud model look great offline and fail in production.
2. **Time-based splits.** Random K-fold on transactions leaks future fraud patterns into training.
3. **Right metric for imbalance.** PR-AUC and recall at a fixed alert rate, not accuracy/ROC-AUC.
4. **Business thresholding.** The threshold maximises `$ fraud caught − $ review cost` subject
   to `max_alert_rate` (the fraud ops team can only review so many cases).
5. **Champion/challenger.** A new model only goes live if it beats production on the test window.
6. **Train/serve consistency.** The exact sklearn `Pipeline` (encoders included) is serialised.
7. **Drift monitoring.** The "new month" file has a deliberate spending shift; `make monitor`
   should flag the amount features.

## Practice exercises (roughly easy → hard)

- Add a `distinct_merchants_24h` and `distinct_countries_24h` feature and a leakage test for each.
- Replace the file registry with **MLflow** tracking + model registry.
- Add **SHAP** explanations for the top alerts in `scored_transactions.csv`.
- Try `imbalanced-learn` (SMOTE / undersampling) and compare against `class_weight`.
- Make the data harder: lower burst size, more legit new devices, and see what breaks.
- Wrap the stages as an **Airflow / Prefect** DAG (one task per CLI command).
- Build a real-time path: Kafka/Redis keeps rolling per-customer state; a FastAPI endpoint
  scores a single swipe in <50ms using that state.
- Add delayed-label backtesting: labels for last month "arrive" later; compute live PR-AUC.

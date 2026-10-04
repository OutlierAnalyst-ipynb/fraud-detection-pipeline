"""Pipeline entrypoint.

    python -m fraud.cli generate   # create synthetic raw data
    python -m fraud.cli train      # validate -> features -> train -> register
    python -m fraud.cli score      # batch-score new transactions with PRODUCTION model
    python -m fraud.cli monitor    # PSI drift: new data vs training reference
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fraud.config import get_logger, load_config
from fraud.data.generate import generate_dataset
from fraud.data.validate import load_raw
from fraud.features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features
from fraud.inference.batch_score import score_batch, write_scores
from fraud.models.registry import ModelRegistry
from fraud.models.train import run_training
from fraud.monitoring.drift import drift_report

log = get_logger("fraud.cli")


def cmd_generate(cfg):
    d = cfg["data"]
    n_new_days = 30
    # Simulate n_days of history + 30 "future" days; the future month is held out
    # as the batch to score / monitor (same customers, so profiles stay consistent).
    tx, cust = generate_dataset(d["n_customers"], d["n_days"] + n_new_days,
                                d["start_date"], d["fraud_rate"], cfg["seed"])
    cutoff = pd.Timestamp(d["start_date"]) + pd.Timedelta(days=d["n_days"])
    hist, new = tx[tx["timestamp"] < cutoff], tx[tx["timestamp"] >= cutoff].copy()

    Path(cfg["paths"]["raw_transactions"]).parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(cfg["paths"]["raw_transactions"], index=False)
    cust.to_csv(cfg["paths"]["raw_customers"], index=False)
    log.info("wrote %d historical transactions (fraud rate %.4f), %d customers",
             len(hist), hist["is_fraud"].mean(), len(cust))

    # Inject a spending shift into the new month (e.g. holiday season) so the
    # drift monitor has something to catch. Labels are dropped: in real life
    # chargebacks/fraud confirmations arrive weeks later.
    new["amount"] = (new["amount"] * 1.6).round(2)
    new.drop(columns="is_fraud").to_csv(cfg["paths"]["scoring_input"], index=False)
    log.info("wrote %d new unlabeled transactions for scoring", len(new))


def cmd_train(cfg):
    tx, cust = load_raw(cfg["paths"]["raw_transactions"], cfg["paths"]["raw_customers"])
    log.info("validated raw data: %d transactions", len(tx))
    feats = build_features(tx, cust)
    Path(cfg["paths"]["features"]).parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(cfg["paths"]["features"], index=False)
    log.info("built %d feature rows -> %s", len(feats), cfg["paths"]["features"])
    run_training(feats, cfg)


def _load_with_lookback(cfg, lookback_days: int = 30):
    """New transactions + the last N days of history, so velocity/device features are correct."""
    new, cust = load_raw(cfg["paths"]["scoring_input"], cfg["paths"]["raw_customers"],
                         require_label=False)
    hist, _ = load_raw(cfg["paths"]["raw_transactions"], cfg["paths"]["raw_customers"])
    start = new["timestamp"].min()
    hist = hist[hist["timestamp"] >= start - pd.Timedelta(days=lookback_days)]
    combined = pd.concat([hist.drop(columns="is_fraud"), new], ignore_index=True)
    return combined, cust, start


def cmd_score(cfg):
    tx, cust, start = _load_with_lookback(cfg)
    scored, _ = score_batch(tx, cust, cfg["paths"]["registry"], score_from=start)
    write_scores(scored, cfg["paths"]["scoring_output"])
    log.info("scores -> %s", cfg["paths"]["scoring_output"])


def cmd_monitor(cfg):
    _, meta, reference = ModelRegistry(cfg["paths"]["registry"]).load()
    tx, cust, start = _load_with_lookback(cfg)
    current = build_features(tx, cust)
    current = current[current["timestamp"] >= start]
    rep = drift_report(reference, current, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
                       cfg["monitoring"]["psi_warn"], cfg["monitoring"]["psi_alert"])
    print(rep.to_string(index=False))
    out = Path(cfg["paths"]["reports"]) / "drift_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model_version": meta["version"],
                               "features": rep.to_dict(orient="records")}, indent=2))
    n_alert = (rep["status"] == "ALERT").sum()
    if n_alert:
        log.warning("%d feature(s) in ALERT drift -> consider retraining", n_alert)


def main():
    p = argparse.ArgumentParser(description="Card fraud detection ML pipeline")
    p.add_argument("command", choices=["generate", "train", "score", "monitor", "all"])
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    steps = {"generate": cmd_generate, "train": cmd_train,
             "score": cmd_score, "monitor": cmd_monitor}
    for name in (steps if args.command == "all" else [args.command]):
        log.info("===== %s =====", name.upper())
        steps[name](cfg)


if __name__ == "__main__":
    main()

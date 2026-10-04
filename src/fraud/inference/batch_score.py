"""Batch scoring job (e.g. nightly / hourly via Airflow).

Note: velocity features need each customer's history, so in production you'd
pull a lookback window from the feature store.  Here we score a file that
contains the lookback + new transactions and only emit scores for the new ones.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud.config import get_logger
from fraud.features.build import FEATURES, build_features
from fraud.models.registry import ModelRegistry

log = get_logger(__name__)


def score_batch(tx: pd.DataFrame, customers: pd.DataFrame, registry_root: str,
                score_from: str | None = None) -> tuple[pd.DataFrame, dict]:
    model, meta, _ = ModelRegistry(registry_root).load()
    feats = build_features(tx, customers)
    if score_from:
        feats = feats[feats["timestamp"] >= score_from]
    feats = feats.copy()
    feats["fraud_score"] = model.predict_proba(feats[FEATURES])[:, 1]
    feats["alert"] = (feats["fraud_score"] >= meta["threshold"]).astype(int)
    feats["model_version"] = meta["version"]
    log.info("scored %d txns with %s, alerts=%d (%.2f%%)", len(feats), meta["version"],
             feats["alert"].sum(), 100 * feats["alert"].mean())
    return feats, meta


def write_scores(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = ["transaction_id", "customer_id", "timestamp", "amount",
            "fraud_score", "alert", "model_version"]
    df[cols].sort_values("fraud_score", ascending=False).to_csv(path, index=False)

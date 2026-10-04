"""Training stage: split -> candidate models -> select -> threshold -> test -> register."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud.config import get_logger
from fraud.features.build import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET
from fraud.models.evaluate import choose_threshold, evaluate
from fraud.models.registry import ModelRegistry

log = get_logger(__name__)


def time_split(df: pd.DataFrame, train_end: str, valid_end: str):
    ts = df["timestamp"]
    train = df[ts < train_end]
    valid = df[(ts >= train_end) & (ts < valid_end)]
    test = df[ts >= valid_end]
    return train, valid, test


def build_candidates(seed: int) -> dict[str, Pipeline]:
    def pre(scale: bool) -> ColumnTransformer:
        return ColumnTransformer([
            ("num", StandardScaler() if scale else "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ])

    return {
        "logreg": Pipeline([
            ("pre", pre(scale=True)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5)),
        ]),
        "hgb": Pipeline([
            ("pre", pre(scale=False)),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.08, max_iter=300, max_leaf_nodes=31,
                l2_regularization=1.0, class_weight="balanced",
                early_stopping=True, validation_fraction=0.15, random_state=seed)),
        ]),
    }


def run_training(features: pd.DataFrame, cfg: dict) -> dict:
    costs, mcfg = cfg["costs"], cfg["model"]
    train, valid, test = time_split(features, cfg["split"]["train_end"], cfg["split"]["valid_end"])
    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        log.info("%-5s rows=%7d fraud_rate=%.4f  [%s -> %s]", name, len(part),
                 part[TARGET].mean(), part["timestamp"].min(), part["timestamp"].max())

    # 1) Fit each candidate on train, compare on valid
    results = {}
    for name, pipe in build_candidates(cfg["seed"]).items():
        if name not in mcfg["candidates"]:
            continue
        pipe.fit(train[FEATURES], train[TARGET])
        s = pipe.predict_proba(valid[FEATURES])[:, 1]
        t, _ = choose_threshold(valid[TARGET].to_numpy(), s, valid["amount"].to_numpy(),
                                costs["review_cost"], costs["fraud_loss_rate"],
                                mcfg["max_alert_rate"])
        m = evaluate(valid[TARGET], s, valid["amount"], t,
                     costs["review_cost"], costs["fraud_loss_rate"])
        results[name] = {"pipeline": pipe, "threshold": t, "valid_metrics": m}
        log.info("candidate=%-6s valid pr_auc=%.4f recall=%.3f precision=%.3f savings=$%.0f",
                 name, m["pr_auc"], m["recall"], m["precision"], m["net_savings"])

    # 2) Select on the primary metric
    best_name = max(results, key=lambda n: results[n]["valid_metrics"][mcfg["primary_metric"]])
    best = results[best_name]
    log.info("selected model: %s", best_name)

    # 3) Refit on train+valid (more recent data), keep valid-chosen threshold, score test once
    final = build_candidates(cfg["seed"])[best_name]
    trval = pd.concat([train, valid])
    final.fit(trval[FEATURES], trval[TARGET])
    s_test = final.predict_proba(test[FEATURES])[:, 1]
    test_m = evaluate(test[TARGET], s_test, test["amount"], best["threshold"],
                      costs["review_cost"], costs["fraud_loss_rate"])
    log.info("TEST pr_auc=%.4f roc_auc=%.4f recall=%.3f precision=%.3f alert_rate=%.4f savings=$%.0f",
             test_m["pr_auc"], test_m["roc_auc"], test_m["recall"], test_m["precision"],
             test_m["alert_rate"], test_m["net_savings"])

    # 4) Register + champion/challenger promotion
    registry = ModelRegistry(cfg["paths"]["registry"])
    reference = trval[FEATURES].sample(min(20000, len(trval)), random_state=cfg["seed"])
    metadata = {
        "model_type": best_name, "threshold": best["threshold"], "features": FEATURES,
        "train_window": [str(trval["timestamp"].min()), str(trval["timestamp"].max())],
        "valid_metrics": {n: r["valid_metrics"] for n, r in results.items()},
        "test_metrics": test_m, "config": {k: cfg[k] for k in ("split", "costs", "model")},
    }
    version = registry.register(final, metadata, reference)
    promoted = registry.maybe_promote(version, mcfg["primary_metric"])
    log.info("registered %s (promoted to PRODUCTION: %s)", version, promoted)

    report = {"version": version, "promoted": promoted, "selected": best_name,
              "test_metrics": test_m}
    Path(cfg["paths"]["reports"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["paths"]["reports"]) / "training_report.json").write_text(
        json.dumps(report, indent=2))
    return report

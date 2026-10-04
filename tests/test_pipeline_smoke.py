"""End-to-end: features -> train -> register -> batch score, on tiny data."""
from fraud.features.build import build_features
from fraud.inference.batch_score import score_batch
from fraud.models.train import run_training


def test_end_to_end(small_data, tmp_path):
    tx, cust = small_data
    cfg = {
        "seed": 0,
        "paths": {"registry": str(tmp_path / "reg"), "reports": str(tmp_path / "rep")},
        "split": {"train_end": "2026-02-01", "valid_end": "2026-02-15"},
        "costs": {"review_cost": 5.0, "fraud_loss_rate": 1.0},
        "model": {"candidates": ["logreg"], "primary_metric": "pr_auc", "max_alert_rate": 0.05},
    }
    report = run_training(build_features(tx, cust), cfg)
    assert report["promoted"] is True
    assert report["test_metrics"]["pr_auc"] > 0.3

    scored, meta = score_batch(tx.drop(columns="is_fraud"), cust, cfg["paths"]["registry"])
    assert scored["fraud_score"].between(0, 1).all()
    assert scored["model_version"].eq(meta["version"]).all()

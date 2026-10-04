"""Business-aware evaluation: metrics + cost-based threshold selection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, roc_auc_score)


def net_savings(y, scores, amounts, threshold, review_cost, loss_rate) -> float:
    """$ fraud caught minus $ spent reviewing alerts."""
    alert = scores >= threshold
    caught = (amounts * loss_rate)[alert & (y == 1)].sum()
    return float(caught - review_cost * alert.sum())


def choose_threshold(y, scores, amounts, review_cost, loss_rate,
                     max_alert_rate: float = 1.0) -> tuple[float, float]:
    """Pick the threshold that maximises net savings within ops capacity."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.5, 0.9995, 400)))
    best_t, best_val = 1.0, -np.inf
    for t in candidates:
        if (scores >= t).mean() > max_alert_rate:
            continue
        val = net_savings(y, scores, amounts, t, review_cost, loss_rate)
        if val > best_val:
            best_t, best_val = float(t), val
    return best_t, float(best_val)


def recall_at_alert_rate(y, scores, rate: float) -> float:
    k = max(1, int(len(scores) * rate))
    top = np.argsort(-scores)[:k]
    return float(y[top].sum() / max(1, y.sum()))


def evaluate(y, scores, amounts, threshold, review_cost, loss_rate) -> dict:
    y = np.asarray(y); scores = np.asarray(scores); amounts = np.asarray(amounts)
    pred = (scores >= threshold).astype(int)
    fraud_amt = float(amounts[y == 1].sum())
    caught_amt = float(amounts[(y == 1) & (pred == 1)].sum())
    return {
        "n": int(len(y)),
        "fraud_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "alert_rate": float(pred.mean()),
        "recall_at_1pct_alerts": recall_at_alert_rate(y, scores, 0.01),
        "dollar_recall": caught_amt / fraud_amt if fraud_amt else 0.0,
        "net_savings": net_savings(y, scores, amounts, threshold, review_cost, loss_rate),
    }

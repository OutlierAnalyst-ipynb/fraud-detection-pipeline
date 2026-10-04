"""Population Stability Index (PSI) drift monitoring.

PSI is the standard drift metric in bank model-risk teams (SR 11-7 ongoing
monitoring).  Rule of thumb: <0.10 stable, 0.10-0.25 watch, >0.25 investigate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi_numeric(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                      # near-constant feature (e.g. binary flags)
        return psi_categorical(expected.astype(str), actual.astype(str))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected.dropna(), edges)[0] / max(1, expected.notna().sum())
    a = np.histogram(actual.dropna(), edges)[0] / max(1, actual.notna().sum())
    return _psi(e, a)


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    cats = sorted(set(expected.unique()) | set(actual.unique()))
    e = expected.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    a = actual.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(e, a)


def _psi(e: np.ndarray, a: np.ndarray, eps: float = 1e-4) -> float:
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, numeric: list[str],
                 categorical: list[str], warn: float, alert: float) -> pd.DataFrame:
    rows = []
    for col in numeric:
        rows.append((col, psi_numeric(reference[col], current[col])))
    for col in categorical:
        rows.append((col, psi_categorical(reference[col].astype(str), current[col].astype(str))))
    out = pd.DataFrame(rows, columns=["feature", "psi"]).sort_values("psi", ascending=False)
    out["status"] = np.select([out["psi"] >= alert, out["psi"] >= warn],
                              ["ALERT", "WARN"], "OK")
    return out.reset_index(drop=True)

"""Point-in-time feature engineering.

The #1 bug in fraud models is LEAKAGE: computing a feature using information
that wasn't available when the card was swiped. Every behavioral feature here
only looks at the customer's *previous* transactions (closed='left', shift(1)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "amount", "log_amount", "hour", "day_of_week", "is_night", "is_weekend",
    "secs_since_last_txn", "txn_count_1h", "txn_count_24h", "amount_sum_24h",
    "hist_amount_mean", "amount_to_hist_mean", "is_new_device", "is_foreign",
    "account_age_days", "amount_to_limit",
]
CATEGORICAL_FEATURES = ["merchant_category", "channel"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ID_COLS = ["transaction_id", "customer_id", "timestamp"]
TARGET = "is_fraud"


def _rolling(df: pd.DataFrame, col: str, window: str, agg: str) -> np.ndarray:
    # df must be sorted by (customer_id, timestamp); closed='left' excludes current txn
    r = (df.set_index("timestamp").groupby("customer_id", sort=False)[col]
           .rolling(window, closed="left"))
    return getattr(r, agg)().fillna(0).to_numpy()


def build_features(tx: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    df = tx.merge(customers, on="customer_id", how="left", validate="many_to_one")
    df = df.sort_values(["customer_id", "timestamp"], kind="stable").reset_index(drop=True)

    # --- transaction-level ---
    df["log_amount"] = np.log1p(df["amount"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_foreign"] = (df["country"] != df["home_country"]).astype(int)
    df["amount_to_limit"] = df["amount"] / df["credit_limit"]

    # --- behavioral / velocity (point-in-time) ---
    g = df.groupby("customer_id", sort=False)
    df["secs_since_last_txn"] = (g["timestamp"].diff().dt.total_seconds()
                                 .fillna(30 * 24 * 3600))
    df["txn_count_1h"] = _rolling(df, "amount", "1h", "count")
    df["txn_count_24h"] = _rolling(df, "amount", "24h", "count")
    df["amount_sum_24h"] = _rolling(df, "amount", "24h", "sum")

    hist_mean = g["amount"].transform(lambda s: s.shift(1).expanding().mean())
    df["hist_amount_mean"] = hist_mean.fillna(df["amount"].median())
    df["amount_to_hist_mean"] = df["amount"] / (df["hist_amount_mean"] + 1.0)

    # First time this customer has ever been seen on this device
    df["is_new_device"] = (df.groupby(["customer_id", "device_id"]).cumcount() == 0).astype(int)

    cols = ID_COLS + FEATURES + ([TARGET] if TARGET in df.columns else [])
    return df[cols].sort_values("timestamp", kind="stable").reset_index(drop=True)

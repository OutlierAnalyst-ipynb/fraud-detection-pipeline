"""Data contracts. Fail fast before bad data reaches a model.

In production this is where you'd plug in Great Expectations / Pandera;
here it's plain pandas so you can see exactly what's checked.
"""
from __future__ import annotations

import pandas as pd

TRANSACTION_SCHEMA = {
    "transaction_id": "object", "customer_id": "object", "timestamp": "datetime",
    "amount": "numeric", "merchant_category": "object", "channel": "object",
    "country": "object", "device_id": "object",
}
CUSTOMER_SCHEMA = {"customer_id": "object", "home_country": "object",
                   "account_age_days": "numeric", "credit_limit": "numeric"}
VALID_CHANNELS = {"card_present", "online", "atm"}


class DataValidationError(ValueError):
    pass


def _check_schema(df: pd.DataFrame, schema: dict[str, str], name: str) -> list[str]:
    errors = []
    missing = set(schema) - set(df.columns)
    if missing:
        errors.append(f"{name}: missing columns {sorted(missing)}")
    for col, kind in schema.items():
        if col not in df.columns:
            continue
        if kind == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"{name}.{col}: expected numeric, got {df[col].dtype}")
        if kind == "datetime" and not pd.api.types.is_datetime64_any_dtype(df[col]):
            errors.append(f"{name}.{col}: expected datetime, got {df[col].dtype}")
    return errors


def validate_transactions(df: pd.DataFrame, require_label: bool = True) -> pd.DataFrame:
    errors = _check_schema(df, TRANSACTION_SCHEMA, "transactions")
    if require_label and "is_fraud" not in df.columns:
        errors.append("transactions: missing label column 'is_fraud'")
    if not errors:
        if df["transaction_id"].duplicated().any():
            errors.append("transactions: duplicate transaction_id")
        if df[list(TRANSACTION_SCHEMA)].isna().any().any():
            nulls = df[list(TRANSACTION_SCHEMA)].isna().sum()
            errors.append(f"transactions: nulls found {nulls[nulls > 0].to_dict()}")
        if (df["amount"] <= 0).any():
            errors.append("transactions: non-positive amounts")
        bad_ch = set(df["channel"].unique()) - VALID_CHANNELS
        if bad_ch:
            errors.append(f"transactions: unknown channels {bad_ch}")
        if require_label and not set(df["is_fraud"].unique()) <= {0, 1}:
            errors.append("transactions: is_fraud must be 0/1")
    if errors:
        raise DataValidationError("\n".join(errors))
    return df


def validate_customers(df: pd.DataFrame) -> pd.DataFrame:
    errors = _check_schema(df, CUSTOMER_SCHEMA, "customers")
    if not errors and df["customer_id"].duplicated().any():
        errors.append("customers: duplicate customer_id")
    if errors:
        raise DataValidationError("\n".join(errors))
    return df


def load_raw(tx_path: str, cust_path: str, require_label: bool = True):
    tx = pd.read_csv(tx_path, parse_dates=["timestamp"])
    cust = pd.read_csv(cust_path)
    return validate_transactions(tx, require_label), validate_customers(cust)

import pytest

from fraud.data.validate import DataValidationError, validate_transactions


def test_valid_data_passes(small_data):
    tx, _ = small_data
    validate_transactions(tx)


def test_negative_amount_fails(small_data):
    tx = small_data[0].copy()
    tx.loc[0, "amount"] = -10
    with pytest.raises(DataValidationError, match="non-positive"):
        validate_transactions(tx)


def test_missing_column_fails(small_data):
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_transactions(small_data[0].drop(columns=["channel"]))


def test_duplicate_ids_fail(small_data):
    tx = small_data[0].copy()
    tx.loc[1, "transaction_id"] = tx.loc[0, "transaction_id"]
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_transactions(tx)

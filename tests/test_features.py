import pandas as pd

from fraud.features.build import FEATURES, build_features


def _toy():
    tx = pd.DataFrame({
        "transaction_id": ["t1", "t2", "t3", "t4"],
        "customer_id": ["c1"] * 4,
        "timestamp": pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:20",
                                     "2026-01-01 10:40", "2026-01-02 12:00"]),
        "amount": [10.0, 20.0, 30.0, 40.0],
        "merchant_category": ["grocery"] * 4,
        "channel": ["online"] * 4,
        "country": ["US", "US", "FR", "US"],
        "device_id": ["d1", "d1", "d2", "d1"],
        "is_fraud": [0, 0, 1, 0],
    })
    cust = pd.DataFrame({"customer_id": ["c1"], "home_country": ["US"],
                         "account_age_days": [100], "credit_limit": [1000]})
    return build_features(tx, cust)


def test_velocity_excludes_current_txn():
    f = _toy()
    # t3 at 10:40 has two prior txns in the last hour, NOT three (no leakage)
    assert f["txn_count_1h"].tolist() == [0, 1, 2, 0]


def test_history_mean_uses_only_past():
    f = _toy()
    assert f.loc[1, "hist_amount_mean"] == 10.0          # only t1
    assert f.loc[3, "hist_amount_mean"] == 20.0          # mean(10,20,30)


def test_new_device_and_foreign_flags():
    f = _toy()
    assert f["is_new_device"].tolist() == [1, 0, 1, 0]
    assert f["is_foreign"].tolist() == [0, 0, 1, 0]


def test_feature_columns_present_and_no_nulls(small_data):
    f = build_features(*small_data)
    assert set(FEATURES) <= set(f.columns)
    assert not f[FEATURES].isna().any().any()

import pytest

from fraud.data.generate import generate_dataset


@pytest.fixture(scope="session")
def small_data():
    return generate_dataset(n_customers=300, n_days=60, start_date="2026-01-01",
                            fraud_rate=0.02, seed=7)

"""Synthetic card-transaction generator.

Real bank data is never available for practice, so we simulate it with the
patterns fraud teams actually see:
  * account takeover -> bursts of transactions in a short window
  * new / unseen device, often online channel
  * high-risk merchant categories (electronics, gift cards, crypto)
  * foreign country, night-time activity, larger amounts
Legit noise (travel, new phones, big purchases) is injected so the problem
is not trivially separable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MCC = ["grocery", "restaurants", "fuel", "retail", "travel", "electronics",
       "gift_cards", "crypto", "utilities", "entertainment"]
MCC_LEGIT_P = [0.25, 0.18, 0.12, 0.18, 0.05, 0.05, 0.02, 0.005, 0.08, 0.065]
MCC_FRAUD_P = [0.03, 0.03, 0.02, 0.12, 0.08, 0.30, 0.25, 0.12, 0.0, 0.05]
CHANNELS = ["card_present", "online", "atm"]
COUNTRIES = ["US", "CA", "MX", "GB", "FR", "NG", "RU", "BR", "CN", "IN"]


def generate_customers(n_customers: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(n_customers)],
        "home_country": rng.choice(["US", "CA", "GB"], n_customers, p=[0.85, 0.1, 0.05]),
        "account_age_days": rng.integers(30, 5000, n_customers),
        "credit_limit": rng.choice([1000, 2500, 5000, 10000, 20000], n_customers,
                                   p=[0.15, 0.3, 0.3, 0.2, 0.05]),
        "avg_spend": rng.lognormal(mean=3.6, sigma=0.6, size=n_customers).round(2),
        "txn_rate": rng.gamma(shape=2.0, scale=0.3, size=n_customers),  # txns per day
    })


def _legit_transactions(cust: pd.DataFrame, start: pd.Timestamp, n_days: int,
                        rng: np.random.Generator) -> pd.DataFrame:
    n_per_cust = rng.poisson(cust["txn_rate"].to_numpy() * n_days)
    idx = np.repeat(np.arange(len(cust)), n_per_cust)
    n = len(idx)
    c = cust.iloc[idx].reset_index(drop=True)

    # Daytime-heavy hour distribution
    hour_p = np.array([1, 0.5, 0.3, 0.2, 0.2, 0.5, 1.5, 3, 4, 5, 5, 6,
                       7, 6, 5, 5, 5, 6, 7, 7, 6, 4, 3, 2], dtype=float)
    hours = rng.choice(24, n, p=hour_p / hour_p.sum())
    days = rng.integers(0, n_days, n)
    ts = (start + pd.to_timedelta(days, unit="D") + pd.to_timedelta(hours, unit="h")
          + pd.to_timedelta(rng.integers(0, 3600, n), unit="s"))

    travelling = rng.random(n) < 0.02
    country = np.where(travelling, rng.choice(COUNTRIES, n), c["home_country"].to_numpy())
    # Each customer mostly uses 1-2 devices; ~2% of legit txns come from a new device
    device = np.array([f"D{cid}_{k}" for cid, k in zip(c["customer_id"], rng.integers(0, 2, n))])
    new_dev = rng.random(n) < 0.02
    device[new_dev] = [f"D{x:09d}" for x in rng.integers(0, 10**9, new_dev.sum())]

    return pd.DataFrame({
        "customer_id": c["customer_id"],
        "timestamp": ts,
        "amount": (c["avg_spend"].to_numpy() * rng.lognormal(0, 0.7, n)).round(2),
        "merchant_category": rng.choice(MCC, n, p=MCC_LEGIT_P),
        "channel": rng.choice(CHANNELS, n, p=[0.65, 0.3, 0.05]),
        "country": country,
        "device_id": device,
        "is_fraud": 0,
    })


def _fraud_transactions(cust: pd.DataFrame, start: pd.Timestamp, n_days: int,
                        n_fraud_target: int, rng: np.random.Generator) -> pd.DataFrame:
    """Account-takeover bursts: 2-9 txns within a few hours of the compromise."""
    rows = []
    n_victims = max(1, n_fraud_target // 5)
    victims = cust.sample(n_victims, random_state=int(rng.integers(1e9)))
    for _, v in victims.iterrows():
        t0 = start + pd.Timedelta(days=float(rng.uniform(1, n_days - 1)))
        burst = rng.integers(2, 10)
        device = f"D{rng.integers(0, 10**9):09d}"
        foreign = rng.random() < 0.6
        for _ in range(burst):
            rows.append({
                "customer_id": v["customer_id"],
                "timestamp": t0 + pd.Timedelta(minutes=float(rng.exponential(25))),
                "amount": round(float(v["avg_spend"] * rng.lognormal(1.0, 0.8)), 2),
                "merchant_category": rng.choice(MCC, p=MCC_FRAUD_P),
                "channel": rng.choice(CHANNELS, p=[0.15, 0.75, 0.10]),
                "country": rng.choice(COUNTRIES[3:]) if foreign else v["home_country"],
                # fraudster sometimes spoofs a known device
                "device_id": device if rng.random() < 0.8 else f"D{v['customer_id']}_0",
                "is_fraud": 1,
            })
    return pd.DataFrame(rows)


def generate_dataset(n_customers: int, n_days: int, start_date: str, fraud_rate: float,
                     seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)
    customers = generate_customers(n_customers, rng)
    legit = _legit_transactions(customers, start, n_days, rng)
    n_fraud = int(len(legit) * fraud_rate / (1 - fraud_rate))
    fraud = _fraud_transactions(customers, start, n_days, n_fraud, rng)

    tx = (pd.concat([legit, fraud], ignore_index=True)
            .sort_values("timestamp").reset_index(drop=True))
    tx.insert(0, "transaction_id", [f"T{i:09d}" for i in range(len(tx))])
    # Customer master table doesn't expose the simulator's hidden parameters
    return tx, customers.drop(columns=["avg_spend", "txn_rate"])

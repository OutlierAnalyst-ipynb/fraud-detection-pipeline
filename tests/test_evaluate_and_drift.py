import numpy as np
import pandas as pd

from fraud.models.evaluate import choose_threshold, net_savings
from fraud.monitoring.drift import psi_numeric


def test_net_savings_math():
    y = np.array([1, 0, 1, 0]); s = np.array([0.9, 0.8, 0.1, 0.0])
    amt = np.array([100.0, 50.0, 200.0, 10.0])
    # alerts on first two: catch $100, pay 2 * $5 reviews
    assert net_savings(y, s, amt, 0.5, review_cost=5, loss_rate=1) == 90


def test_threshold_respects_alert_capacity():
    rng = np.random.default_rng(0)
    s = rng.random(10_000); y = (s > 0.98).astype(int); amt = np.full(10_000, 100.0)
    t, _ = choose_threshold(y, s, amt, 5, 1, max_alert_rate=0.01)
    assert (s >= t).mean() <= 0.01


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(ref, pd.Series(rng.normal(0, 1, 5000))) < 0.1
    assert psi_numeric(ref, pd.Series(rng.normal(1, 1, 5000))) > 0.25

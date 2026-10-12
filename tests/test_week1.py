import sys
from pathlib import Path

sys.path.insert(0, str(Path("backend/src").resolve()))

import numpy as np
import pandas as pd
from econocausal.data.schemas import validate
from econocausal.optimize.economics import greedy_policy, incremental_profit


def test_economics_sleeping_dogs_negative():
    tau = np.array([1.0, 0.0, 0.0, -1.0])
    p = incremental_profit(tau, margin=20.0, unit_cost=5.0)
    assert p[0] == 15.0 and p[3] == -25.0


def test_greedy_respects_budget():
    profit = np.array([10.0, 9.0, 1.0])
    cost = np.array([5.0, 5.0, 5.0])
    pick = greedy_policy(profit, cost, budget=10.0)
    assert (pick * cost).sum() <= 10.0
    assert pick.sum() == 2


def test_schema_rejects_post_treatment():
    df = pd.DataFrame({"T": [0, 1], "Y": [0, 1], "clicked_checkout": [0, 1]})
    try:
        validate(df)
    except ValueError as e:
        assert "post-treatment" in str(e)
    else:
        raise AssertionError("should have rejected post-T column")

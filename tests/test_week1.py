import sys
from pathlib import Path

sys.path.insert(0, str(Path("backend/src").resolve()))

import numpy as np
import pandas as pd
from econocausal.causal.propensity import overlap_gate
from econocausal.data.schemas import validate
from econocausal.optimize.economics import greedy_policy, incremental_profit
from econocausal.optimize.knapsack import knapsack_lp, knapsack_milp, optimize


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


def test_knapsack_lp_respects_budget():
    profit = np.array([10.0, 9.0, 8.0, -1.0])
    cost = np.array([5.0, 5.0, 5.0, 5.0])
    x = knapsack_lp(profit, cost, budget=10.0)
    assert (x * cost).sum() <= 10.0 + 1e-6
    assert x[3] == 0  # negative profit never picked


def test_knapsack_milp_small_exact():
    profit = np.array([10.0, 9.0, 8.0])
    cost = np.array([5.0, 5.0, 5.0])
    x = knapsack_milp(profit, cost, budget=10.0)
    assert x.sum() == 2 and (x * cost).sum() <= 10.0


def test_optimize_empty_when_all_negative():
    sol = optimize(np.array([-1.0, -2.0]), np.array([5.0, 5.0]), budget=100.0)
    assert sol["method"] == "empty" and sol["expected_profit"] == 0.0


def test_overlap_gate_blocks_on_separation():
    n = 200
    df = pd.DataFrame({
        "x1": np.concatenate([np.zeros(n // 2), np.ones(n // 2)]),
        "T": np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int),
        "Y": np.zeros(n, dtype=int),
    })
    gate = overlap_gate(df, ["x1"], lo=0.025, hi=0.975)
    assert gate["block_policy"] is True


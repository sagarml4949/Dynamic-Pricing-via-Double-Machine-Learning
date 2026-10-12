"""Money math: pure numpy, no pandas. incremental_profit = tau*margin - unit_cost."""
from __future__ import annotations

import numpy as np


def incremental_profit(tau: np.ndarray, margin: float, unit_cost: float) -> np.ndarray:
    return np.asarray(tau, dtype=float) * margin - unit_cost


def greedy_policy(profit: np.ndarray, cost: np.ndarray, budget: float) -> np.ndarray:
    order = np.argsort(-(profit / np.maximum(cost, 1e-9)))
    pick = np.zeros(len(profit), dtype=int)
    spent = 0.0
    for i in order:
        if profit[i] <= 0:
            break
        if spent + cost[i] <= budget:
            pick[i] = 1
            spent += cost[i]
    return pick

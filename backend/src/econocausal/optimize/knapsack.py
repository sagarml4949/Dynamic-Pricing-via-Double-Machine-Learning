"""Budget optimizer: LP/HiGHS default, MILP exact for N<=2k, greedy fallback.

Maximize sum(profit_i * x_i) s.t. sum(cost_i * x_i) <= budget, x in {0,1} (LP relax [0,1]).
Pure numpy in, numpy out. Never force-spend: all-negative profit -> empty pick.
"""
from __future__ import annotations

import numpy as np


def knapsack_lp(profit: np.ndarray, cost: np.ndarray, budget: float) -> np.ndarray:
    from scipy.optimize import linprog
    p, c = np.asarray(profit, float), np.asarray(cost, float)
    n = len(p)
    res = linprog(-p, A_ub=[c], b_ub=[budget], bounds=[(0, 1)] * n, method="highs")
    if not res.success:
        return np.zeros(n, dtype=int)
    x = (res.x >= 0.5).astype(int)
    # repair: drop worst ROI until feasible (LP rounding can overflow)
    while (x * c).sum() > budget and x.sum() > 0:
        roi = np.where(x == 1, p / np.maximum(c, 1e-9), np.inf)
        x[int(np.argmin(roi))] = 0
    x[p <= 0] = 0
    return x


def knapsack_milp(profit: np.ndarray, cost: np.ndarray, budget: float) -> np.ndarray:
    from scipy.optimize import milp, LinearConstraint, Bounds
    p, c = np.asarray(profit, float), np.asarray(cost, float)
    n = len(p)
    if n > 2000:
        raise ValueError("milp only for N<=2000; use knapsack_lp")
    res = milp(-p, constraints=[LinearConstraint(c, ub=budget)],
               bounds=Bounds(np.zeros(n), np.ones(n)),
               integrality=np.ones(n))
    if not res.success:
        return np.zeros(n, dtype=int)
    x = (np.asarray(res.x).ravel() >= 0.5).astype(int)
    x[p <= 0] = 0
    return x


def optimize(profit: np.ndarray, cost: np.ndarray, budget: float,
             method: str = "lp") -> dict:
    from .economics import greedy_policy
    p = np.asarray(profit, float)
    if bool((p <= 0).all()):
        return {"assignment": np.zeros(len(p), dtype=int), "method": "empty",
                "expected_profit": 0.0, "spent": 0.0, "note": "all-negative profit"}
    if method == "milp":
        x = knapsack_milp(p, np.asarray(cost, float), budget)
        used = "milp"
    elif method == "greedy":
        x = greedy_policy(p, np.asarray(cost, float), budget)
        used = "greedy"
    else:
        x = knapsack_lp(p, np.asarray(cost, float), budget)
        used = "lp"
    c = np.asarray(cost, float)
    return {"assignment": x, "method": used,
            "expected_profit": float((x * p).sum()),
            "spent": float((x * c).sum())}

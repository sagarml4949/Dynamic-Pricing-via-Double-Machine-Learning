"""QINI, policy value, calibration. Reads predictions, never refits."""
from __future__ import annotations

import numpy as np
import pandas as pd


def qini(df: pd.DataFrame, tau_hat: np.ndarray, y: str = "Y") -> float:
    from .uplift import auuc
    return auuc(df, tau_hat, y)  # AUUC == QINI for binary Y up to scaling


def policy_value_table(df: pd.DataFrame, tau_hat: np.ndarray,
                       budgets: list[int], unit_cost: float = 0.10,
                       margin: float = 20.0, y: str = "Y") -> pd.DataFrame:
    from econocausal.optimize.economics import incremental_profit
    from econocausal.optimize.knapsack import optimize
    profit = incremental_profit(np.asarray(tau_hat), margin, unit_cost)
    cost = np.full(len(df), unit_cost)
    rows = []
    for b in budgets:
        sol = optimize(profit, cost, b * unit_cost, method="lp")
        x = sol["assignment"]
        picked = df.iloc[np.where(x == 1)[0]]
        rows.append({"budget_n": b, "n_picked": int(x.sum()),
                     "expected_profit": sol["expected_profit"],
                     "picked_rate": float(picked[y].mean()) if len(picked) else 0.0})
    return pd.DataFrame(rows)


def calibration_by_decile(df: pd.DataFrame, tau_hat: np.ndarray,
                          y: str = "Y") -> pd.DataFrame:
    d = df.copy()
    d["_tau"] = np.asarray(tau_hat)
    d["_dec"] = pd.qcut(d["_tau"], 10, labels=False, duplicates="drop")
    g = d.groupby("_dec", observed=True)
    out = g.apply(lambda s: pd.Series({
        "mean_tau": s["_tau"].mean(),
        "rate_T1": s.loc[s["T"] == 1, y].mean() if (s["T"] == 1).any() else float("nan"),
        "rate_T0": s.loc[s["T"] == 0, y].mean() if (s["T"] == 0).any() else float("nan"),
        "n": len(s)}), include_groups=False)
    out["uplift"] = out["rate_T1"] - out["rate_T0"]
    return out.reset_index(drop=True)

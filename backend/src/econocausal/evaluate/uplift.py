"""Uplift/AUUC/QINI on eval split. Reads predictions, never refits."""
from __future__ import annotations

import numpy as np
import pandas as pd


def auuc(df: pd.DataFrame, tau_hat: np.ndarray, y: str = "Y") -> float:
    order = np.argsort(-np.asarray(tau_hat))
    d = df.iloc[order].reset_index(drop=True)
    n1 = (d["T"] == 1).cumsum() / max((d["T"] == 1).sum(), 1)
    n0 = (d["T"] == 0).cumsum() / max((d["T"] == 0).sum(), 1)
    r1 = (d[y] * (d["T"] == 1)).cumsum() / max((d["T"] == 1).sum(), 1)
    r0 = (d[y] * (d["T"] == 0)).cumsum() / max((d["T"] == 0).sum(), 1)
    return float(np.trapezoid((r1 - r0).to_numpy(), (n1 + n0).to_numpy() / 2))


def pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(tau_hat) - np.asarray(tau_true)) ** 2)))

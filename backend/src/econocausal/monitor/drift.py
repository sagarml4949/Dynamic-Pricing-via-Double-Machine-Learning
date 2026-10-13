"""Drift + overlap alerts. Read-only over registry/parquet; never mutates runs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=qs)[0] / max(len(expected), 1)
    a = np.histogram(actual, bins=qs)[0] / max(len(actual), 1)
    e, a = np.clip(e, 1e-4, 1), np.clip(a, 1e-4, 1)
    return float(((a - e) * np.log(a / e)).sum())


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame,
                 feats: list[str]) -> dict:
    out = {}
    for c in feats:
        v = psi(ref[c].to_numpy(), cur[c].to_numpy())
        out[c] = {"psi": v, "alert": bool(v > 0.25)}
    return out

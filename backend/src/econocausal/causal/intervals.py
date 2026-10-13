"""Interval helpers: conformal-style bands around tau_hat on eval data.

Forest intervals come from CausalForestDML.effect_interval; this module adds
a model-agnostic residual band for S/T/X/LinearDML using cross-fit residuals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def residual_band(tau_hat: np.ndarray, y: np.ndarray, t: np.ndarray,
                  quantile: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    tau_hat = np.asarray(tau_hat, float)
    # pseudo-outcome residual scale: |Y - mean(Y|T)| as a rough band width
    y = np.asarray(y, float)
    t = np.asarray(t)
    mu1 = y[t == 1].mean() if (t == 1).any() else y.mean()
    mu0 = y[t == 0].mean() if (t == 0).any() else y.mean()
    resid = np.abs(y - np.where(t == 1, mu1, mu0))
    w = float(np.quantile(resid, quantile))
    return tau_hat - w, tau_hat + w


def coverage(tau_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    tau_true = np.asarray(tau_true, float)
    return float(((tau_true >= np.asarray(lo)) & (tau_true <= np.asarray(hi))).mean())

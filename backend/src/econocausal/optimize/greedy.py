"""Greedy ROI policy: explainable fallback + sanity check for LP/MILP."""
from __future__ import annotations

import numpy as np

from .economics import greedy_policy

__all__ = ["greedy_policy", "roi_order"]


def roi_order(profit: np.ndarray, cost: np.ndarray) -> np.ndarray:
    p, c = np.asarray(profit, float), np.asarray(cost, float)
    return np.argsort(-(p / np.maximum(c, 1e-9)))

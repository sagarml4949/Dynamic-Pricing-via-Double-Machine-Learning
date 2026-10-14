"""Synthetic pricing data with known ITE (4 customer types).

Y(0), Y(1) known -> tau_true. T confounded: past targeted lapsed+valuable.
Types: persuadable (0,1), sure_thing (1,1), lost_cause (0,0), sleeping_dog (1,0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(n: int = 20_000, seed: int = 42, sharp: float = 2.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lapsed = rng.binomial(1, 0.35, n).astype(float)
    value = rng.normal(50, 15, n)
    tenure = rng.integers(0, 60, n).astype(float)
    noise = rng.normal(0, 1, (n, 2))
    X = np.column_stack([lapsed, value, tenure, noise])

    # Learnable heterogeneity: type scores are strong functions of X, plus
    # small noise. sharp>1 steepens the mapping (more identified); sharp<1
    # flattens it (harder). Default 2.0 gives corr ceiling ~0.8.
    is_lapsed = lapsed == 1
    hi_val = (value - 50) / 15.0
    lo_ten = (30 - tenure) / 15.0
    # separated regions: persuadable = lapsed+lowTenure, sure = active+hiVal,
    # sleeping = lapsed+hiVal+hiTenure, lost = active+lowVal
    s_pers = sharp * (1.5 * is_lapsed + 1.0 * lo_ten - 0.5 * hi_val)
    s_sure = sharp * (1.5 * (~is_lapsed) + 1.0 * hi_val)
    s_sleep = sharp * (1.0 * is_lapsed + 1.0 * hi_val - 1.0 * lo_ten)
    s_lost = sharp * (1.0 * (~is_lapsed) - 1.0 * hi_val)
    scores = np.column_stack([s_pers, s_sure, s_sleep, s_lost])
    scores += rng.normal(0, 1, scores.shape)  # residual unobserved part
    typ = np.array(["persuadable", "sure_thing", "sleeping_dog", "lost_cause"])[
        scores.argmax(axis=1)]
    y0 = np.where((typ == "sure_thing") | (typ == "sleeping_dog"), 1, 0)
    y1 = np.where((typ == "persuadable") | (typ == "sure_thing"), 1, 0)
    tau_true = (y1 - y0).astype(float)

    # confounded treatment: targeted lapsed + valuable
    logit = -0.5 + 1.6 * lapsed + 0.02 * (value - 50)
    p_t = 1 / (1 + np.exp(-logit))
    t = (rng.random(n) < p_t).astype(int)
    y = np.where(t == 1, y1, y0)

    return pd.DataFrame({
        "lapsed": lapsed, "value": value, "tenure": tenure,
        "z1": X[:, 3], "z2": X[:, 4],
        "T": t, "Y": y, "tau_true": tau_true, "type": typ,
        "p_t_true": p_t,
    })

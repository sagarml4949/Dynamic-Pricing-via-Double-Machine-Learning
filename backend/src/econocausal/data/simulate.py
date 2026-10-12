"""Synthetic pricing data with known ITE (4 customer types).

Y(0), Y(1) known -> tau_true. T confounded: past targeted lapsed+valuable.
Types: persuadable (0,1), sure_thing (1,1), lost_cause (0,0), sleeping_dog (1,0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lapsed = rng.binomial(1, 0.35, n).astype(float)
    value = rng.normal(50, 15, n)
    tenure = rng.integers(0, 60, n).astype(float)
    noise = rng.normal(0, 1, (n, 2))
    X = np.column_stack([lapsed, value, tenure, noise])

    # latent type depends on X: high value + not lapsed -> sure thing mass, etc.
    u = rng.random(n)
    is_lapsed = lapsed == 1
    # persuadable: lapsed & mid value
    p_pers = 0.45 * is_lapsed + 0.10 * (~is_lapsed)
    p_sure = 0.10 * is_lapsed + 0.40 * (~is_lapsed)
    p_sleep = 0.05 + 0.02 * (value > 65)
    typ = np.where(u < p_pers, "persuadable",
           np.where(u < p_pers + p_sure, "sure_thing",
           np.where(u < p_pers + p_sure + p_sleep, "sleeping_dog", "lost_cause")))
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

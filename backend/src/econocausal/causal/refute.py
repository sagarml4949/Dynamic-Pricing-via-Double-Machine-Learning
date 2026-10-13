"""Refutation audit: placebo / subset / add-noise.

Lightweight honest refuters (DoWhy semantics, sklearn implementation so they
run on py3.14 + binary T without version hell). Any fail -> needs_review.
DoWhy graph/identify step lives in docs; estimation stays in EconML (ARCHITECTURE #2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ate_via_s_learner(tr: pd.DataFrame, feats: list[str], seed: int = 42) -> float:
    from .learners import SLearner
    m = SLearner(seed).fit(tr, feats)
    return float(m.predict_tau(tr).mean())


def refute(df: pd.DataFrame, feats: list[str], ate_hat: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    out: dict = {"ate_hat": float(ate_hat)}

    # 1. placebo: shuffle T -> effect should collapse toward 0
    d = df.copy()
    d["T"] = rng.permutation(d["T"].to_numpy())
    ate_placebo = _ate_via_s_learner(d, feats, seed)
    out["placebo_ate"] = ate_placebo
    out["placebo_pass"] = bool(abs(ate_placebo) < 0.5 * abs(ate_hat) + 1e-6)

    # 2. subset: 50% sample -> effect should stay within 50% of full
    sub = df.sample(frac=0.5, random_state=seed).reset_index(drop=True)
    ate_sub = _ate_via_s_learner(sub, feats, seed)
    out["subset_ate"] = ate_sub
    out["subset_pass"] = bool(abs(ate_sub - ate_hat) < 0.5 * abs(ate_hat) + 0.02)

    # 3. add-noise: jitter continuous feats -> effect should be stable
    d2 = df.copy()
    for c in feats:
        if pd.api.types.is_numeric_dtype(d2[c]):
            d2[c] = d2[c] + rng.normal(0, 0.01 * (d2[c].std() + 1e-9), len(d2))
    ate_noisy = _ate_via_s_learner(d2, feats, seed)
    out["noise_ate"] = ate_noisy
    out["noise_pass"] = bool(abs(ate_noisy - ate_hat) < 0.5 * abs(ate_hat) + 0.02)

    out["all_pass"] = bool(out["placebo_pass"] and out["subset_pass"] and out["noise_pass"])
    out["status"] = "pass" if out["all_pass"] else "needs_review"
    return out

"""Propensity + overlap gate (diagnostics only, never an effect).

Lightweight LGBM propensity m(X)=P(T=1|X); reports overlap hist bins,
SMD per covariate (|SMD|>0.1 flagged), ESS, trim fractions.
Gate: >15% mass outside [lo,hi] or ESS < 0.3*N -> block_policy=True.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


def propensity_scores(df: pd.DataFrame, feats: list[str], seed: int = 42) -> np.ndarray:
    m = LGBMClassifier(n_estimators=150, min_child_samples=50,
                       verbose=-1, random_state=seed)
    m.fit(df[feats].to_numpy(), df["T"].to_numpy())
    return m.predict_proba(df[feats].to_numpy())[:, 1]


def smd(df: pd.DataFrame, feats: list[str]) -> dict[str, float]:
    out = {}
    d1, d0 = df[df["T"] == 1], df[df["T"] == 0]
    for c in feats:
        m1, m0 = d1[c].mean(), d0[c].mean()
        v1, v0 = d1[c].var(ddof=1), d0[c].var(ddof=1)
        pooled = np.sqrt((v1 + v0) / 2) if (v1 + v0) > 0 else 1.0
        out[c] = float((m1 - m0) / pooled) if pooled else 0.0
    return out


def overlap_gate(df: pd.DataFrame, feats: list[str], seed: int = 42,
                 lo: float = 0.025, hi: float = 0.975) -> dict:
    p = propensity_scores(df, feats, seed)
    p_clip = np.clip(p, 1e-6, 1 - 1e-6)
    outside = float(((p < lo) | (p > hi)).mean())
    ess = float((p_clip * (1 - p_clip)).sum() / 0.25)  # normalized ESS proxy
    s = smd(df, feats)
    flagged = [k for k, v in s.items() if abs(v) > 0.1]
    hist, edges = np.histogram(p, bins=20, range=(0, 1))
    block = bool(outside > 0.15 or ess < 0.3 * len(df))
    return {"propensity_mean": float(p.mean()),
            "frac_outside_trim": outside,
            "ess": ess, "ess_ratio": ess / len(df),
            "smd": s, "smd_flagged": flagged,
            "hist": hist.tolist(), "hist_edges": edges.tolist(),
            "block_policy": block,
            "lo": lo, "hi": hi}

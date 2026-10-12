"""Seeded train/eval/policy splits, stratified on T."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split(df: pd.DataFrame, seed: int = 42,
          train: float = 0.6, ev: float = 0.2) -> dict[str, pd.DataFrame]:
    tr, rest = train_test_split(df, train_size=train, random_state=seed, stratify=df["T"])
    ev_size = ev / (1 - train)
    e, p = train_test_split(rest, train_size=ev_size, random_state=seed + 1, stratify=rest["T"])
    return {"train": tr.reset_index(drop=True),
            "eval": e.reset_index(drop=True),
            "policy": p.reset_index(drop=True)}

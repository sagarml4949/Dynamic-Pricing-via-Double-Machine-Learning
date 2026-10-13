"""Parquet store: eval frames + tau columns keyed by run_id. Registry has no domain imports."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REG = Path("data/registry")


def save_frame(run_id: str, df: pd.DataFrame, name: str = "eval") -> Path:
    d = REG / run_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.parquet"
    df.to_parquet(p, index=False)
    return p


def load_frame(run_id: str, name: str = "eval") -> pd.DataFrame:
    return pd.read_parquet(REG / run_id / f"{name}.parquet")

"""Loader: CSV/parquet -> validated frame. No causal logic."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schemas import validate


def load_table(path: str | Path, require_cols: list[str] | None = None) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    # normalize common criteo names to canonical T/Y if present
    ren = {}
    if "treatment" in df.columns and "T" not in df.columns:
        ren["treatment"] = "T"
    if "conversion" in df.columns and "Y" not in df.columns and "visit" not in df.columns:
        ren["conversion"] = "Y"
    if ren:
        df = df.rename(columns=ren)
    if require_cols:
        return validate(df, require_cols=require_cols)
    return df

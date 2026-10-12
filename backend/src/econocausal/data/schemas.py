"""Pandera schemas: the only path to prod numbers. Reject, never silently coerce."""
from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series

POST_TREATMENT_DENYLIST = {"checkout_after_discount", "clicked_checkout", "exposure_post"}


class EventFrame(pa.DataFrameModel):
    T: Series[int] = pa.Field(ge=0, le=1, nullable=False)
    Y: Series[int] = pa.Field(ge=0, le=1, nullable=False)

    class Config:
        strict = False
        coerce = False


def validate(df, require_cols: list[str] | None = None):
    leaked = POST_TREATMENT_DENYLIST & set(df.columns)
    if leaked:
        raise ValueError(f"post-treatment leakage columns present: {sorted(leaked)}")
    if require_cols:
        missing = set(require_cols) - set(df.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
    if df[["T", "Y"]].isnull().any().any():
        raise ValueError("null T/Y rows present: drop them before validate")
    return EventFrame.validate(df)

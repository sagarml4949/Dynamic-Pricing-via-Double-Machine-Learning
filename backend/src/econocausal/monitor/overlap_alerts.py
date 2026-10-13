"""Overlap alerts: re-run the gate on fresh data and diff against the run gate."""
from __future__ import annotations

import pandas as pd


def overlap_alert(run_gate: dict, fresh_gate: dict) -> dict:
    d_out = abs(fresh_gate.get("frac_outside_trim", 0)
                - run_gate.get("frac_outside_trim", 0))
    alert = bool(fresh_gate.get("block_policy", False)
                 or d_out > 0.05)
    return {"delta_outside_trim": float(d_out), "alert": alert,
            "fresh_block": bool(fresh_gate.get("block_policy", False))}

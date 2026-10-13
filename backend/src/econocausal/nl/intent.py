"""NL -> typed RunConfig. Rule-based intent parser (no LLM in repo).

Understands: estimator names, budget/cost/margin numbers, target visit/conversion,
source simulator/criteo. Returns a RunConfig dict; API validates it.
"""
from __future__ import annotations

import re

ESTIMATORS = ["causal_forest", "linear_dml", "s_learner", "t_learner", "x_learner"]


def parse(query: str) -> dict:
    q = query.lower()
    cfg: dict = {"seed": 42, "estimator": "linear_dml", "n_folds": 5,
                 "budget": 1000.0, "unit_cost": 0.10, "margin": 20.0,
                 "source": "simulator", "n_rows": 20_000, "target": "visit"}
    for e in ESTIMATORS:
        if e.replace("_", " ") in q or e in q:
            cfg["estimator"] = e
    if "forest" in q:
        cfg["estimator"] = "causal_forest"
    if "criteo" in q:
        cfg["source"] = "criteo"
    if "conversion" in q:
        cfg["target"] = "conversion"
    m = re.search(r"budget\s*(\d+(?:\.\d+)?)", q)
    if m:
        cfg["budget"] = float(m.group(1))
    m = re.search(r"cost\s*(\d+(?:\.\d+)?)", q)
    if m:
        cfg["unit_cost"] = float(m.group(1))
    m = re.search(r"margin\s*(\d+(?:\.\d+)?)", q)
    if m:
        cfg["margin"] = float(m.group(1))
    return cfg

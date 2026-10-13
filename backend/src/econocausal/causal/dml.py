"""DoWhy graph + identification + refutation wrappers (binary T).

Graph: X -> T, X -> Y, T -> Y. Identification: backdoor. Estimation stays in
EconML (ARCHITECTURE decision #2). Refutations here call the real dowhy API;
lightweight sklearn refuters live in causal/refute.py for CI speed.
"""
from __future__ import annotations

import pandas as pd

GRAPH_DOT = """digraph { T; Y; %s; %s -> T; %s -> Y; T -> Y; }"""


def build_graph(feats: list[str]) -> str:
    xs = "; ".join(feats)
    edges_t = "; ".join(f"{c} -> T" for c in feats)
    edges_y = "; ".join(f"{c} -> Y" for c in feats)
    return f"digraph {{ {xs}; T; Y; {edges_t}; {edges_y}; T -> Y; }}"


def identify(feats: list[str]) -> dict:
    return {"estimand": "ATE via backdoor adjustment",
            "graph": build_graph(feats),
            "backdoor_vars": list(feats),
            "assumptions": ["unconfoundedness Y(0),Y(1) indep T|X",
                            "overlap 0<P(T|X)<1", "SUTVA"]}


def dowhy_refute(df: pd.DataFrame, feats: list[str], expected_ate: float,
                 seed: int = 42) -> dict:
    """Real DoWhy refuters: placebo + subset. Slow; use sparingly (CI smoke only)."""
    try:
        import dowhy
        from dowhy import CausalModel
    except Exception as e:
        return {"backend": "dowhy", "status": "skipped", "reason": str(e)[:200]}
    data = df[feats + ["T", "Y"]].copy()
    model = CausalModel(data=data, treatment="T", outcome="Y",
                        graph=build_graph(feats))
    identified = model.identify_effect()
    estimate = model.estimate_effect(identified,
                                     method_name="backdoor.propensity_score_weighting")
    ate = float(estimate.value)
    out: dict = {"backend": "dowhy", "ate_ipw": ate,
                 "expected_ate": float(expected_ate)}
    try:
        p = model.refute_estimate(identified, estimate,
                                  method_name="placebo_treatment_refuter",
                                  placebo_type="permute")
        out["placebo_ate"] = float(p.new_effect)
        out["placebo_pass"] = bool(abs(float(p.new_effect)) < 0.5 * abs(ate) + 1e-6)
    except Exception as e:
        out["placebo_error"] = str(e)[:200]
        out["placebo_pass"] = False
    try:
        s = model.refute_estimate(identified, estimate,
                                  method_name="data_subset_refuter",
                                  subset_fraction=0.5, random_state=seed)
        out["subset_ate"] = float(s.new_effect)
        out["subset_pass"] = bool(abs(float(s.new_effect) - ate) < 0.5 * abs(ate) + 0.02)
    except Exception as e:
        out["subset_error"] = str(e)[:200]
        out["subset_pass"] = False
    out["status"] = ("pass" if out.get("placebo_pass") and out.get("subset_pass")
                     else "needs_review")
    return out

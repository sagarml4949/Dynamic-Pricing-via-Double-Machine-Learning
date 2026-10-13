"""Week 2: Criteo bake-off on confounded resample.

Compares naive / S / T / X / LinearDML / CausalForest on AUUC + policy value
at fixed budget + dollars-wasted-on-Sure-Things logic (predictive vs causal
targeting). No tau_true here -> ranking metrics only, honest per ARCHITECTURE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("backend/src").resolve()))

import numpy as np
import pandas as pd
from econocausal.causal.forest import CausalForestDML
from econocausal.causal.learners import (
    LinearDMLBaseline, SLearner, TLearner, XLearner,
)
from econocausal.causal.propensity import overlap_gate
from econocausal.causal.refute import refute
from econocausal.data.schemas import validate
from econocausal.data.splits import split
from econocausal.evaluate.uplift import auuc
from econocausal.optimize.economics import incremental_profit
from econocausal.optimize.knapsack import optimize
from lightgbm import LGBMClassifier

FEATS = [f"f{i}" for i in range(12)]


def policy_value(df: pd.DataFrame, tau_hat: np.ndarray, budget_n: int,
                 margin: float = 20.0, unit_cost: float = 0.10,
                 y: str = "visit") -> dict:
    profit = incremental_profit(tau_hat, margin, unit_cost)
    cost = np.full(len(df), unit_cost)
    sol = optimize(profit, cost, budget_n * unit_cost, method="lp")
    x = sol["assignment"]
    # honest-ish value: treated-picked outcomes vs control-picked outcomes
    picked = df.iloc[np.where(x == 1)[0]]
    return {"n_picked": int(x.sum()), "expected_profit": sol["expected_profit"],
            "picked_visit_rate": float(picked[y].mean()) if len(picked) else 0.0}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/criteo_confounded.parquet")
    p.add_argument("--y", default="visit", choices=["visit", "conversion"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=60_000)
    p.add_argument("--budget-n", type=int, default=2000)
    a = p.parse_args()

    df = pd.read_parquet(a.data)
    if len(df) > a.max_rows:
        df = df.sample(a.max_rows, random_state=a.seed).reset_index(drop=True)
    df = df.rename(columns={"treatment": "T", "visit": "Y", "conversion": "Yc"})
    df["Y"] = df["Y"] if a.y == "visit" else df["Yc"]
    validate(df, require_cols=FEATS + ["T", "Y"])
    parts = split(df, seed=a.seed)
    tr, ev = parts["train"], parts["eval"]

    gate = overlap_gate(tr, FEATS, seed=a.seed)
    print(f"overlap: outside_trim={gate['frac_outside_trim']:.3f} "
          f"ess_ratio={gate['ess_ratio']:.3f} block={gate['block_policy']} "
          f"smd_flagged={len(gate['smd_flagged'])}")

    models = {
        "s_learner": SLearner(a.seed), "t_learner": TLearner(a.seed),
        "x_learner": XLearner(a.seed), "linear_dml": LinearDMLBaseline(a.seed),
        "causal_forest": CausalForestDML(a.seed, n_estimators=200),
    }
    naive = float(tr[tr["T"] == 1]["Y"].mean() - tr[tr["T"] == 0]["Y"].mean())
    print(f"naive train lift ({a.y}): {naive:.5f}")
    results: dict = {"naive_lift": naive, "overlap_gate": {
        k: v for k, v in gate.items() if k not in ("hist", "hist_edges", "smd")}}
    for name, m in models.items():
        m.fit(tr, FEATS)
        tau = m.predict_tau(ev)
        pv = policy_value(ev, tau, a.budget_n, y="Y")
        interval = ""
        if hasattr(m, "predict_interval"):
            lo, hi = m.predict_interval(ev)
            w = float(np.mean(hi - lo))
            interval = f" meanCIw={w:.4f}"
            results[name] = {"auuc": auuc(ev, tau), **pv, "mean_ci_width": w}
        else:
            results[name] = {"auuc": auuc(ev, tau), **pv}
        print(f"{name:13s} AUUC={results[name]['auuc']:.4f} "
              f"expProfit={pv['expected_profit']:.0f} picked={pv['n_picked']}{interval}")
        if name == "causal_forest":
            ate = float(tau.mean())
            results["refutation"] = refute(tr.sample(min(15000, len(tr)),
                                                     random_state=a.seed).reset_index(drop=True),
                                           FEATS, ate, a.seed)
            print(f"refute: {results['refutation']['status']} "
                  f"(placebo={results['refutation']['placebo_ate']:.4f} vs ate={ate:.4f})")

    # predictive strawman: top-K by P(Y|X) wastes budget on sure things
    clf = LGBMClassifier(n_estimators=150, verbose=-1, random_state=a.seed)
    clf.fit(tr[FEATS].to_numpy(), tr["Y"].to_numpy())
    pred = clf.predict_proba(ev[FEATS].to_numpy())[:, 1]
    pv_pred = policy_value(ev, (pred - pred.mean()) * 0.02, a.budget_n, y="Y")
    results["predictive_targeting"] = {"auuc": auuc(ev, pred), **pv_pred}
    print(f"predictive   AUUC={results['predictive_targeting']['auuc']:.4f} "
          f"expProfit={pv_pred['expected_profit']:.0f} (strawman)")

    out = Path("data/registry") / f"criteo-bakeoff-s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"saved {out}/metrics.json")


if __name__ == "__main__":
    main()

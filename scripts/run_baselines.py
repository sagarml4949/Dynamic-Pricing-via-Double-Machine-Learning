"""Week 1 baseline bake-off on simulator: naive vs S/T/X vs LinearDML.

Writes data/registry/<run> metrics: PEHE (has tau_true), AUUC, ATE error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path("backend/src").resolve()))

from econocausal.causal.learners import (  # noqa: E402
    LinearDMLBaseline, SLearner, TLearner, XLearner,
)
from econocausal.data.simulate import simulate  # noqa: E402
from econocausal.data.splits import split  # noqa: E402
from econocausal.data.schemas import validate  # noqa: E402
from econocausal.evaluate.uplift import auuc, pehe  # noqa: E402
from econocausal.registry.runs import save_run  # noqa: E402

FEATS = ["lapsed", "value", "tenure", "z1", "z2"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    df = simulate(a.n, a.seed)
    validate(df, require_cols=FEATS + ["T", "Y"])
    parts = split(df, seed=a.seed)
    tr, ev = parts["train"], parts["eval"]
    ate_true = float(df["tau_true"].mean())

    models = {
        "s_learner": SLearner(a.seed),
        "t_learner": TLearner(a.seed),
        "x_learner": XLearner(a.seed),
        "linear_dml": LinearDMLBaseline(a.seed),
    }
    metrics: dict = {
        "ate_true": ate_true,
        "naive_ate": float(tr[tr["T"] == 1]["Y"].mean() - tr[tr["T"] == 0]["Y"].mean()),
    }
    for name, m in models.items():
        m.fit(tr, FEATS)
        tau = m.predict_tau(ev)
        metrics[name] = {
            "pehe": pehe(tau, ev["tau_true"].to_numpy()),
            "auuc": auuc(ev, tau),
            "ate_hat": float(tau.mean()),
            "ate_abs_err": abs(float(tau.mean()) - float(ev["tau_true"].mean())),
        }
        print(f"{name:12s} PEHE={metrics[name]['pehe']:.4f} "
              f"AUUC={metrics[name]['auuc']:.4f} "
              f"ATEerr={metrics[name]['ate_abs_err']:.4f}")
    print(f"naive ATE={metrics['naive_ate']:.4f} true={ate_true:.4f}")
    d = save_run(f"sim-baseline-s{a.seed}",
                 {"task": "sim-baseline", "n": a.n, "seed": a.seed}, metrics)
    print(f"saved {d}")


if __name__ == "__main__":
    main()

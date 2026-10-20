"""Honest outer K-fold CV for uplift: refit nuisances per fold, never leak.

Sim (tau_true known): PEHE, ATE error, interval coverage, AUUC.
Criteo (no tau_true): DR pseudo-outcome R2 (Kennedy), R-score (Nie&Wager),
AUUC -- all OUT-OF-FOLD with train-fold nuisances. Paired t-tests included.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path("backend/src").resolve()))

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.model_selection import StratifiedKFold

SIM_FEATS = ["lapsed", "value", "tenure", "z1", "z2"]
CRI_FEATS = [f"f{i}" for i in range(12)]


def make(name: str, seed: int, fast: bool = False):
    from econocausal.causal.forest import CausalForestDML
    from econocausal.causal.learners import (
        LinearDMLBaseline, SLearner, TLearner, XLearner)
    if name == "causal_forest":
        return CausalForestDML(seed, n_estimators=100 if fast else 200)
    if name == "linear_dml":
        return LinearDMLBaseline(seed)
    return {"s_learner": SLearner, "t_learner": TLearner,
            "x_learner": XLearner}[name](seed)


def dr_pseudo_outcome(Y, T, mu1, mu0, e):
    e = np.clip(e, 0.025, 0.975)
    mu_t = np.where(T == 1, mu1, mu0)
    return mu1 - mu0 + (T - e) * (Y - mu_t) / (e * (1 - e))


def fit_nuisances(Xtr, Ttr, Ytr, Xte, seed):
    from lightgbm import LGBMClassifier
    kw = dict(n_estimators=150, min_child_samples=50, verbose=-1, random_state=seed)
    m1 = LGBMClassifier(**kw).fit(Xtr[Ttr == 1], Ytr[Ttr == 1])
    m0 = LGBMClassifier(**kw).fit(Xtr[Ttr == 0], Ytr[Ttr == 0])
    me = LGBMClassifier(n_estimators=100, verbose=-1, random_state=seed).fit(Xtr, Ttr)
    return (m1.predict_proba(Xte)[:, 1], m0.predict_proba(Xte)[:, 1],
            me.predict_proba(Xte)[:, 1])


def evaluate_fold(df_te, tau, mu1, mu0, e, has_truth, y="Y"):
    from econocausal.evaluate.uplift import auuc, pehe
    Y, T = df_te[y].to_numpy(), df_te["T"].to_numpy()
    out = {"auuc": auuc(df_te, tau, y)}
    if has_truth:
        tt = df_te["tau_true"].to_numpy()
        out["pehe"] = pehe(tau, tt)
        out["ate_err"] = abs(float(tau.mean() - tt.mean()))
    dr = dr_pseudo_outcome(Y, T, mu1, mu0, e)
    out["dr_r2"] = float(1 - np.mean((dr - tau) ** 2) / max(np.var(dr), 1e-9))
    mu = np.where(T == 1, mu1, mu0)
    out["r_score"] = float(1 - np.mean(((Y - mu) - tau * (T - e)) ** 2)
                           / max(np.mean((Y - mu) ** 2), 1e-9))
    return out


def run_cv(df, feats, models, n_splits=5, seed=42, y="Y",
           has_truth=False, fast=False, skip_intervals=False):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    T = df["T"].to_numpy()
    per_model: dict = {m: [] for m in models}
    for fold, (tri, tei) in enumerate(skf.split(df, T)):
        dtr = df.iloc[tri].reset_index(drop=True)
        dte = df.iloc[tei].reset_index(drop=True)
        Xtr, Xte = dtr[feats].to_numpy(), dte[feats].to_numpy()
        mu1, mu0, e = fit_nuisances(Xtr, dtr["T"].to_numpy(),
                                    dtr[y].to_numpy(), Xte, seed + fold)
        for m in models:
            t0 = time.time()
            est = make(m, seed + fold, fast)
            est.fit(dtr, feats, y)
            tau = np.asarray(est.predict_tau(dte)).ravel()
            fr = evaluate_fold(dte, tau, mu1, mu0, e, has_truth, y)
            if hasattr(est, "predict_interval") and not skip_intervals:
                try:
                    lo, hi = est.predict_interval(dte)
                    lo, hi = np.asarray(lo).ravel(), np.asarray(hi).ravel()
                    fr["mean_ci_width"] = float(np.mean(hi - lo))
                    if has_truth:
                        tt = dte["tau_true"].to_numpy()
                        fr["coverage_95"] = float(((tt >= lo) & (tt <= hi)).mean())
                except Exception as ex:
                    fr["interval_error"] = str(ex)[:150]
            fr["fit_s"] = round(time.time() - t0, 1)
            per_model[m].append(fr)
            print(f"fold{fold} {m:13s} " + " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in fr.items()), flush=True)
    summary, key = {}, "pehe" if has_truth else "dr_r2"
    for m, folds in per_model.items():
        agg = {}
        for k in folds[0]:
            vs = [f[k] for f in folds if isinstance(f.get(k), (int, float))]
            if vs:
                agg[k] = {"mean": float(np.mean(vs)),
                          "sd": float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0}
        summary[m] = agg
    order = sorted(models, key=lambda m: summary[m][key]["mean"],
                   reverse=(key != "pehe"))
    comp: dict = {"best": order[0], "metric": key,
                  "ranking": order, "paired_tests": {}}
    for m in order[1:]:
        a = np.array([f[key] for f in per_model[order[0]]])
        b = np.array([f[key] for f in per_model[m]])
        d = (a - b) if key == "pehe" else (b - a)
        ts = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
        pv = 2 * float(t_dist.sf(abs(ts), len(d) - 1))
        comp["paired_tests"][f"{order[0]}_vs_{m}"] = {
            "mean_diff": float(d.mean()), "t": float(ts),
            "p": round(pv, 4), "significant_05": bool(pv < 0.05)}
    return {"per_fold": per_model, "summary": summary,
            "comparison": comp, "n_splits": n_splits, "seed": seed}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["sim", "criteo"], default="sim")
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--skip-intervals", action="store_true",
                   help="skip predict_interval in CV (coverage measured separately)")
    p.add_argument("--models", nargs="*", default=["s_learner", "t_learner",
                 "x_learner", "linear_dml", "causal_forest"])
    p.add_argument("--target", choices=["visit", "conversion"], default="visit",
                   help="criteo outcome: visit (dense) or conversion (money metric)")
    a = p.parse_args()
    if a.source == "sim":
        from econocausal.data.simulate import simulate
        df = simulate(a.n, a.seed)
        feats, y, truth = SIM_FEATS, "Y", True
    else:
        df = pd.read_parquet("data/processed/criteo_confounded.parquet")
        df = df.rename(columns={"treatment": "T", "visit": "Y", "conversion": "Yc"})
        df["Y"] = df["Y"] if a.target == "visit" else df["Yc"]
        if len(df) > a.n:
            df = df.sample(a.n, random_state=a.seed).reset_index(drop=True)
        feats, y, truth = CRI_FEATS, "Y", False
    print(f"CV {a.source}/{a.target} n={len(df)} folds={a.folds} models={a.models}")
    res = run_cv(df, feats, a.models, a.folds, a.seed, y, truth, a.fast,
                 skip_intervals=a.skip_intervals)
    out = Path("data/registry") / f"cv-{a.source}-{a.target}-s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        {"summary": res["summary"], "comparison": res["comparison"],
         "source": a.source, "target": a.target, "n": len(df)}, indent=2))
    print(json.dumps(res["comparison"], indent=2))
    for m, agg in res["summary"].items():
        print(f"{m:13s} " + " ".join(
            f"{k}={v['mean']:.4f}+-{v['sd']:.4f}" for k, v in agg.items()
            if k in ("pehe", "ate_err", "auuc", "dr_r2", "r_score",
                     "coverage_95", "mean_ci_width")))
    print(f"saved {out}/summary.json")


if __name__ == "__main__":
    main()

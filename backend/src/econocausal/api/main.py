"""Thin FastAPI glue: validate -> dispatch -> serialize. No business logic here."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="EconoCausal", version="0.2.0")


class RunConfig(BaseModel):
    seed: int = 42
    estimator: Literal["linear_dml", "causal_forest", "s_learner", "t_learner", "x_learner"] = "linear_dml"
    nuisance: Literal["lightgbm"] = "lightgbm"
    n_folds: int = Field(default=5, ge=2, le=10)
    trim: tuple[float, float] = (0.025, 0.975)
    budget: float = Field(default=1000.0, gt=0)
    unit_cost: float = Field(default=5.0, ge=0)
    margin: float = Field(default=20.0, gt=0)
    source: Literal["simulator", "criteo"] = "simulator"
    n_rows: int = Field(default=20_000, ge=1000, le=200_000)
    target: Literal["visit", "conversion"] = "visit"


def _pkg():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    return sys


_RUNS: dict[str, dict] = {}


@app.post("/v1/runs")
def create_run(cfg: RunConfig):
    run_id = uuid.uuid4().hex[:12]
    _RUNS[run_id] = {"config": cfg.model_dump(), "status": "created"}
    return {"run_id": run_id, "status": "created", "config": cfg.model_dump()}


@app.post("/v1/runs/{run_id}/fit")
def fit_run(run_id: str):
    run = _RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run_id")
    _pkg()
    import numpy as np
    import pandas as pd
    cfg = run["config"]
    if cfg["source"] == "simulator":
        from econocausal.data.simulate import simulate
        df = simulate(cfg["n_rows"], cfg["seed"])
        feats = ["lapsed", "value", "tenure", "z1", "z2"]
    else:
        df = pd.read_parquet("data/processed/criteo_confounded.parquet")
        df = df.rename(columns={"treatment": "T", "visit": "Y", "conversion": "Yc"})
        df["Y"] = df["Y"] if cfg["target"] == "visit" else df["Yc"]
        feats = [f"f{i}" for i in range(12)]
        if len(df) > cfg["n_rows"]:
            df = df.sample(cfg["n_rows"], random_state=cfg["seed"]).reset_index(drop=True)
    from econocausal.causal.propensity import overlap_gate
    from econocausal.data.schemas import validate
    from econocausal.data.splits import split
    validate(df, require_cols=feats + ["T", "Y"])
    parts = split(df, seed=cfg["seed"])
    tr, ev = parts["train"], parts["eval"]
    gate = overlap_gate(tr, feats, seed=cfg["seed"], lo=cfg["trim"][0], hi=cfg["trim"][1])
    name = cfg["estimator"]
    if name == "causal_forest":
        from econocausal.causal.forest import CausalForestDML
        m = CausalForestDML(cfg["seed"], n_estimators=200, n_folds=cfg["n_folds"])
    else:
        from econocausal.causal.learners import (
            LinearDMLBaseline, SLearner, TLearner, XLearner)
        m = {"linear_dml": LinearDMLBaseline(cfg["seed"], cfg["n_folds"]),
             "s_learner": SLearner(cfg["seed"]), "t_learner": TLearner(cfg["seed"]),
             "x_learner": XLearner(cfg["seed"])}[name]
    m.fit(tr, feats)
    tau = np.asarray(m.predict_tau(ev)).ravel()
    lo = hi = None
    if hasattr(m, "predict_interval"):
        lo, hi = m.predict_interval(ev)
        lo, hi = np.asarray(lo).ravel(), np.asarray(hi).ravel()
    run.update({"status": "fitted", "tau": tau.tolist(),
                "lo": lo.tolist() if lo is not None else None,
                "hi": hi.tolist() if hi is not None else None,
                "eval_n": len(ev),
                "gate": {k: v for k, v in gate.items()
                         if k not in ("hist", "hist_edges", "smd")}})
    Path("data/registry").mkdir(parents=True, exist_ok=True)
    (Path("data/registry") / f"{run_id}.json").write_text(json.dumps(
        {"run_id": run_id, "config": cfg,
         "ate_hat": float(tau.mean()), "gate": run["gate"]}, indent=2))
    return {"run_id": run_id, "status": "fitted",
            "ate_hat": float(tau.mean()), "eval_n": len(ev),
            "overlap_gate": run["gate"]}


@app.get("/v1/runs/{run_id}/effects")
def get_effects(run_id: str):
    run = _RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run_id")
    if run.get("tau") is None:
        raise HTTPException(409, "run not fitted: POST /v1/runs/{id}/fit first")
    return {"run_id": run_id, "status": "fitted",
            "tau_hat": run["tau"], "lo": run["lo"], "hi": run["hi"]}


@app.post("/v1/runs/{run_id}/policy:optimize")
def optimize_policy(run_id: str):
    run = _RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run_id")
    if run.get("tau") is None:
        raise HTTPException(409, "run not fitted: POST /v1/runs/{id}/fit first")
    _pkg()
    import numpy as np
    from econocausal.optimize.economics import incremental_profit
    from econocausal.optimize.knapsack import optimize
    cfg = run["config"]
    tau = np.asarray(run["tau"])
    profit = incremental_profit(tau, cfg["margin"], cfg["unit_cost"])
    cost = np.full(len(tau), cfg["unit_cost"])
    sol = optimize(profit, cost, cfg["budget"], method="lp")
    return {"run_id": run_id, "status": "fitted",
            "assignment": np.asarray(sol["assignment"]).tolist(),
            "expected_profit": sol["expected_profit"],
            "spent": sol["spent"], "method": sol["method"]}


@app.get("/v1/runs/{run_id}/report")
def get_report(run_id: str):
    run = _RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run_id")
    _pkg()
    import numpy as np
    metrics = {}
    if run.get("tau") is not None:
        tau = np.asarray(run["tau"])
        metrics = {"ate_hat": float(tau.mean()),
                   "frac_positive_tau": float((tau > 0).mean())}
    return {"run_id": run_id, "config": run["config"], "status": run["status"],
            "metrics": metrics, "overlap_gate": run.get("gate", {}),
            "provenance": {"mocked": False, "api_version": "0.2.0"}}

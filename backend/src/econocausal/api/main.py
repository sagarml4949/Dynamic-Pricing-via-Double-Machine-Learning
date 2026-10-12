"""Thin FastAPI glue: validate -> dispatch -> serialize. No business logic here."""
from __future__ import annotations

import uuid
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="EconoCausal", version="0.1.0")


class RunConfig(BaseModel):
    seed: int = 42
    estimator: Literal["linear_dml", "causal_forest", "s_learner", "t_learner", "x_learner"] = "linear_dml"
    nuisance: Literal["lightgbm"] = "lightgbm"
    n_folds: int = Field(default=5, ge=2, le=10)
    trim: tuple[float, float] = (0.025, 0.975)
    budget: float = Field(default=1000.0, gt=0)
    unit_cost: float = Field(default=5.0, ge=0)
    margin: float = Field(default=20.0, gt=0)


_RUNS: dict[str, dict] = {}


@app.post("/v1/runs")
def create_run(cfg: RunConfig):
    run_id = uuid.uuid4().hex[:12]
    _RUNS[run_id] = {"config": cfg.model_dump(), "status": "created"}
    return {"run_id": run_id, "status": "created", "config": cfg.model_dump()}


@app.get("/v1/runs/{run_id}/effects")
def get_effects(run_id: str):
    run = _RUNS.get(run_id)
    if run is None:
        from fastapi import HTTPException
        raise HTTPException(404, "unknown run_id")
    # Week 1 stub: mocked effects allowed only until estimators wire in (Week 1 exit).
    return {"run_id": run_id, "status": "mocked-week1",
            "tau_hat": [], "lo": [], "hi": [],
            "note": "wire to causal.learners in Week 1 close-out"}


@app.post("/v1/runs/{run_id}/policy:optimize")
def optimize_policy(run_id: str):
    run = _RUNS.get(run_id)
    if run is None:
        from fastapi import HTTPException
        raise HTTPException(404, "unknown run_id")
    return {"run_id": run_id, "status": "mocked-week1",
            "assignment": [], "expected_profit": 0.0}


@app.get("/v1/runs/{run_id}/report")
def get_report(run_id: str):
    run = _RUNS.get(run_id)
    if run is None:
        from fastapi import HTTPException
        raise HTTPException(404, "unknown run_id")
    return {"run_id": run_id, "config": run["config"],
            "metrics": {}, "refutations": {}, "provenance": {"mocked": True}}

import sys
from pathlib import Path

sys.path.insert(0, str(Path("backend/src").resolve()))

import numpy as np
import pandas as pd
from econocausal.causal.dml import build_graph, identify
from econocausal.causal.intervals import coverage, residual_band
from econocausal.causal.propensity import overlap_gate
from econocausal.data.schemas import validate
from econocausal.evaluate.calibration import calibration_by_decile
from econocausal.evaluate.qini import policy_value_table, qini
from econocausal.monitor.drift import drift_report, psi
from econocausal.monitor.overlap_alerts import overlap_alert
from econocausal.nl.intent import parse
from econocausal.optimize.economics import greedy_policy, incremental_profit
from econocausal.optimize.greedy import roi_order
from econocausal.optimize.knapsack import knapsack_lp, knapsack_milp, optimize
from econocausal.registry.store import load_frame, save_frame


def test_graph_identify_lists_backdoor():
    g = identify(["a", "b"])
    assert g["backdoor_vars"] == ["a", "b"]
    assert "T -> Y" in g["graph"]
    assert "unconfoundedness" in g["assumptions"][0]


def test_nl_parses_estimator_budget_source():
    cfg = parse("run causal forest on criteo with budget 500")
    assert cfg["estimator"] == "causal_forest"
    assert cfg["source"] == "criteo" and cfg["budget"] == 500.0


def test_intervals_cover_and_band():
    tau = np.array([0.1, 0.2, 0.3])
    lo, hi = residual_band(tau, np.array([1, 0, 1]), np.array([1, 0, 1]))
    assert ((lo <= tau) & (tau <= hi)).all()
    assert coverage(np.array([0.15]), np.array([0.0]), np.array([0.2])) == 1.0


def test_qini_and_policy_table():
    df = pd.DataFrame({"T": [1, 1, 0, 0], "Y": [1, 0, 1, 0]})
    tau = np.array([0.5, 0.4, -0.1, -0.2])
    assert isinstance(qini(df, tau), float)
    tab = policy_value_table(df, tau, budgets=[1, 2], unit_cost=0.1)
    assert list(tab.columns) == ["budget_n", "n_picked", "expected_profit", "picked_rate"]


def test_calibration_deciles():
    df = pd.DataFrame({"T": [1, 0] * 10, "Y": [1, 0] * 10})
    tau = np.linspace(-0.1, 0.3, 20)
    cal = calibration_by_decile(df, tau)
    assert "uplift" in cal.columns and len(cal) > 0


def test_drift_and_overlap_alert():
    ref = pd.DataFrame({"x": np.zeros(200)})
    same = pd.DataFrame({"x": np.zeros(200)})
    assert drift_report(ref, same, ["x"])["x"]["alert"] is False
    assert psi(np.zeros(100), np.zeros(100)) < 0.05
    a = overlap_alert({"frac_outside_trim": 0.01}, {"frac_outside_trim": 0.10,
                                                   "block_policy": False})
    assert a["alert"] is True


def test_registry_store_roundtrip(tmp_path, monkeypatch):
    import econocausal.registry.store as st
    monkeypatch.setattr(st, "REG", tmp_path)
    df = pd.DataFrame({"a": [1, 2]})
    p = save_frame("r1", df)
    assert p.exists()
    assert load_frame("r1").equals(df)


def test_roi_order_sorts_descending():
    o = roi_order(np.array([1.0, 5.0, 3.0]), np.array([1.0, 1.0, 1.0]))
    assert list(o) == [1, 2, 0]

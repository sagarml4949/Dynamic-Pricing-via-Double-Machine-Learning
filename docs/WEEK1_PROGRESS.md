# Week 1 Progress — EconoCausal: Dynamic Pricing via Double ML
**Period:** 10 Oct → 12 Oct 2026 (Week 1 of 10 Oct–10 Nov timeline)
**Repo:** https://github.com/sagarml4949/Dynamic-Pricing-via-Double-Machine-Learning (2 commits, main clean)
**Status:** Week 1 exit criteria met — design + data + simulator + baselines + API stub + CI green.

## 1. Goal vs outcome
Goal (ARCHITECTURE.md Appendix A): design contract, clean-clone skeleton, synthetic simulator with tau_true, pandera schemas + overlap payload + /v1/runs stub, checked-in openapi.
Outcome: all done except full React dashboard and CausalForest primary (correctly deferred to Week 2). Tests 3/3 pass. No secrets or data committed.

## 2. Architecture (docs/ARCHITECTURE.md, 214 lines)
- System: React (DAG/Overlap/ITE/Policy/Audit) → FastAPI /v1/* → data → causal → evaluate/optimize → registry. Dependency rule enforced.
- Pipeline with gates: validate (pandera 422) → seeded split → propensity/overlap gate (blocks policy if >15% mass outside [0.025,0.975]) → 5-fold DML → ITE+CI → DoWhy refute (needs_review on fail) → economics (tau*margin−cost) → knapsack (HiGHS default, MILP N≤2k) → report with run_id+sha+seed+config_hash.
- Lineup: naive (bias demo), S/T/X (baselines), LinearDML (ship-first), CausalForestDML (primary W2). Policies: random/predictive/knapsack/oracle(synthetic-gated).
- 12 decisions + assumptions (unconfoundedness, overlap, SUTVA, no post-T, binary-T) + 10 risks with fallbacks.

## 3. Data: Criteo Uplift (docs/DATA_CRITEO.md)
- Source: public HF mirror criteo/criteo-uplift (v2.1 unbiased, CC-BY-NC-SA-4.0, 13.98M rows). Kaggle slug artembet/criteo-uplift-dataset attempted; KGAT token returned 403 → fell back to HF, token never written to disk.
- Schema: f0..f11 float, treatment 0/1, exposure 0/1, visit + conversion binary. Zero nulls.
- Sample (10k strided): T-rate 0.909, visit 0.0120 control vs 0.0197 treated (+0.0077), conversion 0.00219 vs 0.00253 (+0.00034).
- Gotchas found: file is RCT-sorted (first ~12M rows treated — never head()); treatment (auction entry, randomized) ≠ exposure (actually shown).
- Local: data/raw/criteo_sample.parquet + full file in HF cache. scripts/fetch_criteo.py supports Kaggle/HF-full/HF-sample.

## 4. Confounded resample (scripts/make_confounded.py)
Simulated targeted history: keep treated high-f0 w.p.→1, drop treated low-f0 (keep_lo 0.15), controls untouched, seed 42, n=200k→127.6k.
- Visit naive: RCT +0.01040 → confounded −0.00136; bias −0.01176 (−113%, sign flip).
- Conversion naive: +0.00095 → +0.00041; bias −0.00053 (−56%).
- Treated rate 0.851→0.766. Receipt: data/processed/criteo_confounded.json.
- Learning: targeting bent toward high-f0 users who visit less, so bias went negative; Week 2 flips direction (target low-f0) so naive overstates — the classic story.

## 5. Simulator + baselines (scripts/run_baselines.py, data/registry/sim-baseline-s42/)
Simulator (backend/src/econocausal/data/simulate.py): 4 types (persuadable/sure_thing/lost_cause/sleeping_dog), known tau_true, confounded T~sigmoid(lapsed+value). n=20k seed 42, ATE_true 0.1676.
| model | PEHE ↓ | AUUC ↑ | ATE err ↓ |
| S-learner | 0.4713 | 0.1693 | 0.0128 |
| T-learner | 0.5069 | 0.1458 | 0.0210 |
| X-learner | 0.4816 | 0.1609 | 0.0156 |
| LinearDML | 0.4643 | 0.1627 | 0.0102 |
| naive ATE | — | — | 0.0360 (0.1317 vs 0.1676, −21%) |
LinearDML best PEHE/ATE; S-learner best AUUC. Confounding story holds on sim.

## 6. Backend + API + quality
- Packages: data (simulate/schemas/splits), causal (S/T/X/LinearDML, one interface), optimize (pure-numpy economics+greedy), evaluate (AUUC/PEHE), registry (hash+save_run), api (thin FastAPI).
- API verified via TestClient: POST /v1/runs → 200 run_id; GET effects / POST policy:optimize / GET report all 200 (mocked-week1, to be wired W2). frontend/openapi.json checked in.
- Tests 3/3: sleeping-dog profit −25, greedy respects budget, schema rejects post-T column.
- Tooling: backend/pyproject.toml (pinned), Makefile (setup/test/lint/run-api/synth/baselines), CI (py3.11 + pytest).
- Fixes: df.T→df["T"] transpose bug; EconML LinearDML needs LGBMRegressor nuisances + discrete_treatment=True on this version.

## 7. Known gaps / risks
- Env: dev on Python 3.14, spec 3.11 — pyproject pins ==3.11 so clean-clone install needs 3.11 venv/Docker (W2 setup).
- EconML nuisance-as-regressor workaround vs ARCHITECTURE's classifier wording — reconcile on version pin.
- Criteo-confounded bake-off, CausalForest, DoWhy refutations, knapsack wiring, overlap gate, React dashboard → Week 2.
- Repo visibility: confirm private per company policy. Commit identity currently placeholder EconoCausal Dev.

## 8. Week 2 plan
1. Flip confounded targeting (low-f0) + Criteo bake-off with QINI/AUUC/policy-value + dollars-wasted-on-Sure-Things table.
2. CausalForestDML primary + honest intervals; pin env on 3.11.
3. DoWhy graph + placebo/subset refutations; overlap gate enforcement.
4. Wire estimators into API (drop mocked-week1) + knapsack optimizer; start React dashboard (DAG + overlap + ITE + policy).

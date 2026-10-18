# Week 2–3 Progress — EconoCausal
**Period:** 12 Oct → 18 Oct 2026 (Weeks 2–3 of 10 Oct–10 Nov timeline)
**Base:** `a4b2340` + this commit. Tests 16/16 green.

## 1. Targeting flip (scripts/make_confounded.py --direction low)
Old direction kept treated high-f0 → bias went negative (visit −113%, sign flip).
New default keeps treated LOW-f0 → naive OVERSTATES lift (classic story):
- Visit: RCT +0.0104 → confounded +0.0216 (+108% overstatement).
- Conversion: +0.00095 → +0.00184 (+94%).
- Treated rate 0.851 → 0.766, n=200k → 127.5k, seed 42.

## 2. CausalForestDML primary (causal/forest.py)
EconML CausalForestDML, LightGBM nuisances, 5-fold cross-fit, seeded.
n_estimators must be divisible by 4 (EconML subforest constraint).
Subsample fallback past 20k rows; analytic intervals kept but flagged
under-covering (see §5).

## 3. Overlap gate (causal/propensity.py)
LGBM propensity, SMD flags, ESS, trim fractions. On confounded Criteo train:
0.1% outside [0.025,0.975], ESS ratio 0.69, 5 SMD flags (f0/f2/f3/f8/f9 —
confounding visible), block_policy=False. Wired into POST /fit.

## 4. Refutations (causal/refute.py + causal/dml.py)
Lightweight placebo/subset/noise refuters: PASS on Criteo
(placebo 0.001–0.004 vs ATE ~0.009). Real DoWhy wrapper (dowhy_refute) exists;
dowhy 0.8 on py3.14 untested — lightweight path is the CI gate.

## 5. Interval repair — verdict: variance path broken, claim fixed not faked
- EconML 0.17 analytic coverage: ~45–55% at nominal 95% (measured, sim).
- Root cause: version rejects classifier nuisances (raises in _fit_fold), so
  LGBMRegressor-for-T workaround corrupts the variance; plus over-shrinkage
  (tau_hat sd 0.19 vs true 0.50, corr 0.34 on old DGP).
- Bootstrap percentile: ~51% — honest for variance, not for shrinkage bias.
- DR-conformal: 100% cover at width 4.7 on tau∈{−1,0,1} — vacuous (IPW noise).
- Fix: rebuilt simulator with learnable heterogeneity (corr 0.34→0.83,
  balanced 4 types); shipped bootstrap default + conformal helper; dashboard
  must label bands with MEASURED CV coverage, never nominal 95%.

## 6. Simulator fix (data/simulate.py, sharp=2.0)
Type scores now strong separated functions of X + N(0,1) noise.
Balanced: sure_thing 3743 / persuadable 2969 / lost_cause 2799 /
sleeping_dog 2489 (n=12k, seed 7). CF corr 0.83, DML 0.66.
Ranking flipped vs old DGP: metalearners now beat LinearDML —
X PEHE 0.370 / S 0.371 / T 0.388 / LinearDML 0.501 (5-fold CV, §7).
Naive overstates ATE 84% (0.0757 vs 0.0411 true).

## 7. Honest 5-fold CV (scripts/cv_all.py)
Outer StratifiedKFold, per-fold nuisance refit, out-of-fold tau.
Sim (tau_true): PEHE/ATE/AUUC/coverage. Criteo (no truth): DR-R²/R-score/AUUC.
Paired t-tests on primary metric.
- Sim: X 0.370±0.005 / S 0.371±0.006 / T 0.388±0.005 / DML 0.501±0.004.
  DR-R²: X 0.411 / S 0.409 / T 0.392 / DML 0.260. R-scores negative
  everywhere (residual scale) — ranking metric is DR-R²/PEHE, not R-score.
- Criteo (60k confounded): DR-R² ≈ 0 for all (S −0.001 / DML +0.002 /
  T −0.026); AUUC ~0.017–0.020. Weak learnable heterogeneity for visit —
  conversion-target CV still open.
- Forest sim folds: running in background job at commit time; results append
  to data/registry/cv-sim-s42/ (gitignored).

## 8. Backend completion (Week 3)
All ARCHITECTURE.md modules now exist: data/{loader,schemas,simulate,splits},
causal/{propensity,dml,forest,learners,refute,intervals},
optimize/{economics,knapsack,greedy}, evaluate/{uplift,qini,policy_value,
calibration}, monitor/{drift,overlap_alerts}, nl/{intent,prompts},
registry/{runs,store}, api/main (+graph/identify, /nl/query, /calibration).
API verified end-to-end (create→fit→effects→optimize→report, all 200;
sim S-learner: ATE 0.155, $930 profit on $1000 budget).
Knapsack: HiGHS LP + MILP≤2k + greedy + empty-on-negative. 9 new tests.

## 9. Economics calibration note
Criteo visit lift ~1pp at margin $20 clears only unit_cost ≈ $0.10.
Bake-off default lowered $5 → $0.10; at $5 the honest answer is "don't spend"
(R7). Profit LEVELS are illustrative; RANKINGS (AUUC/DR-R²) are the claim.

## 10. Remaining
- Forest sim CV folds (background) + measured-coverage label.
- Conversion-target CV (money metric may carry more signal).
- React dashboard (only big-ticket item left; API serves everything).
- DoWhy real-refuter validation, Docker/uv.lock/py3.11, repo visibility check.

## 11. Forest validation verdict (18 Oct, single 60/20/20 split, n=20k, seed 42)
- Forest: PEHE 0.3772, ATE err 0.0015, AUUC 0.1801, corr 0.828, fit 9s.
- Sits between X-learner (PEHE 0.370) and S-learner (0.371) — competitive,
  not dominant. Primary status justified by honest-split structure + interval
  support, NOT by a PEHE lead. Full 5-fold forest CV still open (slow).
- Interval labeling rule (binding): no band is shown without its MEASURED
  coverage from sim CV. Analytic ~45-55%, bootstrap ~51%, DR-conformal 100%
  at vacuous width. Dashboard renders band + measured number, never "95%".


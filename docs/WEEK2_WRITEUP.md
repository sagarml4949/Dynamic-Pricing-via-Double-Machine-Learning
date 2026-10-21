# Week 2 write-up (19 Oct – 26 Oct)

**Repo:** `Dynamic-Pricing-via-Double-Machine-Learning` · **Commits:** 3 (`9a5f2e9`, `418f821`, `24e662d`)

## What I built this week

**1. Finished the honest cross-validation for the causal forest.**
The forest is the primary model, so I ran the full 5-fold outer CV on the simulator (n=20k, seed 42) and compared it against all four baselines:

| Model | PEHE (↓) | AUUC (↑) | DR-R² (↑) |
|---|---|---|---|
| Causal forest | **0.3691 ± 0.0064** | 0.1905 ± 0.0117 | 0.4110 ± 0.0082 |
| X-learner | 0.370 | — | 0.411 |
| S-learner | 0.371 | — | 0.409 |
| T-learner | 0.388 | — | 0.392 |
| LinearDML | 0.501 | — | 0.260 |

The forest comes out best, but I deliberately did **not** claim a win: the gap to the X- and S-learners (~0.001–0.002) is inside fold-to-fold noise. The honest verdict is *competitive, not dominant* — the forest's status rests on the honest-split structure and its support for intervals, not on a PEHE lead. The ranking also flipped from my first hypothesis: the meta-learners now beat LinearDML, which is only a good baseline when heterogeneity is linear.

**2. Made the CV harness practical.** Bootstrap interval refits pushed a single fold from ~9 seconds to 140–666 seconds, which made full CV infeasible. I decoupled the two: `--skip-intervals` runs the point-estimate CV fast (~8–19 s/fold), and coverage is measured as a separate, explicit step.

**3. Ran the conversion-target CV** (the money metric that was the last open question from the Criteo side). Added a `--target visit|conversion` flag to the harness and ran 5-fold CV on 60k confounded Criteo rows for the conversion outcome:

- DR-R²: forest **+0.0023 ± 0.0158** (the only positive) / S-learner −0.0027 / LinearDML −0.0049 / X-learner −0.0135 / T-learner −0.0374
- AUUC ~0.001–0.002 for every model; R-scores negative throughout

Verdict: **the conversion outcome carries no more learnable signal than the visit outcome at this sample size.** The sparsity (0.36% conversion rate) cancels out the advantage of targeting a monetary outcome. So the Criteo story stays what the data supports — weak heterogeneity, honest model ranking, illustrative dollars — rather than a stronger claim I can't back.

**4. Scaffolded the React dashboard.** Vite + React + TypeScript + Tailwind + Plotly + React Flow, with a typed API client written against the checked-in `openapi.json` and a Vite dev proxy to the backend on :8000. Four views wired: causal DAG / identification, ITE distribution, policy optimisation with expected economics, and the run report. The scaffold is committed; `npm install` still needs to be run on a machine with npm registry access (it timed out on mine).

## Fixes and blockers resolved

- **Interval coverage was badly broken and I refused to hide it.** EconML 0.17's analytic intervals cover only 32–55% at a nominal 95%. I traced the root cause (the version rejects classifier nuisances, so the variance path is corrupted) and fixed the underlying problem instead of the symptom: I rebuilt the simulator with genuinely learnable heterogeneity, which lifted forest↔truth correlation from ~0.34 to ~0.83. Bootstrap also under-covers because point estimates are shrunk, and DR-conformal covers but with vacuous width. I therefore set a **binding rule for the whole project: no band is ever shown without its measured coverage — never a nominal "95%"**.
- **The simulator was too weak to learn from.** Type scores are now strong separated functions of X plus noise, with balanced response types (sure-thing / persuadable / lost-cause / sleeping-dog). This is what flipped the model ranking.
- **Naive comparison overstates the lift by 84%** (0.0757 vs 0.0411 true in the simulator; +108% on visit, +94% on conversion for Criteo) — the confounded-vs-RCT comparison now holds up in both directions of targeting.

## Blockers and open items

- `npm install` is blocked on this machine (npm registry unreachable — it hung on `npm create vite`), so the dashboard is scaffolded but not yet installed/built/screenshotted. This is environmental, not a code issue.
- The real DoWhy refuter (`dowhy_refute`) is still unvalidated on the installed Python 3.14 environment; the lightweight placebo/subset/noise refuters remain the CI gate until that's resolved.
- Hardening (uv.lock, Python 3.11 standardisation, Docker, repo visibility check) is still open.

## Learnings

- **A weak simulator is a lie detector, but only if you fix it honestly.** My first "no model beats the baseline" result was partly the simulator's fault. Rebuilding the DGP made heterogeneity learnable — and then the meta-learners beat LinearDML, which is the more interesting result.
- **Bootstrap intervals are expensive and still not honest.** Separating point-estimate CV from coverage measurement was the unlock for finishing the forest evaluation at all.
- **The sparser metric is not the better one.** I expected conversion to be the more valuable target; the CV said it is statistically indistinguishable from visit, and the money-metric advantage evaporates at 0.36% conversion rate.
- **"Best by a hair" is not a win.** I'm reporting the forest as competitive rather than dominant because the fold noise exceeds the gap, and that is the number I would have to defend in review.

## Next step (the one with the highest leverage)

**Get the dashboard running end-to-end and screenshot it** — install dependencies on a machine with registry access, start the API, click through create → fit → effects → policy → report, and fix whatever breaks (CORS/proxy, type drift against the OpenAPI contract, and wiring the measured-coverage labels into the band displays so the UI never implies nominal 95%). A working, screenshot-able demo changes the conversation more than another backend refinement, and it also forces the honest-coverage rule to be visible in the product rather than only in the docs.

After that: validate or explicitly document the DoWhy refuter path, lock the environment (uv.lock + Python 3.11), and add Docker so the whole stack — API + dashboard — runs from a clean clone.


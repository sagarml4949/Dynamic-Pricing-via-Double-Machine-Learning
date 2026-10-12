# Data card: Criteo Uplift Prediction (Week 1 slice)

Source: `criteo/criteo-uplift` on Hugging Face (public mirror of Criteo AI Lab v2.1,
Diemert et al. AdKDD 2018). CC-BY-NC-SA-4.0, research only.
Kaggle slug `artembet/criteo-uplift-dataset` is the same family; use it via
`scripts/fetch_criteo.py --full --source kaggle` once your KGAT token is fresh
(current token returned 403 on 12 Oct 2026 -> fell back to HF, no token needed).

Local: `data/raw/criteo_sample.parquet` (10k strided rows, full file cached in HF hub).
Full: 13,979,592 rows, ~311MB parquet. Columns: f0..f11 float, treatment 0/1,
exposure 0/1, visit 0/1, conversion 0/1. Zero nulls in sample.

## Sample profile (10k, seeded stride)

- treatment rate 0.909, exposure|T=1 ~0.017 (exposure=0 whenever treatment=0)
- visit: 0.0120 control vs 0.0197 treated -> naive lift +0.0077
- conversion: 0.00219 vs 0.00253 -> naive lift +0.00034
- File is RCT-sorted (first ~12M rows treated); always stride/shuffle, never head().

## Mapping to EconoCausal (ARCHITECTURE.md)

- X = f0..f11 (pre-treatment, anonymized). T = `treatment` (auction participation;
  randomized, so unconfoundedness holds by design here). Y = `visit` (primary,
  denser) or `conversion` (sparse, real money metric). `exposure` kept for
  treatment-vs-exposure sensitivity only, never as default T.
- Confounding note: this file is near-RCT, so DML's de-biasing has little to do.
  To honor the project story (targeted history), Week 2 builds a confounded
  resample (drop treated low-value users) where naive is biased and DML must fix it.
- Persuadable/Sure-Thing/etc. are latent here (no tau_true); use QINI/AUUC/policy
  value, not PEHE. PEHE only on simulator.

## Next

- `scripts/fetch_criteo.py` supports Kaggle full + HF full + HF sample.
- `data/processed/` gets train/eval/policy splits (seeded, stratify T) in Week 1 slice.
- Economics: set margin/cost per run config; conversion is the profit event.

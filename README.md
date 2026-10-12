# EconoCausal: Dynamic Pricing via Double ML

Standard ML predicts who will buy. We estimate who buys **because of** a discount (ITE)
and spend a fixed budget only where it adds profit.

- **Data:** Criteo Uplift Prediction (public HF mirror `criteo/criteo-uplift`, CC-BY-NC-SA-4.0).
  Run `python scripts/fetch_criteo.py --n-rows 200000` for a sample (no token).
  Kaggle full needs `KAGGLE_API_TOKEN` in env (never committed).
- **Design:** see `docs/ARCHITECTURE.md`. Pipeline: validate → split → propensity/overlap
  gate → Double ML (5-fold) → ITE + intervals → refutation → economics → knapsack → report.
- **Status:** Week 1. Confounded resample built (`scripts/make_confounded.py`);
  S/T/X + LinearDML baselines next.

## Quickstart

```bash
python scripts/fetch_criteo.py --n-rows 200000 --out data/raw/criteo_sample.parquet
python scripts/make_confounded.py --n 200000 --seed 42
```

## Privacy

Private repo until company approves otherwise. Never commit tokens (`.env`),
raw/processed data (`data/` is gitignored), or confidential internship docs.

"""Fetch Criteo Uplift Prediction Dataset.

Priority:
  1. Kaggle (full, ~25M or 13.9M rows) if KAGGLE_API_TOKEN is valid in env.
     PowerShell:  $env:KAGGLE_API_TOKEN="KGAT_..."
     Never commit the token. It is read from env only.
  2. Hugging Face public mirror `criteo/criteo-uplift` (14M rows, CC-BY-NC-SA-4.0)
     No token needed. Used as automatic fallback + for fast dev samples.

Usage:
  python scripts/fetch_criteo.py --n-rows 200000 --out data/raw/criteo_sample.parquet
  python scripts/fetch_criteo.py --full --source kaggle --slug artembet/criteo-uplift-dataset
  python scripts/fetch_criteo.py --full --source hf

Schema (both sources):
  f0..f11: float features (anonymized) | treatment: 0/1 (auction participation)
  exposure: 0/1 (actually shown) | visit, conversion: binary outcomes
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

EXPECTED = [f"f{i}" for i in range(12)] + ["treatment", "exposure", "visit", "conversion"]


def fetch_hf_sample(n_rows: int, out: Path) -> Path:
    from datasets import load_dataset

    print(f"[hf] streaming criteo/criteo-uplift, taking {n_rows} rows ...")
    ds = load_dataset("criteo/criteo-uplift", split="train", streaming=True)
    rows = list(ds.take(n_rows))
    import pandas as pd

    df = pd.DataFrame(rows)
    # HF parquet preview sometimes lacks some cols in streaming dict; normalize
    for c in EXPECTED:
        if c not in df.columns:
            print(f"[hf] WARNING missing col {c}", file=sys.stderr)
    df.to_parquet(out, index=False)
    print(f"[hf] wrote {out} shape={df.shape}")
    print(df.head(3).to_string())
    return out


def fetch_hf_full(out_dir: Path) -> Path:
    from datasets import load_dataset

    print("[hf] downloading full criteo/criteo-uplift (parquet, ~311MB) ...")
    ds = load_dataset("criteo/criteo-uplift", split="train")
    out = out_dir / "criteo_full.parquet"
    ds.to_parquet(str(out))
    print(f"[hf] wrote {out} rows={len(ds)}")
    return out


def fetch_kaggle(slug: str, out_dir: Path) -> Path:
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token.startswith("KGAT_"):
        raise SystemExit(
            "KAGGLE_API_TOKEN missing/invalid in env. Set it for this session only:\n"
            '  PowerShell: $env:KAGGLE_API_TOKEN="KGAT_..."\n'
            "  Get a fresh one at https://www.kaggle.com/settings/api -> Generate New Token.\n"
            "  (403 = expired/revoked token or no dataset consent. Do NOT put it in files.)"
        )
    try:
        import kagglehub

        print(f"[kaggle] downloading {slug} ...")
        path = kagglehub.dataset_download(slug)
        print(f"[kaggle] cached at {path}")
        return Path(path)
    except Exception as e:
        raise SystemExit(
            f"[kaggle] download failed ({e}).\n"
            "Fallback: python scripts/fetch_criteo.py --full --source hf\n"
            "If 403 persists, regenerate token at https://www.kaggle.com/settings/api"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-rows", type=int, default=200_000)
    p.add_argument("--out", type=str, default="data/raw/criteo_sample.parquet")
    p.add_argument("--full", action="store_true")
    p.add_argument("--source", choices=["hf", "kaggle"], default="hf")
    p.add_argument("--slug", default="artembet/criteo-uplift-dataset")
    a = p.parse_args()

    if a.full and a.source == "kaggle":
        fetch_kaggle(a.slug, RAW)
    elif a.full:
        fetch_hf_full(RAW)
    else:
        fetch_hf_sample(a.n_rows, Path(a.out))


if __name__ == "__main__":
    main()

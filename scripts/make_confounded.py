"""Build confounded resample: simulate targeted history from near-RCT Criteo data.

Story: past campaigns targeted high-value users. We mimic by keeping treated
high-value rows with high prob and dropping treated low-value rows, while
keeping all controls. Result: T correlates with X (value) AND Y -> naive biased.

Value proxy: f0 (7362 uniques, widest spread; f9/f4 near-constant in sample).
  keep_prob(T=1) = 0.15 + 0.85 * rank(f0)   -> high-f0 treated kept, low-f0 dropped
Controls untouched. Seed fixed. Writes data/processed/criteo_confounded.parquet
+ a JSON sidecar with RCT vs confounded naive lifts (bias receipt).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)
FEATS = [f"f{i}" for i in range(12)]


def load_base(n: int, seed: int) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("criteo/criteo-uplift", split="train")
    N = len(ds)
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(N, size=min(n, N), replace=False).tolist())
    df = ds.select(idx).to_pandas()
    return df


def confound(df: pd.DataFrame, seed: int, keep_lo: float = 0.15) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    rank = df["f0"].rank(pct=True).to_numpy()
    keep_p = keep_lo + (1 - keep_lo) * rank
    keep_p[df["treatment"].to_numpy() == 0] = 1.0  # controls untouched
    keep = rng.random(len(df)) < keep_p
    out = df.loc[keep].reset_index(drop=True)
    out["confound_weight"] = keep_p[keep]
    return out


def naive_lift(d: pd.DataFrame, y: str) -> float:
    return float(d.loc[d.treatment == 1, y].mean() - d.loc[d.treatment == 0, y].mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="data/processed/criteo_confounded.parquet")
    a = p.parse_args()

    base = load_base(a.n, a.seed)
    rct_lift_visit = naive_lift(base, "visit")
    rct_lift_conv = naive_lift(base, "conversion")
    conf = confound(base, a.seed)
    conf_lift_visit = naive_lift(conf, "visit")
    conf_lift_conv = naive_lift(conf, "conversion")

    conf.to_parquet(a.out, index=False)
    sidecar = {
        "seed": a.seed,
        "base_n": len(base),
        "confounded_n": len(conf),
        "treated_rate_base": float(base.treatment.mean()),
        "treated_rate_confounded": float(conf.treatment.mean()),
        "rct_naive_visit_lift": rct_lift_visit,
        "confounded_naive_visit_lift": conf_lift_visit,
        "bias_visit": conf_lift_visit - rct_lift_visit,
        "rct_naive_conv_lift": rct_lift_conv,
        "confounded_naive_conv_lift": conf_lift_conv,
        "bias_conv": conf_lift_conv - rct_lift_conv,
    }
    Path(a.out).with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    print(json.dumps(sidecar, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

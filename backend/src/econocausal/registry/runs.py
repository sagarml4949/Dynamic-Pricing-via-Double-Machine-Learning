"""Run registry: parquet + JSON sidecars, content hashes. No domain imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REG = Path("data/registry")
REG.mkdir(parents=True, exist_ok=True)


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def save_run(run_id: str, cfg: dict, metrics: dict) -> Path:
    d = REG / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({**cfg, "config_hash": config_hash(cfg)}, indent=2))
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return d

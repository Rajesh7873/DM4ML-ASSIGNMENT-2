"""Stage 1 - Data ingestion.

Reads users / items / interactions from CSV (or generates documented synthetic data first),
loads everything as *strings* so no silent type coercion happens before validation, and writes an
ingestion log (row counts, columns, MD5 checksums) for lineage/reproducibility.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd

from src.ingestion.synthetic import generate_to_csv
from src.utils import get_logger, write_json

log = get_logger("ingestion")
TABLES = ["users", "items", "interactions"]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_raw(cfg: dict) -> dict[str, pd.DataFrame]:
    raw_dir = cfg["_paths"]["raw"]
    return {t: pd.read_csv(raw_dir / f"{t}.csv", dtype=str, keep_default_na=True) for t in TABLES}


def run(cfg: dict) -> dict:
    t0 = time.time()
    raw_dir = cfg["_paths"]["raw"]
    src = cfg["data"]["source"]
    if src == "synthetic":
        log.info("Generating synthetic data (seed=%s)", cfg["seed"])
        generate_to_csv(cfg, raw_dir)
    elif src == "csv":
        csv_dir = Path(cfg["data"]["csv_dir"])
        if csv_dir.resolve() != raw_dir.resolve():
            for t in TABLES:
                (raw_dir / f"{t}.csv").write_bytes((csv_dir / f"{t}.csv").read_bytes())
    else:
        raise ValueError(f"Unknown data.source: {src}")

    log_rows = {}
    for t, df in load_raw(cfg).items():
        f = raw_dir / f"{t}.csv"
        log_rows[t] = {"rows": len(df), "columns": list(df.columns), "md5": _md5(f),
                       "missing_cells": int(df.isna().sum().sum())}
        log.info("Ingested %-12s rows=%d cols=%d md5=%s", t, len(df), df.shape[1], log_rows[t]["md5"][:8])
    summary = {"source": src, "tables": log_rows, "seconds": round(time.time() - t0, 2)}
    write_json(summary, cfg["_paths"]["reports"] / "ingestion_log.json")
    return summary

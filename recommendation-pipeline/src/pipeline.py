"""Plain-Python stage runner (no Prefect needed). Shared by the Prefect flow and `--no-prefect` mode."""
from __future__ import annotations

import importlib
import json
import time
from datetime import datetime, timezone

from src.utils import ROOT, load_config

RUN_LOG = ROOT / "reports" / "run_log.jsonl"

# (task name, "module:function") in dependency order
STAGES = [
    ("ingest", "src.ingestion.ingest:run"),
    ("validate", "src.validation.validate:run"),
    ("prepare", "src.preparation.prepare:run"),
    ("transform", "src.transformation.transform:run"),
    ("features", "src.features.feature_store:build_features"),
    ("train", "src.training.train:run"),
    ("evaluate", "src.evaluation.evaluate:run"),
    ("report", "src.evaluation.report:run"),
]
FN = dict(STAGES)


def run_stage(name: str, config_path: str | None = None):
    """Import + run one stage, recording timing and status in reports/run_log.jsonl."""
    mod_name, fn = FN[name].rsplit(":", 1)
    cfg = load_config(config_path)
    t0, started = time.time(), datetime.now(timezone.utc).isoformat(timespec="seconds")
    status, err = "COMPLETED", None
    try:
        return getattr(importlib.import_module(mod_name), fn)(cfg)
    except Exception as e:  # noqa: BLE001
        status, err = "FAILED", repr(e)
        raise
    finally:
        with open(RUN_LOG, "a") as f:
            f.write(json.dumps({"stage": name, "status": status, "started": started,
                                "seconds": round(time.time() - t0, 2), "error": err}) + "\n")


def run_sequential(config_path: str | None = None):
    RUN_LOG.parent.mkdir(exist_ok=True)
    RUN_LOG.write_text("")
    for name, _ in STAGES:
        run_stage(name, config_path)

"""Shared helpers: config loading, path resolution, logging, JSON IO."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_paths"] = {}
    for key, rel in cfg["paths"].items():
        p = ROOT / rel
        p.mkdir(parents=True, exist_ok=True)
        cfg["_paths"][key] = p
    return cfg


def get_logger(name: str) -> logging.Logger:
    parent = logging.getLogger("recpipe")
    if not parent.handlers:
        parent.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        parent.addHandler(sh)
        (ROOT / "reports").mkdir(exist_ok=True)
        fh = logging.FileHandler(ROOT / "reports" / "pipeline.log", mode="a")
        fh.setFormatter(fmt)
        parent.addHandler(fh)
    return logging.getLogger(f"recpipe.{name}")


def write_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def read_json(path: Path):
    return json.loads(Path(path).read_text())

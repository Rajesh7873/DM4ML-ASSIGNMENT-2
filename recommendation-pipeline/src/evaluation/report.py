"""Final pipeline task: collect stage outputs into reports/pipeline_summary.json."""
from __future__ import annotations

import mlflow

from src.utils import read_json, write_json


def run(cfg: dict) -> dict:
    R = cfg["_paths"]["reports"]
    summary = {
        "ingestion": read_json(R / "ingestion_log.json"),
        "validation": {t: s for t, s in read_json(R / "validation_report.json")["tables"].items()},
        "split": read_json(R / "split_meta.json"),
        "evaluation": read_json(R / "evaluation_results.json"),
    }
    write_json(summary, R / "pipeline_summary.json")
    # portable export of the MLflow experiment (params + metrics of every run)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    runs = mlflow.search_runs(experiment_names=[cfg["mlflow"]["experiment_name"]])
    keep = [c for c in runs.columns if c.startswith(("params.", "metrics.", "tags.mlflow.runName"))]
    runs[["run_id", *keep]].to_csv(R / "mlflow_runs_summary.csv", index=False)
    return {"summary_file": "reports/pipeline_summary.json"}

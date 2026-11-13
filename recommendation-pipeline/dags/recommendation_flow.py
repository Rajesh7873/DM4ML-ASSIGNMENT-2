"""Stage 8 - Orchestration with Prefect.

DAG:  ingest -> validate -> prepare -> transform -> features -> train -> evaluate -> report

Each stage is a Prefect task; dependencies are declared with `wait_for`, so a stage only starts after its
upstream stage succeeded and the flow stops at the first failure (e.g. the data-quality gate).
Every stage logs start/end/status to reports/run_log.jsonl (used for the run-summary figure).

Run once:        python run_pipeline.py
Schedule daily:  python dags/recommendation_flow.py --serve        (cron: 02:00 every day)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefect import flow, task  # noqa: E402

from src.pipeline import RUN_LOG, run_stage  # noqa: E402


@task(name="ingest", retries=1, retry_delay_seconds=2)
def ingest(config_path=None):
    return run_stage("ingest", config_path)


@task(name="validate")          # no retries: a failed quality gate is deterministic
def validate(config_path=None):
    return run_stage("validate", config_path)


@task(name="prepare", retries=1, retry_delay_seconds=2)
def prepare(config_path=None):
    return run_stage("prepare", config_path)


@task(name="transform", retries=1, retry_delay_seconds=2)
def transform(config_path=None):
    return run_stage("transform", config_path)


@task(name="features", retries=1, retry_delay_seconds=2)
def features(config_path=None):
    return run_stage("features", config_path)


@task(name="train")
def train(config_path=None):
    return run_stage("train", config_path)


@task(name="evaluate")
def evaluate(config_path=None):
    return run_stage("evaluate", config_path)


@task(name="report")
def report(config_path=None):
    return run_stage("report", config_path)


@flow(name="recommendation-pipeline", log_prints=True)
def recommendation_pipeline(config_path: str | None = None):
    RUN_LOG.parent.mkdir(exist_ok=True)
    RUN_LOG.write_text("")                                   # fresh log per run
    a = ingest.submit(config_path)
    b = validate.submit(config_path, wait_for=[a])
    c = prepare.submit(config_path, wait_for=[b])
    d = transform.submit(config_path, wait_for=[c])
    e = features.submit(config_path, wait_for=[d])
    f = train.submit(config_path, wait_for=[e])
    g = evaluate.submit(config_path, wait_for=[f])
    h = report.submit(config_path, wait_for=[g])
    return h.result()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        recommendation_pipeline.serve(name="daily-recommendation-pipeline", cron="0 2 * * *")
    else:
        recommendation_pipeline()

"""Entry point.  python run_pipeline.py [--config configs/config.yaml] [--no-prefect]

Default: run the Prefect flow (dags/recommendation_flow.py). --no-prefect runs the same stages sequentially
in plain Python (useful if Prefect is not installed).
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-prefect", action="store_true")
    a = ap.parse_args()
    log = Path(__file__).parent / "reports" / "pipeline.log"
    log.parent.mkdir(exist_ok=True)
    log.write_text("")
    if a.no_prefect:
        from src.pipeline import run_sequential
        run_sequential(a.config)
    else:
        from dags.recommendation_flow import recommendation_pipeline
        recommendation_pipeline(a.config)
    from scripts.make_run_summary import make
    make()


if __name__ == "__main__":
    main()

"""Stage 6 - Model training with MLflow experiment tracking.

Trains (1) popularity baseline, (2) ALS matrix-factorisation CF, (3) hybrid CF + content + popularity.
The hybrid blend weights are tuned on a validation slice carved from the END of the training period
(time-based, no test leakage). Parameters, timings and artefacts are logged to MLflow; run ids are saved
to models/run_registry.json so the evaluation stage can attach its metrics to the same runs.
"""
from __future__ import annotations

import itertools
import pickle
import time

import mlflow
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.evaluation.evaluate import build_eval_set, evaluate_ranking
from src.features.feature_store import FeatureStore, genre_profiles
from src.training.models import HybridRecommender, MFRecommender, PopularityRecommender
from src.utils import get_logger, read_json, write_json

log = get_logger("training")


def _tune_hybrid(cfg, fs, train, uids, iids, u_map, i_map, G):
    """Grid-search hybrid weights on a time-based validation slice of the training data."""
    h, thr, pk = cfg["training"]["hybrid"], cfg["preparation"]["relevance_threshold"], cfg["evaluation"]["primary_k"]
    cut = train.timestamp.quantile(1 - h["val_fraction"])
    sub, val = train[train.timestamp < cut], train[train.timestamp >= cut]
    M_sub = sp.csr_matrix((sub.rating.values, (u_map[sub.user_id].values, i_map[sub.item_id].values)),
                          shape=(len(uids), len(iids)))
    val = val[val.user_id.isin(sub.user_id) & val.item_id.isin(sub.item_id)]
    _, ui, rel = build_eval_set(val, u_map, i_map, thr)
    base = MFRecommender(cfg["seed"]).fit(M_sub, uids, iids, fs, cfg)
    profiles = genre_profiles(sub, fs.item_genre, thr).reindex(uids).fillna(0.0).to_numpy(float)
    log_pop = np.log1p(np.bincount(i_map[sub.item_id].values, minlength=len(iids)).astype(float))
    grid = []
    for wc, wp in itertools.product(h["w_content_grid"], h["w_pop_grid"]):
        hyb = HybridRecommender(base, wc, wp)
        hyb.profiles, hyb.G, hyb.log_pop = profiles, fs.item_genre_matrix(iids), log_pop
        r = evaluate_ranking(hyb, M_sub, ui, rel, G, [pk])
        grid.append({"w_content": wc, "w_pop": wp, f"val_ndcg@{pk}": r[f"ndcg@{pk}"], f"val_diversity@{pk}": r[f"diversity@{pk}"]})
        log.info("  tune w_content=%.2f w_pop=%.2f -> val NDCG@%d=%.4f", wc, wp, pk, r[f"ndcg@{pk}"])
    best = max(grid, key=lambda g: g[f"val_ndcg@{pk}"])
    return best, grid


def run(cfg: dict) -> dict:
    t0 = time.time()
    P = cfg["_paths"]
    M = sp.load_npz(P["processed"] / "user_item_matrix.npz").tocsr()
    idx = read_json(P["processed"] / "matrix_index.json")
    uids, iids = np.array(idx["user_ids"]), np.array(idx["item_ids"])
    u_map, i_map = pd.Series(np.arange(len(uids)), index=uids), pd.Series(np.arange(len(iids)), index=iids)
    train = pd.read_parquet(P["processed"] / "train.parquet")
    fs = FeatureStore(cfg)
    G = fs.item_genre_matrix(iids)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])   # artifacts default to ./mlruns (run from project root)

    best, grid = None, None
    if cfg["training"]["hybrid"]["tune"]:
        log.info("Tuning hybrid weights on a time-based validation slice")
        best, grid = _tune_hybrid(cfg, fs, train, uids, iids, u_map, i_map, G)
        log.info("Best hybrid weights: %s", best)
    else:
        best = {"w_content": 0.3, "w_pop": 0.1}

    common = {"seed": cfg["seed"], "n_users": len(uids), "n_items": len(iids), "n_train_interactions": int(M.nnz),
              "train_cutoff": idx["train_cutoff"], "feature_store_as_of": fs.meta["as_of"]}
    registry = {"models": {}}
    base_mf = None
    for name in ["popularity", "als_mf", "hybrid"]:
        ts = time.time()
        if name == "popularity":
            model = PopularityRecommender().fit(M, uids, iids, fs, cfg)
        elif name == "als_mf":
            model = base_mf = MFRecommender(cfg["seed"]).fit(M, uids, iids, fs, cfg)
        else:
            model = HybridRecommender(base_mf, best["w_content"], best["w_pop"]).fit(M, uids, iids, fs, cfg)
        secs = time.time() - ts
        path = P["models"] / f"{name}.pkl"
        pickle.dump(model, open(path, "wb"))
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({**common, **{k: str(v) for k, v in model.params().items()}})
            mlflow.set_tags({"model_type": name, "stage": "training"})
            mlflow.log_metric("train_seconds", secs)
            mlflow.log_artifact(str(path))
            mlflow.log_artifact(str(P["reports"] / "feature_catalog.md"))
            if name == "hybrid" and grid:
                write_json({"best": best, "grid": grid}, P["models"] / "hybrid_tuning.json")
                mlflow.log_artifact(str(P["models"] / "hybrid_tuning.json"))
                pk = cfg['evaluation']['primary_k']
                mlflow.log_metric(f"val_ndcg_at_{pk}", best[f"val_ndcg@{pk}"])
            registry["models"][name] = {"run_id": run.info.run_id, "file": path.name, "train_seconds": round(secs, 2)}
        log.info("Trained %-10s in %.1fs (MLflow run %s)", name, secs, run.info.run_id[:8])
    write_json(registry, P["models"] / "run_registry.json")
    return {"models": registry["models"], "best_hybrid": best, "seconds": round(time.time() - t0, 2)}

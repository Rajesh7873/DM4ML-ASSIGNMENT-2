"""Stage 4 - Data transformation: derived datasets built from the TRAIN split only.

  * user_activity_summary  - per-user activity statistics
  * item_popularity        - per-item counts, mean rating, Bayesian rating, popularity rank
  * item_genre_matrix      - multi-hot item x genre matrix (content side-information)
  * user_item_matrix.npz   - sparse CSR rating matrix (+ matrix_index.json id <-> row/col mapping)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils import get_logger, write_json

log = get_logger("transformation")


def item_genre_matrix(items: pd.DataFrame) -> pd.DataFrame:
    g = items.set_index("item_id")["category"].str.get_dummies(sep="|")
    g.columns = [f"genre_{c}" for c in g.columns]
    return g.astype("int8").sort_index()


def run(cfg: dict) -> dict:
    t0 = time.time()
    p = cfg["_paths"]
    train = pd.read_parquet(p["processed"] / "train.parquet")
    items = pd.read_parquet(p["processed"] / "items_clean.parquet")
    cutoff = pd.Timestamp(__import__("json").loads((p["reports"] / "split_meta.json").read_text())["cutoff"])
    prior = cfg["features"]["bayes_prior_count"]

    ua = train.groupby("user_id").agg(
        n_interactions=("item_id", "count"), n_distinct_items=("item_id", "nunique"),
        avg_rating=("rating", "mean"), first_interaction=("timestamp", "min"), last_interaction=("timestamp", "max"))
    ua["active_days"] = (ua.last_interaction.dt.normalize() - ua.first_interaction.dt.normalize()).dt.days + 1
    ua["interactions_per_active_day"] = ua.n_interactions / ua.active_days
    ua = ua.reset_index()

    gm = train.rating.mean()
    ip = train.groupby("item_id").agg(n_interactions=("user_id", "count"), n_users=("user_id", "nunique"),
                                      avg_rating=("rating", "mean"), rating_sum=("rating", "sum"))
    ip["bayes_avg_rating"] = (prior * gm + ip.rating_sum) / (prior + ip.n_interactions)
    ip["popularity_rank"] = ip.n_interactions.rank(ascending=False, method="min").astype(int)
    ip = ip.drop(columns="rating_sum").reset_index()

    genres = item_genre_matrix(items)
    user_ids = np.sort(train.user_id.unique())
    item_ids = np.sort(train.item_id.unique())
    u_idx = pd.Series(np.arange(len(user_ids)), index=user_ids)
    i_idx = pd.Series(np.arange(len(item_ids)), index=item_ids)
    M = sp.csr_matrix((train.rating.values, (u_idx[train.user_id].values, i_idx[train.item_id].values)),
                      shape=(len(user_ids), len(item_ids)))

    ua.to_parquet(p["processed"] / "user_activity_summary.parquet", index=False)
    ip.to_parquet(p["processed"] / "item_popularity.parquet", index=False)
    genres.reset_index().to_parquet(p["processed"] / "item_genre_matrix.parquet", index=False)
    sp.save_npz(p["processed"] / "user_item_matrix.npz", M)
    write_json({"user_ids": user_ids.tolist(), "item_ids": item_ids.tolist(), "global_mean": gm,
                "train_cutoff": str(cutoff)}, p["processed"] / "matrix_index.json")

    density = M.nnz / (M.shape[0] * M.shape[1])
    meta = {"matrix_shape": list(M.shape), "nnz": int(M.nnz), "density": density,
            "n_genres": genres.shape[1], "seconds": round(time.time() - t0, 2)}
    log.info("Interaction matrix %s nnz=%d density=%.3f", M.shape, M.nnz, density)
    return meta

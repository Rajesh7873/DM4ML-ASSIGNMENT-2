"""Stage 5 - Feature management: a small file-based offline feature store.

`build_features(cfg)` materialises user / item / pair features (as of the train cutoff, train data only).
`FeatureStore` is the SINGLE access path for those features, used both by training (models.py) and by
prediction (recommend.py), so train-time and serve-time features are identical (no training/serving skew).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.features.registry import FEATURES
from src.utils import get_logger, write_json

log = get_logger("features")


def genre_profiles(train: pd.DataFrame, item_genre: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """User x genre preference profile (rows sum to 1). Uses liked items; falls back to all items."""
    G = item_genre.reindex(columns=item_genre.columns)
    liked = train[train.rating >= threshold]
    def prof(df):
        m = df.merge(G, left_on="item_id", right_index=True, how="left").groupby("user_id")[list(G.columns)].sum()
        return m
    p_all, p_liked = prof(train), prof(liked)
    p = p_liked.reindex(p_all.index).fillna(0.0)
    empty = p.sum(1) == 0
    p.loc[empty] = p_all.loc[empty]
    p = p.div(p.sum(1).replace(0, 1), axis=0)
    p.columns = [c.replace("genre_", "genre_pref_") for c in p.columns]
    return p


def build_features(cfg: dict) -> dict:
    t0 = time.time()
    P, prior, thr = cfg["_paths"], cfg["features"]["bayes_prior_count"], cfg["preparation"]["relevance_threshold"]
    train = pd.read_parquet(P["processed"] / "train.parquet")
    test = pd.read_parquet(P["processed"] / "test.parquet")
    users = pd.read_parquet(P["processed"] / "users_clean.parquet")
    items = pd.read_parquet(P["processed"] / "items_clean.parquet")
    genre = pd.read_parquet(P["processed"] / "item_genre_matrix.parquet").set_index("item_id")
    cutoff = pd.Timestamp(json.loads((P["processed"] / "matrix_index.json").read_text())["train_cutoff"])
    day = 86400.0

    # ---------------- user features
    g = train.groupby("user_id").agg(
        user_avg_rating=("rating", "mean"), user_rating_std=("rating", "std"),
        user_activity_count=("item_id", "count"), user_distinct_items=("item_id", "nunique"),
        _last=("timestamp", "max"), _first=("timestamp", "min"))
    g["user_rating_std"] = g["user_rating_std"].fillna(0.0)
    g["user_days_since_last_interaction"] = (cutoff - g["_last"]).dt.total_seconds() / day
    g["user_tenure_days"] = (g["_last"] - g["_first"]).dt.total_seconds() / day
    prof = genre_profiles(train, genre, thr)
    g = g.drop(columns=["_last", "_first"]).join(prof)
    pref_cols = list(prof.columns)
    g["user_preferred_category"] = prof.idxmax(axis=1).str.replace("genre_pref_", "", regex=False)
    g["user_preferred_category_share"] = prof.max(axis=1)
    u = users.set_index("user_id")
    g["user_days_since_signup"] = (cutoff - u.loc[g.index, "signup_date"]).dt.total_seconds() / day
    g["user_age"], g["user_age_group"], g["user_country"] = u.loc[g.index, "age"], u.loc[g.index, "age_group"], u.loc[g.index, "country"]
    user_feats = g.reset_index()

    # ---------------- item features
    gm = train.rating.mean()
    h = train.groupby("item_id").agg(
        item_popularity=("user_id", "count"), item_unique_users=("user_id", "nunique"),
        item_avg_rating=("rating", "mean"), _sum=("rating", "sum"), _last=("timestamp", "max"))
    h["item_bayes_rating"] = (prior * gm + h["_sum"]) / (prior + h["item_popularity"])
    h["item_popularity_rank"] = h["item_popularity"].rank(ascending=False, method="min").astype(int)
    h["item_popularity_pct"] = h["item_popularity"].rank(pct=True)
    h["item_days_since_last_interaction"] = (cutoff - h["_last"]).dt.total_seconds() / day
    it = items.set_index("item_id").loc[h.index]
    h["item_age_years"] = cutoff.year - it["release_year"]
    h["item_n_categories"] = genre.loc[h.index].sum(1)
    h["item_primary_category"] = it["category"].str.split("|").str[0]
    item_feats = h.drop(columns=["_sum", "_last"]).reset_index()

    user_feats.to_parquet(P["features"] / "user_features.parquet", index=False)
    item_feats.to_parquet(P["features"] / "item_features.parquet", index=False)
    write_json({"as_of": str(cutoff), "global_mean_rating": gm, "genre_pref_columns": pref_cols,
                "genre_columns": list(genre.columns)}, P["features"] / "feature_meta.json")

    # ---------------- pair features, materialised for train + test rows (test uses TRAIN-time features only)
    fs = FeatureStore(cfg)
    for name, df in [("train", train), ("test", test)]:
        fs.pair_features(df[["user_id", "item_id"]]).assign(rating=df["rating"].values).to_parquet(
            P["features"] / f"interaction_features_{name}.parquet", index=False)

    write_catalog(cfg, user_feats, item_feats)
    meta = {"as_of": str(cutoff), "n_user_features": user_feats.shape[1] - 1, "n_item_features": item_feats.shape[1] - 1,
            "n_users": len(user_feats), "n_items": len(item_feats), "seconds": round(time.time() - t0, 2)}
    log.info("Features built: users=%d (%d cols) items=%d (%d cols)", len(user_feats), meta["n_user_features"],
             len(item_feats), meta["n_item_features"])
    return meta


def write_catalog(cfg: dict, user_feats: pd.DataFrame, item_feats: pd.DataFrame) -> None:
    rep = cfg["_paths"]["reports"]
    cat = pd.DataFrame(FEATURES, columns=["feature", "entity", "dtype", "description", "definition", "source"])
    cat.to_csv(rep / "feature_catalog.csv", index=False)
    lines = ["# Feature Catalog", "",
             "Features are computed **as of the train cutoff using train data only** (no leakage) and are served through "
             "`src/features/feature_store.py::FeatureStore`, which is used by both training and prediction.", "",
             "| Feature | Entity | Type | Description | Definition | Source |", "|---|---|---|---|---|---|"]
    for r in FEATURES:
        lines.append("| `{}` | {} | {} | {} | `{}` | {} |".format(*r))
    lines += ["", "## Materialised tables (`data/features/`)", "",
              "| File | Rows | Columns | Key |", "|---|---:|---:|---|",
              f"| user_features.parquet | {len(user_feats):,} | {user_feats.shape[1]} | user_id |",
              f"| item_features.parquet | {len(item_feats):,} | {item_feats.shape[1]} | item_id |",
              "| interaction_features_train/test.parquet | one row per interaction | 3 + rating | (user_id, item_id) |", ""]
    (rep / "feature_catalog.md").write_text("\n".join(lines))


class FeatureStore:
    """Read-side API over the materialised features. Used for training AND prediction."""

    def __init__(self, cfg: dict):
        d = cfg["_paths"]["features"]
        self.user = pd.read_parquet(d / "user_features.parquet").set_index("user_id")
        self.item = pd.read_parquet(d / "item_features.parquet").set_index("item_id")
        meta = json.loads((d / "feature_meta.json").read_text())
        self.meta = meta
        self.pref_cols, self.genre_cols = meta["genre_pref_columns"], meta["genre_columns"]
        genre = pd.read_parquet(cfg["_paths"]["processed"] / "item_genre_matrix.parquet").set_index("item_id")
        self.item_genre = genre[self.genre_cols]

    def get_user_features(self, user_ids) -> pd.DataFrame:
        return self.user.reindex(user_ids)

    def get_item_features(self, item_ids) -> pd.DataFrame:
        return self.item.reindex(item_ids)

    def user_profile_matrix(self, user_ids) -> np.ndarray:
        return self.user.reindex(user_ids)[self.pref_cols].fillna(0.0).to_numpy(float)

    def item_genre_matrix(self, item_ids) -> np.ndarray:
        return self.item_genre.reindex(item_ids).fillna(0).to_numpy(float)

    def affinity_matrix(self, user_ids, item_ids) -> np.ndarray:
        """Dense user x item category affinity: (profile_u . genre_i) / n_categories_i."""
        G = self.item_genre_matrix(item_ids)
        return (self.user_profile_matrix(user_ids) @ G.T) / np.maximum(G.sum(1), 1)[None, :]

    def pair_features(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Interaction features for (user_id, item_id) pairs - identical code path for train and serving."""
        uf, itf = self.get_user_features(pairs.user_id.values), self.get_item_features(pairs.item_id.values)
        P = uf[self.pref_cols].fillna(0.0).to_numpy(float)
        G = self.item_genre.reindex(pairs.item_id.values).fillna(0).to_numpy(float)
        aff = (P * G).sum(1) / np.maximum(G.sum(1), 1)
        gmean = self.meta["global_mean_rating"]
        return pd.DataFrame({
            "user_id": pairs.user_id.values, "item_id": pairs.item_id.values,
            "user_item_category_affinity": aff,
            "item_avg_minus_user_avg": itf["item_bayes_rating"].fillna(gmean).values - uf["user_avg_rating"].fillna(gmean).values,
            "item_popularity_pct_x_activity": itf["item_popularity_pct"].fillna(0).values * np.log1p(uf["user_activity_count"].fillna(0).values),
        })

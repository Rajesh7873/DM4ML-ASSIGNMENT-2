"""Stage 3 - Data preparation.

Cleans the validated tables (imputation, standardisation, de-duplication) and creates the
train/test datasets with a TIME-BASED split (no random shuffling -> no look-ahead leakage).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.utils import get_logger, write_json

log = get_logger("preparation")


def clean_users(u: pd.DataFrame, v: dict) -> pd.DataFrame:
    u = u.copy()
    u.loc[(u.age < v["age_min"]) | (u.age > v["age_max"]), "age"] = np.nan   # invalid -> missing
    u["age"] = u["age"].fillna(u["age"].median()).round().astype("int64")
    u["gender"] = u["gender"].fillna("unknown").str.lower().str.strip().replace("", "unknown")
    u["country"] = u["country"].fillna("unknown").str.upper().str.strip().replace("", "UNKNOWN").replace("UNKNOWN", "unknown")
    u["age_group"] = pd.cut(u["age"], [0, 24, 34, 44, 54, 200], labels=["<25", "25-34", "35-44", "45-54", "55+"]).astype(str)
    return u.sort_values("user_id").reset_index(drop=True)


def clean_items(i: pd.DataFrame, v: dict, now: pd.Timestamp) -> pd.DataFrame:
    i = i.copy()
    i["title"] = i["title"].fillna("").str.strip()
    i.loc[i.title == "", "title"] = "Untitled item " + i.loc[i.title == "", "item_id"].astype(str)
    i["category"] = i["category"].fillna("unknown").str.lower().str.strip().replace("", "unknown")
    i["categories"] = i["category"].apply(lambda s: "|".join(sorted(set(x.strip() for x in s.split("|")))))
    i.loc[(i.release_year < v["release_year_min"]) | (i.release_year > now.year), "release_year"] = np.nan
    i["release_year"] = i["release_year"].fillna(i["release_year"].median()).astype("int64")
    return i.drop(columns="category").rename(columns={"categories": "category"}).sort_values("item_id").reset_index(drop=True)


def clean_interactions(x: pd.DataFrame) -> pd.DataFrame:
    x = x.sort_values(["timestamp", "user_id", "item_id"])
    before = len(x)
    x = x.drop_duplicates(["user_id", "item_id"], keep="last")        # latest rating per (user, item) wins
    log.info("De-duplicated (user,item) pairs: %d -> %d rows", before, len(x))
    return x.reset_index(drop=True)


def time_split(x: pd.DataFrame, test_fraction: float):
    cutoff = x["timestamp"].quantile(1 - test_fraction)
    return x[x.timestamp < cutoff].copy(), x[x.timestamp >= cutoff].copy(), cutoff


def run(cfg: dict) -> dict:
    t0 = time.time()
    p, v = cfg["_paths"], cfg["validation"]
    now = pd.Timestamp(v["reference_date"])
    users = clean_users(pd.read_parquet(p["processed"] / "validated_users.parquet"), v)
    items = clean_items(pd.read_parquet(p["processed"] / "validated_items.parquet"), v, now)
    inter = clean_interactions(pd.read_parquet(p["processed"] / "validated_interactions.parquet"))

    train, test, cutoff = time_split(inter, cfg["preparation"]["test_fraction"])
    test["user_in_train"] = test.user_id.isin(train.user_id)
    test["item_in_train"] = test.item_id.isin(train.item_id)
    warm = test.user_in_train & test.item_in_train

    users.to_parquet(p["processed"] / "users_clean.parquet", index=False)
    items.to_parquet(p["processed"] / "items_clean.parquet", index=False)
    inter.to_parquet(p["processed"] / "interactions_clean.parquet", index=False)
    train.to_parquet(p["processed"] / "train.parquet", index=False)
    test.to_parquet(p["processed"] / "test.parquet", index=False)

    meta = {
        "cutoff": str(cutoff), "train_rows": len(train), "test_rows": len(test),
        "test_rows_warm": int(warm.sum()), "test_rows_cold_user": int((~test.user_in_train).sum()),
        "test_rows_cold_item": int((~test.item_in_train).sum()),
        "train_range": [str(train.timestamp.min()), str(train.timestamp.max())],
        "test_range": [str(test.timestamp.min()), str(test.timestamp.max())],
        "n_users": len(users), "n_items": len(items), "n_interactions_clean": len(inter),
        "seconds": round(time.time() - t0, 2),
    }
    write_json(meta, p["reports"] / "split_meta.json")
    log.info("Time split at %s -> train=%d test=%d (warm-start test rows=%d)", cutoff, len(train), len(test), warm.sum())
    return meta

"""Prediction CLI - top-K recommendations for a user using a trained model and the SAME FeatureStore.

    python recommend.py --user 42 --model hybrid --k 10
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.features.feature_store import FeatureStore  # noqa: E402
from src.training import models  # noqa: E402,F401  (needed for unpickling)
from src.utils import load_config, read_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--model", default="hybrid", choices=["popularity", "als_mf", "hybrid"])
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    cfg = load_config()
    P = cfg["_paths"]
    idx = read_json(P["processed"] / "matrix_index.json")
    uids, iids = np.array(idx["user_ids"]), np.array(idx["item_ids"])
    if a.user not in set(uids):
        sys.exit(f"user {a.user} has no training history (cold start) - fall back to the popularity model")
    u = int(np.flatnonzero(uids == a.user)[0])
    model = pickle.load(open(P["models"] / f"{a.model}.pkl", "rb"))
    M = sp.load_npz(P["processed"] / "user_item_matrix.npz").tocsr()
    s = model.score([u])[0].copy()
    s[M[u].indices] = -np.inf                                   # drop items already seen
    top = np.argsort(-s)[: a.k]
    fs = FeatureStore(cfg)
    items = pd.read_parquet(P["processed"] / "items_clean.parquet").set_index("item_id")
    uf = fs.get_user_features([a.user]).iloc[0]
    print(f"User {a.user}: avg rating {uf.user_avg_rating:.2f}, {int(uf.user_activity_count)} interactions, "
          f"preferred category = {uf.user_preferred_category}")
    pf = fs.pair_features(pd.DataFrame({"user_id": a.user, "item_id": iids[top]}))
    for rank, (j, aff) in enumerate(zip(top, pf.user_item_category_affinity), 1):
        print(f"{rank:2d}. item {iids[j]:4d} | {items.loc[iids[j], 'title']} | {items.loc[iids[j], 'category']:<28} | "
              f"score {s[j]:.3f} | category affinity {aff:.2f}")


if __name__ == "__main__":
    main()

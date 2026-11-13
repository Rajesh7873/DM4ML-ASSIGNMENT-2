"""Stage 7 - Model evaluation (Precision@K, Recall@K, NDCG@K, Coverage, Diversity, RMSE, MAE).

Protocol: warm-start users with >=1 relevant (rating >= threshold) item in the test period. Items already
seen in train are excluded from recommendations. Results are written to reports/ and logged to MLflow.
"""
from __future__ import annotations

import json
import pickle
import time

import mlflow
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.evaluation import metrics as m
from src.utils import get_logger, read_json, write_json

log = get_logger("evaluation")


def build_eval_set(df: pd.DataFrame, u_map: pd.Series, i_map: pd.Series, thr: float):
    """df: interaction rows (already warm). Returns (user ids, user row idx, list of relevant item-idx sets)."""
    rel = df[df.rating >= thr].groupby("user_id").item_id.apply(lambda s: set(i_map[s].values))
    users = np.array(rel.index)
    return users, u_map[users].values, list(rel.values)


def evaluate_ranking(model, M, user_idx, relsets, item_genre, ks) -> dict:
    scores = model.score(user_idx)
    seen = M[user_idx].toarray() > 0
    out = {}
    for k in ks:
        tk = m.top_k(scores, seen, k)
        out.update(m.ranking_metrics(tk, relsets, k))
        out[f"coverage@{k}"] = m.coverage(tk, M.shape[1])
        out[f"diversity@{k}"] = m.diversity(tk, item_genre)
    return out


def run(cfg: dict) -> dict:
    t0 = time.time()
    P, ev, thr = cfg["_paths"], cfg["evaluation"], cfg["preparation"]["relevance_threshold"]
    ks, pk = ev["k_values"], ev["primary_k"]
    M = sp.load_npz(P["processed"] / "user_item_matrix.npz").tocsr()
    idx = read_json(P["processed"] / "matrix_index.json")
    uids, iids = np.array(idx["user_ids"]), np.array(idx["item_ids"])
    u_map, i_map = pd.Series(np.arange(len(uids)), index=uids), pd.Series(np.arange(len(iids)), index=iids)
    test = pd.read_parquet(P["processed"] / "test.parquet")
    test = test[test.user_in_train & test.item_in_train]
    genre = pd.read_parquet(P["processed"] / "item_genre_matrix.parquet").set_index("item_id")
    G = genre.reindex(iids).to_numpy(float)
    items = pd.read_parquet(P["processed"] / "items_clean.parquet").set_index("item_id")

    eval_users, ui, relsets = build_eval_set(test, u_map, i_map, thr)
    tu, ti, y = u_map[test.user_id].values, i_map[test.item_id].values, test.rating.values
    log.info("Evaluating on %d users / %d warm test interactions (%d relevant users)", len(eval_users), len(test), len(ui))

    registry = read_json(P["models"] / "run_registry.json")
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    rows, sample_recs = [], []
    for name, info in registry["models"].items():
        model = pickle.load(open(P["models"] / info["file"], "rb"))
        res = evaluate_ranking(model, M, ui, relsets, G, ks)
        pred = model.predict_rating(tu, ti)
        res.update(m.rating_errors(y, pred) if pred is not None else {"rmse": np.nan, "mae": np.nan})
        res["model"] = name
        rows.append(res)
        # a few human-readable recommendations
        top = m.top_k(model.score(ui[:3]), M[ui[:3]].toarray() > 0, 5)
        for u, rec in zip(eval_users[:3], top):
            sample_recs.append({"model": name, "user_id": u, "recommended": "; ".join(
                f"{iids[j]} ({items.loc[iids[j], 'title']}, {items.loc[iids[j], 'category']})" for j in rec)})
        with mlflow.start_run(run_id=info["run_id"]):
            mlflow.log_metrics({k.replace("@", "_at_"): float(v) for k, v in res.items() if k != "model" and not pd.isna(v)})
        log.info("%-11s P@%d=%.4f R@%d=%.4f NDCG@%d=%.4f cov=%.3f div=%.3f", name, pk, res[f"precision@{pk}"], pk,
                 res[f"recall@{pk}"], pk, res[f"ndcg@{pk}"], res[f"coverage@{pk}"], res[f"diversity@{pk}"])

    df = pd.DataFrame(rows).set_index("model")
    cols = [f"{c}@{k}" for k in ks for c in ["precision", "recall", "ndcg", "coverage", "diversity"]] + ["rmse", "mae"]
    df = df[cols]
    df.to_csv(P["reports"] / "evaluation_results.csv")
    write_json({"protocol": {"eval_users": len(eval_users), "warm_test_rows": len(test), "relevance_threshold": thr,
                             "k_values": ks, "catalog_size": len(iids)}, "results": df.reset_index().to_dict("records")},
               P["reports"] / "evaluation_results.json")
    pd.DataFrame(sample_recs).to_csv(P["reports"] / "sample_recommendations.csv", index=False)
    (P["reports"] / "evaluation_results.md").write_text(_md(df, ks, pk, len(eval_users), thr, len(iids)))
    _plot(df, pk, P["reports"] / "model_comparison.png")
    # attach the artefacts to every model's run
    for info in registry["models"].values():
        with mlflow.start_run(run_id=info["run_id"]):
            for f in ["evaluation_results.csv", "evaluation_results.md", "model_comparison.png"]:
                mlflow.log_artifact(str(P["reports"] / f))
    return {"results": df.reset_index().to_dict("records"), "eval_users": len(eval_users), "seconds": round(time.time() - t0, 2)}


def _md(df, ks, pk, n_users, thr, n_items) -> str:
    L = ["# Model Evaluation Results", "",
         f"Evaluation users: **{n_users}** (warm-start, >=1 test item rated >= {thr:g}); catalogue = {n_items} items; "
         "train-seen items excluded from recommendations; time-based split.", ""]
    for k in ks:
        L += [f"## K = {k}", "", "| Model | Precision@K | Recall@K | NDCG@K | Coverage | Diversity |", "|---|---:|---:|---:|---:|---:|"]
        for n, r in df.iterrows():
            L.append(f"| {n} | {r[f'precision@{k}']:.4f} | {r[f'recall@{k}']:.4f} | {r[f'ndcg@{k}']:.4f} | "
                     f"{r[f'coverage@{k}']:.3f} | {r[f'diversity@{k}']:.3f} |")
        L.append("")
    L += ["## Rating prediction (all warm test interactions)", "", "| Model | RMSE | MAE |", "|---|---:|---:|"]
    for n, r in df.iterrows():
        L.append(f"| {n} | " + (f"{r['rmse']:.4f} | {r['mae']:.4f} |" if not pd.isna(r["rmse"]) else "n/a (ranking-only) | n/a |"))
    L += ["", "*Diversity* = 1 - mean pairwise cosine similarity of genre vectors inside a list (higher = more varied). "
          "*Coverage* = share of the catalogue appearing in at least one user's list.", ""]
    return "\n".join(L)


def _plot(df, pk, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [f"precision@{pk}", f"recall@{pk}", f"ndcg@{pk}", f"coverage@{pk}", f"diversity@{pk}"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 3.2))
    colors = ["#9aa5b1", "#2f6fb2", "#d9822b"]
    for ax, met in zip(axes, metrics):
        ax.bar(df.index, df[met], color=colors[: len(df)])
        ax.set_title(met.upper().replace("@", "@"), fontsize=10)
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for j, v in enumerate(df[met]):
            ax.text(j, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

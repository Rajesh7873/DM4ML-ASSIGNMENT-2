"""Ranking and rating metrics (binary relevance)."""
from __future__ import annotations

import numpy as np


def top_k(scores: np.ndarray, seen_mask: np.ndarray, k: int) -> np.ndarray:
    """Top-k item indices per row, never recommending items the user already interacted with in train."""
    s = np.where(seen_mask, -np.inf, scores)
    part = np.argpartition(-s, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, part, 1), axis=1)
    return np.take_along_axis(part, order, 1)


def ranking_metrics(topk: np.ndarray, relevant: list[set], k: int) -> dict:
    prec, rec, ndcg = [], [], []
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for row, rel in zip(topk, relevant):
        hits = np.array([i in rel for i in row[:k]], dtype=float)
        prec.append(hits.sum() / k)
        rec.append(hits.sum() / len(rel))
        dcg = (hits * disc).sum()
        idcg = disc[: min(len(rel), k)].sum()
        ndcg.append(dcg / idcg)
    return {f"precision@{k}": float(np.mean(prec)), f"recall@{k}": float(np.mean(rec)), f"ndcg@{k}": float(np.mean(ndcg))}


def coverage(topk: np.ndarray, n_items: int) -> float:
    """Share of the catalogue that appears in at least one user's recommendation list."""
    return len(np.unique(topk)) / n_items


def diversity(topk: np.ndarray, item_genre: np.ndarray) -> float:
    """Mean intra-list diversity = 1 - mean pairwise cosine similarity of genre vectors in each list."""
    V = item_genre / np.maximum(np.linalg.norm(item_genre, axis=1, keepdims=True), 1e-9)
    k = topk.shape[1]
    out = []
    for row in topk:
        S = V[row] @ V[row].T
        out.append(1.0 - (S.sum() - np.trace(S)) / (k * (k - 1)))
    return float(np.mean(out))


def rating_errors(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    e = y_true - y_pred
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e)))}

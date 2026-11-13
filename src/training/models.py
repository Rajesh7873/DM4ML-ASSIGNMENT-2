"""Recommender models. All share the same interface so training / evaluation are model-agnostic:

    score(user_idx)               -> (n_users, n_items) ranking scores (higher = better)
    predict_rating(u_idx, i_idx)  -> predicted ratings, or None if the model is ranking-only

Row / column indices refer to the train interaction matrix (see matrix_index.json).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _zscore_rows(a: np.ndarray) -> np.ndarray:
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)


# ------------------------------------------------------------------------------------------ ALS
def _als_side(C: sp.csr_matrix, Y: np.ndarray, reg: float, implicit: bool, R: sp.csr_matrix | None = None):
    """Solve one side of ALS. implicit=True: Hu-Koren-Volinsky weighted ALS; else ridge on explicit residuals."""
    n, f = C.shape[0], Y.shape[1]
    out = np.zeros((n, f))
    eye = np.eye(f)
    YtY = Y.T @ Y
    for u in range(n):
        s, e = C.indptr[u], C.indptr[u + 1]
        if s == e:
            continue
        idx, vals = C.indices[s:e], C.data[s:e]
        Yu = Y[idx]
        if implicit:   # vals = confidence - 1 ; preference = 1 on observed entries
            A = YtY + (Yu.T * vals) @ Yu + reg * eye
            b = (Yu * (1.0 + vals)[:, None]).sum(0)
        else:          # vals = residual rating ; weighted-lambda regularisation
            A = Yu.T @ Yu + reg * (e - s) * eye
            b = Yu.T @ vals
        out[u] = np.linalg.solve(A, b)
    return out


class PopularityRecommender:
    """Baseline: every user gets the globally most popular items (non-personalised)."""
    name = "popularity"

    def fit(self, M, user_ids, item_ids, fs, cfg):
        f = fs.get_item_features(item_ids)
        self.pop = f["item_popularity"].to_numpy(float)
        self.item_rating = f["item_bayes_rating"].to_numpy(float)
        return self

    def params(self):
        return {"strategy": "item_popularity (train interaction count)", "rating_predictor": "item Bayesian mean rating"}

    def score(self, user_idx):
        return np.tile(self.pop, (len(user_idx), 1))

    def predict_rating(self, u_idx, i_idx):
        return np.clip(self.item_rating[i_idx], 1, 5)


class MFRecommender:
    """Personalised collaborative filtering via matrix factorisation (pure NumPy).

    * ranking  : implicit-feedback ALS (every interaction = positive signal, confidence grows with rating)
    * rating   : explicit biased MF  r_hat = mu + b_u + b_i + p_u . q_i  (ALS on bias residuals)
    """
    name = "als_mf"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, M: sp.csr_matrix, user_ids, item_ids, fs, cfg):
        a, b = cfg["training"]["als"], cfg["training"]["biased_mf"]
        rng = np.random.default_rng(self.seed)
        self.a, self.b = a, b
        # ---- implicit ALS for ranking
        C = M.copy().astype(float)
        C.data = a["alpha"] * C.data / 5.0
        CT = C.T.tocsr()
        X = rng.normal(0, 0.01, (M.shape[0], a["factors"]))
        Y = rng.normal(0, 0.01, (M.shape[1], a["factors"]))
        for _ in range(a["iterations"]):
            X = _als_side(C, Y, a["reg"], True)
            Y = _als_side(CT, X, a["reg"], True)
        self.X, self.Y = X, Y
        # ---- explicit biased MF for rating prediction
        R = M.tocsr().astype(float)
        self.mu = R.data.mean()
        cnt_i = np.diff(R.tocsc().indptr)
        bi = np.asarray(R.sum(0)).ravel() - self.mu * cnt_i
        self.bi = bi / (cnt_i + b["bias_reg"])
        cnt_u = np.diff(R.indptr)
        rowsum = np.asarray(R.sum(1)).ravel()
        resid_i = sp.csr_matrix((self.bi[R.indices], R.indices, R.indptr), shape=R.shape).sum(1).A1
        self.bu = (rowsum - self.mu * cnt_u - resid_i) / (cnt_u + b["bias_reg"])
        E = R.copy()
        E.data = R.data - self.mu - self.bu[np.repeat(np.arange(R.shape[0]), cnt_u)] - self.bi[R.indices]
        ET = E.T.tocsr()
        P = rng.normal(0, 0.1, (R.shape[0], b["factors"]))
        Q = rng.normal(0, 0.1, (R.shape[1], b["factors"]))
        for _ in range(b["iterations"]):
            P = _als_side(E, Q, b["reg"], False)
            Q = _als_side(ET, P, b["reg"], False)
        self.P, self.Q = P, Q
        return self

    def params(self):
        return {**{f"als_{k}": v for k, v in self.a.items()}, **{f"bmf_{k}": v for k, v in self.b.items()}}

    def score(self, user_idx):
        return self.X[user_idx] @ self.Y.T

    def predict_rating(self, u_idx, i_idx):
        r = self.mu + self.bu[u_idx] + self.bi[i_idx] + (self.P[u_idx] * self.Q[i_idx]).sum(1)
        return np.clip(r, 1, 5)


class HybridRecommender:
    """Hybrid: z-scored blend of CF score + content (category-affinity feature) + popularity prior.

    score = z(ALS) + w_content * z(affinity) + w_pop * z(log1p(popularity))
    Affinity and popularity come from the FeatureStore, so train-time and serve-time features match.
    Ranking-only (no rating prediction).
    """
    name = "hybrid"

    def __init__(self, base: MFRecommender, w_content: float = 0.3, w_pop: float = 0.1):
        self.base, self.w_content, self.w_pop = base, w_content, w_pop

    def fit(self, M, user_ids, item_ids, fs, cfg):
        self.profiles = fs.user_profile_matrix(user_ids)                 # (n_users, n_genres)
        self.G = fs.item_genre_matrix(item_ids)                          # (n_items, n_genres)
        self.log_pop = np.log1p(fs.get_item_features(item_ids)["item_popularity"].to_numpy(float))
        return self

    def params(self):
        return {"w_cf": 1.0, "w_content": self.w_content, "w_pop": self.w_pop, "base_model": self.base.name}

    def score(self, user_idx):
        aff = (self.profiles[user_idx] @ self.G.T) / np.maximum(self.G.sum(1), 1)[None, :]
        s = _zscore_rows(self.base.score(user_idx))
        s = s + self.w_content * _zscore_rows(aff)
        s = s + self.w_pop * _zscore_rows(np.tile(self.log_pop, (len(user_idx), 1)))
        return s

    def predict_rating(self, u_idx, i_idx):
        return None

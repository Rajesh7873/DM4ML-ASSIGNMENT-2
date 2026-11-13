"""Synthetic movie-platform data generator (users, items, interactions) with injected data-quality problems.

The clean data has real taste structure (genre preferences + latent factors + popularity skew), so a
personalised model has something to learn. A controlled amount of dirt is then injected so the
validation stage has realistic work to do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GENRES = ["action", "comedy", "drama", "sci-fi", "romance", "thriller",
          "documentary", "animation", "horror", "fantasy"]
COUNTRIES = ["IN", "US", "UK", "DE", "FR", "BR", "CA", "AU"]
TS_FMT = "%Y-%m-%d %H:%M:%S"


def generate_clean(cfg: dict):
    d = cfg["data"]
    rng = np.random.default_rng(cfg["seed"])
    n_u, n_i, G, D = d["n_users"], d["n_items"], len(GENRES), 4
    start, end = pd.Timestamp(d["start_date"]), pd.Timestamp(d["end_date"])
    span = (end - start).total_seconds()

    # ---- items
    genre_w = rng.dirichlet(np.ones(G) * 3)
    n_g = rng.choice([1, 2, 3], size=n_i, p=[0.45, 0.40, 0.15])
    item_G = np.zeros((n_i, G))
    for i in range(n_i):
        item_G[i, rng.choice(G, size=n_g[i], replace=False, p=genre_w)] = 1
    pop = rng.lognormal(0, 1.1, n_i)
    quality = rng.normal(0, 0.45, n_i)
    q = rng.normal(0, 1, (n_i, D))
    items = pd.DataFrame({
        "item_id": np.arange(1, n_i + 1),
        "title": [f"Movie {i:04d}" for i in range(1, n_i + 1)],
        "category": ["|".join(GENRES[g] for g in np.flatnonzero(row)) for row in item_G],
        "release_year": rng.integers(1990, 2024, n_i),
    })

    # ---- users
    taste = rng.dirichlet(np.ones(G) * 0.35, n_u)
    p = rng.normal(0, 1, (n_u, D))
    ub = rng.normal(0, 0.35, n_u)
    signup = start + pd.to_timedelta(rng.uniform(0, 0.7, n_u) * span, unit="s")
    users = pd.DataFrame({
        "user_id": np.arange(1, n_u + 1),
        "age": rng.integers(16, 70, n_u),
        "gender": rng.choice(["m", "f", "other"], n_u, p=[0.48, 0.48, 0.04]),
        "country": rng.choice(COUNTRIES, n_u),
        "signup_date": signup.strftime("%Y-%m-%d"),
    })

    # ---- utility: genre affinity + latent taste
    gen_aff = (taste @ item_G.T) / item_G.sum(1)[None, :]
    lat = (p @ q.T) / np.sqrt(D)
    z_gen = (gen_aff - gen_aff.mean()) / gen_aff.std()
    z_lat = lat / lat.std()
    util = 0.9 * z_gen + 0.6 * z_lat
    rating = 3.4 + ub[:, None] + quality[None, :] + 0.55 * util + rng.normal(0, 0.5, util.shape)
    rating = np.clip(np.rint(rating), 1, 5).astype(int)

    # ---- interactions: selection depends on popularity AND taste
    sel = pop[None, :] * np.exp(0.8 * util)
    sel = sel / sel.sum(1, keepdims=True)
    counts = np.clip(rng.lognormal(np.log(d["mean_interactions_per_user"]), 0.7, n_u), 5, 250).astype(int)
    rows = []
    for u in range(n_u):
        chosen = rng.choice(n_i, size=min(counts[u], n_i), replace=False, p=sel[u])
        frac = rng.uniform(0, 1, len(chosen))
        t0 = signup[u]
        ts = t0 + pd.to_timedelta(frac * (end - t0).total_seconds(), unit="s")
        for c, t in zip(chosen, ts):
            rows.append((u + 1, c + 1, rating[u, c], t.floor("s")))
    inter = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    return users, items, inter


def inject_noise(users, items, inter, cfg):
    """Return object-typed frames with realistic problems injected (rates scale with data.dirty_rate)."""
    rng = np.random.default_rng(cfg["seed"] + 1)
    r = cfg["data"]["dirty_rate"]
    users, items, inter = users.astype(object), items.astype(object), inter.copy()
    inter["timestamp"] = inter["timestamp"].dt.strftime(TS_FMT)
    inter = inter.astype(object)

    def pick(df, frac):
        return rng.choice(df.index.to_numpy(), size=max(1, int(len(df) * frac)), replace=False)

    # ---- users
    users.loc[pick(users, r * 1.7), "age"] = np.nan                         # missing age
    bad_age = pick(users, r * 0.35)
    users.loc[bad_age, "age"] = rng.choice([-5, 0, 150, 999], len(bad_age))   # invalid age
    users.loc[pick(users, r), "country"] = np.nan
    users.loc[pick(users, r * 0.7), "gender"] = np.nan
    users.loc[pick(users, r * 0.2), "signup_date"] = "31/02/2024"            # malformed date
    users = pd.concat([users, users.loc[pick(users, r * 0.35)]], ignore_index=True)  # exact duplicates

    # ---- items
    items.loc[pick(items, r * 1.3), "title"] = np.nan
    items.loc[pick(items, r), "category"] = np.nan
    items.loc[pick(items, r * 0.2), "release_year"] = 2099
    items = pd.concat([items, items.loc[pick(items, r * 0.35)]], ignore_index=True)

    # ---- interactions
    signup_map = users.drop_duplicates("user_id").set_index("user_id")["signup_date"]
    inter.loc[pick(inter, r * 0.45), "user_id"] = "unknown"                  # wrong dtype
    idx = pick(inter, r * 0.8)
    inter.loc[idx, "rating"] = rng.choice([0, 6, -1, 10, 99], len(idx))      # out-of-range rating
    inter.loc[pick(inter, r * 0.35), "rating"] = np.nan                      # missing rating
    inter.loc[pick(inter, r * 0.1), "rating"] = "five"                       # non-numeric rating
    inter.loc[pick(inter, r * 0.3), "timestamp"] = "not-a-date"
    inter.loc[pick(inter, r * 0.3), "timestamp"] = "2024-13-45 99:00:00"
    inter.loc[pick(inter, r * 0.3), "timestamp"] = "2031-05-01 00:00:00"    # future timestamp
    idx = pick(inter, r * 0.5)
    inter.loc[idx, "user_id"] = rng.integers(90000, 99999, len(idx))         # orphan user
    idx = pick(inter, r * 0.5)
    inter.loc[idx, "item_id"] = rng.integers(90000, 99999, len(idx))         # orphan item
    # interaction before the user signed up
    for i in pick(inter, r * 0.17):
        s = signup_map.get(inter.at[i, "user_id"])
        if isinstance(s, str) and "/" not in s:
            inter.at[i, "timestamp"] = (pd.Timestamp(s) - pd.Timedelta(days=int(rng.integers(1, 60)))).strftime(TS_FMT)
    # legitimate re-ratings (same user+item, later timestamp, new rating) -> resolved in preparation
    re = inter.loc[pick(inter, r * 0.7)].copy()
    ts = pd.to_datetime(re["timestamp"], format=TS_FMT, errors="coerce")
    ok = ts.notna() & (ts < pd.Timestamp("2025-01-01"))
    new_ts = (ts + pd.to_timedelta(rng.integers(1, 30, len(re)), unit="D")).dt.strftime(TS_FMT)
    re["timestamp"] = new_ts.where(ok, re["timestamp"])
    re["rating"] = rng.integers(1, 6, len(re))
    inter = pd.concat([inter, re], ignore_index=True)
    inter = pd.concat([inter, inter.loc[pick(inter, r * 0.5)]], ignore_index=True)  # exact duplicates
    return users, items, inter.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)


def generate_to_csv(cfg: dict, out_dir) -> dict:
    users, items, inter = generate_clean(cfg)
    u, i, x = inject_noise(users, items, inter, cfg)
    u.to_csv(out_dir / "users.csv", index=False)
    i.to_csv(out_dir / "items.csv", index=False)
    x.to_csv(out_dir / "interactions.csv", index=False)
    return {"users": len(u), "items": len(i), "interactions": len(x)}

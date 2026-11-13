"""Unit tests for validation rules, metrics, and split logic.  Run: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import metrics as m  # noqa: E402
from src.preparation.prepare import clean_interactions, time_split  # noqa: E402
from src.validation.validate import validate_interactions, validate_users  # noqa: E402

V = {"rating_min": 1, "rating_max": 5, "age_min": 10, "age_max": 100}
NOW = pd.Timestamp("2025-01-01")


def _users():
    raw = pd.DataFrame({"user_id": ["1", "2", "2", "x", "5"], "age": ["30", None, None, "40", "200"],
                        "gender": ["m", "f", "f", "m", "m"], "country": ["IN"] * 5,
                        "signup_date": ["2023-01-01"] * 5})
    return validate_users(raw, V, NOW)


def test_user_validation_flags_dup_dtype_and_keeps_warnings():
    clean, c = _users()
    assert sorted(clean.user_id) == [1, 2, 5]            # exact duplicate of '2' and non-integer 'x' rejected
    fails = {r["check"]: r["failed"] for r in c.results}
    assert fails["exact_duplicate_row"] == 1 and fails["invalid_dtype_user_id"] == 1
    assert fails["missing_age"] == 2 and fails["age_out_of_range"] == 1   # warnings: rows kept


def test_interaction_validation_rules():
    users, _ = _users()
    items = pd.DataFrame({"item_id": [10, 11]})
    raw = pd.DataFrame({
        "user_id": ["1", "1", "1", "1", "1", "1", "9"], "item_id": ["10", "10", "11", "11", "10", "99", "10"],
        "rating": ["4", "6", "five", None, "3", "3", "3"],
        "timestamp": ["2024-01-01 10:00:00", "2024-01-01 10:00:00", "2024-01-01 10:00:00", "2024-01-01 10:00:00",
                      "2031-01-01 00:00:00", "2024-01-01 10:00:00", "2024-01-01 10:00:00"]})
    clean, c = validate_interactions(raw, V, NOW, users, items)
    assert len(clean) == 1 and clean.iloc[0].rating == 4.0
    fails = {r["check"]: r["failed"] for r in c.results}
    assert fails["rating_out_of_range"] == 1 and fails["invalid_dtype_rating"] == 1 and fails["missing_rating"] == 1
    assert fails["future_timestamp"] == 1 and fails["orphan_item_id"] == 1 and fails["orphan_user_id"] == 1


def test_latest_rating_wins_and_split_is_chronological():
    x = pd.DataFrame({"user_id": [1, 1, 2, 3], "item_id": [5, 5, 6, 7], "rating": [2.0, 5.0, 3.0, 4.0],
                      "timestamp": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"])})
    x = clean_interactions(x)
    assert len(x) == 3 and x[(x.user_id == 1)].rating.iloc[0] == 5.0
    tr, te, cut = time_split(x, 0.34)
    assert tr.timestamp.max() < te.timestamp.min()


def test_ranking_metrics_known_values():
    topk = np.array([[1, 2, 3]])
    r = m.ranking_metrics(topk, [{1, 3}], 3)
    assert abs(r["precision@3"] - 2 / 3) < 1e-9 and abs(r["recall@3"] - 1.0) < 1e-9
    expected = (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
    assert abs(r["ndcg@3"] - expected) < 1e-9


def test_topk_excludes_seen_and_coverage_diversity():
    scores = np.array([[0.9, 0.8, 0.7, 0.1]])
    tk = m.top_k(scores, np.array([[True, False, False, False]]), 2)
    assert list(tk[0]) == [1, 2]
    assert m.coverage(np.array([[0, 1], [1, 2]]), 4) == 0.75
    G = np.array([[1, 0], [1, 0], [0, 1]], float)
    assert m.diversity(np.array([[0, 1]]), G) == 0.0 and m.diversity(np.array([[0, 2]]), G) == 1.0

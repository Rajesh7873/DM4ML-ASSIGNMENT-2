"""Stage 2 - Data validation.

Rule-based checks (missing values, duplicates, data types, rating range, timestamp validity,
referential integrity). Each check has a severity:

  * error   -> the record is REJECTED and written to data/errors/<table>_errors.csv with its reasons
  * warning -> the record is KEPT and the issue is repaired in the preparation stage (e.g. impute age)

Outputs: validated typed Parquet tables, error datasets, reports/validation_report.{md,json}.
A quality gate aborts the pipeline if too large a share of any table is rejected.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.ingestion.ingest import load_raw
from src.utils import get_logger, write_json

log = get_logger("validation")
TS_FMT = "%Y-%m-%d %H:%M:%S"


class DataQualityError(RuntimeError):
    pass


class _Checker:
    """Collects per-row reasons and per-check statistics for one table."""

    def __init__(self, table: str, df: pd.DataFrame):
        self.table, self.df = table, df
        self.err = pd.Series("", index=df.index, dtype=object)
        self.warn = pd.Series("", index=df.index, dtype=object)
        self.results: list[dict] = []

    def add(self, name: str, mask: pd.Series, severity: str, desc: str, n_checked: int | None = None):
        mask = mask.fillna(False).astype(bool)
        target = self.err if severity == "error" else self.warn
        target[mask] = target[mask] + name + ";"
        self.results.append({
            "table": self.table, "check": name, "severity": severity, "description": desc,
            "checked": int(n_checked if n_checked is not None else len(self.df)),
            "failed": int(mask.sum()),
            "action": "rejected -> errors dataset" if severity == "error" else "kept, repaired in preparation",
        })

    @property
    def error_mask(self) -> pd.Series:
        return self.err != ""


def _strip(s: pd.Series) -> pd.Series:
    return s.astype("object").where(s.isna(), s.astype(str).str.strip())


def _missing(s: pd.Series) -> pd.Series:
    return s.isna() | (_strip(s).astype(str) == "")


def _to_int(s: pd.Series) -> pd.Series:
    n = pd.to_numeric(_strip(s), errors="coerce")
    return n.where(n.notna() & (n == n.round()))


def _dup_checks(c: _Checker, key_col: str, key: pd.Series):
    exact = c.df.duplicated(keep="first")
    pk = key.notna() & key.duplicated(keep="first") & ~exact
    c.add("exact_duplicate_row", exact, "error", "Row is an exact copy of an earlier row")
    c.add("duplicate_primary_key", pk, "error", f"{key_col} repeated with different content")


# ----------------------------------------------------------------------------------------------
def validate_users(raw: pd.DataFrame, v: dict, now: pd.Timestamp):
    c = _Checker("users", raw)
    uid_missing, uid = _missing(raw.user_id), _to_int(raw.user_id)
    c.add("missing_user_id", uid_missing, "error", "user_id is empty")
    c.add("invalid_dtype_user_id", uid.isna() & ~uid_missing, "error", "user_id is not an integer")
    _dup_checks(c, "user_id", uid)
    age = pd.to_numeric(_strip(raw.age), errors="coerce")
    c.add("missing_age", _missing(raw.age), "warning", "age is empty -> imputed with median")
    c.add("age_out_of_range", age.notna() & ((age < v["age_min"]) | (age > v["age_max"])), "warning",
          f"age outside [{v['age_min']}, {v['age_max']}] -> set to missing, then imputed")
    c.add("missing_country", _missing(raw.country), "warning", "country is empty -> 'unknown'")
    c.add("missing_gender", _missing(raw.gender), "warning", "gender is empty -> 'unknown'")
    signup = pd.to_datetime(_strip(raw.signup_date), format="%Y-%m-%d", errors="coerce")
    c.add("invalid_signup_date", signup.isna() & ~_missing(raw.signup_date), "error", "signup_date is not YYYY-MM-DD")
    c.add("missing_signup_date", _missing(raw.signup_date), "error", "signup_date is empty")
    c.add("future_signup_date", signup > now, "error", "signup_date is after the reference date")
    good = ~c.error_mask
    clean = pd.DataFrame({
        "user_id": uid[good].astype("int64"), "age": age[good].astype("float64"),
        "gender": _strip(raw.gender)[good], "country": _strip(raw.country)[good],
        "signup_date": signup[good]}).reset_index(drop=True)
    return clean, c


def validate_items(raw: pd.DataFrame, v: dict, now: pd.Timestamp):
    c = _Checker("items", raw)
    iid_missing, iid = _missing(raw.item_id), _to_int(raw.item_id)
    c.add("missing_item_id", iid_missing, "error", "item_id is empty")
    c.add("invalid_dtype_item_id", iid.isna() & ~iid_missing, "error", "item_id is not an integer")
    _dup_checks(c, "item_id", iid)
    yr = pd.to_numeric(_strip(raw.release_year), errors="coerce")
    c.add("missing_title", _missing(raw.title), "warning", "title is empty -> placeholder title")
    c.add("missing_category", _missing(raw.category), "warning", "category is empty -> 'unknown'")
    c.add("release_year_out_of_range", yr.notna() & ((yr < v["release_year_min"]) | (yr > now.year)), "warning",
          "release_year outside plausible range -> set to missing, then imputed")
    good = ~c.error_mask
    clean = pd.DataFrame({
        "item_id": iid[good].astype("int64"), "title": _strip(raw.title)[good],
        "category": _strip(raw.category)[good], "release_year": yr[good].astype("float64")}).reset_index(drop=True)
    return clean, c


def validate_interactions(raw: pd.DataFrame, v: dict, now: pd.Timestamp,
                          users: pd.DataFrame, items: pd.DataFrame):
    c = _Checker("interactions", raw)
    uid_m, iid_m = _missing(raw.user_id), _missing(raw.item_id)
    uid, iid = _to_int(raw.user_id), _to_int(raw.item_id)
    c.add("missing_user_id", uid_m, "error", "user_id is empty")
    c.add("missing_item_id", iid_m, "error", "item_id is empty")
    c.add("invalid_dtype_user_id", uid.isna() & ~uid_m, "error", "user_id is not an integer")
    c.add("invalid_dtype_item_id", iid.isna() & ~iid_m, "error", "item_id is not an integer")
    # rating
    r_missing = _missing(raw.rating)
    rating = pd.to_numeric(_strip(raw.rating), errors="coerce")
    c.add("missing_rating", r_missing, "error", "rating is empty")
    c.add("invalid_dtype_rating", rating.isna() & ~r_missing, "error", "rating is not numeric")
    c.add("rating_out_of_range", rating.notna() & ((rating < v["rating_min"]) | (rating > v["rating_max"])),
          "error", f"rating outside [{v['rating_min']}, {v['rating_max']}]")
    # timestamp
    t_missing = _missing(raw.timestamp)
    ts = pd.to_datetime(_strip(raw.timestamp), format=TS_FMT, errors="coerce")
    c.add("missing_timestamp", t_missing, "error", "timestamp is empty")
    c.add("invalid_timestamp", ts.isna() & ~t_missing, "error", f"timestamp is not '{TS_FMT}'")
    c.add("future_timestamp", ts > now, "error", "timestamp is after the reference date")
    # duplicates
    c.add("exact_duplicate_row", raw.duplicated(keep="first"), "error", "Row is an exact copy of an earlier row")
    pair_dup = uid.notna() & iid.notna() & pd.DataFrame({"u": uid, "i": iid}).duplicated(keep=False)
    c.add("repeat_user_item_pair", pair_dup & ~raw.duplicated(keep=False), "warning",
          "same (user, item) rated more than once -> latest rating kept in preparation")
    # referential integrity (against the *validated* dimension tables)
    c.add("orphan_user_id", uid.notna() & ~uid.isin(users.user_id), "error", "user_id not found in validated users")
    c.add("orphan_item_id", iid.notna() & ~iid.isin(items.item_id), "error", "item_id not found in validated items")
    # temporal consistency
    signup = uid.map(users.set_index("user_id")["signup_date"])
    c.add("interaction_before_signup", ts.notna() & signup.notna() & (ts.dt.normalize() < signup),
          "error", "interaction happened before the user's signup_date")
    good = ~c.error_mask
    clean = pd.DataFrame({
        "user_id": uid[good].astype("int64"), "item_id": iid[good].astype("int64"),
        "rating": rating[good].astype("float64"), "timestamp": ts[good]}).reset_index(drop=True)
    return clean, c


# ----------------------------------------------------------------------------------------------
def _report_md(stats: dict, results: list[dict]) -> str:
    lines = ["# Data Validation Report", "",
             f"Reference date: `{stats['reference_date']}` | Quality gate: max {stats['max_error_rate']:.0%} rejected per table", "",
             "## Summary", "", "| Table | Records in | Valid | Rejected | Rejected % | Warnings (rows) |",
             "|---|---:|---:|---:|---:|---:|"]
    for t, s in stats["tables"].items():
        lines.append(f"| {t} | {s['records_in']:,} | {s['valid']:,} | {s['rejected']:,} | "
                     f"{s['rejected_pct']:.2f}% | {s['warning_rows']:,} |")
    lines += ["", "## Check results", "",
              "| Table | Check | Severity | Failed | Checked | Failed % | Description | Action |",
              "|---|---|---|---:|---:|---:|---|---|"]
    for r in results:
        pct = 100 * r["failed"] / max(r["checked"], 1)
        lines.append(f"| {r['table']} | `{r['check']}` | {r['severity']} | {r['failed']:,} | {r['checked']:,} | "
                     f"{pct:.2f}% | {r['description']} | {r['action']} |")
    lines += ["", "*Note:* a row can fail several checks, so per-check counts can sum to more than the rejected total.", ""]
    return "\n".join(lines)


def run(cfg: dict) -> dict:
    t0 = time.time()
    v, paths = cfg["validation"], cfg["_paths"]
    now = pd.Timestamp(v["reference_date"])
    raw = load_raw(cfg)

    users, cu = validate_users(raw["users"], v, now)
    items, ci = validate_items(raw["items"], v, now)
    inter, cx = validate_interactions(raw["interactions"], v, now, users, items)

    stats, results = {"reference_date": v["reference_date"], "max_error_rate": v["max_error_rate"], "tables": {}}, []
    for name, checker, clean in [("users", cu, users), ("items", ci, items), ("interactions", cx, inter)]:
        err_df = checker.df[checker.error_mask].copy()
        err_df["error_reasons"] = checker.err[checker.error_mask]
        err_df.to_csv(paths["errors"] / f"{name}_errors.csv", index=False)
        clean.to_parquet(paths["processed"] / f"validated_{name}.parquet", index=False)
        n_in, n_bad = len(checker.df), int(checker.error_mask.sum())
        stats["tables"][name] = {"records_in": n_in, "valid": len(clean), "rejected": n_bad,
                                 "rejected_pct": 100 * n_bad / n_in, "warning_rows": int((checker.warn != "").sum())}
        results += checker.results
        log.info("%-12s in=%d valid=%d rejected=%d (%.2f%%) warning_rows=%d", name, n_in, len(clean), n_bad,
                 100 * n_bad / n_in, stats["tables"][name]["warning_rows"])
        for r in checker.results:
            if r["failed"]:
                log.info("   [%s] %-28s failed=%d", r["severity"].upper(), r["check"], r["failed"])

    stats["checks"] = results
    stats["seconds"] = round(time.time() - t0, 2)
    write_json(stats, paths["reports"] / "validation_report.json")
    (paths["reports"] / "validation_report.md").write_text(_report_md(stats, results))

    worst = max(s["rejected_pct"] for s in stats["tables"].values()) / 100
    if worst > v["max_error_rate"]:
        raise DataQualityError(f"Quality gate failed: {worst:.1%} rejected > {v['max_error_rate']:.0%}")
    log.info("Quality gate passed (worst table rejected %.2f%% <= %.0f%%)", 100 * worst, 100 * v["max_error_rate"])
    return {"tables": stats["tables"], "seconds": stats["seconds"]}

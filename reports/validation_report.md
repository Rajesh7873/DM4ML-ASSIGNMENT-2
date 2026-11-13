# Data Validation Report

Reference date: `2025-01-01` | Quality gate: max 15% rejected per table

## Summary

| Table | Records in | Valid | Rejected | Rejected % | Warnings (rows) |
|---|---:|---:|---:|---:|---:|
| users | 1,513 | 1,493 | 20 | 1.32% | 136 |
| items | 605 | 600 | 5 | 0.83% | 34 |
| interactions | 59,007 | 52,736 | 6,271 | 10.63% | 1,954 |

## Check results

| Table | Check | Severity | Failed | Checked | Failed % | Description | Action |
|---|---|---|---:|---:|---:|---|---|
| users | `missing_user_id` | error | 0 | 1,513 | 0.00% | user_id is empty | rejected -> errors dataset |
| users | `invalid_dtype_user_id` | error | 0 | 1,513 | 0.00% | user_id is not an integer | rejected -> errors dataset |
| users | `exact_duplicate_row` | error | 13 | 1,513 | 0.86% | Row is an exact copy of an earlier row | rejected -> errors dataset |
| users | `duplicate_primary_key` | error | 0 | 1,513 | 0.00% | user_id repeated with different content | rejected -> errors dataset |
| users | `missing_age` | warning | 63 | 1,513 | 4.16% | age is empty -> imputed with median | kept, repaired in preparation |
| users | `age_out_of_range` | warning | 13 | 1,513 | 0.86% | age outside [10, 100] -> set to missing, then imputed | kept, repaired in preparation |
| users | `missing_country` | warning | 38 | 1,513 | 2.51% | country is empty -> 'unknown' | kept, repaired in preparation |
| users | `missing_gender` | warning | 26 | 1,513 | 1.72% | gender is empty -> 'unknown' | kept, repaired in preparation |
| users | `invalid_signup_date` | error | 7 | 1,513 | 0.46% | signup_date is not YYYY-MM-DD | rejected -> errors dataset |
| users | `missing_signup_date` | error | 0 | 1,513 | 0.00% | signup_date is empty | rejected -> errors dataset |
| users | `future_signup_date` | error | 0 | 1,513 | 0.00% | signup_date is after the reference date | rejected -> errors dataset |
| items | `missing_item_id` | error | 0 | 605 | 0.00% | item_id is empty | rejected -> errors dataset |
| items | `invalid_dtype_item_id` | error | 0 | 605 | 0.00% | item_id is not an integer | rejected -> errors dataset |
| items | `exact_duplicate_row` | error | 5 | 605 | 0.83% | Row is an exact copy of an earlier row | rejected -> errors dataset |
| items | `duplicate_primary_key` | error | 0 | 605 | 0.00% | item_id repeated with different content | rejected -> errors dataset |
| items | `missing_title` | warning | 19 | 605 | 3.14% | title is empty -> placeholder title | kept, repaired in preparation |
| items | `missing_category` | warning | 15 | 605 | 2.48% | category is empty -> 'unknown' | kept, repaired in preparation |
| items | `release_year_out_of_range` | warning | 3 | 605 | 0.50% | release_year outside plausible range -> set to missing, then imputed | kept, repaired in preparation |
| interactions | `missing_user_id` | error | 0 | 59,007 | 0.00% | user_id is empty | rejected -> errors dataset |
| interactions | `missing_item_id` | error | 0 | 59,007 | 0.00% | item_id is empty | rejected -> errors dataset |
| interactions | `invalid_dtype_user_id` | error | 649 | 59,007 | 1.10% | user_id is not an integer | rejected -> errors dataset |
| interactions | `invalid_dtype_item_id` | error | 0 | 59,007 | 0.00% | item_id is not an integer | rejected -> errors dataset |
| interactions | `missing_rating` | error | 504 | 59,007 | 0.85% | rating is empty | rejected -> errors dataset |
| interactions | `invalid_dtype_rating` | error | 147 | 59,007 | 0.25% | rating is not numeric | rejected -> errors dataset |
| interactions | `rating_out_of_range` | error | 1,144 | 59,007 | 1.94% | rating outside [1, 5] | rejected -> errors dataset |
| interactions | `missing_timestamp` | error | 0 | 59,007 | 0.00% | timestamp is empty | rejected -> errors dataset |
| interactions | `invalid_timestamp` | error | 874 | 59,007 | 1.48% | timestamp is not '%Y-%m-%d %H:%M:%S' | rejected -> errors dataset |
| interactions | `future_timestamp` | error | 477 | 59,007 | 0.81% | timestamp is after the reference date | rejected -> errors dataset |
| interactions | `exact_duplicate_row` | error | 732 | 59,007 | 1.24% | Row is an exact copy of an earlier row | rejected -> errors dataset |
| interactions | `repeat_user_item_pair` | warning | 1,954 | 59,007 | 3.31% | same (user, item) rated more than once -> latest rating kept in preparation | kept, repaired in preparation |
| interactions | `orphan_user_id` | error | 1,030 | 59,007 | 1.75% | user_id not found in validated users | rejected -> errors dataset |
| interactions | `orphan_item_id` | error | 732 | 59,007 | 1.24% | item_id not found in validated items | rejected -> errors dataset |
| interactions | `interaction_before_signup` | error | 245 | 59,007 | 0.42% | interaction happened before the user's signup_date | rejected -> errors dataset |

*Note:* a row can fail several checks, so per-check counts can sum to more than the rejected total.

"""Feature registry: single source of truth for every feature (name, entity, definition, source).

The feature catalog (reports/feature_catalog.md|csv) is generated from this list, so the documentation
can never drift from the code. All features are computed AS OF the train cutoff, from train data only.
"""
FEATURES = [
    # ---- user features (entity: user_id)
    ("user_avg_rating", "user", "float", "Mean rating given by the user", "mean(rating) over user's train interactions", "train.parquet"),
    ("user_rating_std", "user", "float", "Spread of the user's ratings (0 if one rating)", "std(rating), NaN->0", "train.parquet"),
    ("user_activity_count", "user", "int", "Number of interactions by the user", "count(interactions)", "train.parquet"),
    ("user_distinct_items", "user", "int", "Number of distinct items the user interacted with", "nunique(item_id)", "train.parquet"),
    ("user_days_since_last_interaction", "user", "float", "Recency: days between the user's last interaction and the train cutoff", "(cutoff - max(timestamp)) / 1 day", "train.parquet"),
    ("user_tenure_days", "user", "float", "Days between the user's first and last train interaction", "(max(ts) - min(ts)) / 1 day", "train.parquet"),
    ("user_days_since_signup", "user", "float", "Account age at the train cutoff", "(cutoff - signup_date) / 1 day", "users_clean.parquet"),
    ("user_preferred_category", "user", "str", "Genre with the highest weight in the user's profile", "argmax(genre_pref_*)", "train + item_genre_matrix"),
    ("user_preferred_category_share", "user", "float", "Share of the profile held by the preferred genre (taste concentration)", "max(genre_pref_*)", "train + item_genre_matrix"),
    ("genre_pref_<genre> (x11)", "user", "float", "Normalised genre preference profile (sums to 1); built from liked items (rating >= 4), falling back to all items", "L1-normalise(sum of item genre vectors)", "train + item_genre_matrix"),
    ("user_age / user_age_group / user_country", "user", "int/str", "Cleaned demographic attributes", "imputed + standardised in preparation", "users_clean.parquet"),
    # ---- item features (entity: item_id)
    ("item_popularity", "item", "int", "Number of train interactions with the item", "count(interactions)", "train.parquet"),
    ("item_unique_users", "item", "int", "Number of distinct users who interacted with the item", "nunique(user_id)", "train.parquet"),
    ("item_avg_rating", "item", "float", "Mean rating of the item", "mean(rating)", "train.parquet"),
    ("item_bayes_rating", "item", "float", "Mean rating shrunk toward the global mean (robust for rarely-rated items)", "(C*global_mean + sum(rating)) / (C + n), C=10", "train.parquet"),
    ("item_popularity_rank", "item", "int", "Rank by interaction count (1 = most popular)", "rank(-item_popularity)", "train.parquet"),
    ("item_popularity_pct", "item", "float", "Popularity percentile in [0,1]", "rank(item_popularity)/N", "train.parquet"),
    ("item_days_since_last_interaction", "item", "float", "Days between the item's last interaction and the train cutoff", "(cutoff - max(timestamp)) / 1 day", "train.parquet"),
    ("item_age_years", "item", "int", "Item age at the train cutoff", "cutoff.year - release_year", "items_clean.parquet"),
    ("item_n_categories", "item", "int", "Number of genres attached to the item", "sum(genre_*)", "item_genre_matrix"),
    ("item_primary_category", "item", "str", "First (alphabetical) genre of the item", "first(category)", "items_clean.parquet"),
    # ---- interaction / pair features (entity: user_id x item_id) - computed on demand
    ("user_item_category_affinity", "pair", "float", "How well the item's genres match the user's genre profile", "(genre_pref_u . genre_i) / n_categories_i", "user + item features"),
    ("item_avg_minus_user_avg", "pair", "float", "Item quality relative to the user's rating level", "item_bayes_rating - user_avg_rating", "user + item features"),
    ("item_popularity_pct_x_activity", "pair", "float", "Popularity percentile weighted by how active the user is (log scale)", "item_popularity_pct * log1p(user_activity_count)", "user + item features"),
]

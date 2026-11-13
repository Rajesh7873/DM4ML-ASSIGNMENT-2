# Model Evaluation Results

Evaluation users: **1345** (warm-start, >=1 test item rated >= 4); catalogue = 599 items; train-seen items excluded from recommendations; time-based split.

## K = 5

| Model | Precision@K | Recall@K | NDCG@K | Coverage | Diversity |
|---|---:|---:|---:|---:|---:|
| popularity | 0.0512 | 0.0521 | 0.0630 | 0.033 | 0.849 |
| als_mf | 0.0943 | 0.1090 | 0.1225 | 0.648 | 0.695 |
| hybrid | 0.1285 | 0.1524 | 0.1735 | 0.641 | 0.195 |

## K = 10

| Model | Precision@K | Recall@K | NDCG@K | Coverage | Diversity |
|---|---:|---:|---:|---:|---:|
| popularity | 0.0502 | 0.1124 | 0.0833 | 0.055 | 0.850 |
| als_mf | 0.0742 | 0.1721 | 0.1403 | 0.748 | 0.730 |
| hybrid | 0.1016 | 0.2321 | 0.1956 | 0.758 | 0.233 |

## Rating prediction (all warm test interactions)

| Model | RMSE | MAE |
|---|---:|---:|
| popularity | 0.8950 | 0.7314 |
| als_mf | 0.8038 | 0.6221 |
| hybrid | n/a (ranking-only) | n/a |

*Diversity* = 1 - mean pairwise cosine similarity of genre vectors inside a list (higher = more varied). *Coverage* = share of the catalogue appearing in at least one user's list.

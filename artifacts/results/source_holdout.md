# Source Holdout

## False positive rate by publisher provenance

| Model | FPR, seen publishers | FPR, unseen publisher | Difference | 95% CI | p | Acc. seen | Acc. unseen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.0727 | 0.2427 | 0.1700 | [+0.1521, +0.1879] | 5.74e-76 | 0.9343 | 0.8493 |
| TextCNN | 0.0307 | 0.2470 | 0.2163 | [+0.1998, +0.2330] | 1.63e-144 | 0.9497 | 0.8415 |
| Ensemble (soft voting) | 0.0327 | 0.2167 | 0.1840 | [+0.1680, +0.2002] | 2.38e-113 | 0.9587 | 0.8667 |

## Split composition

| Quantity | Articles |
| --- | --- |
| train_total | 18826 |
| train_legit | 9413 |
| train_fake | 9413 |
| validation_total | 2690 |
| test_seen_legit | 3000 |
| test_fake | 3000 |
| test_unseen_legit | 3000 |
| dateline_articles_excluded | 21289 |
| training_balanced | 1 |

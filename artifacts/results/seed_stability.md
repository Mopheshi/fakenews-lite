# Seed Stability

## Stability across seeds (mean ± SD)

| Model | accuracy | f1-fake | macro-f1 | roc-auc | n seeds |
| --- | --- | --- | --- | --- | --- |
| XGBoost | 0.9582 ± 0.0006 | 0.9596 ± 0.0006 | 0.9581 ± 0.0006 | 0.9924 ± 0.0002 | 5 |
| TextCNN | 0.9746 ± 0.0031 | 0.9754 ± 0.0031 | 0.9746 ± 0.0031 | 0.9968 ± 0.0008 | 5 |
| Ensemble (soft voting) | 0.9781 ± 0.0020 | 0.9788 ± 0.0020 | 0.9781 ± 0.0020 | 0.9971 ± 0.0005 | 5 |
| Ensemble (weighted voting) | 0.9781 ± 0.0018 | 0.9788 ± 0.0018 | 0.9780 ± 0.0018 | 0.9971 ± 0.0005 | 5 |
| Ensemble (stacking) | 0.9785 ± 0.0018 | 0.9791 ± 0.0017 | 0.9784 ± 0.0018 | 0.9972 ± 0.0005 | 5 |

## Accuracy with bootstrap confidence intervals

| Model | Accuracy | 95% CI |
| --- | --- | --- |
| XGBoost | 0.9575 | [0.9542, 0.9608] |
| TextCNN | 0.9775 | [0.9750, 0.9799] |
| Ensemble (soft voting) | 0.9797 | [0.9773, 0.9819] |
| Ensemble (weighted voting) | 0.9798 | [0.9774, 0.9820] |
| Ensemble (stacking) | 0.9799 | [0.9775, 0.9821] |

## Paired McNemar tests on the reference split

| Comparison | A right / B wrong | B right / A wrong | Accuracy difference | 95% CI | p (McNemar) | Significant at 0.05 |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost vs TextCNN | 187 | 476 | -0.0200 | [-0.0235, -0.0166] | 4.829e-29 | yes |
| XGBoost vs Ensemble (soft voting) | 86 | 406 | -0.0222 | [-0.0252, -0.0192] | 6.749e-47 | yes |
| XGBoost vs Ensemble (weighted voting) | 88 | 409 | -0.0223 | [-0.0252, -0.0193] | 1.006e-46 | yes |
| XGBoost vs Ensemble (stacking) | 112 | 435 | -0.0224 | [-0.0255, -0.0193] | 3.986e-43 | yes |
| TextCNN vs Ensemble (soft voting) | 70 | 101 | -0.0021 | [-0.0040, -0.0003] | 0.02178 | yes |
| TextCNN vs Ensemble (weighted voting) | 67 | 99 | -0.0022 | [-0.0040, -0.0005] | 0.01613 | yes |
| TextCNN vs Ensemble (stacking) | 41 | 75 | -0.0024 | [-0.0038, -0.0009] | 0.002184 | yes |
| Ensemble (soft voting) vs Ensemble (weighted voting) | 2 | 3 | -0.0001 | [-0.0003, +0.0002] | 1 | no |
| Ensemble (soft voting) vs Ensemble (stacking) | 31 | 34 | -0.0002 | [-0.0013, +0.0009] | 0.8041 | no |
| Ensemble (weighted voting) vs Ensemble (stacking) | 29 | 31 | -0.0001 | [-0.0012, +0.0009] | 0.8973 | no |

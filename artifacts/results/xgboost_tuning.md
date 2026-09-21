# Xgboost Tuning

## Fixed versus tuned XGBoost

| Configuration | Accuracy | F1 (fake) | ROC-AUC | Fit seconds |
| --- | --- | --- | --- | --- |
| Fixed (as used in the paper) | 0.9575 | 0.9591 | 0.9925 | 46.8000 |
| Tuned (20 candidates, 3-fold) | 0.9651 | 0.9663 | 0.9946 | 2553.1000 |
| Difference (tuned - fixed) | 0.0076 | 0.0072 | 0.0021 |  |

## Selected hyperparameters

| Hyperparameter | Fixed | Tuned |
| --- | --- | --- |
| colsample_bytree | 0.9000 | 1.0000 |
| gamma | 0.1000 | 0.1000 |
| learning_rate | 0.1000 | 0.2000 |
| max_depth | 7 | 6 |
| min_child_weight | 3 | 7 |
| n_estimators | 200 | 500 |
| subsample | 0.9000 | 0.8000 |

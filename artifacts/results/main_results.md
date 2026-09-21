# Main Results

## Overall performance

| Model | Accuracy | Precision (fake) | Recall (fake) | F1 (fake) | Macro-F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.9575 | 0.9495 | 0.9689 | 0.9591 | 0.9574 | 0.9925 |
| TextCNN | 0.9775 | 0.9714 | 0.9853 | 0.9783 | 0.9775 | 0.9971 |
| Ensemble (soft voting) | 0.9797 | 0.9734 | 0.9875 | 0.9804 | 0.9797 | 0.9975 |
| Ensemble (weighted voting) | 0.9798 | 0.9734 | 0.9876 | 0.9805 | 0.9797 | 0.9975 |
| Ensemble (stacking) | 0.9799 | 0.9763 | 0.9848 | 0.9805 | 0.9799 | 0.9976 |

## Per-class performance

| Model | Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- | --- |
| XGBoost | Legit | 0.9663 | 0.9455 | 0.9558 | 7006 |
| XGBoost | Fake | 0.9495 | 0.9689 | 0.9591 | 7419 |
| TextCNN | Legit | 0.9842 | 0.9693 | 0.9767 | 7006 |
| TextCNN | Fake | 0.9714 | 0.9853 | 0.9783 | 7419 |
| Ensemble (soft voting) | Legit | 0.9865 | 0.9715 | 0.9789 | 7006 |
| Ensemble (soft voting) | Fake | 0.9734 | 0.9875 | 0.9804 | 7419 |
| Ensemble (weighted voting) | Legit | 0.9867 | 0.9715 | 0.9790 | 7006 |
| Ensemble (weighted voting) | Fake | 0.9734 | 0.9876 | 0.9805 | 7419 |
| Ensemble (stacking) | Legit | 0.9837 | 0.9747 | 0.9792 | 7006 |
| Ensemble (stacking) | Fake | 0.9763 | 0.9848 | 0.9805 | 7419 |

## Error breakdown

| Model | False positives (legit flagged fake) | False negatives (fake missed) | False positive rate |
| --- | --- | --- | --- |
| XGBoost | 382 | 231 | 0.0545 |
| TextCNN | 215 | 109 | 0.0307 |
| Ensemble (soft voting) | 200 | 93 | 0.0285 |
| Ensemble (weighted voting) | 200 | 92 | 0.0285 |
| Ensemble (stacking) | 177 | 113 | 0.0253 |

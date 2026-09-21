# Pos Ablation

## Class-conditional POS ratios

| POS category | Mean (legit) | Mean (fake) | Difference | Cohen's d | Magnitude | p |
| --- | --- | --- | --- | --- | --- | --- |
| Nouns | 0.3362 | 0.3395 | 0.0033 | 0.0421 | negligible | 8.95e-09 |
| Verbs | 0.1510 | 0.1481 | -0.0030 | -0.0948 | negligible | 5.73e-38 |
| Adjectives | 0.0688 | 0.0654 | -0.0034 | -0.1625 | negligible | 1.92e-106 |
| Adverbs | 0.0277 | 0.0358 | 0.0081 | 0.5167 | medium | 0.00e+00 |
| Pronouns | 0.0390 | 0.0515 | 0.0125 | 0.5019 | medium | 0.00e+00 |
| Modals | 0.0097 | 0.0088 | -0.0009 | -0.1092 | negligible | 1.28e-48 |

## XGBoost ablation

| Configuration | Accuracy | F1 (fake) | Macro-F1 | ROC-AUC |
| --- | --- | --- | --- | --- |
| SVD only (300 features) | 0.9560 | 0.9577 | 0.9559 | 0.9926 |
| SVD + POS (306 features) | 0.9575 | 0.9591 | 0.9574 | 0.9925 |
| Difference (POS - no POS) | 0.0015 | 0.0015 | 0.0015 | -0.0001 |

## POS feature importance (gain)

| POS category | Gain | Share of total gain | Rank (of 306) |
| --- | --- | --- | --- |
| Nouns | 13.0517 | 0.0058 | 30 |
| Verbs | 12.4891 | 0.0055 | 32 |
| Adjectives | 4.6873 | 0.0021 | 166 |
| Adverbs | 25.6861 | 0.0114 | 7 |
| Pronouns | 13.7618 | 0.0061 | 23 |
| Modals | 4.3116 | 0.0019 | 196 |

## Ensemble-level effect

| Configuration | Accuracy | F1 (fake) | ROC-AUC |
| --- | --- | --- | --- |
| Ensemble with POS | 0.9797 | 0.9804 | 0.9975 |
| Ensemble without POS | 0.9797 | 0.9804 | 0.9975 |

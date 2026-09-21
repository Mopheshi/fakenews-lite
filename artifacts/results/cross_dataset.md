# Cross Dataset

## Transfer to an unseen corpus

| Model | Corpus | n | Accuracy | 95% CI | F1 (fake) | ROC-AUC | FPR | Mean P(fake) | Oracle threshold | Oracle accuracy | WELFake accuracy | Drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | All Perez-Rosas | 980 | 0.6010 | [0.570, 0.631] | 0.6899 | 0.6692 | 0.6857 | 0.7439 | 0.9134 | 0.6235 | 0.9575 | 0.3565 |
| XGBoost | Celebrity | 500 | 0.6080 | [0.565, 0.650] | 0.7066 | 0.6935 | 0.7280 | 0.8006 | 0.9607 | 0.6500 | 0.9575 | 0.3495 |
| XGBoost | FakeNewsAMT | 480 | 0.5938 | [0.549, 0.637] | 0.6712 | 0.6509 | 0.6417 | 0.6848 | 0.8322 | 0.6146 | 0.9575 | 0.3638 |
| TextCNN | All Perez-Rosas | 980 | 0.5551 | [0.524, 0.586] | 0.6722 | 0.6259 | 0.8020 | 0.8164 | 0.9194 | 0.6071 | 0.9775 | 0.4224 |
| TextCNN | Celebrity | 500 | 0.5820 | [0.538, 0.624] | 0.6904 | 0.6419 | 0.7680 | 0.8183 | 0.9564 | 0.6160 | 0.9775 | 0.3955 |
| TextCNN | FakeNewsAMT | 480 | 0.5271 | [0.482, 0.571] | 0.6534 | 0.6094 | 0.8375 | 0.8145 | 0.8869 | 0.6104 | 0.9775 | 0.4505 |
| Ensemble (soft voting) | All Perez-Rosas | 980 | 0.5776 | [0.546, 0.608] | 0.6883 | 0.6765 | 0.7776 | 0.7802 | 0.8749 | 0.6316 | 0.9797 | 0.4021 |
| Ensemble (soft voting) | Celebrity | 500 | 0.5880 | [0.544, 0.630] | 0.7014 | 0.7026 | 0.7920 | 0.8095 | 0.9349 | 0.6520 | 0.9797 | 0.3917 |
| Ensemble (soft voting) | FakeNewsAMT | 480 | 0.5667 | [0.522, 0.610] | 0.6740 | 0.6529 | 0.7625 | 0.7497 | 0.8460 | 0.6229 | 0.9797 | 0.4130 |

## FakeNewsAMT by topical domain

| Model | Domain | n | Accuracy | ROC-AUC | FPR | Mean P(fake) |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost | business | 80 | 0.5625 | 0.6500 | 0.5000 | 0.5137 |
| XGBoost | education | 80 | 0.5500 | 0.6194 | 0.7500 | 0.7548 |
| XGBoost | entertainment | 80 | 0.5625 | 0.6112 | 0.8000 | 0.8050 |
| XGBoost | politics | 80 | 0.6750 | 0.7644 | 0.3750 | 0.5503 |
| XGBoost | sports | 80 | 0.5875 | 0.5950 | 0.7250 | 0.7355 |
| XGBoost | technology | 80 | 0.6250 | 0.6837 | 0.7000 | 0.7499 |
| TextCNN | business | 80 | 0.5375 | 0.6475 | 0.8500 | 0.8120 |
| TextCNN | education | 80 | 0.5375 | 0.6894 | 0.8000 | 0.7791 |
| TextCNN | entertainment | 80 | 0.5250 | 0.5481 | 0.9000 | 0.9013 |
| TextCNN | politics | 80 | 0.5375 | 0.5931 | 0.7250 | 0.6987 |
| TextCNN | sports | 80 | 0.4625 | 0.5787 | 0.9500 | 0.8720 |
| TextCNN | technology | 80 | 0.5625 | 0.6119 | 0.8000 | 0.8239 |
| Ensemble (soft voting) | business | 80 | 0.6000 | 0.6575 | 0.6250 | 0.6628 |
| Ensemble (soft voting) | education | 80 | 0.5375 | 0.6763 | 0.8000 | 0.7669 |
| Ensemble (soft voting) | entertainment | 80 | 0.5125 | 0.5988 | 0.9500 | 0.8531 |
| Ensemble (soft voting) | politics | 80 | 0.6125 | 0.7181 | 0.5500 | 0.6245 |
| Ensemble (soft voting) | sports | 80 | 0.5125 | 0.5925 | 0.9500 | 0.8037 |
| Ensemble (soft voting) | technology | 80 | 0.6250 | 0.6656 | 0.7000 | 0.7869 |

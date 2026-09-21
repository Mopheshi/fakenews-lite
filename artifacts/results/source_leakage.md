# Source Leakage

## Performance under source-marker removal

| Condition | Model | Accuracy | F1 (fake) | ROC-AUC | Delta accuracy |
| --- | --- | --- | --- | --- | --- |
| Unmodified | XGBoost | 0.9575 | 0.9591 | 0.9925 | 0.0000 |
| Unmodified | TextCNN | 0.9775 | 0.9783 | 0.9971 | 0.0000 |
| Unmodified | Ensemble (soft voting) | 0.9797 | 0.9804 | 0.9975 | 0.0000 |
| Wire markers removed | XGBoost | 0.9546 | 0.9562 | 0.9909 | -0.0029 |
| Wire markers removed | TextCNN | 0.9665 | 0.9672 | 0.9953 | -0.0110 |
| Wire markers removed | Ensemble (soft voting) | 0.9746 | 0.9752 | 0.9961 | -0.0051 |
| Wire + outlets removed | XGBoost | 0.9496 | 0.9515 | 0.9901 | -0.0079 |
| Wire + outlets removed | TextCNN | 0.9652 | 0.9660 | 0.9946 | -0.0123 |
| Wire + outlets removed | Ensemble (soft voting) | 0.9714 | 0.9722 | 0.9955 | -0.0082 |

## Blocklist composition

| Preset | Tokens removed | Examples |
| --- | --- | --- |
| none | 0 | (none) |
| wire | 15 | afp, ansa, ap, apimages, epa, getty, ians, interfax |
| extended | 58 | abc, activistpost, afp, ansa, ap, apimages, bbc, bloomberg |

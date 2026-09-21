# Efficiency

## Hardware

| Property | Value |
| --- | --- |
| cpu_model | 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz |
| cpu_clock_mhz | 2803.0 |
| cpu_clock_basis | rated |
| cpu_identifier | Intel64 Family 6 Model 140 Stepping 1, GenuineIntel |
| architecture | AMD64 |
| physical_cores | 4 |
| logical_cores | 8 |
| total_ram_gb | 15.8 |
| gpus_visible | 0 |

## Software and threading

| Property | Value |
| --- | --- |
| os | Windows-11-10.0.26200-SP0 |
| python | 3.13.5 |
| python_implementation | CPython |
| packages.numpy | 2.3.5 |
| packages.scipy | 1.16.3 |
| packages.pandas | 2.3.3 |
| packages.sklearn | 1.8.0 |
| packages.tensorflow | 2.20.0 |
| packages.xgboost | 3.1.2 |
| packages.nltk | 3.9.2 |
| thread_environment.OMP_NUM_THREADS | unset |
| thread_environment.MKL_NUM_THREADS | unset |
| thread_environment.OPENBLAS_NUM_THREADS | unset |
| thread_environment.NUMEXPR_NUM_THREADS | unset |
| thread_environment.VECLIB_MAXIMUM_THREADS | unset |
| thread_environment.TF_NUM_INTRAOP_THREADS | unset |
| thread_environment.TF_NUM_INTEROP_THREADS | unset |
| tensorflow_threads.intra_op | 0 |
| tensorflow_threads.inter_op | 0 |
| tensorflow_threads.note | 0 means TensorFlow chooses, normally one thread per logical core |

## Thread configuration during the latency benchmark

| Property | Value |
| --- | --- |
| Latency benchmark threads | unconstrained |
| TensorFlow intra/inter op | TensorFlow default (one thread per logical core) |
| openblas (libscipy_openblas) | 8 threads, version 0.3.30 |
| openblas (libscipy_openblas) | 8 threads, version 0.3.29.dev |
| openmp (vcomp) | 8 threads, version None |

## Parameter accounting

| Component | Count | Note |
| --- | --- | --- |
| TextCNN - learned weights | 203,393 | convolution filters, dense layer, output layer |
| TextCNN - frozen GloVe embeddings | 5,000,000 | pre-trained, never updated during training |
| TextCNN - total | 5,203,393 | learned + frozen |
| XGBoost - trees | 200 | max depth 7 |
| XGBoost - split + leaf nodes | 35,426 | 17,613 splits, 17,813 leaves |
| XGBoost - stored values | 53,039 | two per split (feature, threshold) plus one per leaf |

## Published transformer sizes (citations, not measurements)

| Model | Parameters | Ratio vs our learned parameters | Ratio vs our total footprint |
| --- | --- | --- | --- |
| BERT-base | 110,000,000 | 429x | 21x |
| DistilBERT | 66,000,000 | 257x | 13x |
| RoBERTa-base | 125,000,000 | 487x | 24x |
| GPT-2 (124M) | 124,000,000 | 484x | 24x |

## Training wall-clock

| Stage | Seconds |
| --- | --- |
| TF-IDF + SVD fit/transform | 173.7000 |
| XGBoost training | 23.5000 |
| Sequence encoding | 53.8000 |
| TextCNN training | 642.5000 |
| Total (excluding one-off preprocessing) | 893.5000 |

## Inference latency (single article, CPU)

| Measure | Value |
| --- | --- |
| Mean | 364.79 ms |
| Median | 261.32 ms |
| 95th percentile | 717.66 ms |
| 99th percentile | 855.29 ms |
| Throughput | 2.7 articles/s |
| Mean article length | 3,255 characters |
| Model load time | 1.16 s |
| Peak resident memory | 1214 MB |

## Model artifacts on disk

| File | Size (MB) |
| --- | --- |
| encoders.pkl | 38.3810 |
| ensemble.json | 0.0000 |
| manifest.json | 0.0010 |
| text_cnn.keras | 21.4530 |
| xgboost.json | 2.1550 |
| Total | 61.9900 |

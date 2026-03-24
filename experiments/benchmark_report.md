# 🐟 markrel Benchmark Report
Generated: 2026-03-21 23:23:12
============================================================

## Summary
| Configuration | Accuracy | F1 Score | AUC-ROC | Speed (s/sec) |
|--------------|----------|----------|---------|---------------|
| Cosine Only | 0.800 | 0.808 | 0.839 | 17812 |
| Jaccard Only | 0.800 | 0.808 | 0.839 | 16000 |
| Euclidean Only | 0.800 | 0.808 | 0.839 | 21576 |
| Cosine + Jaccard | 0.800 | 0.808 | 0.839 | 12495 |
| Cosine + Jaccard + Overlap | 0.800 | 0.808 | 0.839 | 9200 |
| Uniform (5 bins) | 0.780 | 0.784 | 0.823 | 18807 |
| Uniform (10 bins) | 0.800 | 0.808 | 0.839 | 18008 |
| Uniform (20 bins) | 0.800 | 0.808 | 0.839 | 18180 |
| Quantile (5 bins) | 0.590 | 0.506 | 0.669 | 19518 |
| Quantile (10 bins) | 0.750 | 0.747 | 0.798 | 17853 |
| Quantile (20 bins) | 0.790 | 0.796 | 0.831 | 17829 |
| Mean Combination | 0.800 | 0.808 | 0.839 | 12844 |
| Bayesian Combination | 0.800 | 0.808 | 0.839 | 13083 |

## Detailed Results

### Cosine Only
- **Train samples**: 400
- **Test samples**: 100
- **Config**: `{'metrics': ['cosine'], 'n_bins': 10}`

**Metrics:**
- Accuracy: 0.800
- Precision: 1.000
- Recall: 0.677
- F1 Score: 0.808
- AUC-ROC: 0.839
- Train time: 0.03s
- Prediction time: 0.01s
- Throughput: 17812 samples/sec

### Jaccard Only
- **Train samples**: 400
- **Test samples**: 100
- **Config**: `{'metrics': ['jaccard'], 'n_bins': 10}`

**Metrics:**
- Accuracy: 0.800
- Precision: 1.000
- Recall: 0.677
- F1 Score: 0.808
- AUC-ROC: 0.839
- Train time: 0.03s
- Prediction time: 0.01s
- Throughput: 16000 samples/sec

### Euclidean Only
- **Train samples**: 400
- **Test samples**: 100
- **Config**: `{'metrics': ['euclidean'], 'n_bins': 10}`

**Metrics:**
- Accuracy: 0.800
- Precision: 1.000
- Recall: 0.677
- F1 Score: 0.808
- AUC-ROC: 0.839
- Train time: 0.02s
- Prediction time: 0.00s
- Throughput: 21576 samples/sec

### Cosine + Jaccard
- **Train samples**: 400
- **Test samples**: 100
- **Config**: `{'metrics': ['cosine', 'jaccard'], 'n_bins': 10}`

**Metrics:**
- Accuracy: 0.800
- Precision: 1.000
- Recall: 0.677
- F1 Score: 0.808
- AUC-ROC: 0.839
- Train time: 0.03s
- Prediction time: 0.01s
- Throughput: 12495 samples/sec

### Cosine + Jaccard + Overlap
- **Train samples**: 400
- **Test samples**: 100
- **Config**: `{'metrics': ['cosine', 'jaccard', 'overlap'], 'n_bins': 10}`

**Metrics:**
- Accuracy: 0.800
- Precision: 1.000
- Recall: 0.677
- F1 Score: 0.808
- AUC-ROC: 0.839
- Train time: 0.05s
- Prediction time: 0.01s
- Throughput: 9200 samples/sec

### Uniform (5 bins)
- **Train samples**: 400
- **Test samples**: 100
- **Config**: `{'metrics': ['cosine'], 'n_bins': 5, 'bin_strategy': 'uniform'}`

**Metrics:**
- Accuracy: 0.780
- Precision: 1.000
- Recall: 0.645
- F1 Score: 0.784
- AUC-ROC: 0.823
- Train time: 0.02s
- Prediction time: 0.01s
- Throughput: 18807 samples/sec

### Uniform (10 bins)
- **Train samples**: 400
- **Test samples**: 100
- **Config**: `{'metrics': ['cosine'], 'n_bins': 10, 'bin_strategy': 'uniform'}`

**Metrics:**
- Accuracy: 0.800
- Precision: 1.000
- Recall: 0.677
- F1 Score: 0.808
- AUC-ROC: 0.839
- Train time: 0.02s
- Prediction time: 0.01s
- Throughput: 18008 samples/sec

### Uniform (20 bins)
- **Train samples**: 400
- **Test samples**: 100
- **Config**: `{'metrics': ['cosine'], 'n_bins': 20, 'bin_strategy': 'uniform'}`

**Metrics:**
- Accuracy: 0.800
- Precision: 1.000
- Recall: 0.677
- F1 Score: 0.808
- AUC-ROC: 0.839
- Train time: 0.03s
- Prediction time: 0.01s
- Throughput: 18180 samples/sec

### Quantile (5 bins)
- **Train samples**: 400
- **Test samples**: 100
- **Config**: `{'metrics': ['cosine'], 'n_bins': 5, 'bin_strategy': 'quantile'}`

**Metrics:**
- Accuracy: 0.590
- Precision: 1.000
- Recall: 0.339
- F1 Score: 0.506
- AUC-ROC: 0.669
- Train time: 0.03s
- Prediction time: 0.01s
- Throughput: 19518 samples/sec

### Quantile (10 bins)
- **Train samples**: 400
- **Test samples**: 100
- **Config**: `{'metrics': ['cosine'], 'n_bins': 10, 'bin_strategy': 'quantile'}`

**Metrics:**
- Accuracy: 0.750
- Precision: 1.000
- Recall: 0.597
- F1 Score: 0.747
- AUC-ROC: 0.798
- Train time: 0.03s
- Prediction time: 0.01s
- Throughput: 17853 samples/sec

### Quantile (20 bins)
- **Train samples**: 400
- **Test samples**: 100
- **Config**: `{'metrics': ['cosine'], 'n_bins': 20, 'bin_strategy': 'quantile'}`

**Metrics:**
- Accuracy: 0.790
- Precision: 1.000
- Recall: 0.661
- F1 Score: 0.796
- AUC-ROC: 0.831
- Train time: 0.03s
- Prediction time: 0.01s
- Throughput: 17829 samples/sec

### Mean Combination
- **Train samples**: 400
- **Test samples**: 100
- **Config**: `{'metrics': ['cosine', 'jaccard'], 'combine_rule': 'mean'}`

**Metrics:**
- Accuracy: 0.800
- Precision: 1.000
- Recall: 0.677
- F1 Score: 0.808
- AUC-ROC: 0.839
- Train time: 0.03s
- Prediction time: 0.01s
- Throughput: 12844 samples/sec

### Bayesian Combination
- **Train samples**: 400
- **Test samples**: 100
- **Config**: `{'metrics': ['cosine', 'jaccard'], 'combine_rule': 'bayesian'}`

**Metrics:**
- Accuracy: 0.800
- Precision: 1.000
- Recall: 0.677
- F1 Score: 0.808
- AUC-ROC: 0.839
- Train time: 0.03s
- Prediction time: 0.01s
- Throughput: 13083 samples/sec

## Key Findings

1. **Best F1 Score**: Cosine Only (0.808)
2. **Fastest**: Euclidean Only (21576 s/sec)
3. **Best AUC**: Cosine Only (0.839)

## Recommendations
Based on these benchmarks:

1. **For accuracy**: Use multiple metrics with bayesian combination
2. **For speed**: Use single metric (cosine) with fewer bins
3. **Best balance**: Cosine + Jaccard with 10 bins
4. **Binning**: Quantile works better for imbalanced data

# 🐟 markrel WikiQA Embeddings Benchmark Report

**Date:** 2026-03-22 22:26:59

**Dataset:** WikiQA Real Dataset (from Hugging Face)

**Embedding Model:** all-MiniLM-L6-v2

## Results (with Optimized Thresholds)

| Model | Accuracy | F1 | AUC | Threshold | Speed |
|-------|----------|-----|-----|-----------|-------|
| markrel (Cosine) | 0.937 | 0.322 | 0.799 | 0.130 | 64046/s |
| markrel (Cosine + Jaccard) | 0.936 | 0.324 | 0.799 | 0.040 | 25770/s |
| markrel (Cosine + Euclidean) | 0.937 | 0.322 | 0.800 | 0.070 | 39854/s |
| markrel (All metrics) | 0.950 | 0.232 | 0.796 | 0.010 | 9375/s |

## Detailed Results

### markrel (Cosine)
- Threshold: 0.1300
- Accuracy: 0.9372
- Precision: 0.3309
- Recall: 0.3140
- F1: 0.3222
- AUC: 0.7989
- Train time: 0.359s
- Pred time: 0.096s


### markrel (Cosine + Jaccard)
- Threshold: 0.0400
- Accuracy: 0.9363
- Precision: 0.3264
- Recall: 0.3208
- F1: 0.3236
- AUC: 0.7986
- Train time: 0.847s
- Pred time: 0.239s


### markrel (Cosine + Euclidean)
- Threshold: 0.0700
- Accuracy: 0.9372
- Precision: 0.3309
- Recall: 0.3140
- F1: 0.3222
- AUC: 0.7996
- Train time: 0.557s
- Pred time: 0.155s


### markrel (All metrics)
- Threshold: 0.0100
- Accuracy: 0.9496
- Precision: 0.4196
- Recall: 0.1604
- F1: 0.2321
- AUC: 0.7962
- Train time: 2.353s
- Pred time: 0.658s


# 🐟 markrel WikiQA 1024-Dim Embedding Benchmark Report

**Date:** 2026-03-23 14:28:57

**Dataset:** WikiQA (Question-Answer Relevance)

**Embedding Model:** sentence-transformers/all-roberta-large-v1

**Embedding Dimension:** 1024

## Results Summary

| Model | Accuracy | Precision | Recall | F1 | AUC | Avg Prec | Speed |
|-------|----------|-----------|--------|-----|-----|----------|-------|
| markrel (Cosine only) | 0.907 | 0.247 | 0.468 | 0.323 | 0.828 | 0.223 | 54205/s |
| markrel (Cosine + Euclidean) | 0.897 | 0.237 | 0.522 | 0.326 | 0.832 | 0.239 | 37634/s |
| markrel (Cosine + Jaccard) | 0.909 | 0.253 | 0.464 | 0.327 | 0.831 | 0.247 | 23938/s |
| markrel (All metrics) | 0.936 | 0.324 | 0.311 | 0.317 | 0.832 | 0.258 | 8571/s |

## Detailed Results

### markrel (Cosine only)
- **Embedding:** sentence-transformers/all-roberta-large-v1 (1024 dims)
- **Metrics Used:** ['cosine']
- **Bins:** 10 (uniform)
- **Optimal Threshold:** 0.0810

**Performance:**
- Accuracy: 0.9069
- Precision: 0.2468
- Recall: 0.4676
- F1 Score: 0.3231
- AUC-ROC: 0.8279
- Average Precision: 0.2226

**Timing:**
- Train time: 0.426s
- Pred time: 0.114s
- Throughput: 54205 samples/sec


### markrel (Cosine + Euclidean)
- **Embedding:** sentence-transformers/all-roberta-large-v1 (1024 dims)
- **Metrics Used:** ['cosine', 'euclidean']
- **Bins:** 10 (uniform)
- **Optimal Threshold:** 0.0070

**Performance:**
- Accuracy: 0.8973
- Precision: 0.2368
- Recall: 0.5222
- F1 Score: 0.3259
- AUC-ROC: 0.8323
- Average Precision: 0.2393

**Timing:**
- Train time: 0.612s
- Pred time: 0.164s
- Throughput: 37634 samples/sec


### markrel (Cosine + Jaccard)
- **Embedding:** sentence-transformers/all-roberta-large-v1 (1024 dims)
- **Metrics Used:** ['cosine', 'jaccard']
- **Bins:** 10 (uniform)
- **Optimal Threshold:** 0.0350

**Performance:**
- Accuracy: 0.9093
- Precision: 0.2528
- Recall: 0.4642
- F1 Score: 0.3273
- AUC-ROC: 0.8312
- Average Precision: 0.2474

**Timing:**
- Train time: 1.011s
- Pred time: 0.258s
- Throughput: 23938 samples/sec


### markrel (All metrics)
- **Embedding:** sentence-transformers/all-roberta-large-v1 (1024 dims)
- **Metrics Used:** ['cosine', 'euclidean', 'jaccard', 'overlap', 'dice']
- **Bins:** 10 (uniform)
- **Optimal Threshold:** 0.0010

**Performance:**
- Accuracy: 0.9364
- Precision: 0.3238
- Recall: 0.3106
- F1 Score: 0.3171
- AUC-ROC: 0.8324
- Average Precision: 0.2577

**Timing:**
- Train time: 2.617s
- Pred time: 0.719s
- Throughput: 8571 samples/sec


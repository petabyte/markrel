# 🐟 markrel WikiQA Benchmark Report

**Date:** 2026-03-22 22:21:24

**Dataset:** WikiQA-Style Question-Answer Relevance

## Results

| Model | Accuracy | F1 | AUC | Speed |
|-------|----------|-----|-----|-------|
| markrel (Cosine) | 0.952 | 0.000 | 0.622 | 6494/s |
| markrel (Cosine + Jaccard) | 0.952 | 0.000 | 0.628 | 5001/s |
| markrel (All metrics) | 0.952 | 0.000 | 0.659 | 3271/s |

## Detailed Results

### markrel (Cosine)
- Accuracy: 0.9525
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000
- AUC: 0.6223
- Train time: 4.523s
- Pred time: 0.949s


### markrel (Cosine + Jaccard)
- Accuracy: 0.9525
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000
- AUC: 0.6284
- Train time: 5.508s
- Pred time: 1.233s


### markrel (All metrics)
- Accuracy: 0.9525
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000
- AUC: 0.6595
- Train time: 7.714s
- Pred time: 1.885s


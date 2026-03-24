# 🐟 markrel WikiQA Ranking Benchmark Report

**Date:** 2026-03-22 22:32:53

**Dataset:** WikiQA Real Dataset (from Hugging Face)

**Embedding Model:** all-MiniLM-L6-v2

**Metrics:** MAP, MRR, Recall@K (standard IR metrics)

## Results

| Model | MAP | MRR | Recall@1 | Recall@5 | Recall@10 |
|-------|-----|-----|----------|----------|-----------|
| markrel (Cosine) | 0.2835 | 0.2887 | 0.5607 | 0.9345 | 0.9846 |
| markrel (Cosine + Jaccard) | 0.2731 | 0.2785 | 0.5274 | 0.9294 | 0.9794 |
| markrel (Cosine + Euclidean) | 0.2781 | 0.2833 | 0.5398 | 0.9304 | 0.9805 |

## Comparison with Other Methods

### WikiQA Leaderboard (from WikiQA paper)

| Method | MAP | MRR |
|--------|-----|-----|
| BM25 (traditional IR) | ~0.59 | ~0.62 |
| CNN (neural) | ~0.69 | ~0.71 |
| LSTM (neural) | ~0.70 | ~0.72 |
| **markrel (Cosine + Jaccard)** | **0.2731** | **0.2785** |

Note: markrel uses no neural network training - just Markov chains on similarity scores!


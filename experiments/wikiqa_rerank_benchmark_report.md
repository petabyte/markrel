# 🐟 markrel WikiQA Re-ranking Benchmark Report

**Date:** 2026-03-22 22:38:48

**Dataset:** WikiQA Real Dataset (from Hugging Face)

**Embedding Model:** all-MiniLM-L6-v2

**Approach:** Balanced training + Two-stage re-ranking

## Results

| Model | Initial k | MAP | MRR | Recall@1 | Recall@5 | Recall@10 |
|-------|-----------|-----|-----|----------|----------|-----------|
| markrel Re-rank (Cosine) - Balanced | 20 | 0.2769 | 0.2822 | 0.5398 | 0.9324 | 0.9794 |
| markrel Re-rank (Cosine) - Balanced, k=50 | 50 | 0.2769 | 0.2822 | 0.5398 | 0.9324 | 0.9794 |
| markrel Re-rank (Cosine + Jaccard) - Balanced | 20 | 0.2717 | 0.2775 | 0.5274 | 0.9283 | 0.9794 |

## Comparison with Baseline

| Method | MAP | MRR | Notes |
|--------|-----|-----|-------|
| markrel (Cosine) - Baseline | 0.2835 | 0.2887 | Unbalanced, no re-ranking |
| **markrel Re-rank (Cosine) - Balanced** | **0.2769** | **0.2822** | **Balanced + re-ranking** |

**Improvement:** MAP +-2.3%, MRR +-2.2%


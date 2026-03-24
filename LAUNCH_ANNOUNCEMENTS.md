# 🚀 markrel Launch Announcement Template

## Twitter/X Thread

**Tweet 1/5** 🐟
```
🚀 Introducing markrel: Markov chains for document relevance!

Like a school of mackerel swimming through the sea of documents, markrel traces probabilistic paths to find the most relevant matches.

Thread 🧵👇
```

**Tweet 2/5** ⚡
```
Why Markov chains?

Traditional approaches use fixed thresholds:
❌ "Cosine > 0.7 = relevant" (ignores your data)

markrel learns from YOUR data:
✅ Each similarity bin learns P(relevance)
✅ Captures non-linear patterns
✅ Domain-specific tuning
```

**Tweet 3/5** 📊
```
Benchmarks on WikiQA (6K samples):

• BGE-M3: F1=0.343, AUC=0.815
• RoBERTa-large: F1=0.323
• MiniLM: F1=0.322

Optimized configs:
• F1: 35 bins, euclidean
• Recall: 7 bins, euclidean  
• Precision: 24 bins, cos+euc
```

**Tweet 4/5** 🎯
```
Use cases:
🔍 Semantic search reranking
📧 Email classification
📄 Document similarity
🤖 Chatbot response selection

Speed: 50K+ inferences/sec
No GPU required!
```

**Tweet 5/5** 🛠️
```
Try it now:

pip install markrel

Docs: https://github.com/petabyte/markrel
Benchmarks: https://github.com/petabyte/markrel#benchmarks

Star ⭐ if you find it useful!

#Python #MachineLearning #NLP
```

---

## LinkedIn Post

```
🎉 Excited to share markrel — a new Python library for document relevance prediction using Markov chains!

The Problem:
Traditional relevance models use fixed thresholds (e.g., "cosine similarity > 0.7"). This ignores domain-specific patterns and assumes linear relationships.

The Solution:
markrel uses Markov chains to learn P(relevance) from your data. Each similarity "bin" learns its own probability, capturing non-linear patterns like:
• High similarity → Almost certainly relevant
• Medium similarity → Context-dependent
• Low similarity → Probably not relevant

Key Results:
📊 Benchmarked on WikiQA dataset (6,165 samples)
🏆 F1=0.370 with BGE-M3 embeddings
⚡ 50,000+ predictions/second
🐟 Fast, lightweight, no GPU required

Perfect for:
✅ Semantic search reranking
✅ Email/document classification  
✅ Chatbot response selection
✅ Real-time filtering pipelines

Quick Start:
pip install markrel

Links:
📖 GitHub: https://github.com/petabyte/markrel
📊 Benchmarks: See README for detailed results

Would love your feedback! Drop a comment or open an issue.

#MachineLearning #Python #NLP #OpenSource
```

---

## Hacker News Post

**Title**: Show HN: markrel – Markov chain document relevance in Python

**Body**:
```
markrel uses Markov chains to predict document relevance, learning non-linear patterns from your data instead of using fixed thresholds.

The intuition: similarity scores get discretized into bins (Markov states), and each bin learns P(relevant | bin) from training data. This captures patterns like "cosine 0.8 almost always means relevant, but cosine 0.5 depends on context."

Benchmarks on WikiQA (6K question-answer pairs, 4.8% positive):
- BGE-M3 embeddings: F1=0.343, AUC=0.815
- RoBERTa-large: F1=0.323
- MiniLM-L6: F1=0.322

Speed: 50K+ inferences/sec on CPU. Works with any embeddings (BERT, OpenAI, sentence-transformers) or TF-IDF.

Installation: pip install markrel

GitHub: https://github.com/petabyte/markrel

Would appreciate feedback on the approach and implementation!
```

---

## Reddit r/MachineLearning

**Title**: [P] markrel: Markov chain document relevance prediction

**Body**:
```
markrel is a Python library for predicting document relevance using Markov chains and similarity metrics.

**Why Markov chains?**

Traditional approaches often use fixed thresholds (cosine > 0.7 = relevant). This:
- Ignores domain-specific patterns
- Assumes linear relationships
- Can't adapt to your data

Markrel discretizes similarity scores into bins and learns P(relevant | bin) for each. This captures non-linear patterns automatically.

**Benchmarks (WikiQA dataset, 6,165 test samples):**

| Embedding | F1 | AUC | Speed |
|-----------|-----|-----|-------|
| BGE-M3 | 0.343 | 0.815 | 51K/s |
| RoBERTa | 0.323 | 0.828 | 54K/s |
| MiniLM | 0.322 | 0.799 | 61K/s |

**Key findings from experiments:**
- BGE-M3 outperforms larger models
- Single metrics (euclidean) often beat combinations
- Uniform binning consistently beats quantile
- 35 bins optimal for F1, 7 for recall

**Use cases:**
- Semantic search reranking
- Email classification
- Chatbot response selection
- Real-time document filtering

GitHub: https://github.com/petabyte/markrel

Feedback welcome!
```

---

## Dev.to / Medium Article

**Title**: "School Your Documents with Markov Chains: Introducing markrel"

**Outline**:
1. The problem with fixed thresholds
2. How Markov chains solve it
3. Benchmark results
4. Code examples
5. When to use markrel
6. Call to action

**Link**: (Publish and include in promotion)

---

## Email Template (for ML Newsletters)

**Subject**: New open-source library: markrel for document relevance

**Body**:
```
Hi [Name],

I wanted to share markrel, a new Python library I've been working on for document relevance prediction.

What makes it different:
• Uses Markov chains instead of fixed thresholds
• Learns from your data, not generic rules
• 50K+ inferences/second
• Works with any embeddings

Benchmarks show it outperforms traditional approaches on WikiQA (F1=0.37 with BGE-M3).

Would you consider featuring it in [Newsletter Name]? Happy to provide more details or a guest post.

GitHub: https://github.com/petabyte/markrel

Thanks!
[Your Name]
```

---

## 🎨 Visual Assets Needed

1. **Mackerel mascot/logo** (fish swimming in a school pattern)
2. **Architecture diagram** (input → embeddings → Markov chain → output)
3. **Benchmark results chart** (bar chart comparing F1 scores)
4. **GIF demo** (quick code demo in terminal)
5. **Social media cards** (1200x628 for Twitter/LinkedIn)

---

**Ready to launch? Pick a date and start with Tweet 1!** 🚀

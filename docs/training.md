# 🐟 markrel Training Guide

This guide covers how to train a Markov relevance model on your query-document pairs.

## Overview

markrel learns the probability of document relevance given similarity values. During training, it:

1. Computes similarity scores between query-document pairs across all configured metrics
2. Discretizes continuous similarity values into bins (Markov states)
3. Learns P(relevant | bin) from your labeled training data
4. Stores these probabilities for later prediction

## Quick Start

```python
from markrel import MarkovRelevanceModel

# Initialize model
model = MarkovRelevanceModel(
    metrics=["cosine", "jaccard"],
    n_bins=10,
    bin_strategy="uniform",
)

# Train on your data
model.fit(
    queries=train_queries,      # List of query strings or vectors
    documents=train_docs,       # List of document strings or vectors
    labels=train_labels,        # 1=relevant, 0=not relevant
)

# Save for later
import pickle
with open("markrel_model.pkl", "wb") as f:
    pickle.dump(model, f)
```

## Training Data Format

### Text Input (Recommended for Beginners)

```python
queries = [
    "machine learning tutorial",
    "deep learning guide",
    "cooking recipes",
]

documents = [
    "intro to machine learning algorithms",
    "neural network architectures explained",
    "best pasta recipes collection",
]

labels = [1, 1, 0]  # 1=relevant, 0=not relevant
```

markrel automatically:
- Fits a TF-IDF vectorizer on all text
- Transforms queries and documents to vectors
- Computes similarities

### Pre-computed Vectors (Advanced)

If you have embeddings from models like sentence-transformers:

```python
import numpy as np

# Shape: (n_samples, embedding_dim)
query_vectors = np.random.randn(100, 384)
doc_vectors = np.random.randn(100, 384)
labels = [1, 0, 1, ...]  # 100 labels

model = MarkovRelevanceModel(
    metrics=["cosine", "euclidean"],
    use_text_vectorizer=False,  # Important!
)
model.fit(query_vectors, doc_vectors, labels)
```

## Configuration Options

### Similarity Metrics

Choose which metrics to use:

```python
# Single metric (simplest)
model = MarkovRelevanceModel(metrics=["cosine"])

# Multiple metrics (more robust)
model = MarkovRelevanceModel(
    metrics=["cosine", "jaccard", "overlap", "dice"]
)
```

**Available metrics:**
- `cosine` - Best for dense embeddings [-1, 1]
- `euclidean` - Distance converted to similarity [0, 1]
- `manhattan` - L1 distance [0, 1]
- `jaccard` - Set overlap [0, 1]
- `overlap` - Szymkiewicz-Simpson coefficient [0, 1]
- `dice` - Sørensen-Dice coefficient [0, 1]
- `dot_product` - Raw dot product (unbounded)
- `chebyshev` - L∞ distance [0, 1]

### Binning Strategy

```python
# Uniform bins (equal width)
model = MarkovRelevanceModel(
    n_bins=10,
    bin_strategy="uniform",
)

# Quantile bins (equal frequency, data-adaptive)
model = MarkovRelevanceModel(
    n_bins=10,
    bin_strategy="quantile",
)
```

**When to use each:**
- **uniform**: Similarity values spread evenly across range
- **quantile**: Similarity values clustered in certain ranges (recommended for imbalanced data)

### Smoothing

Laplace smoothing prevents zero probabilities:

```python
# Default (moderate smoothing)
model = MarkovRelevanceModel(smoothing=1.0)

# Less smoothing (more confident when you have lots of data)
model = MarkovRelevanceModel(smoothing=0.5)

# More smoothing (safer with small datasets)
model = MarkovRelevanceModel(smoothing=2.0)
```

### Combination Rules

When using multiple metrics, combine their probabilities:

```python
# Bayesian (product of odds) - recommended
model = MarkovRelevanceModel(
    metrics=["cosine", "jaccard"],
    combine_rule="bayesian",
)

# Mean (simple average)
model = MarkovRelevanceModel(
    metrics=["cosine", "jaccard"],
    combine_rule="mean",
)
```

**Recommendation:** Use `bayesian` for multiple metrics. It properly combines independent evidence.

## Training Best Practices

### Data Requirements

- **Minimum**: 20-30 labeled pairs per metric
- **Recommended**: 100+ pairs for good probability estimates
- **Per bin**: At least 2-3 examples per bin for stable estimates

### Label Quality

```python
# Good: Clear relevance labels
labels = [1, 1, 1, 0, 0, 0, 1, 0]  # Clear yes/no

# Avoid: Uncertain/mixed labels
labels = [0.5, 0.3, 0.8, ...]  # ❌ markrel expects binary
```

### Balanced Classes

markrel works best with roughly balanced data:

```python
# Good: ~50% relevant, ~50% not relevant
labels = [1, 1, 1, 1, 0, 0, 0, 0]

# Acceptable: Some imbalance ok
labels = [1, 1, 1, 1, 1, 0, 0]  # 71% relevant

# Avoid: Extreme imbalance
labels = [1, 1, 1, 1, 1, 1, 0]  # 86% relevant - may bias model
```

## Inspecting the Trained Model

### Model Summary

```python
summary = model.summary()
print(summary)
```

Output:
```python
{
    'metrics': ['cosine', 'jaccard'],
    'combine_rule': 'bayesian',
    'chains': {
        'cosine': {
            'metric': 'cosine',
            'n_bins': 10,
            'total_relevant': 50,
            'total_not_relevant': 50,
            'states': [
                {
                    'bin': 0,
                    'range': [0.0, 0.161),
                    'n': 10,
                    'p_relevant': 0.091  # Low similarity → low relevance
                },
                {
                    'bin': 9,
                    'range': [0.804, 1.0),
                    'n': 10,
                    'p_relevant': 0.917  # High similarity → high relevance
                },
            ]
        }
    }
}
```

### View Learned Probabilities

```python
# Get P(relevant | bin) for a specific metric
cosine_probs = model.get_metric_probabilities("cosine")
print(cosine_probs)
# [0.091, 0.182, 0.273, 0.364, 0.455, 0.545, 0.636, 0.727, 0.818, 0.917]
```

## Advanced Training

### Incremental Training

markrel doesn't support incremental training. For new data, retrain from scratch:

```python
# Combine old and new data
all_queries = old_queries + new_queries
all_docs = old_docs + new_docs
all_labels = old_labels + new_labels

# Retrain
model = MarkovRelevanceModel(...)
model.fit(all_queries, all_docs, all_labels)
```

### Cross-Validation

```python
from sklearn.model_selection import KFold
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in kf.split(queries):
    model = MarkovRelevanceModel(metrics=["cosine"])
    
    # Split data
    q_train = [queries[i] for i in train_idx]
    d_train = [documents[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    
    q_val = [queries[i] for i in val_idx]
    d_val = [documents[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]
    
    # Train and evaluate
    model.fit(q_train, d_train, y_train)
    preds = model.predict(q_val, d_val)
    
    accuracy = np.mean(preds == y_val)
    scores.append(accuracy)

print(f"CV Accuracy: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
```

## Troubleshooting

### "All predictions are 0.5"

**Cause**: Not enough data per bin, or all similarities in same range.

**Fix**: 
- Increase `smoothing` (e.g., 2.0)
- Use `quantile` binning
- Collect more diverse training data

### "Model overfits"

**Symptom**: Perfect training accuracy, poor validation accuracy.

**Fix**:
- Reduce `n_bins` (fewer states = simpler model)
- Increase `smoothing`
- Get more training data

### "Predictions seem random"

**Cause**: Similarity metric doesn't match your data type.

**Fix**:
- For sparse text: use `jaccard`, `overlap`, `dice`
- For dense embeddings: use `cosine`, `euclidean`

## Example: Complete Training Pipeline

```python
from markrel import MarkovRelevanceModel
import pickle

# 1. Prepare data
train_queries = [...]  # Your queries
train_docs = [...]     # Your documents
train_labels = [...]   # 1=relevant, 0=not

# 2. Configure model
model = MarkovRelevanceModel(
    metrics=["cosine", "jaccard"],
    n_bins=10,
    bin_strategy="quantile",
    smoothing=1.0,
    combine_rule="bayesian",
)

# 3. Train
print("Training markrel model...")
model.fit(train_queries, train_docs, train_labels)

# 4. Inspect
summary = model.summary()
print(f"Trained on {summary['chains']['cosine']['total_relevant']} relevant pairs")
print(f"Using metrics: {summary['metrics']}")

# 5. Save
with open("markrel_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to markrel_model.pkl")
```

## Next Steps

Once trained, see [inference.md](inference.md) for prediction examples.

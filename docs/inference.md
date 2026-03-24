# 🐟 markrel Inference Guide

This guide covers how to use a trained markrel model to predict document relevance.

## Overview

Once trained, markrel can predict the probability that a document is relevant to a query by:

1. Computing similarity scores between query and document
2. Looking up the learned P(relevant | similarity_bin) for each metric
3. Combining probabilities across metrics
4. Returning a final relevance probability

## Quick Start

```python
from markrel import MarkovRelevanceModel
import pickle

# Load trained model
with open("markrel_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict relevance probability
probs = model.predict_proba(
    queries=["machine learning tutorial"],
    documents=["neural networks explained"]
)
print(f"Relevance probability: {probs[0]:.3f}")

# Binary prediction
predictions = model.predict(
    queries=["ml tutorial", "cooking recipe"],
    documents=["neural nets", "pasta guide"],
    threshold=0.5
)
print(predictions)  # [1, 0] - relevant or not
```

## Prediction Methods

### `predict_proba()` - Get Probabilities

Returns the probability (0-1) that each query-document pair is relevant.

```python
probs = model.predict_proba(
    queries=["query1", "query2", "query3"],
    documents=["doc1", "doc2", "doc3"]
)
# Returns: array([0.923, 0.234, 0.567])
```

**Use when:**
- You need confidence scores
- You want to rank documents by relevance
- You're building a retrieval system

### `predict()` - Get Binary Labels

Returns 1 (relevant) or 0 (not relevant) based on a threshold.

```python
labels = model.predict(
    queries=["query1", "query2"],
    documents=["doc1", "doc2"],
    threshold=0.5  # Default: 0.5
)
# Returns: array([1, 0])
```

**Use when:**
- You need yes/no decisions
- You're classifying pairs

## Input Formats

### Text Input

Same format as training - markrel handles vectorization:

```python
queries = [
    "machine learning tutorial",
    "deep learning guide",
]

documents = [
    "intro to machine learning",
    "neural network basics",
]

probs = model.predict_proba(queries, documents)
```

### Pre-computed Vectors

If the model was trained with `use_text_vectorizer=False`:

```python
import numpy as np

query_vectors = np.random.randn(10, 384)   # (n_samples, embedding_dim)
doc_vectors = np.random.randn(10, 384)

probs = model.predict_proba(query_vectors, doc_vectors)
```

⚠️ **Important**: If model was trained with text, you must pass text. If trained with vectors, pass vectors.

## Common Use Cases

### Document Retrieval

Rank documents by relevance to a query:

```python
query = "machine learning tutorial"
candidates = [
    "neural networks explained",
    "python for beginners",
    "deep learning architectures",
    "cooking recipes",
]

# Score all candidates
probs = model.predict_proba(
    queries=[query] * len(candidates),
    documents=candidates
)

# Rank by relevance
ranked = sorted(
    zip(candidates, probs),
    key=lambda x: x[1],
    reverse=True
)

for doc, prob in ranked:
    print(f"{prob:.3f}: {doc}")

# Output:
# 0.923: deep learning architectures
# 0.856: neural networks explained
# 0.234: python for beginners
# 0.123: cooking recipes
```

### Batch Processing

Process many pairs efficiently:

```python
# Batch of 1000 pairs
queries = load_queries()      # List of 1000 queries
documents = load_documents()  # List of 1000 documents

probs = model.predict_proba(queries, documents)

# Filter high-confidence relevant pairs
threshold = 0.7
relevant_indices = np.where(probs > threshold)[0]
print(f"Found {len(relevant_indices)} relevant pairs")
```

### Threshold Tuning

Find the best threshold for your use case:

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

# Get validation predictions
val_probs = model.predict_proba(val_queries, val_docs)

# Try different thresholds
thresholds = np.linspace(0.1, 0.9, 50)
best_f1 = 0
best_thresh = 0.5

for thresh in thresholds:
    preds = (val_probs >= thresh).astype(int)
    f1 = f1_score(val_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"Best threshold: {best_thresh:.2f} (F1: {best_f1:.3f})")

# Use optimized threshold for predictions
predictions = model.predict(test_queries, test_docs, threshold=best_thresh)
```

### Confidence-Based Filtering

Only keep high-confidence predictions:

```python
probs = model.predict_proba(queries, documents)

# High confidence: relevant
high_conf_rel = probs > 0.9

# High confidence: not relevant
high_conf_nrel = probs < 0.1

# Uncertain: needs review
uncertain = (probs >= 0.1) & (probs <= 0.9)

print(f"High confidence relevant: {np.sum(high_conf_rel)}")
print(f"High confidence not relevant: {np.sum(high_conf_nrel)}")
print(f"Needs human review: {np.sum(uncertain)}")
```

## Debugging Predictions

### Inspect Similarity Scores

See what similarities the model is computing:

```python
similarities = model.predict_similarities(
    queries=["machine learning"],
    documents=["neural networks"]
)

print(similarities)
# {
#     'cosine': array([0.756]),
#     'jaccard': array([0.333]),
# }
```

### Compare Single vs Multiple Metrics

```python
# Single metric model
model_cosine = MarkovRelevanceModel(metrics=["cosine"])
model_cosine.fit(train_q, train_d, train_labels)

# Multi-metric model
model_multi = MarkovRelevanceModel(
    metrics=["cosine", "jaccard", "overlap"]
)
model_multi.fit(train_q, train_d, train_labels)

# Compare predictions
q = "ml tutorial"
d = "neural nets"

prob_cosine = model_cosine.predict_proba([q], [d])[0]
prob_multi = model_multi.predict_proba([q], [d])[0]

print(f"Cosine only: {prob_cosine:.3f}")
print(f"Multi-metric: {prob_multi:.3f}")
```

### Analyze Why a Prediction Was Made

```python
def explain_prediction(model, query, document):
    """Explain why a prediction was made."""
    
    # Get similarities
    sims = model.predict_similarities([query], [document])
    
    # Get learned probabilities
    print(f"Query: '{query}'")
    print(f"Document: '{document}'")
    print("\nSimilarity scores:")
    for metric, sim in sims.items():
        prob = model.chains_[metric].p_relevant(sim[0])
        print(f"  {metric}: sim={sim[0]:.3f} → P(relevant)={prob:.3f}")
    
    # Final combined probability
    final_prob = model.predict_proba([query], [document])[0]
    print(f"\nCombined probability: {final_prob:.3f}")
    print(f"Prediction: {'RELEVANT' if final_prob > 0.5 else 'NOT RELEVANT'}")

# Usage
explain_prediction(model, "machine learning", "neural networks")
```

## Performance Optimization

### Batch Size

Process in batches for large datasets:

```python
def predict_in_batches(model, queries, documents, batch_size=100):
    """Predict in batches to manage memory."""
    all_probs = []
    
    for i in range(0, len(queries), batch_size):
        batch_q = queries[i:i+batch_size]
        batch_d = documents[i:i+batch_size]
        probs = model.predict_proba(batch_q, batch_d)
        all_probs.extend(probs)
    
    return np.array(all_probs)

# Process 10,000 pairs
probs = predict_in_batches(model, queries, documents, batch_size=100)
```

### Pre-computed Embeddings

For repeated inference with same documents:

```python
from sentence_transformers import SentenceTransformer

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Pre-compute document embeddings
doc_texts = load_documents()
doc_embeddings = embedder.encode(doc_texts)

# For each query
for query_text in user_queries:
    query_embedding = embedder.encode([query_text])
    
    # Compute similarities manually
    similarities = cosine_similarity(
        query_embedding,
        doc_embeddings
    )
    
    # Use markrel's learned probabilities
    # (requires model trained on embeddings)
```

## Integration Examples

### Flask API

```python
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model once at startup
with open("markrel_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    query = data["query"]
    document = data["document"]
    
    prob = model.predict_proba([query], [document])[0]
    
    return jsonify({
        "relevance_probability": float(prob),
        "is_relevant": bool(prob > 0.5)
    })

if __name__ == "__main__":
    app.run()
```

### CLI Tool

```python
#!/usr/bin/env python3
"""CLI for markrel predictions."""

import argparse
import pickle

def main():
    parser = argparse.ArgumentParser(description="Predict document relevance")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument("--doc", required=True, help="Document text")
    parser.add_argument("--threshold", type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Load model
    with open(args.model, "rb") as f:
        model = pickle.load(f)
    
    # Predict
    prob = model.predict_proba([args.query], [args.doc])[0]
    label = "RELEVANT" if prob > args.threshold else "NOT RELEVANT"
    
    print(f"Query: {args.query}")
    print(f"Document: {args.doc}")
    print(f"Probability: {prob:.3f}")
    print(f"Prediction: {label}")

if __name__ == "__main__":
    main()
```

Usage:
```bash
python predict_cli.py \
    --model markrel_model.pkl \
    --query "machine learning" \
    --doc "neural networks explained"
```

## Common Patterns

### Re-ranking Search Results

```python
def rerank_results(query, search_results):
    """Re-rank search results using markrel."""
    
    # Extract document texts
    doc_texts = [r["text"] for r in search_results]
    
    # Score with markrel
    probs = model.predict_proba(
        queries=[query] * len(doc_texts),
        documents=doc_texts
    )
    
    # Add scores and re-sort
    for result, prob in zip(search_results, probs):
        result["markrel_score"] = prob
    
    return sorted(search_results, key=lambda x: x["markrel_score"], reverse=True)
```

### Evaluation Pipeline

```python
from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_model(model, test_queries, test_docs, test_labels):
    """Evaluate model performance."""
    
    # Get predictions
    probs = model.predict_proba(test_queries, test_docs)
    preds = model.predict(test_queries, test_docs)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, preds)
    auc = roc_auc_score(test_labels, probs)
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"AUC-ROC: {auc:.3f}")
    
    return {
        "accuracy": accuracy,
        "auc": auc,
        "probs": probs,
        "preds": preds,
    }
```

## Error Handling

### Model Not Fitted

```python
model = MarkovRelevanceModel()

# This will raise RuntimeError
try:
    model.predict_proba(["q"], ["d"])
except RuntimeError as e:
    print(f"Error: {e}")
    # Load a trained model instead
```

### Dimension Mismatch

```python
# If using pre-computed vectors, ensure dimensions match
try:
    probs = model.predict_proba(queries, documents)
except ValueError as e:
    print(f"Input error: {e}")
    # Check that queries and documents have same length
```

## Tips for Production

1. **Load model once**: Don't load the model on every request
2. **Cache embeddings**: If using text, cache vectorized queries
3. **Set timeouts**: markrel predictions are fast but batch wisely
4. **Log predictions**: Track model performance over time
5. **Monitor drift**: Retrain if distribution changes

## Next Steps

- See [training.md](training.md) for training guide
- Check out `demo.py` for working examples
- Run `pytest tests/` to see more usage patterns

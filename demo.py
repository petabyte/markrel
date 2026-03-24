#!/usr/bin/env python3
"""
Demo script for MarkovRelevanceModel.

This demonstrates training a Markov chain model for document relevance prediction
and using it to score new query-document pairs.
"""

import numpy as np
from markrel import MarkovRelevanceModel


def demo_text_data():
    """Demo with text queries and documents using TF-IDF vectorization."""
    print("=" * 60)
    print("Demo 1: Text-based Query-Document Relevance Prediction")
    print("=" * 60)
    
    # Training data: query-document pairs with relevance labels
    train_queries = [
        "machine learning tutorial",
        "deep learning guide",
        "neural networks basics",
        "python programming",
        "data science course",
        "cooking recipes",
        "baking tutorial",
        "italian cuisine",
        "machine learning research",
        "artificial intelligence",
    ]
    
    train_documents = [
        "introduction to machine learning algorithms",
        "deep neural networks explained",
        "basics of neural networks",
        "python for beginners",
        "data science fundamentals",
        "best pasta recipes",
        "how to bake sourdough",
        "authentic italian cooking",
        "latest ML research papers",
        "AI and neural networks",
    ]
    
    # Labels: 1 = relevant, 0 = not relevant
    train_labels = [1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
    
    # Create and train model
    print(f"\nTraining with {len(train_queries)} query-document pairs...")
    model = MarkovRelevanceModel(
        metrics=["cosine", "jaccard"],
        n_bins=5,
        combine_rule="bayesian",
    )
    model.fit(train_queries, train_documents, train_labels)
    
    print("Model trained successfully!")
    
    # Test on new query-document pairs
    test_queries = [
        "machine learning",
        "neural networks",
        "cooking pasta",
        "deep learning",
        "baking bread",
    ]
    
    test_documents = [
        "supervised learning algorithms",
        "convolutional neural networks",
        "italian recipes for dinner",
        "neural network architectures",
        "sourdough bread making",
    ]
    
    print("\n" + "-" * 60)
    print("Predicting relevance for test pairs:")
    print("-" * 60)
    
    for q, d in zip(test_queries, test_documents):
        prob = model.predict_proba([q], [d])[0]
        label = "RELEVANT" if prob > 0.5 else "NOT RELEVANT"
        print(f"\nQuery: '{q}'")
        print(f"Document: '{d}'")
        print(f"Relevance Probability: {prob:.3f} ({label})")
    
    # Show learned probabilities
    print("\n" + "-" * 60)
    print("Learned Markov Chain (Cosine Similarity):")
    print("-" * 60)
    cosine_probs = model.get_metric_probabilities("cosine")
    summary = model.summary()
    for state_info in summary["chains"]["cosine"]["states"]:
        bin_range = state_info["range"]
        p_rel = state_info["p_relevant"]
        n = state_info["n"]
        print(f"  Bin [{bin_range[0]:.3f}, {bin_range[1]:.3f}): "
              f"P(relevant)={p_rel:.3f}, n={n}")


def demo_vector_data():
    """Demo with pre-computed vector representations."""
    print("\n" + "=" * 60)
    print("Demo 2: Vector-based Relevance Prediction")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate synthetic vector data
    dim = 50
    n_samples = 100
    
    # Create relevant pairs: similar vectors
    queries_rel = np.random.randn(n_samples // 2, dim)
    docs_rel = queries_rel + np.random.randn(n_samples // 2, dim) * 0.3
    labels_rel = np.ones(n_samples // 2)
    
    # Create non-relevant pairs: dissimilar vectors
    queries_nrel = np.random.randn(n_samples // 2, dim)
    docs_nrel = np.random.randn(n_samples // 2, dim)  # Independent
    labels_nrel = np.zeros(n_samples // 2)
    
    queries = np.vstack([queries_rel, queries_nrel])
    docs = np.vstack([docs_rel, docs_nrel])
    labels = np.concatenate([labels_rel, labels_nrel])
    
    print(f"\nTraining with {n_samples} vector pairs (dim={dim})...")
    
    model = MarkovRelevanceModel(
        metrics=["cosine", "euclidean"],
        n_bins=10,
        bin_strategy="quantile",
        use_text_vectorizer=False,  # Pre-computed vectors
    )
    model.fit(queries, docs, labels)
    
    print("Model trained!")
    
    # Test on new data
    test_queries = np.random.randn(20, dim)
    test_docs_similar = test_queries + np.random.randn(20, dim) * 0.3
    test_docs_different = np.random.randn(20, dim)
    
    probs_similar = model.predict_proba(test_queries, test_docs_similar)
    probs_different = model.predict_proba(test_queries, test_docs_different)
    
    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)
    print(f"Mean relevance probability for similar documents: "
          f"{np.mean(probs_similar):.3f}")
    print(f"Mean relevance probability for different documents: "
          f"{np.mean(probs_different):.3f}")
    print(f"\nThe model correctly assigns higher relevance to similar documents!")


def demo_multiple_metrics():
    """Demo showing how multiple metrics are combined."""
    print("\n" + "=" * 60)
    print("Demo 3: Multiple Metric Combination")
    print("=" * 60)
    
    queries = [
        "deep learning neural networks",
        "machine learning algorithms",
        "cooking recipes",
        "python programming",
        "baking cakes",
        "artificial intelligence",
    ]
    
    documents = [
        "neural networks and deep learning",
        "ML algorithms and models",
        "best cooking recipes",
        "python coding tutorial",
        "cake baking guide",
        "AI and machine learning",
    ]
    
    labels = [1, 1, 0, 1, 0, 1]
    
    print("\nComparing single vs multiple metrics:")
    print("-" * 60)
    
    # Single metric
    model_cosine = MarkovRelevanceModel(metrics=["cosine"], n_bins=4)
    model_cosine.fit(queries, documents, labels)
    
    # Multiple metrics
    model_multi = MarkovRelevanceModel(
        metrics=["cosine", "jaccard", "overlap"],
        n_bins=4,
        combine_rule="bayesian",
    )
    model_multi.fit(queries, documents, labels)
    
    # Test pair
    test_q = "neural networks"
    test_d = "deep learning models"
    
    prob_single = model_cosine.predict_proba([test_q], [test_d])[0]
    prob_multi = model_multi.predict_proba([test_q], [test_d])[0]
    
    print(f"\nTest Pair:")
    print(f"  Query: '{test_q}'")
    print(f"  Document: '{test_d}'")
    print(f"\nRelevance Probability:")
    print(f"  Cosine only: {prob_single:.3f}")
    print(f"  Cosine + Jaccard + Overlap: {prob_multi:.3f}")
    print(f"\nMultiple metrics can provide more robust predictions!")


def demo_inspection():
    """Demo model inspection and state analysis."""
    print("\n" + "=" * 60)
    print("Demo 4: Model Inspection")
    print("=" * 60)
    
    queries = ["q"] * 20
    documents = [f"doc with words {i}" for i in range(20)]
    labels = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    model = MarkovRelevanceModel(
        metrics=["cosine", "euclidean"],
        n_bins=5,
    )
    model.fit(queries, documents, labels)
    
    print("\nModel Summary:")
    print("-" * 60)
    summary = model.summary()
    print(f"Metrics: {summary['metrics']}")
    print(f"Combination Rule: {summary['combine_rule']}")
    print(f"\nMarkov Chains:")
    
    for metric_name, chain_summary in summary["chains"].items():
        print(f"\n  {metric_name}:")
        print(f"    Total bins: {chain_summary['n_bins']}")
        print(f"    Relevant samples: {chain_summary['total_relevant']}")
        print(f"    Non-relevant samples: {chain_summary['total_not_relevant']}")
        
        print(f"    State Probabilities:")
        for state in chain_summary['states']:
            print(f"      Bin {state['bin']}: [{state['range'][0]:.3f}, "
                  f"{state['range'][1]:.3f}) -> P(rel)={state['p_relevant']:.3f}")


if __name__ == "__main__":
    demo_text_data()
    demo_vector_data()
    demo_multiple_metrics()
    demo_inspection()
    
    print("\n" + "=" * 60)
    print("Demo complete! Run 'pytest tests/' to run unit tests.")
    print("=" * 60)

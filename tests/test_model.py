"""Unit tests for MarkovRelevanceModel."""

import numpy as np
import pytest

from markrel import MarkovRelevanceModel
from markrel.model import MarkovRelevanceModel


class TestMarkovRelevanceModel:
    """Test suite for the main MarkovRelevanceModel class."""

    def test_basic_fit_predict_with_text(self):
        """Test basic fitting and prediction with text inputs."""
        queries = [
            "machine learning",
            "deep learning",
            "cooking recipes",
            "neural networks",
        ]
        documents = [
            "neural networks and machine learning",
            "deep neural networks",
            "best pasta recipes",
            "artificial intelligence",
        ]
        labels = [1, 1, 0, 1]  # 1=relevant, 0=not relevant

        model = MarkovRelevanceModel(metrics=["cosine"], n_bins=5)
        model.fit(queries, documents, labels)

        # Predict on new data
        test_queries = ["machine learning", "cooking"]
        test_docs = ["artificial intelligence", "italian recipes"]
        probs = model.predict_proba(test_queries, test_docs)

        assert len(probs) == 2
        assert all(0 <= p <= 1 for p in probs)
        # ML-related query should have higher relevance
        assert probs[0] > probs[1]

    def test_fit_predict_with_vectors(self):
        """Test fitting and prediction with pre-computed vectors."""
        np.random.seed(42)
        
        # Create synthetic vector data
        n_samples = 50
        dim = 20
        
        # Relevant pairs: similar vectors
        queries_rel = np.random.randn(n_samples // 2, dim)
        docs_rel = queries_rel + np.random.randn(n_samples // 2, dim) * 0.1
        labels_rel = np.ones(n_samples // 2)
        
        # Non-relevant pairs: dissimilar vectors
        queries_nrel = np.random.randn(n_samples // 2, dim)
        docs_nrel = np.random.randn(n_samples // 2, dim)
        labels_nrel = np.zeros(n_samples // 2)
        
        queries = np.vstack([queries_rel, queries_nrel])
        docs = np.vstack([docs_rel, docs_nrel])
        labels = np.concatenate([labels_rel, labels_nrel])
        
        model = MarkovRelevanceModel(
            metrics=["cosine", "euclidean"],
            n_bins=8,
            use_text_vectorizer=False,
        )
        model.fit(queries, docs, labels)
        
        # Test predictions
        test_queries = np.random.randn(10, dim)
        test_docs_similar = test_queries + np.random.randn(10, dim) * 0.1
        test_docs_different = np.random.randn(10, dim)
        
        probs_similar = model.predict_proba(test_queries, test_docs_similar)
        probs_different = model.predict_proba(test_queries, test_docs_different)
        
        # Similar documents should have higher relevance
        assert np.mean(probs_similar) > np.mean(probs_different)

    def test_binary_prediction(self):
        """Test binary prediction with threshold."""
        queries = ["test query"] * 10
        documents = [f"doc {i}" for i in range(10)]
        labels = [1] * 5 + [0] * 5
        
        model = MarkovRelevanceModel(metrics=["jaccard"], n_bins=4)
        model.fit(queries, documents, labels)
        
        preds = model.predict(queries[:3], documents[:3], threshold=0.5)
        assert len(preds) == 3
        assert all(isinstance(p, (int, np.integer)) for p in preds)

    def test_multiple_metrics(self):
        """Test model with multiple similarity metrics."""
        queries = ["machine learning tutorial"] * 4
        documents = [
            "deep learning guide",
            "neural network basics", 
            "baking tutorial",
            "data science",
        ]
        labels = [1, 1, 0, 1]
        
        model = MarkovRelevanceModel(
            metrics=["cosine", "jaccard", "overlap"],
            n_bins=6,
        )
        model.fit(queries, documents, labels)
        
        probs = model.predict_proba(["ml guide"], ["learning tutorial"])
        assert len(probs) == 1
        assert 0 <= probs[0] <= 1

    def test_quantile_binning(self):
        """Test quantile-based binning strategy."""
        queries = ["query"] * 10
        documents = [f"doc {i}" for i in range(10)]
        labels = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
        
        model = MarkovRelevanceModel(
            metrics=["cosine"],
            n_bins=5,
            bin_strategy="quantile",
        )
        model.fit(queries, documents, labels)
        
        probs = model.predict_proba(["query"], ["doc 0"])
        assert len(probs) == 1

    def test_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="Unknown similarity metric"):
            MarkovRelevanceModel(metrics=["invalid_metric"])

    def test_unfitted_predict(self):
        """Test that predicting before fitting raises RuntimeError."""
        model = MarkovRelevanceModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(["query"], ["doc"])

    def test_mismatched_samples(self):
        """Test that mismatched query/doc counts raise error."""
        model = MarkovRelevanceModel(use_text_vectorizer=False)
        queries = np.random.randn(5, 10)
        docs = np.random.randn(3, 10)
        labels = [1] * 5
        
        with pytest.raises(ValueError, match="same number of samples"):
            model.fit(queries, docs, labels)

    def test_summary(self):
        """Test model summary output."""
        queries = ["q1", "q2", "q3"]
        documents = ["d1", "d2", "d3"]
        labels = [1, 0, 1]
        
        model = MarkovRelevanceModel(metrics=["cosine"], n_bins=3)
        model.fit(queries, documents, labels)
        
        summary = model.summary()
        assert "metrics" in summary
        assert "cosine" in summary["chains"]
        assert "combine_rule" in summary

    def test_get_metric_probabilities(self):
        """Test retrieving learned probabilities for a metric."""
        queries = ["q"] * 10
        documents = [f"d{i}" for i in range(10)]
        labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
        
        model = MarkovRelevanceModel(metrics=["cosine"], n_bins=5)
        model.fit(queries, documents, labels)
        
        probs = model.get_metric_probabilities("cosine")
        assert len(probs) == 5
        assert all(0 <= p <= 1 for p in probs)

    def test_mean_combination_rule(self):
        """Test the mean combination rule."""
        queries = ["q"] * 8
        documents = [f"d{i}" for i in range(8)]
        labels = [1, 1, 1, 0, 0, 0, 1, 0]
        
        model = MarkovRelevanceModel(
            metrics=["cosine", "euclidean"],
            combine_rule="mean",
        )
        model.fit(queries, documents, labels)
        
        probs = model.predict_proba(["q"], ["d0"])
        assert len(probs) == 1


class TestSimilarityMetrics:
    """Test similarity metric functions."""

    def test_cosine_same_vectors(self):
        """Cosine similarity of identical vectors should be 1."""
        from markrel.similarity import cosine_similarity
        
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        """Cosine similarity of orthogonal vectors should be 0."""
        from markrel.similarity import cosine_similarity
        
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_euclidean_similarity_range(self):
        """Euclidean-based similarity should be in (0, 1]."""
        from markrel.similarity import euclidean_similarity
        
        v1 = np.array([1.0, 2.0])
        v2 = np.array([3.0, 4.0])
        sim = euclidean_similarity(v1, v2)
        assert 0 < sim <= 1.0

    def test_jaccard_similarity(self):
        """Jaccard similarity on binary vectors."""
        from markrel.similarity import jaccard_similarity
        
        v1 = np.array([1, 1, 0, 0])
        v2 = np.array([1, 0, 1, 0])
        # Intersection = [1], Union = [1, 1, 1, 0, 0, 1, 0] counting non-zero
        # Actually: v1>0 = [1,1,0,0], v2>0 = [1,0,1,0]
        # Intersection indices where both >0: index 0
        # Union indices where either >0: indices 0, 1, 2
        # Jaccard = 1/3
        sim = jaccard_similarity(v1, v2)
        assert sim == pytest.approx(1.0 / 3.0)

    def test_compute_all(self):
        """Test computing all metrics."""
        from markrel.similarity import compute_all
        
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.0, 2.0, 3.0])
        
        results = compute_all(v1, v2)
        assert "cosine" in results
        assert "euclidean" in results
        assert all(0 <= v <= 1 or -1 <= v <= 1 for v in results.values())


class TestStateDiscretizer:
    """Test StateDiscretizer functionality."""

    def test_uniform_binning(self):
        """Test uniform binning strategy."""
        from markrel.states import StateDiscretizer
        
        values = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        discretizer = StateDiscretizer(n_bins=5, strategy="uniform")
        discretizer.fit(values)
        
        indices = discretizer.transform(np.array([0.1, 0.5, 0.9]))
        assert len(indices) == 3
        assert all(0 <= i < 5 for i in indices)

    def test_quantile_binning(self):
        """Test quantile binning strategy."""
        from markrel.states import StateDiscretizer
        
        values = np.random.randn(100)
        discretizer = StateDiscretizer(n_bins=10, strategy="quantile")
        discretizer.fit(values)
        
        # Each bin should have roughly equal number of samples
        indices = discretizer.transform(values)
        unique, counts = np.unique(indices, return_counts=True)
        # With 100 samples and 10 bins, each should have ~10
        assert all(5 <= c <= 20 for c in counts)

    def test_unfitted_transform(self):
        """Test that transform before fit raises error."""
        from markrel.states import StateDiscretizer
        
        discretizer = StateDiscretizer(n_bins=5)
        with pytest.raises(RuntimeError, match="Call fit"):
            discretizer.transform(np.array([0.5]))


class TestMetricChain:
    """Test MetricChain functionality."""

    def test_chain_fit(self):
        """Test fitting a MetricChain."""
        from markrel.transitions import MetricChain
        
        similarities = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        chain = MetricChain(metric_name="cosine", n_bins=3)
        chain.fit(similarities, labels)
        
        assert chain._fitted
        assert chain.total_relevant_ == 3
        assert chain.total_not_relevant_ == 3

    def test_chain_probability(self):
        """Test relevance probability computation."""
        from markrel.transitions import MetricChain
        
        # Perfect separation: low similarity = not relevant, high = relevant
        similarities = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        chain = MetricChain(metric_name="cosine", n_bins=3)
        chain.fit(similarities, labels)
        
        # Low similarity should have low probability
        low_prob = chain.p_relevant(0.15)
        # High similarity should have high probability  
        high_prob = chain.p_relevant(0.85)
        
        assert low_prob < high_prob

    def test_chain_summary(self):
        """Test chain summary output."""
        from markrel.transitions import MetricChain
        
        similarities = np.array([0.1, 0.5, 0.9])
        labels = np.array([0, 1, 1])
        
        chain = MetricChain(metric_name="cosine", n_bins=2)
        chain.fit(similarities, labels)
        
        summary = chain.summary()
        assert summary["metric"] == "cosine"
        assert "states" in summary
        assert len(summary["states"]) > 0

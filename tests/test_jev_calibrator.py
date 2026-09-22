"""Unit tests for JevCalibrator."""

import numpy as np
import pytest

from markrel import MarkovRelevanceModel
from markrel.integrations import JevCalibrator


class TestJevCalibrator:
    """Test suite for the JevCalibrator recalibration wrapper."""

    def _miscalibrated_data(self, n=2000, seed=0):
        """Simulate an external scorer (e.g. Jev) that is systematically
        overconfident relative to the true relevance rate."""
        rng = np.random.default_rng(seed)
        true_p = rng.uniform(0, 1, n)
        jev_probs = np.clip(true_p * 1.4 + rng.normal(0, 0.05, n), 0, 1)
        labels = rng.binomial(1, true_p)
        return jev_probs, labels

    def test_fit_predict_basic(self):
        jev_probs, labels = self._miscalibrated_data()

        cal = JevCalibrator(n_bins=15, bin_strategy="quantile")
        cal.fit(jev_probs, labels)

        probs = cal.predict_proba([0.2, 0.5, 0.8, 0.95])
        assert len(probs) == 4
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_recalibration_corrects_overconfidence(self):
        """A raw score of 0.95 should map to a lower recalibrated
        probability when the input scorer is overconfident."""
        jev_probs, labels = self._miscalibrated_data()

        cal = JevCalibrator(n_bins=15, bin_strategy="quantile")
        cal.fit(jev_probs, labels)

        recalibrated = cal.predict_proba([0.95])[0]
        assert recalibrated < 0.95

    def test_predict_binary(self):
        jev_probs, labels = self._miscalibrated_data()

        cal = JevCalibrator(n_bins=10)
        cal.fit(jev_probs, labels)

        preds = cal.predict([0.05, 0.99], threshold=0.5)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_before_fit_raises(self):
        cal = JevCalibrator()
        with pytest.raises(RuntimeError):
            cal.predict_proba([0.5])

    def test_mismatched_lengths_raise(self):
        cal = JevCalibrator()
        with pytest.raises(ValueError):
            cal.fit([0.1, 0.2, 0.3], [1, 0])

    def test_clip_input_true_by_default(self):
        """Minor float drift outside [0, 1] should be clipped, not raise."""
        jev_probs, labels = self._miscalibrated_data(n=200)
        cal = JevCalibrator(n_bins=5)
        cal.fit(jev_probs, labels)

        # Should not raise even with slightly out-of-range values.
        probs = cal.predict_proba([1.0001, -0.0001])
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_clip_input_false_raises_out_of_range(self):
        cal = JevCalibrator(n_bins=5, clip_input=False)
        cal.fit([0.1, 0.5, 0.9], [0, 1, 1])
        with pytest.raises(ValueError):
            cal.predict_proba([1.5])

    def test_summary_and_bin_edges(self):
        jev_probs, labels = self._miscalibrated_data(n=500)
        cal = JevCalibrator(n_bins=8, bin_strategy="quantile")
        cal.fit(jev_probs, labels)

        summary = cal.summary()
        assert "states" in summary
        assert len(cal.bin_edges()) == len(summary["states"])

    def test_combine_with_embedding_model(self):
        """JevCalibrator's recalibrated scores should combine cleanly
        with a markrel embedding-based MarkovRelevanceModel."""
        rng = np.random.default_rng(1)
        n = 200
        queries = [f"q{i} machine learning" for i in range(n)]
        documents = [
            f"d{i} deep learning content" if rng.random() < 0.5 else f"d{i} cooking recipe"
            for i in range(n)
        ]
        labels = rng.integers(0, 2, n)
        jev_probs = rng.uniform(0, 1, n)

        embed_model = MarkovRelevanceModel(metrics=["cosine"], n_bins=8)
        embed_model.fit(queries, documents, labels)

        cal = JevCalibrator(n_bins=8, bin_strategy="quantile")
        cal.fit(jev_probs, labels)

        combined = cal.combine_with(jev_probs, embed_model, queries, documents, rule="bayesian")
        assert combined.shape == (n,)
        assert np.all((combined >= 0.0) & (combined <= 1.0))

        combined_mean = cal.combine_with(jev_probs, embed_model, queries, documents, rule="mean")
        assert combined_mean.shape == (n,)

    def test_combine_with_invalid_rule_raises(self):
        rng = np.random.default_rng(2)
        n = 50
        queries = [f"q{i}" for i in range(n)]
        documents = [f"d{i}" for i in range(n)]
        labels = rng.integers(0, 2, n)
        jev_probs = rng.uniform(0, 1, n)

        embed_model = MarkovRelevanceModel(metrics=["cosine"], n_bins=5)
        embed_model.fit(queries, documents, labels)

        cal = JevCalibrator(n_bins=5)
        cal.fit(jev_probs, labels)

        with pytest.raises(ValueError):
            cal.combine_with(jev_probs, embed_model, queries, documents, rule="bogus")

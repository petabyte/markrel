"""
Main MarkovRelevanceModel class.

This is the high-level API that orchestrates:
1. Text vectorization (optional - can also accept pre-computed vectors)
2. Similarity computation across multiple metrics
3. Per-metric Markov chain fitting
4. Relevance prediction via probability combination
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Union

import numpy as np

from .similarity import SIMILARITY_METRICS, get_metric
from .transitions import MetricChain
from .vectorizer import TextVectorizer, coerce_to_matrix, InputType


@dataclass
class MarkovRelevanceModel:
    """Markov chain model for document relevance prediction.

    The model treats each similarity metric as a Markov chain where:
    - States (nodes) are discretized similarity value ranges (bins)
    - Transition probabilities encode P(relevant | similarity_bin)

    Multiple metrics can be combined using Bayesian or log-likelihood
    combination rules.

    Parameters
    ----------
    metrics :
        List of similarity metric names to use. Default: ["cosine"].
    n_bins :
        Number of bins (Markov states) per metric.
    bin_strategy :
        "uniform" (equal-width) or "quantile" (equal-frequency) binning.
    smoothing :
        Laplace smoothing parameter for probability estimates.
    combine_rule :
        "bayesian" (product of odds) or "mean" (average probabilities).
    use_text_vectorizer :
        If True, fit a TF-IDF vectorizer on text inputs. If False,
        assume inputs are already numeric vectors.

    Attributes
    ----------
    chains_ :
        Dictionary mapping metric name -> fitted MetricChain.
    vectorizer_ :
        Fitted TextVectorizer (if use_text_vectorizer=True).
    _fitted :
        Whether the model has been fitted.

    Examples
    --------
    >>> model = MarkovRelevanceModel(metrics=["cosine", "euclidean"])
    >>> model.fit(train_queries, train_docs, train_labels)
    >>> probs = model.predict(test_queries, test_docs)
    """

    metrics: list[str] = field(default_factory=lambda: ["cosine"])
    n_bins: int = 10
    bin_strategy: str = "uniform"
    smoothing: float = 1.0
    combine_rule: str = "bayesian"
    use_text_vectorizer: bool = True

    chains_: dict[str, MetricChain] = field(init=False, repr=False)
    vectorizer_: TextVectorizer | None = field(init=False, repr=False)
    _fitted: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        # Validate metrics
        for m in self.metrics:
            get_metric(m)  # raises ValueError if unknown
        
        if self.combine_rule not in {"bayesian", "mean"}:
            raise ValueError(f"combine_rule must be 'bayesian' or 'mean', got {self.combine_rule!r}")
        
        self.chains_ = {}
        self.vectorizer_ = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        queries: InputType,
        documents: InputType,
        labels: Sequence[bool | int],
    ) -> "MarkovRelevanceModel":
        """Fit the Markov chains on training data.

        Parameters
        ----------
        queries :
            Training queries. Either a list of strings or a 2-D numpy array
            of pre-computed feature vectors.
        documents :
            Training documents. Same format as queries.
        labels :
            Binary relevance labels (1=relevant, 0=not relevant).

        Returns
        -------
        self : MarkovRelevanceModel
            The fitted model.
        """
        labels_arr = np.asarray(labels, dtype=bool)
        
        # Handle text inputs
        if self.use_text_vectorizer:
            self.vectorizer_ = TextVectorizer()
            # Combine queries and documents to fit shared vocabulary
            all_texts = []
            if isinstance(queries, (list, tuple)) and len(queries) > 0 and isinstance(queries[0], str):
                all_texts.extend(queries)
            if isinstance(documents, (list, tuple)) and len(documents) > 0 and isinstance(documents[0], str):
                all_texts.extend(documents)
            
            if all_texts:
                self.vectorizer_.fit(all_texts)
        
        # Convert to matrices
        query_vecs = coerce_to_matrix(queries, self.vectorizer_)
        doc_vecs = coerce_to_matrix(documents, self.vectorizer_)
        
        if query_vecs.shape[0] != doc_vecs.shape[0]:
            raise ValueError(
                f"queries and documents must have same number of samples, "
                f"got {query_vecs.shape[0]} and {doc_vecs.shape[0]}"
            )
        
        # Fit a Markov chain for each metric
        self.chains_ = {}
        for metric_name in self.metrics:
            # Compute similarities for this metric
            metric_fn = get_metric(metric_name)
            similarities = np.array([
                metric_fn(query_vecs[i], doc_vecs[i])
                for i in range(query_vecs.shape[0])
            ])
            
            # Create and fit the chain
            chain = MetricChain(
                metric_name=metric_name,
                n_bins=self.n_bins,
                bin_strategy=self.bin_strategy,
                smoothing=self.smoothing,
            )
            chain.fit(similarities, labels_arr)
            self.chains_[metric_name] = chain
        
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        queries: InputType,
        documents: InputType,
    ) -> np.ndarray:
        """Predict relevance probabilities for query-document pairs.

        Parameters
        ----------
        queries :
            Query texts or pre-computed vectors.
        documents :
            Document texts or pre-computed vectors.

        Returns
        -------
        probs : np.ndarray
            1-D array of relevance probabilities (values in [0, 1]).
        """
        self._check_fitted()
        
        query_vecs = coerce_to_matrix(queries, self.vectorizer_)
        doc_vecs = coerce_to_matrix(documents, self.vectorizer_)
        
        if query_vecs.shape[0] != doc_vecs.shape[0]:
            raise ValueError(
                f"queries and documents must have same number of samples"
            )
        
        n_samples = query_vecs.shape[0]
        
        # Collect probabilities from each metric
        all_probs = np.zeros((n_samples, len(self.metrics)))
        
        for j, metric_name in enumerate(self.metrics):
            metric_fn = get_metric(metric_name)
            similarities = np.array([
                metric_fn(query_vecs[i], doc_vecs[i])
                for i in range(n_samples)
            ])
            all_probs[:, j] = self.chains_[metric_name].p_relevant_batch(similarities)
        
        # Combine probabilities
        if self.combine_rule == "bayesian":
            # Product of odds, then convert back to probability
            # odds = p / (1-p), combined_odds = product(odds), p = odds / (1 + odds)
            # Avoid division by zero by clipping
            all_probs = np.clip(all_probs, 1e-10, 1 - 1e-10)
            odds = all_probs / (1 - all_probs)
            combined_odds = np.prod(odds, axis=1)
            combined_probs = combined_odds / (1 + combined_odds)
        else:  # mean
            combined_probs = np.mean(all_probs, axis=1)
        
        return combined_probs

    def predict(
        self,
        queries: InputType,
        documents: InputType,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict binary relevance labels.

        Parameters
        ----------
        queries :
            Query texts or pre-computed vectors.
        documents :
            Document texts or pre-computed vectors.
        threshold :
            Probability threshold for positive prediction (default 0.5).

        Returns
        -------
        labels : np.ndarray
            Binary predictions (0 or 1).
        """
        probs = self.predict_proba(queries, documents)
        return (probs >= threshold).astype(int)

    def predict_similarities(
        self,
        queries: InputType,
        documents: InputType,
    ) -> dict[str, np.ndarray]:
        """Return raw similarity values for each metric.

        Useful for debugging and inspection.

        Parameters
        ----------
        queries :
            Query texts or pre-computed vectors.
        documents :
            Document texts or pre-computed vectors.

        Returns
        -------
        similarities : dict
            Dictionary mapping metric name to array of similarity values.
        """
        self._check_fitted()
        
        query_vecs = coerce_to_matrix(queries, self.vectorizer_)
        doc_vecs = coerce_to_matrix(documents, self.vectorizer_)
        
        n_samples = query_vecs.shape[0]
        result = {}
        
        for metric_name in self.metrics:
            metric_fn = get_metric(metric_name)
            similarities = np.array([
                metric_fn(query_vecs[i], doc_vecs[i])
                for i in range(n_samples)
            ])
            result[metric_name] = similarities
        
        return result

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

    def summary(self) -> dict:
        """Return a summary of fitted model statistics."""
        self._check_fitted()
        return {
            "metrics": self.metrics,
            "combine_rule": self.combine_rule,
            "chains": {name: chain.summary() for name, chain in self.chains_.items()},
        }

    def get_metric_probabilities(self, metric_name: str) -> np.ndarray:
        """Get the learned P(relevant | bin) for a specific metric.

        Parameters
        ----------
        metric_name :
            Name of the metric.

        Returns
        -------
        probs : np.ndarray
            1-D array of probabilities for each bin.
        """
        self._check_fitted()
        if metric_name not in self.chains_:
            raise ValueError(f"Unknown metric: {metric_name}")
        chain = self.chains_[metric_name]
        return np.array([state.relevance_probability(chain.smoothing) 
                         for state in chain.states_])

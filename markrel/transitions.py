"""
Transition probability tables for the relevance Markov chain.

Each ``MetricChain`` owns one ``StateDiscretizer`` and an array of
``MarkovState`` nodes.  After fitting, it answers two questions for any
similarity value:

1. P(relevant | similarity_bin)          — posterior probability
2. log P(bin | relevant) / P(bin | not_relevant) — log-likelihood ratio
   used by the sequential chain combiner in the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .states import MarkovState, StateDiscretizer


# ---------------------------------------------------------------------------
# MetricChain — one Markov chain per similarity metric
# ---------------------------------------------------------------------------

@dataclass
class MetricChain:
    """Complete Markov chain for a single similarity metric.

    Attributes
    ----------
    metric_name:
        Identifier matching a key in ``SIMILARITY_METRICS``.
    n_bins:
        Requested number of bins (actual may differ for quantile strategy).
    bin_strategy:
        ``'uniform'`` or ``'quantile'``.
    smoothing:
        Laplace pseudo-count for probability estimates (default 1.0).
    """

    metric_name: str
    n_bins: int = 10
    bin_strategy: str = "uniform"
    smoothing: float = 1.0

    # Populated by fit()
    discretizer_: StateDiscretizer = field(init=False, repr=False)
    states_: list[MarkovState] = field(init=False, repr=False)
    total_relevant_: int = field(init=False, default=0)
    total_not_relevant_: int = field(init=False, default=0)
    _fitted: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.discretizer_ = StateDiscretizer(self.n_bins, self.bin_strategy)
        self.states_ = []

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        similarities: np.ndarray,
        relevance: np.ndarray,
    ) -> "MetricChain":
        """Fit bin edges and accumulate per-bin relevance counts.

        Parameters
        ----------
        similarities:
            1-D float array of similarity values (one per training pair).
        relevance:
            1-D boolean/int array of relevance labels aligned with
            ``similarities``.
        """
        similarities = np.asarray(similarities, dtype=float)
        relevance = np.asarray(relevance, dtype=bool)

        if similarities.shape != relevance.shape:
            raise ValueError("similarities and relevance must have the same length.")

        # Fit discretizer and map values to bin indices
        bin_indices = self.discretizer_.fit_transform(similarities)
        n_bins = self.discretizer_.actual_n_bins

        # Build MarkovState nodes
        self.states_ = []
        for i in range(n_bins):
            lo, hi = self.discretizer_.bin_edges(i)
            self.states_.append(
                MarkovState(
                    bin_index=i,
                    bin_lower=lo,
                    bin_upper=hi,
                    metric_name=self.metric_name,
                )
            )

        # Accumulate counts
        for idx, label in zip(bin_indices, relevance):
            state = self.states_[int(idx)]
            if label:
                state.relevant_count += 1
                self.total_relevant_ += 1
            else:
                state.not_relevant_count += 1
                self.total_not_relevant_ += 1

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Probability queries
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"MetricChain '{self.metric_name}' has not been fitted.")

    def bin_index(self, similarity: float) -> int:
        """Discretise a single similarity value to its bin index."""
        self._check_fitted()
        return int(self.discretizer_.transform(np.array([similarity]))[0])

    def p_relevant(self, similarity: float) -> float:
        """P(relevant | similarity) — posterior probability with smoothing."""
        self._check_fitted()
        idx = self.bin_index(similarity)
        return self.states_[idx].relevance_probability(self.smoothing)

    def log_likelihood_ratio(self, similarity: float) -> float:
        """log P(sim_bin | relevant) - log P(sim_bin | not_relevant).

        This is the Markov chain *edge weight* used when combining multiple
        metrics sequentially.
        """
        self._check_fitted()
        idx = self.bin_index(similarity)
        return self.states_[idx].log_likelihood_ratio(
            self.total_relevant_,
            self.total_not_relevant_,
            self.smoothing,
        )

    # ------------------------------------------------------------------
    # Vectorised helpers (used by the model for batch prediction)
    # ------------------------------------------------------------------

    def p_relevant_batch(self, similarities: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return P(relevant | sim) for an array of similarity values."""
        self._check_fitted()
        sims = np.asarray(similarities, dtype=float)
        indices = self.discretizer_.transform(sims)
        probs = np.array(
            [self.states_[i].relevance_probability(self.smoothing) for i in indices]
        )
        return probs

    def log_likelihood_ratio_batch(
        self, similarities: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        """Return log-likelihood ratios for an array of similarity values."""
        self._check_fitted()
        sims = np.asarray(similarities, dtype=float)
        indices = self.discretizer_.transform(sims)
        llrs = np.array(
            [
                self.states_[i].log_likelihood_ratio(
                    self.total_relevant_, self.total_not_relevant_, self.smoothing
                )
                for i in indices
            ]
        )
        return llrs

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def transition_table(self) -> np.ndarray:
        """Return (n_bins, 2) array of [P(not_relevant|bin), P(relevant|bin)]."""
        self._check_fitted()
        probs = np.array(
            [[1.0 - s.relevance_probability(self.smoothing),
              s.relevance_probability(self.smoothing)]
             for s in self.states_]
        )
        return probs

    def summary(self) -> dict:
        """Human-readable summary of the chain's learned probabilities."""
        self._check_fitted()
        return {
            "metric": self.metric_name,
            "n_bins": self.discretizer_.actual_n_bins,
            "total_relevant": self.total_relevant_,
            "total_not_relevant": self.total_not_relevant_,
            "states": [
                {
                    "bin": i,
                    "range": [s.bin_lower, s.bin_upper],
                    "n": s.total_count,
                    "p_relevant": round(s.relevance_probability(self.smoothing), 4),
                }
                for i, s in enumerate(self.states_)
            ],
        }

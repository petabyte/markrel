"""
Markov state definitions and similarity-value discretisation.

The relevance-prediction Markov chain maps each observed similarity value to a
discrete *state* (a similarity bin).  A separate chain (set of bins) is
maintained for every metric so that each metric can have its own natural range
and data-driven bin boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# MarkovState
# ---------------------------------------------------------------------------

@dataclass
class MarkovState:
    """A single node in the Markov chain representing one similarity bin.

    Attributes
    ----------
    bin_index:
        Zero-based position of this bin within its metric's chain.
    bin_lower, bin_upper:
        Inclusive lower and exclusive upper edge of the similarity range.
    metric_name:
        Name of the similarity metric this state belongs to.
    relevant_count:
        Number of training examples whose similarity fell in this bin AND were
        labelled relevant.
    not_relevant_count:
        Analogous count for non-relevant training examples.
    """

    bin_index: int
    bin_lower: float
    bin_upper: float
    metric_name: str
    relevant_count: int = field(default=0)
    not_relevant_count: int = field(default=0)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def total_count(self) -> int:
        return self.relevant_count + self.not_relevant_count

    @property
    def center(self) -> float:
        return (self.bin_lower + self.bin_upper) / 2.0

    def relevance_probability(self, smoothing: float = 1.0) -> float:
        """P(relevant | this state) with additive (Laplace) smoothing.

        Parameters
        ----------
        smoothing:
            Pseudo-count added to numerator and denominator (default 1.0).
            Set to 0 for maximum-likelihood estimate.
        """
        return (self.relevant_count + smoothing) / (self.total_count + 2.0 * smoothing)

    def log_likelihood_ratio(
        self,
        total_relevant: int,
        total_not_relevant: int,
        smoothing: float = 1.0,
    ) -> float:
        """log P(state | relevant) - log P(state | not_relevant).

        Used in the sequential (chain) combination of multiple metrics where
        each metric contributes a log-likelihood ratio update to the relevance
        belief.

        Parameters
        ----------
        total_relevant, total_not_relevant:
            Total relevant/non-relevant training examples *across all bins* for
            this metric (needed to compute marginal likelihoods).
        smoothing:
            Laplace pseudo-count per bin.
        """
        n_rel = total_relevant + self.total_count  # avoid div-by-zero when 0
        n_nrel = total_not_relevant + self.total_count

        p_bin_given_rel = (self.relevant_count + smoothing) / (
            total_relevant + smoothing * n_rel
        )
        p_bin_given_nrel = (self.not_relevant_count + smoothing) / (
            total_not_relevant + smoothing * n_nrel
        )
        p_bin_given_rel = max(p_bin_given_rel, 1e-12)
        p_bin_given_nrel = max(p_bin_given_nrel, 1e-12)
        return float(np.log(p_bin_given_rel) - np.log(p_bin_given_nrel))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MarkovState(metric={self.metric_name!r}, "
            f"bin=[{self.bin_lower:.3f}, {self.bin_upper:.3f}), "
            f"P(rel)={self.relevance_probability():.3f}, n={self.total_count})"
        )


# ---------------------------------------------------------------------------
# StateDiscretizer
# ---------------------------------------------------------------------------

class StateDiscretizer:
    """Fits bin edges to a set of similarity values and maps new values to bins.

    Parameters
    ----------
    n_bins:
        Number of bins (Markov states) to create.
    strategy:
        ``'uniform'`` — equal-width bins between observed min and max.
        ``'quantile'`` — equal-frequency bins (data-adaptive).
    """

    def __init__(self, n_bins: int = 10, strategy: str = "uniform") -> None:
        if strategy not in {"uniform", "quantile"}:
            raise ValueError(f"strategy must be 'uniform' or 'quantile', got {strategy!r}")
        self.n_bins = n_bins
        self.strategy = strategy
        self.bin_edges_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, values: np.ndarray) -> "StateDiscretizer":
        """Learn bin edges from observed similarity values.

        Parameters
        ----------
        values:
            1-D array of similarity scores seen during training.
        """
        values = np.asarray(values, dtype=float)
        if values.ndim != 1:
            raise ValueError("values must be a 1-D array.")

        if self.strategy == "uniform":
            lo, hi = float(values.min()), float(values.max())
            # Extend edges slightly so max value falls inside the last bin
            self.bin_edges_ = np.linspace(lo, hi + 1e-9, self.n_bins + 1)
        else:  # quantile
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            edges = np.percentile(values, quantiles)
            edges[-1] += 1e-9  # ensure max value is inside
            # Deduplicate edges that collapse (can happen with many ties)
            unique = np.unique(edges)
            if len(unique) < 2:
                unique = np.array([edges[0], edges[0] + 1e-9])
            self.bin_edges_ = unique

        return self

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Map similarity values to zero-based bin indices.

        Values outside the training range are clipped to the nearest bin.
        """
        if self.bin_edges_ is None:
            raise RuntimeError("Call fit() before transform().")
        values = np.asarray(values, dtype=float)
        # np.digitize returns 1-based indices; subtract 1 for zero-based
        indices = np.digitize(values, self.bin_edges_[1:], right=False)
        return np.clip(indices, 0, self.actual_n_bins - 1)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(values).transform(values)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def actual_n_bins(self) -> int:
        """Number of bins after fitting (may differ from n_bins for quantile)."""
        if self.bin_edges_ is None:
            return self.n_bins
        return len(self.bin_edges_) - 1

    def bin_edges(self, bin_index: int) -> tuple[float, float]:
        """Return (lower, upper) edge of a bin by index."""
        if self.bin_edges_ is None:
            raise RuntimeError("Call fit() first.")
        return (float(self.bin_edges_[bin_index]), float(self.bin_edges_[bin_index + 1]))

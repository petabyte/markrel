"""
JevCalibrator — domain recalibration for Jev (or any external) probabilities,
built on markrel's Markov-chain binning machinery.

Jev's output is a calibrated probability for *its own* judgment task, not
for your domain's actual base rate. This wraps markrel.MetricChain to learn
a histogram-binning calibration curve: P(actually relevant | Jev said p),
fit against your own ground-truth labels.

It deliberately bypasses markrel.MarkovRelevanceModel — that class expects
to compute similarity from query/document embeddings internally. Here the
"similarity" is already a scalar (Jev's probability), so MetricChain is
used directly as a standalone calibration layer.

Example
-------
    >>> from markrel.integrations import JevCalibrator
    >>> import numpy as np
    >>>
    >>> jev_probs = np.array([0.91, 0.83, 0.4, 0.12, 0.77, ...])
    >>> domain_labels = np.array([1, 0, 0, 0, 1, ...])  # independent ground truth
    >>>
    >>> cal = JevCalibrator(n_bins=20, bin_strategy="quantile")
    >>> cal.fit(jev_probs, domain_labels)
    >>>
    >>> cal.predict_proba([0.83])       # recalibrated probability
    >>> cal.predict([0.83], threshold=0.5)
    >>> cal.summary()                   # inspect the learned bin table
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..transitions import MetricChain
from ..model import MarkovRelevanceModel


@dataclass
class JevCalibrator:
    """Recalibrates Jev's (or any model's) output probability against your
    own domain labels, using markrel's bin-based Markov chain.

    Parameters
    ----------
    n_bins :
        Number of calibration bins. Quantile binning with 15-25 bins is a
        reasonable default; too few bins under-corrects, too many bins
        starve individual bins of training examples.
    bin_strategy :
        "quantile" (equal-frequency, recommended — robust to Jev's score
        distribution being lumpy or skewed) or "uniform" (equal-width).
    smoothing :
        Laplace smoothing pseudo-count. Higher values pull sparse bins
        toward the overall base rate instead of overfitting to a handful
        of labels.
    clip_input :
        If True, silently clips input probabilities to [0, 1] before
        binning (guards against minor float drift from an upstream API).

    Attributes
    ----------
    chain_ :
        The fitted markrel.MetricChain doing the actual bin lookups.
    """

    n_bins: int = 20
    bin_strategy: str = "quantile"
    smoothing: float = 1.0
    clip_input: bool = True

    chain_: MetricChain | None = field(init=False, default=None, repr=False)
    _fitted: bool = field(init=False, default=False)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        jev_probs: Sequence[float] | np.ndarray,
        labels: Sequence[bool | int],
    ) -> "JevCalibrator":
        """Fit the calibration curve.

        Parameters
        ----------
        jev_probs :
            Jev's raw P(relevant) output for each training pair.
        labels :
            Independent ground-truth relevance labels for the *same*
            pairs. These must not themselves be derived from Jev, or the
            calibrator just learns to reproduce Jev's own scores.

        Returns
        -------
        self
        """
        probs = self._prepare(jev_probs)
        labels_arr = np.asarray(labels, dtype=bool)

        if probs.shape != labels_arr.shape:
            raise ValueError(
                f"jev_probs and labels must have the same length, "
                f"got {probs.shape[0]} and {labels_arr.shape[0]}"
            )

        self.chain_ = MetricChain(
            metric_name="jev_prob",
            n_bins=self.n_bins,
            bin_strategy=self.bin_strategy,
            smoothing=self.smoothing,
        )
        self.chain_.fit(probs, labels_arr)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, jev_probs: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return recalibrated P(relevant) for each input Jev score."""
        self._check_fitted()
        probs = self._prepare(jev_probs)
        return self.chain_.p_relevant_batch(probs)

    def predict(
        self,
        jev_probs: Sequence[float] | np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Return binary predictions after recalibration.

        Note the threshold is applied to the *recalibrated* probability,
        not Jev's raw score — that's the whole point of calibrating first.
        """
        return (self.predict_proba(jev_probs) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Combining with a markrel embedding-based model
    # ------------------------------------------------------------------

    def combine_with(
        self,
        jev_probs: Sequence[float] | np.ndarray,
        other_model: MarkovRelevanceModel,
        queries,
        documents,
        rule: str = "bayesian",
    ) -> np.ndarray:
        """Combine recalibrated Jev probabilities with a markrel embedding
        model's probabilities (e.g. cosine similarity trained on your own
        embeddings), using the same odds-product logic markrel uses to
        combine its own metrics.

        Parameters
        ----------
        jev_probs :
            Raw Jev scores for the same pairs as ``queries``/``documents``.
        other_model :
            A fitted MarkovRelevanceModel (e.g. trained on embedding
            similarity) to combine with.
        queries, documents :
            Inputs to ``other_model.predict_proba``.
        rule :
            "bayesian" (product of odds — recommended when the two
            signals are roughly independent) or "mean".
        """
        self._check_fitted()
        p_jev = self.predict_proba(jev_probs)
        p_other = other_model.predict_proba(queries, documents)

        if rule == "bayesian":
            probs = np.clip(np.stack([p_jev, p_other], axis=1), 1e-10, 1 - 1e-10)
            odds = probs / (1 - probs)
            combined_odds = np.prod(odds, axis=1)
            return combined_odds / (1 + combined_odds)
        elif rule == "mean":
            return np.mean(np.stack([p_jev, p_other], axis=1), axis=1)
        else:
            raise ValueError(f"rule must be 'bayesian' or 'mean', got {rule!r}")

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Human-readable bin table: what Jev said vs. what was actually true."""
        self._check_fitted()
        return self.chain_.summary()

    def bin_edges(self) -> list[tuple[float, float]]:
        """Return the (lo, hi) edges of each calibration bin."""
        self._check_fitted()
        return [self.chain_.discretizer_.bin_edges(i) for i in range(len(self.chain_.states_))]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare(self, jev_probs) -> np.ndarray:
        probs = np.asarray(jev_probs, dtype=float)
        if self.clip_input:
            probs = np.clip(probs, 0.0, 1.0)
        elif np.any((probs < 0.0) | (probs > 1.0)):
            raise ValueError(
                "jev_probs must be in [0, 1]. Pass clip_input=True to "
                "auto-clip minor float drift instead of raising."
            )
        return probs

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("JevCalibrator has not been fitted. Call fit() first.")

"""
Similarity metric implementations for document relevance scoring.

Each metric takes two numpy arrays (vector representations of a query and
document) and returns a float in a consistent range — typically [0, 1] for
set-based metrics or normalized variants, and [-1, 1] for cosine.  The
StateDiscretizer handles any range by fitting bin edges to observed values.
"""

from __future__ import annotations

import numpy as np
from typing import Callable


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]; returns 0.0 for zero vectors."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance converted to similarity in (0, 1]."""
    dist = float(np.linalg.norm(a - b))
    return 1.0 / (1.0 + dist)


def manhattan_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan (L1) distance converted to similarity in (0, 1]."""
    dist = float(np.sum(np.abs(a - b)))
    return 1.0 / (1.0 + dist)


def dot_product_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Raw dot product — unbounded; useful for unit-normalized embeddings."""
    return float(np.dot(a, b))


def jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Jaccard similarity on binarised term vectors in [0, 1].
    Elements > 0 are treated as present.
    """
    a_set = a > 0
    b_set = b > 0
    intersection = int(np.sum(a_set & b_set))
    union = int(np.sum(a_set | b_set))
    if union == 0:
        return 0.0
    return intersection / union


def overlap_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """
    Overlap (Szymkiewicz-Simpson) coefficient in [0, 1].
    Measures the fraction of the smaller set that overlaps with the larger.
    """
    a_set = a > 0
    b_set = b > 0
    intersection = int(np.sum(a_set & b_set))
    min_size = min(int(np.sum(a_set)), int(np.sum(b_set)))
    if min_size == 0:
        return 0.0
    return intersection / min_size


def dice_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Dice (Sørensen-Dice) coefficient in [0, 1]."""
    a_set = a > 0
    b_set = b > 0
    intersection = int(np.sum(a_set & b_set))
    denom = int(np.sum(a_set)) + int(np.sum(b_set))
    if denom == 0:
        return 0.0
    return (2 * intersection) / denom


def chebyshev_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Chebyshev (L-inf) distance converted to similarity in (0, 1]."""
    dist = float(np.max(np.abs(a - b)))
    return 1.0 / (1.0 + dist)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: All built-in metric names and their callables.
SIMILARITY_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "cosine": cosine_similarity,
    "euclidean": euclidean_similarity,
    "manhattan": manhattan_similarity,
    "dot_product": dot_product_similarity,
    "jaccard": jaccard_similarity,
    "overlap": overlap_coefficient,
    "dice": dice_similarity,
    "chebyshev": chebyshev_similarity,
}


def get_metric(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return a metric callable by name, raising ValueError for unknowns."""
    if name not in SIMILARITY_METRICS:
        raise ValueError(
            f"Unknown similarity metric '{name}'. "
            f"Available: {sorted(SIMILARITY_METRICS)}"
        )
    return SIMILARITY_METRICS[name]


def compute_all(
    a: np.ndarray,
    b: np.ndarray,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Compute multiple similarity scores between two vectors.

    Parameters
    ----------
    a, b:
        Feature vectors of equal length.
    metrics:
        Names of metrics to compute.  Defaults to all built-in metrics.

    Returns
    -------
    dict mapping metric name -> similarity value.
    """
    chosen = metrics if metrics is not None else list(SIMILARITY_METRICS)
    return {name: get_metric(name)(a, b) for name in chosen}

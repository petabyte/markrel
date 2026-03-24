"""
Text-to-vector conversion utilities.

The model accepts either pre-computed numpy arrays or raw text strings.
When strings are passed, a ``TextVectorizer`` wraps scikit-learn's
``TfidfVectorizer`` and converts the sparse TF-IDF matrix to dense arrays.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer


InputType = Union[np.ndarray, Sequence[str]]


def _is_string_input(X: InputType) -> bool:
    """Return True if X looks like a sequence of text strings."""
    if isinstance(X, np.ndarray):
        return False
    if isinstance(X, (list, tuple)) and len(X) > 0 and isinstance(X[0], str):
        return True
    return False


class TextVectorizer:
    """Thin wrapper around TF-IDF vectorisation for text inputs.

    Parameters
    ----------
    max_features:
        Maximum vocabulary size passed to ``TfidfVectorizer``.
    **tfidf_kwargs:
        Additional keyword arguments forwarded to ``TfidfVectorizer``.
    """

    def __init__(self, max_features: int = 10_000, **tfidf_kwargs) -> None:
        self.max_features = max_features
        self._tfidf_kwargs = tfidf_kwargs
        self._vectorizer: _TfidfVectorizer | None = None

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, texts: Sequence[str]) -> "TextVectorizer":
        """Fit the TF-IDF vocabulary on *texts*."""
        self._vectorizer = _TfidfVectorizer(
            max_features=self.max_features,
            **self._tfidf_kwargs,
        )
        self._vectorizer.fit(texts)
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Convert texts to a dense float32 matrix of shape (n, vocab_size)."""
        if self._vectorizer is None:
            raise RuntimeError("Call fit() before transform().")
        sparse = self._vectorizer.transform(texts)
        return sparse.toarray().astype(np.float32)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        if self._vectorizer is None or self._vectorizer.vocabulary_ is None:
            return 0
        return len(self._vectorizer.vocabulary_)


def coerce_to_matrix(X: InputType, vectorizer: TextVectorizer | None = None) -> np.ndarray:
    """Ensure *X* is a 2-D float numpy array.

    Parameters
    ----------
    X:
        Either a 2-D numpy array or a sequence of strings.
    vectorizer:
        Fitted ``TextVectorizer``.  Required when *X* contains strings.
    """
    if isinstance(X, np.ndarray):
        if X.ndim != 2:
            raise ValueError(f"Expected a 2-D array, got shape {X.shape}.")
        return X.astype(float)

    if _is_string_input(X):
        if vectorizer is None:
            raise ValueError(
                "A fitted TextVectorizer must be supplied when X contains strings."
            )
        return vectorizer.transform(list(X))

    # Fallback: try converting directly
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Could not coerce input to a 2-D array (shape={arr.shape}).")
    return arr

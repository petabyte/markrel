"""
🐟 markrel - A Markov chain-based document relevance predictor.

Swim through your documents with markrel! Like a school of mackerel navigating
the seas, this library traces probabilistic paths through similarity space to
find relevant documents.

markrel treats each similarity metric as a Markov chain where:
- States (nodes) are discretized similarity value ranges (bins)
- Transition probabilities encode P(relevant | similarity_bin)

Example:
    >>> from markrel import MarkovRelevanceModel
    >>> model = MarkovRelevanceModel(metrics=["cosine", "euclidean"])
    >>> model.fit(queries, documents, labels)
    >>> relevance_prob = model.predict(query, document)
"""

from .model import MarkovRelevanceModel
from .similarity import (
    SIMILARITY_METRICS,
    cosine_similarity,
    euclidean_similarity,
    manhattan_similarity,
    dot_product_similarity,
    jaccard_similarity,
    overlap_coefficient,
    dice_similarity,
    chebyshev_similarity,
    get_metric,
    compute_all,
)
from .states import MarkovState, StateDiscretizer
from .transitions import MetricChain
from .vectorizer import TextVectorizer, coerce_to_matrix

__version__ = "0.1.0"

__all__ = [
    # Main model
    "MarkovRelevanceModel",
    # Similarity metrics
    "SIMILARITY_METRICS",
    "cosine_similarity",
    "euclidean_similarity",
    "manhattan_similarity",
    "dot_product_similarity",
    "jaccard_similarity",
    "overlap_coefficient",
    "dice_similarity",
    "chebyshev_similarity",
    "get_metric",
    "compute_all",
    # State management
    "MarkovState",
    "StateDiscretizer",
    # Transitions
    "MetricChain",
    # Vectorization
    "TextVectorizer",
    "coerce_to_matrix",
]

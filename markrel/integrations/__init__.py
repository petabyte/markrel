"""
Optional integrations between markrel and external relevance/reranking
tools.

These modules build on markrel's public API (MetricChain,
MarkovRelevanceModel) rather than modifying it, so they carry no extra
runtime dependencies beyond numpy.
"""

from .jev_calibrator import JevCalibrator

__all__ = ["JevCalibrator"]

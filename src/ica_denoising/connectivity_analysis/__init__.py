"""Derived connectivity analysis from already-saved artifacts (Section 5.7).

This subpackage computes graph and rank statistics from connectivity matrices
and trace arrays that are already saved in ``outputs/connectivity/``. No
estimator is rerun; the goal is to enrich the current paper with derived
metrics (densities, overlaps, degree changes, rank loss, enrichment tests).
"""

from __future__ import annotations

from .config import ConnectivityAnalysisConfig, NodeAnnotations
from .enrichment import (
    EnrichmentResult,
    directional_enrichment_test,
    group_enrichment_test,
    ipsilateral_enrichment_test,
)
from .loaders import load_trace_array, load_variant_matrices
from .metrics import (
    DegreeChange,
    binarize,
    condition_number,
    degree_changes,
    edge_count,
    edge_density,
    effective_rank,
    jaccard_overlap,
    normalized_hamming,
    numerical_rank,
    reciprocal_fraction,
    stable_rank,
)
from .summary import compare_estimators, summarize_recording

__all__ = [
    "ConnectivityAnalysisConfig",
    "NodeAnnotations",
    "EnrichmentResult",
    "DegreeChange",
    "binarize",
    "edge_count",
    "edge_density",
    "jaccard_overlap",
    "normalized_hamming",
    "reciprocal_fraction",
    "degree_changes",
    "numerical_rank",
    "effective_rank",
    "stable_rank",
    "condition_number",
    "group_enrichment_test",
    "ipsilateral_enrichment_test",
    "directional_enrichment_test",
    "load_variant_matrices",
    "load_trace_array",
    "summarize_recording",
    "compare_estimators",
]

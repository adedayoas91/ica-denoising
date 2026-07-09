"""Configuration objects for derived connectivity analysis (Section 5.7).

The connectivity analysis operates on already-saved connectivity matrices and
trace arrays, so the configuration only needs to describe thresholding, rank
tolerances, and permutation-test determinism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ConnectivityAnalysisConfig", "NodeAnnotations"]


@dataclass(frozen=True)
class ConnectivityAnalysisConfig:
    """Immutable configuration for derived connectivity statistics.

    Attributes:
        edge_threshold: Absolute magnitude above which an off-diagonal weight is
            counted as an edge. The default of ``0.0`` treats any nonzero
            off-diagonal entry as an edge.
        rank_tolerance: Relative tolerance (multiplied by the largest singular
            value) used to decide which singular values count towards the
            numerical rank. When ``None`` the NumPy default
            ``max(shape) * eps * s_max`` is used.
        n_permutations: Number of label/coordinate permutations drawn for the
            enrichment tests.
        random_state: Seed for ``numpy.random.default_rng`` so permutation tests
            are deterministic.
        n_past: Conditioning order key used when falling back to a per-variant
            connectivity pickle that maps ``n_past`` to a matrix.
    """

    edge_threshold: float = 0.0
    rank_tolerance: float | None = None
    n_permutations: int = 1000
    random_state: int = 0
    n_past: int = 2

    def rng(self) -> np.random.Generator:
        """Return a fresh deterministic random generator for permutation tests.

        Returns:
            A ``numpy.random.Generator`` seeded with ``random_state``.
        """

        return np.random.default_rng(self.random_state)


@dataclass(frozen=True)
class NodeAnnotations:
    """Per-node annotations used for enrichment tests.

    All arrays are indexed by node (matching the connectivity matrix order).

    Attributes:
        group_labels: Categorical labels per node (e.g. ``"emitter"`` /
            ``"receiver"`` for v2a-RSNs). Used by :func:`group_enrichment_test`.
        source_group: Label treated as the edge source for group enrichment.
        target_group: Label treated as the edge target for group enrichment.
        side: Lateralisation label per node (e.g. ``"L"`` / ``"R"`` for
            motorneurons). Used by ipsilateral enrichment.
        rostro_caudal: Rostro-caudal ordinal coordinate per node. Used by
            :func:`directional_enrichment_test`; larger values are caudal.
    """

    group_labels: np.ndarray | None = None
    source_group: str = "emitter"
    target_group: str = "receiver"
    side: np.ndarray | None = None
    rostro_caudal: np.ndarray | None = None

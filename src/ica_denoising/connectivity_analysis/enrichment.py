"""Permutation-based edge-enrichment tests for connectivity graphs.

These tests keep the observed edge structure fixed and permute the node
annotations (labels or coordinates) to build a null distribution. The greater
-tail permutation p-value is ``(1 + #(null >= observed)) / (n_permutations + 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .metrics import binarize

__all__ = [
    "EnrichmentResult",
    "group_enrichment_test",
    "ipsilateral_enrichment_test",
    "directional_enrichment_test",
]


@dataclass(frozen=True)
class EnrichmentResult:
    """Result of a permutation enrichment test.

    Attributes:
        statistic_name: Human-readable name of the observed statistic.
        observed: Observed statistic value on the true annotations.
        p_value: Greater-tail permutation p-value.
        null_mean: Mean of the null distribution.
        null_std: Standard deviation of the null distribution.
        null_q05: 5th percentile of the null distribution.
        null_q95: 95th percentile of the null distribution.
        n_permutations: Number of permutations drawn.
        n_edges: Number of directed edges in the graph.
    """

    statistic_name: str
    observed: float
    p_value: float
    null_mean: float
    null_std: float
    null_q05: float
    null_q95: float
    n_permutations: int
    n_edges: int


def _run_permutation_test(
    statistic_name: str,
    annotation: np.ndarray,
    observed_fn: Callable[[np.ndarray], float],
    n_edges: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> EnrichmentResult:
    """Run a generic node-annotation permutation test.

    Args:
        statistic_name: Name recorded in the result.
        annotation: Per-node annotation array that gets permuted.
        observed_fn: Maps a (possibly permuted) annotation array to a statistic.
        n_edges: Number of directed edges (recorded; tests are vacuous if zero).
        n_permutations: Number of permutations to draw.
        rng: Deterministic random generator.

    Returns:
        An :class:`EnrichmentResult` summarising observed value and null.
    """

    observed = float(observed_fn(annotation))
    if n_edges == 0 or n_permutations <= 0:
        return EnrichmentResult(
            statistic_name=statistic_name,
            observed=observed,
            p_value=1.0,
            null_mean=float("nan"),
            null_std=float("nan"),
            null_q05=float("nan"),
            null_q95=float("nan"),
            n_permutations=max(n_permutations, 0),
            n_edges=n_edges,
        )
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        permuted = rng.permutation(annotation)
        null[i] = observed_fn(permuted)
    p_value = float((1 + np.sum(null >= observed)) / (n_permutations + 1))
    return EnrichmentResult(
        statistic_name=statistic_name,
        observed=observed,
        p_value=p_value,
        null_mean=float(np.mean(null)),
        null_std=float(np.std(null)),
        null_q05=float(np.percentile(null, 5)),
        null_q95=float(np.percentile(null, 95)),
        n_permutations=n_permutations,
        n_edges=n_edges,
    )


def group_enrichment_test(
    matrix: np.ndarray,
    group_labels: np.ndarray,
    source_group: str,
    target_group: str,
    *,
    threshold: float = 0.0,
    n_permutations: int = 1000,
    rng: np.random.Generator | None = None,
) -> EnrichmentResult:
    """Test enrichment of edges from ``source_group`` to ``target_group``.

    The statistic is the fraction of directed edges whose source node carries
    ``source_group`` and whose target node carries ``target_group`` (e.g.
    emitter -> receiver for v2a-RSNs). The null permutes the node labels.

    Args:
        matrix: Weighted directed connectivity matrix.
        group_labels: Categorical label per node.
        source_group: Label of the edge source group.
        target_group: Label of the edge target group.
        threshold: Edge magnitude threshold.
        n_permutations: Number of label permutations.
        rng: Optional deterministic generator; created if ``None``.

    Returns:
        The :class:`EnrichmentResult` for the directed group enrichment.

    Raises:
        ValueError: If ``group_labels`` length does not match the matrix order.
    """

    adj = binarize(matrix, threshold)
    labels = np.asarray(group_labels)
    if labels.shape[0] != adj.shape[0]:
        raise ValueError("group_labels length must match the matrix dimension.")
    rng = rng if rng is not None else np.random.default_rng()
    sources, targets = np.nonzero(adj)
    n_edges = int(sources.size)

    def statistic(perm_labels: np.ndarray) -> float:
        if n_edges == 0:
            return 0.0
        is_src = perm_labels[sources] == source_group
        is_tgt = perm_labels[targets] == target_group
        return float(np.mean(is_src & is_tgt))

    return _run_permutation_test(
        statistic_name=f"frac_edges_{source_group}_to_{target_group}",
        annotation=labels,
        observed_fn=statistic,
        n_edges=n_edges,
        n_permutations=n_permutations,
        rng=rng,
    )


def ipsilateral_enrichment_test(
    matrix: np.ndarray,
    side_labels: np.ndarray,
    *,
    threshold: float = 0.0,
    n_permutations: int = 1000,
    rng: np.random.Generator | None = None,
) -> EnrichmentResult:
    """Test enrichment of ipsilateral (same-side) edges.

    The statistic is the fraction of directed edges whose source and target
    share the same side label (e.g. ``"L"``/``"R"`` for motorneurons). The null
    permutes the side labels across nodes.

    Args:
        matrix: Weighted directed connectivity matrix.
        side_labels: Side label per node.
        threshold: Edge magnitude threshold.
        n_permutations: Number of label permutations.
        rng: Optional deterministic generator; created if ``None``.

    Returns:
        The :class:`EnrichmentResult` for ipsilateral enrichment.

    Raises:
        ValueError: If ``side_labels`` length does not match the matrix order.
    """

    adj = binarize(matrix, threshold)
    labels = np.asarray(side_labels)
    if labels.shape[0] != adj.shape[0]:
        raise ValueError("side_labels length must match the matrix dimension.")
    rng = rng if rng is not None else np.random.default_rng()
    sources, targets = np.nonzero(adj)
    n_edges = int(sources.size)

    def statistic(perm_labels: np.ndarray) -> float:
        if n_edges == 0:
            return 0.0
        return float(np.mean(perm_labels[sources] == perm_labels[targets]))

    return _run_permutation_test(
        statistic_name="frac_ipsilateral_edges",
        annotation=labels,
        observed_fn=statistic,
        n_edges=n_edges,
        n_permutations=n_permutations,
        rng=rng,
    )


def directional_enrichment_test(
    matrix: np.ndarray,
    coordinate: np.ndarray,
    *,
    threshold: float = 0.0,
    n_permutations: int = 1000,
    rng: np.random.Generator | None = None,
) -> EnrichmentResult:
    """Test enrichment of edges aligned with increasing coordinate.

    For a rostro-caudal coordinate (larger == caudal), the statistic is the
    fraction of directed edges whose target coordinate exceeds the source
    coordinate (rostral -> caudal). The null permutes coordinates across nodes.

    Args:
        matrix: Weighted directed connectivity matrix.
        coordinate: Ordinal coordinate per node.
        threshold: Edge magnitude threshold.
        n_permutations: Number of coordinate permutations.
        rng: Optional deterministic generator; created if ``None``.

    Returns:
        The :class:`EnrichmentResult` for directional enrichment.

    Raises:
        ValueError: If ``coordinate`` length does not match the matrix order.
    """

    adj = binarize(matrix, threshold)
    coord = np.asarray(coordinate, dtype=float)
    if coord.shape[0] != adj.shape[0]:
        raise ValueError("coordinate length must match the matrix dimension.")
    rng = rng if rng is not None else np.random.default_rng()
    sources, targets = np.nonzero(adj)
    n_edges = int(sources.size)

    def statistic(perm_coord: np.ndarray) -> float:
        if n_edges == 0:
            return 0.0
        return float(np.mean(perm_coord[targets] > perm_coord[sources]))

    return _run_permutation_test(
        statistic_name="frac_rostral_to_caudal_edges",
        annotation=coord,
        observed_fn=statistic,
        n_edges=n_edges,
        n_permutations=n_permutations,
        rng=rng,
    )

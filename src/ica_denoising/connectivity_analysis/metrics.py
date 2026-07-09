"""Pure graph and rank metrics for derived connectivity analysis.

Every function operates on plain NumPy arrays so it can be unit-tested without
touching the saved artifacts. All graph metrics zero the diagonal before
counting, and binary adjacency is defined as ``abs(W) > threshold``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "binarize",
    "edge_count",
    "edge_density",
    "jaccard_overlap",
    "normalized_hamming",
    "reciprocal_fraction",
    "DegreeChange",
    "degree_changes",
    "numerical_rank",
    "effective_rank",
    "stable_rank",
    "condition_number",
]


def _check_square(matrix: np.ndarray) -> np.ndarray:
    """Validate and return a 2D square float array.

    Args:
        matrix: Candidate connectivity matrix.

    Returns:
        The matrix as a float ``ndarray``.

    Raises:
        ValueError: If the matrix is not 2D and square.
    """

    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected a square 2D matrix, got shape {arr.shape}.")
    return arr


def binarize(matrix: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Return the binary off-diagonal adjacency of a weighted matrix.

    Args:
        matrix: Weighted (possibly directed) connectivity matrix.
        threshold: Absolute magnitude above which an entry is an edge.

    Returns:
        An integer adjacency matrix with a zeroed diagonal where
        ``adj[i, j] == 1`` iff ``abs(matrix[i, j]) > threshold``.
    """

    arr = _check_square(matrix)
    adj = (np.abs(arr) > threshold).astype(int)
    np.fill_diagonal(adj, 0)
    return adj


def edge_count(matrix: np.ndarray, threshold: float = 0.0) -> int:
    """Count directed off-diagonal edges.

    Args:
        matrix: Weighted connectivity matrix.
        threshold: Edge magnitude threshold.

    Returns:
        Number of off-diagonal entries exceeding the threshold.
    """

    return int(binarize(matrix, threshold).sum())


def edge_density(matrix: np.ndarray, threshold: float = 0.0) -> float:
    """Compute the directed off-diagonal edge density.

    Args:
        matrix: Weighted connectivity matrix.
        threshold: Edge magnitude threshold.

    Returns:
        ``edges / (n * (n - 1))`` or ``0.0`` when ``n < 2``.
    """

    adj = binarize(matrix, threshold)
    n = adj.shape[0]
    possible = n * (n - 1)
    if possible == 0:
        return 0.0
    return float(adj.sum()) / float(possible)


def jaccard_overlap(
    matrix_a: np.ndarray, matrix_b: np.ndarray, threshold: float = 0.0
) -> float:
    """Jaccard overlap of the edge sets of two graphs.

    Args:
        matrix_a: First weighted/binary matrix.
        matrix_b: Second weighted/binary matrix.
        threshold: Edge magnitude threshold.

    Returns:
        ``|edges_a & edges_b| / |edges_a | edges_b|``. Two edgeless graphs are
        defined to overlap perfectly and return ``1.0``.
    """

    adj_a = binarize(matrix_a, threshold)
    adj_b = binarize(matrix_b, threshold)
    if adj_a.shape != adj_b.shape:
        raise ValueError("Matrices must share the same shape for Jaccard overlap.")
    intersection = int(np.logical_and(adj_a, adj_b).sum())
    union = int(np.logical_or(adj_a, adj_b).sum())
    if union == 0:
        return 1.0
    return intersection / union


def normalized_hamming(
    matrix_a: np.ndarray, matrix_b: np.ndarray, threshold: float = 0.0
) -> float:
    """Normalized Hamming distance between two adjacency matrices.

    Args:

        matrix_a: First weighted/binary matrix.
        matrix_b: Second weighted/binary matrix.
        threshold: Edge magnitude threshold.

    Returns:
        Fraction of off-diagonal positions whose edge presence disagrees.
    """

    adj_a = binarize(matrix_a, threshold)
    adj_b = binarize(matrix_b, threshold)
    if adj_a.shape != adj_b.shape:
        raise ValueError("Matrices must share the same shape for Hamming distance.")
    n = adj_a.shape[0]
    possible = n * (n - 1)
    if possible == 0:
        return 0.0
    differing = int((adj_a != adj_b).sum())  # diagonal is identical (both zero)
    return differing / possible


def reciprocal_fraction(matrix: np.ndarray, threshold: float = 0.0) -> float:
    """Fraction of directed edges whose reverse edge also exists.

    Args:
        matrix: Weighted directed connectivity matrix.
        threshold: Edge magnitude threshold.

    Returns:
        ``(# directed edges with present reverse) / (# directed edges)`` or
        ``0.0`` when there are no edges.
    """

    adj = binarize(matrix, threshold)
    total = int(adj.sum())
    if total == 0:
        return 0.0
    reciprocal = int(np.logical_and(adj, adj.T).sum())
    return reciprocal / total


@dataclass(frozen=True)
class DegreeChange:
    """Per-node degree changes of a variant graph relative to a baseline.

    Edges are interpreted as ``W[i, j]`` being a directed edge ``i -> j``, so
    out-degree is the row sum and in-degree the column sum of the adjacency.

    Attributes:
        in_degree_raw: In-degree per node for the baseline graph.
        out_degree_raw: Out-degree per node for the baseline graph.
        in_degree_variant: In-degree per node for the variant graph.
        out_degree_variant: Out-degree per node for the variant graph.
        in_degree_delta: ``in_degree_variant - in_degree_raw``.
        out_degree_delta: ``out_degree_variant - out_degree_raw``.
    """

    in_degree_raw: np.ndarray
    out_degree_raw: np.ndarray
    in_degree_variant: np.ndarray
    out_degree_variant: np.ndarray
    in_degree_delta: np.ndarray
    out_degree_delta: np.ndarray


def degree_changes(
    variant: np.ndarray, baseline: np.ndarray, threshold: float = 0.0
) -> DegreeChange:
    """Compute per-node in/out-degree changes versus a baseline graph.

    Args:
        variant: Weighted directed matrix for the variant (e.g. a BSS method).
        baseline: Weighted directed matrix for the baseline (e.g. ``raw``).
        threshold: Edge magnitude threshold.

    Returns:
        A :class:`DegreeChange` with baseline, variant, and delta degree arrays.

    Raises:
        ValueError: If the two matrices do not share the same shape.
    """

    adj_v = binarize(variant, threshold)
    adj_b = binarize(baseline, threshold)
    if adj_v.shape != adj_b.shape:
        raise ValueError("Variant and baseline matrices must share the same shape.")
    out_v = adj_v.sum(axis=1)
    in_v = adj_v.sum(axis=0)
    out_b = adj_b.sum(axis=1)
    in_b = adj_b.sum(axis=0)
    return DegreeChange(
        in_degree_raw=in_b,
        out_degree_raw=out_b,
        in_degree_variant=in_v,
        out_degree_variant=out_v,
        in_degree_delta=in_v - in_b,
        out_degree_delta=out_v - out_b,
    )


def _singular_values(matrix: np.ndarray) -> np.ndarray:
    """Return singular values of a 2D matrix in descending order.

    Args:
        matrix: Any 2D array (e.g. a trace matrix of shape neurons x frames).

    Returns:
        A 1D array of singular values.

    Raises:
        ValueError: If the input is not 2D.
    """

    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {arr.shape}.")
    return np.linalg.svd(arr, compute_uv=False)


def numerical_rank(matrix: np.ndarray, rank_tolerance: float | None = None) -> int:
    """Numerical rank via a tolerance on singular values.

    Args:
        matrix: 2D matrix (e.g. a trace representation).
        rank_tolerance: Relative tolerance multiplied by the largest singular
            value. When ``None`` the NumPy default ``max(shape) * eps * s_max``
            is used.

    Returns:
        Number of singular values strictly above the tolerance.
    """

    svals = _singular_values(matrix)
    if svals.size == 0:
        return 0
    s_max = float(svals[0])
    if s_max == 0.0:
        return 0
    if rank_tolerance is None:
        threshold = s_max * max(matrix.shape) * float(np.finfo(float).eps)
    else:
        threshold = s_max * float(rank_tolerance)
    return int((svals > threshold).sum())


def effective_rank(matrix: np.ndarray) -> float:
    """Effective rank via the entropy of normalized singular values.

    Implements the spectral entropy effective rank of Roy & Vetterli (2007):
    ``exp(-sum p_i log p_i)`` where ``p_i = s_i / sum(s)``.

    Args:
        matrix: 2D matrix (e.g. a trace representation).

    Returns:
        The effective rank, a real number in ``[1, rank]`` (``0.0`` for a zero
        matrix).
    """

    svals = _singular_values(matrix)
    svals = svals[svals > 0]
    total = float(svals.sum())
    if total == 0.0:
        return 0.0
    p = svals / total
    entropy = float(-np.sum(p * np.log(p)))
    return float(np.exp(entropy))


def stable_rank(matrix: np.ndarray) -> float:
    """Stable rank ``||A||_F^2 / ||A||_2^2``.

    Args:
        matrix: 2D matrix.

    Returns:
        The stable (numerical) rank, or ``0.0`` for a zero matrix.
    """

    svals = _singular_values(matrix)
    if svals.size == 0 or svals[0] == 0.0:
        return 0.0
    return float(np.sum(svals**2) / (svals[0] ** 2))


def condition_number(matrix: np.ndarray) -> float:
    """Spectral condition number ``s_max / s_min``.

    Args:
        matrix: 2D matrix.

    Returns:
        The 2-norm condition number; ``inf`` when the smallest singular value is
        zero.
    """

    svals = _singular_values(matrix)
    if svals.size == 0:
        return float("inf")
    s_min = float(svals[-1])
    if s_min == 0.0:
        return float("inf")
    return float(svals[0] / s_min)

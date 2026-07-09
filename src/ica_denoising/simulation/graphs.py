"""Ground-truth directed graph families for the simulation benchmark.

Conventions
-----------
* The lag-coefficient tensor is ``A[lag, target, source]`` so that a linear VAR
  step reads ``x[t] = sum_lag A[lag] @ x[t - (lag + 1)] + innovation``.
* The collapsed directed adjacency returned for graph-recovery scoring uses the
  ``[source, target]`` convention (an entry ``adj[s, t] != 0`` means ``s -> t``)
  to match the estimator output (``CGCResult.lag_only``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GraphConfig

__all__ = ["GroundTruthGraph", "build_graph", "companion_spectral_radius"]


@dataclass(frozen=True)
class GroundTruthGraph:
    """A sampled ground-truth directed graph.

    Attributes:
        coefficients: Lag tensor ``A[lag, target, source]`` of shape
            ``(lag_order, n, n)``.
        adjacency: Collapsed binary adjacency ``[source, target]`` of shape
            ``(n, n)``.
        node_groups: Per-node group label (e.g. emitter/receiver, side).
        positions: Per-node 2D positions of shape ``(n, 2)``.
        edge_categories: Mapping ``(source, target) -> category``.
        n_observed: Number of observed nodes (first ``n_observed`` indices).
    """

    coefficients: np.ndarray
    adjacency: np.ndarray
    node_groups: np.ndarray
    positions: np.ndarray
    edge_categories: dict[tuple[int, int], str]
    n_observed: int


def companion_spectral_radius(coefficients: np.ndarray) -> float:
    """Return the spectral radius of the VAR companion matrix.

    Args:
        coefficients: Lag tensor ``A[lag, target, source]``.

    Returns:
        The largest absolute eigenvalue of the companion matrix.
    """

    p, n, _ = coefficients.shape
    companion = np.zeros((n * p, n * p))
    companion[:n, :] = np.hstack([coefficients[lag] for lag in range(p)])
    if p > 1:
        companion[n:, : n * (p - 1)] = np.eye(n * (p - 1))
    eigenvalues = np.linalg.eigvals(companion)
    return float(np.max(np.abs(eigenvalues)))


def _rescale_to_radius(coefficients: np.ndarray, target: float) -> np.ndarray:
    """Rescale a lag tensor so the companion spectral radius equals ``target``."""

    radius = companion_spectral_radius(coefficients)
    if radius <= 0:
        return coefficients
    return coefficients * (target / radius)


def _sample_weights(
    rng: np.random.Generator, mask: np.ndarray, cfg: GraphConfig
) -> np.ndarray:
    """Sample signed coefficients on the permitted edges of ``mask``."""

    p = cfg.lag_order
    n = mask.shape[0]
    coeffs = np.zeros((p, n, n))
    for lag in range(p):
        magnitude = rng.uniform(cfg.weight_low, cfg.weight_high, size=(n, n))
        sign = rng.choice((-1.0, 1.0), size=(n, n))
        # mask is [source, target]; coefficient tensor is [target, source].
        coeffs[lag] = (magnitude * sign * mask).T
    return coeffs


def _sparse_random_mask(rng: np.random.Generator, n: int, density: float) -> np.ndarray:
    mask = (rng.random((n, n)) < density).astype(float)
    np.fill_diagonal(mask, 0.0)
    return mask


def _feedforward_modular_mask(
    rng: np.random.Generator, n: int, n_modules: int, density: float
) -> tuple[np.ndarray, np.ndarray]:
    """Feed-forward graph: emitter module -> receiver module."""

    groups = np.zeros(n, dtype=int)
    half = n // 2
    groups[half:] = 1  # 0 = emitter, 1 = receiver
    mask = np.zeros((n, n))
    emitters = np.where(groups == 0)[0]
    receivers = np.where(groups == 1)[0]
    for s in emitters:
        for t in receivers:
            if rng.random() < max(density, 0.05) * 2:
                mask[s, t] = 1.0
    # a little within-emitter recurrence
    for s in emitters:
        for t in emitters:
            if s != t and rng.random() < density / 2:
                mask[s, t] = 1.0
    np.fill_diagonal(mask, 0.0)
    return mask, groups


def _bilateral_chain_mask(
    rng: np.random.Generator, n: int, n_contra: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilateral chain: ipsilateral rostral->caudal edges + few contralateral."""

    side = np.zeros(n, dtype=int)
    side[n // 2 :] = 1  # 0 = left, 1 = right
    rostro_caudal = np.tile(np.arange((n + 1) // 2), 2)[:n].astype(float)
    mask = np.zeros((n, n))
    for s in range(n):
        for t in range(n):
            if s == t:
                continue
            same_side = side[s] == side[t]
            forward = rostro_caudal[t] == rostro_caudal[s] + 1
            if same_side and forward:
                mask[s, t] = 1.0
    # controlled contralateral edges
    contra_candidates = [
        (s, t)
        for s in range(n)
        for t in range(n)
        if s != t and side[s] != side[t]
    ]
    rng.shuffle(contra_candidates)
    for s, t in contra_candidates[:n_contra]:
        mask[s, t] = 1.0
    return mask, side, rostro_caudal


def build_graph(cfg: GraphConfig, rng: np.random.Generator) -> GroundTruthGraph:
    """Sample a :class:`GroundTruthGraph` for the configured family."""

    n = cfg.n_observed + cfg.n_hidden
    positions = rng.uniform(0.0, 1.0, size=(n, 2))

    if cfg.family == "sparse_random":
        mask = _sparse_random_mask(rng, n, cfg.edge_density)
        groups = np.zeros(n, dtype=int)
    elif cfg.family == "feedforward_modular":
        mask, groups = _feedforward_modular_mask(rng, n, cfg.n_modules, cfg.edge_density)
    elif cfg.family == "bilateral_chain":
        n_contra = max(1, int(round(cfg.edge_density * n)))
        mask, side, rostro = _bilateral_chain_mask(rng, n, n_contra)
        groups = side
        positions = np.column_stack([side.astype(float), rostro])
    else:
        raise ValueError(f"Unknown graph family: {cfg.family}")

    coefficients = _sample_weights(rng, mask, cfg)
    coefficients = _rescale_to_radius(coefficients, cfg.spectral_radius)

    adjacency = (mask != 0).astype(int)
    np.fill_diagonal(adjacency, 0)

    edge_categories: dict[tuple[int, int], str] = {}
    for s in range(n):
        for t in range(n):
            if adjacency[s, t]:
                edge_categories[(s, t)] = _categorize_edge(cfg, groups, positions, s, t)

    return GroundTruthGraph(
        coefficients=coefficients,
        adjacency=adjacency,
        node_groups=groups,
        positions=positions,
        edge_categories=edge_categories,
        n_observed=cfg.n_observed,
    )


def _categorize_edge(
    cfg: GraphConfig,
    groups: np.ndarray,
    positions: np.ndarray,
    s: int,
    t: int,
) -> str:
    if cfg.family == "feedforward_modular":
        if groups[s] == 0 and groups[t] == 1:
            return "emitter_to_receiver"
        return "within_module"
    if cfg.family == "bilateral_chain":
        if groups[s] == groups[t]:
            return "ipsilateral"
        return "contralateral"
    return "generic"

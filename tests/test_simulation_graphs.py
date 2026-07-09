from __future__ import annotations

import numpy as np

from ica_denoising.simulation.config import GraphConfig
from ica_denoising.simulation.graphs import (
    build_graph,
    companion_spectral_radius,
)


def test_spectral_radius_below_target():
    cfg = GraphConfig(family="sparse_random", n_observed=10, spectral_radius=0.85)
    graph = build_graph(cfg, np.random.default_rng(0))
    radius = companion_spectral_radius(graph.coefficients)
    assert radius <= cfg.spectral_radius + 1e-6


def test_families_build():
    for family in ("sparse_random", "feedforward_modular", "bilateral_chain"):
        cfg = GraphConfig(family=family, n_observed=8)
        graph = build_graph(cfg, np.random.default_rng(1))
        assert graph.adjacency.shape == (8, 8)
        assert np.all(np.diag(graph.adjacency) == 0)


def test_one_edge_orientation():
    # Construct a graph manually: only edge source 0 -> target 1.
    # Build a known single edge and verify the VAR step direction.
    coeffs = np.zeros((1, 3, 3))
    coeffs[0, 1, 0] = 0.6  # target=1, source=0
    x = np.zeros((3, 50))
    x[0, 0] = 1.0
    for t in range(1, 50):
        x[:, t] = coeffs[0] @ x[:, t - 1]
    # node 1 should respond to node 0's history, node 2 should not.
    assert abs(x[1, 1]) > 0
    assert np.allclose(x[2], 0.0)


def test_deterministic_graph():
    cfg = GraphConfig(family="sparse_random", n_observed=6)
    g1 = build_graph(cfg, np.random.default_rng(7))
    g2 = build_graph(cfg, np.random.default_rng(7))
    assert np.array_equal(g1.adjacency, g2.adjacency)
    assert np.allclose(g1.coefficients, g2.coefficients)

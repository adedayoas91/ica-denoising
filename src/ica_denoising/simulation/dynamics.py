"""Observation-independent neural dynamics generators.

Two generators are provided (Section 5.1):

* ``linear_var`` -- a linear Gaussian VAR (estimator sanity condition).
* ``rate_spike`` -- a nonnegative-rate / spike model (realism stress test).

Both support independent innovations, optional unobserved common drivers
(hidden confounding), reproducible burn-in removal, and an optional behavior
feedback edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import DynamicsConfig
from .graphs import GroundTruthGraph

__all__ = ["NeuralActivity", "simulate_dynamics"]


@dataclass(frozen=True)
class NeuralActivity:
    """Generated neural activity.

    Attributes:
        states: Latent neural states ``(n_nodes, T)`` (all nodes incl. hidden).
        observed: Observed-node states ``(n_observed, T)``.
        spikes: Optional spike counts ``(n_observed, T)`` for the rate model.
        confounder: Optional common-driver series ``(T,)`` or ``None``.
        feedback: Optional lagged feedback series ``(T,)`` or ``None``.
    """

    states: np.ndarray
    observed: np.ndarray
    spikes: Optional[np.ndarray]
    confounder: Optional[np.ndarray]
    feedback: Optional[np.ndarray] = None


def _innovations(
    rng: np.random.Generator, n: int, total: int, dist: str, scale: float
) -> np.ndarray:
    if dist == "gaussian":
        return rng.normal(0.0, scale, size=(n, total))
    if dist == "laplace":
        return rng.laplace(0.0, scale / np.sqrt(2.0), size=(n, total))
    raise ValueError(f"Unknown innovation distribution: {dist}")


def _common_driver(rng: np.random.Generator, total: int) -> np.ndarray:
    driver = np.zeros(total)
    for t in range(1, total):
        driver[t] = 0.95 * driver[t - 1] + rng.normal(0.0, 1.0)
    return driver


def simulate_dynamics(
    graph: GroundTruthGraph,
    cfg: DynamicsConfig,
    n_frames: int,
    rng: np.random.Generator,
) -> NeuralActivity:
    """Simulate neural dynamics on ``graph`` for ``n_frames`` retained frames."""

    coeffs = graph.coefficients  # [lag, target, source]
    p, n, _ = coeffs.shape
    total = n_frames + cfg.burn_in
    innov_scale = 1.0 / max(cfg.snr, 1e-6)
    innovations = _innovations(rng, n, total, cfg.innovation, innov_scale)

    confounder = None
    if cfg.hidden_confounding:
        confounder = _common_driver(rng, total)
        loadings = rng.uniform(0.2, 0.6, size=n)
        innovations = innovations + loadings[:, None] * confounder[None, :]

    feedback = None
    feedback_loadings = None
    if cfg.behavior_feedback:
        feedback = np.zeros(total)
        feedback_loadings = rng.uniform(-0.15, 0.15, size=n)

    states = np.zeros((n, total))
    for t in range(p, total):
        value = innovations[:, t].copy()
        for lag in range(p):
            value = value + coeffs[lag] @ states[:, t - lag - 1]
        if feedback is not None and feedback_loadings is not None:
            # Lagged closed-loop drive: current state never uses contemporaneous
            # behavior, but a low-dimensional proxy from previous observed state
            # can feed back into all nodes.
            proxy = float(np.tanh(np.mean(states[: graph.n_observed, t - 1])))
            feedback[t] = 0.9 * feedback[t - 1] + 0.1 * proxy
            value = value + feedback_loadings * feedback[t - 1]
        states[:, t] = value

    states = states[:, cfg.burn_in :]
    if confounder is not None:
        confounder = confounder[cfg.burn_in :]
    if feedback is not None:
        feedback = feedback[cfg.burn_in :]

    observed = states[: graph.n_observed]
    spikes = None
    if cfg.model == "rate_spike":
        rate = np.log1p(np.exp(observed))  # softplus -> nonnegative rate
        spikes = rng.poisson(np.clip(rate, 0.0, 50.0)).astype(float)
        observed = spikes
    elif cfg.model != "linear_var":
        raise ValueError(f"Unknown dynamics model: {cfg.model}")

    return NeuralActivity(
        states=states,
        observed=observed,
        spikes=spikes,
        confounder=confounder,
        feedback=feedback,
    )

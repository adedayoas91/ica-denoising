"""Behavior generation from latent neural state.

Behavior is generated from the latent neural state BEFORE calcium corruption so
the benchmark can distinguish behavior preservation from artifact tracking
(Section 5.1). The main behavior target uses a one-sided (causal) filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BehaviorConfig

__all__ = ["BehaviorSignals", "simulate_behavior"]


@dataclass(frozen=True)
class BehaviorSignals:
    """Generated behavior signals and their generative metadata."""

    vigor: np.ndarray  # (T,)
    angle: np.ndarray  # (T,)
    bout: np.ndarray  # (T,) binary
    parents: np.ndarray  # indices of neural parents
    behavior_lag: int
    bout_threshold: float


def _causal_smooth(signal: np.ndarray, window: int) -> np.ndarray:
    """One-sided moving-average filter (uses present and past only)."""

    if window <= 1:
        return signal.copy()
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, signal[0]), signal])
    return np.convolve(padded, kernel, mode="valid")


def simulate_behavior(
    states: np.ndarray,
    cfg: BehaviorConfig,
    rng: np.random.Generator,
) -> BehaviorSignals:
    """Generate vigor, angle, and bout signals from latent ``states``.

    Args:
        states: Latent neural states ``(n_nodes, T)`` (pre-calcium).
        cfg: Behavior configuration.
        rng: Random generator.
    """

    n, t = states.shape
    n_parents = min(cfg.n_parents, n)
    parents = rng.choice(n, size=n_parents, replace=False)
    weights = rng.uniform(0.5, 1.5, size=n_parents) * rng.choice((-1.0, 1.0), size=n_parents)

    lag = cfg.behavior_lag
    drive = np.zeros(t)
    parent_states = states[parents]
    for k, w in enumerate(weights):
        shifted = np.zeros(t)
        if lag < t:
            shifted[lag:] = parent_states[k, : t - lag]
        drive = drive + w * shifted

    drive = drive + rng.normal(0.0, cfg.noise * (np.std(drive) or 1.0), size=t)
    vigor = _causal_smooth(np.abs(drive), cfg.smooth_window)

    angle = _causal_smooth(np.tanh(drive), cfg.smooth_window)

    threshold = float(np.quantile(vigor, cfg.bout_quantile))
    bout = (vigor >= threshold).astype(int)

    return BehaviorSignals(
        vigor=vigor,
        angle=angle,
        bout=bout,
        parents=np.asarray(parents),
        behavior_lag=lag,
        bout_threshold=threshold,
    )

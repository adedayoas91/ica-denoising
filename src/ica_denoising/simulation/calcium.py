"""Calcium observation model.

Converts neural activity/spikes into clean fluorescence (the trace-recovery
oracle) before any artifact injection (Section 5.1):

    neural state or spikes
        -> causal one-sided exponential decay kernel
        -> cell-specific gain and baseline
        -> measurement noise
        -> clean fluorescence oracle
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .config import CalciumConfig

__all__ = ["CleanFluorescence", "simulate_calcium"]


@dataclass(frozen=True)
class CleanFluorescence:
    """Clean fluorescence oracle and its generative parameters."""

    fluorescence: np.ndarray  # (n_neurons, T)
    gains: np.ndarray
    baselines: np.ndarray
    decay_constants: np.ndarray


def _exponential_filter(signal: np.ndarray, decay: float) -> np.ndarray:
    """Apply a causal one-sided exponential (AR(1)) kernel along time."""

    alpha = float(np.exp(-1.0 / max(decay, 1e-6)))
    # y[t] = alpha * y[t-1] + x[t]
    return lfilter([1.0], [1.0, -alpha], signal, axis=-1)


def simulate_calcium(
    activity: np.ndarray,
    cfg: CalciumConfig,
    sample_rate_hz: float,
    rng: np.random.Generator,
) -> CleanFluorescence:
    """Generate the clean fluorescence oracle from neural ``activity``.

    Args:
        activity: Observed neural activity / spikes ``(n_neurons, T)``.
        cfg: Calcium configuration.
        sample_rate_hz: Acquisition sampling rate in Hz.
        rng: Random generator for heterogeneous parameters and noise.
    """

    n, t = activity.shape
    if not cfg.enabled:
        clean = activity.copy()
        return CleanFluorescence(
            fluorescence=clean,
            gains=np.ones(n),
            baselines=np.zeros(n),
            decay_constants=np.zeros(n),
        )

    decay_frames = cfg.decay_seconds * sample_rate_hz
    jitter = rng.uniform(1.0 - cfg.decay_jitter, 1.0 + cfg.decay_jitter, size=n)
    decay_constants = np.clip(decay_frames * jitter, 1e-3, None)

    convolved = np.zeros_like(activity, dtype=float)
    for i in range(n):
        convolved[i] = _exponential_filter(activity[i], decay_constants[i])

    gains = rng.uniform(cfg.gain_low, cfg.gain_high, size=n)
    baselines = rng.uniform(0.0, 0.1, size=n)
    fluorescence = gains[:, None] * convolved + baselines[:, None]

    if cfg.poisson_noise:
        scaled = np.clip(fluorescence, 0.0, None)
        fluorescence = fluorescence + rng.normal(0.0, np.sqrt(scaled + 1e-9))

    signal_std = float(np.std(fluorescence)) or 1.0
    fluorescence = fluorescence + rng.normal(
        0.0, cfg.measurement_noise * signal_std, size=fluorescence.shape
    )

    if cfg.saturation > 0.0:
        fluorescence = cfg.saturation * np.tanh(fluorescence / cfg.saturation)

    return CleanFluorescence(
        fluorescence=fluorescence,
        gains=gains,
        baselines=baselines,
        decay_constants=decay_constants,
    )

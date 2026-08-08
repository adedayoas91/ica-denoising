"""Artifact model for the simulation benchmark.

Artifacts are added AFTER clean fluorescence is generated and preserve neuron
identity (Section 5.1). Strength is set by an achieved artifact-to-signal ratio
rather than an arbitrary coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .config import ArtifactConfig

__all__ = ["ArtifactBundle", "inject_artifacts"]


@dataclass(frozen=True)
class ArtifactBundle:
    """Injected artifacts and the corrupted traces."""

    corrupted: np.ndarray  # (n_neurons, T)
    artifact: np.ndarray  # total injected artifact (n_neurons, T)
    sources: dict[str, np.ndarray]  # per-type source time courses
    loadings: dict[str, np.ndarray]
    achieved_asr: float


def _smooth_drift(rng: np.random.Generator, t: int, timescale: float) -> np.ndarray:
    raw = rng.normal(0.0, 1.0, size=t)
    alpha = float(np.exp(-1.0 / max(timescale, 1e-6)))
    return lfilter([1.0 - alpha], [1.0, -alpha], raw)


def _motion_impulses(
    rng: np.random.Generator, t: int, rate: float, decay: float
) -> np.ndarray:
    impulses = (rng.random(t) < rate).astype(float) * rng.normal(0.0, 1.0, size=t)
    alpha = float(np.exp(-1.0 / max(decay, 1e-6)))
    return lfilter([1.0], [1.0, -alpha], impulses)


def _neural_band(rng: np.random.Generator, t: int) -> np.ndarray:
    raw = rng.normal(0.0, 1.0, size=t)
    # mild AR(2) to overlap the calcium band
    return lfilter([1.0], [1.0, -0.6, -0.2], raw)


def _build_sources(
    cfg: ArtifactConfig, n: int, t: int, rng: np.random.Generator
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    sources: dict[str, np.ndarray] = {}
    loadings: dict[str, np.ndarray] = {}
    for kind in cfg.types:
        if kind == "drift":
            sources[kind] = _smooth_drift(rng, t, cfg.drift_timescale)
            loadings[kind] = rng.uniform(0.3, 1.0, size=n)
        elif kind == "motion":
            sources[kind] = _motion_impulses(rng, t, cfg.motion_rate, cfg.motion_decay)
            loadings[kind] = rng.uniform(0.5, 1.0, size=n)
        elif kind == "neuropil":
            sources[kind] = _smooth_drift(rng, t, cfg.drift_timescale / 4)
            loadings[kind] = rng.uniform(0.6, 1.0, size=n)
        elif kind == "neural_band":
            sources[kind] = _neural_band(rng, t)
            loadings[kind] = rng.uniform(0.3, 0.8, size=n)
        elif kind == "crosstalk":
            continue  # handled separately as a mixing matrix
        else:
            raise ValueError(f"Unknown artifact type: {kind}")
    return sources, loadings


def inject_artifacts(
    clean: np.ndarray,
    cfg: ArtifactConfig,
    rng: np.random.Generator,
) -> ArtifactBundle:
    """Inject configured artifacts into ``clean`` fluorescence.

    Args:
        clean: Clean fluorescence oracle ``(n_neurons, T)``.
        cfg: Artifact configuration.
        rng: Random generator.

    Returns:
        An :class:`ArtifactBundle` with corrupted traces and saved sources.
    """

    n, t = clean.shape
    if not cfg.types:
        return ArtifactBundle(
            corrupted=clean.copy(),
            artifact=np.zeros_like(clean),
            sources={},
            loadings={},
            achieved_asr=0.0,
        )

    sources, loadings = _build_sources(cfg, n, t, rng)
    artifact = np.zeros_like(clean, dtype=float)
    for kind, source in sources.items():
        artifact = artifact + loadings[kind][:, None] * source[None, :] * cfg.severity

    if "crosstalk" in cfg.types:
        mixing = np.eye(n) + cfg.crosstalk_strength * rng.normal(0.0, 1.0, size=(n, n))
        np.fill_diagonal(mixing, 1.0)
        crosstalk = mixing @ clean - clean
        artifact = artifact + crosstalk
        sources["crosstalk"] = np.array([0.0])
        loadings["crosstalk"] = mixing.diagonal()

    # Scale to the requested artifact-to-signal ratio (power based).
    signal_power = float(np.mean(clean**2)) or 1.0
    artifact_power = float(np.mean(artifact**2))
    if artifact_power > 0 and cfg.artifact_to_signal > 0:
        scale = np.sqrt(cfg.artifact_to_signal * signal_power / artifact_power)
        artifact = artifact * scale

    corrupted = clean + artifact

    if cfg.nonfinite_fraction > 0:
        n_bad = int(cfg.nonfinite_fraction * t)
        if n_bad > 0:
            bad_frames = rng.choice(t, size=n_bad, replace=False)
            corrupted[:, bad_frames] = np.nan

    achieved = float(np.mean(artifact**2) / signal_power) if signal_power else 0.0
    return ArtifactBundle(
        corrupted=corrupted,
        artifact=artifact,
        sources=sources,
        loadings=loadings,
        achieved_asr=achieved,
    )

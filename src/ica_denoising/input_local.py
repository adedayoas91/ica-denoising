"""Fully input-local evaluation transforms (Section 5.3).

This module replaces the historical "build targets before evaluation" boundary
with transforms that are fit on *outer-training frames only* and then applied to
every frame. Each fitted transform records an audit entry describing what was
fit and on which fold/scope, so the strict runner can prove that no held-out or
future behavior sample influenced any fitted parameter.

Provided transforms:

* :class:`PerNeuronImputer` - fold-local per-neuron mean imputer for raw traces.
* :func:`causal_moving_average` - one-sided (trailing) vigor smoother that never
  reads future samples and resets at block boundaries.
* :func:`fit_bout_threshold` - bout threshold fitted on training targets only.
* :func:`build_input_local_targets` - turns a raw tail-angle signal into
  :class:`~ica_denoising.behavior_decoding.BehaviorTargets` using only training
  frames to fit the smoother state and bout threshold.

All of these are pure functions / frozen dataclasses with full audit support.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Sequence

import numpy as np

from ica_denoising.behavior_decoding import (
    BehaviorTargets,
    bin_signal_to_frames,
)

logger = logging.getLogger(__name__)

__all__ = [
    "InputLocalConfig",
    "PerNeuronImputer",
    "InputTransformAudit",
    "causal_moving_average",
    "fit_per_neuron_imputer",
    "fit_bout_threshold",
    "build_input_local_targets",
    "contiguous_segments",
]


@dataclass(frozen=True)
class InputLocalConfig:
    """Configuration for input-local target construction.

    Attributes:
        bout_quantile: Quantile of training vigor used as the bout threshold.
        vigor_smooth_window: Window for the one-sided causal vigor smoother.
        impute_strategy: Per-neuron imputation strategy (only ``"mean"`` for now).
    """

    bout_quantile: float = 0.75
    vigor_smooth_window: int = 3
    impute_strategy: str = "mean"


@dataclass(frozen=True)
class InputTransformAudit:
    """Audit record for a single fitted input transform.

    Attributes:
        transform: Transform name, e.g. ``per_neuron_imputer``.
        fit_scope: Scope the transform was fit on, e.g. ``training_frames_only``.
        fold: Outer fold id, or ``-1`` when fold-agnostic.
        n_fit_frames: Number of frames used to fit the transform.
        uses_future_samples: Whether the transform reads future behavior samples.
        detail: Free-form description of the fitted parameters.
    """

    transform: str
    fit_scope: str
    fold: int
    n_fit_frames: int
    uses_future_samples: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        """Return a plain-dict view suitable for a DataFrame row."""
        return {
            "transform": self.transform,
            "fit_scope": self.fit_scope,
            "fold": int(self.fold),
            "n_fit_frames": int(self.n_fit_frames),
            "uses_future_samples": bool(self.uses_future_samples),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PerNeuronImputer:
    """Per-neuron mean imputer fitted on training frames only.

    Attributes:
        means: Per-neuron fitted means, shape ``(n_neurons,)``.
        fold: Outer fold id this imputer was fit on.
        n_fit_frames: Number of training frames used.
    """

    means: np.ndarray
    fold: int = -1
    n_fit_frames: int = 0

    def transform(self, traces: np.ndarray) -> np.ndarray:
        """Replace non-finite entries with the fitted per-neuron means.

        Args:
            traces: Raw traces, shape ``(n_frames, n_neurons)``.

        Returns:
            A copy of ``traces`` with non-finite entries imputed.
        """
        values = np.array(traces, dtype=float, copy=True)
        if values.ndim != 2:
            raise ValueError("traces must be 2D (time x neurons).")
        if values.shape[1] != self.means.size:
            raise ValueError(
                f"Expected {self.means.size} neurons, got {values.shape[1]}."
            )
        nonfinite = ~np.isfinite(values)
        if nonfinite.any():
            fill = np.broadcast_to(self.means, values.shape)
            values[nonfinite] = fill[nonfinite]
        return values

    def audit(self) -> InputTransformAudit:
        """Return the audit record for this fitted imputer."""
        return InputTransformAudit(
            transform="per_neuron_imputer",
            fit_scope="training_frames_only",
            fold=self.fold,
            n_fit_frames=self.n_fit_frames,
            uses_future_samples=False,
            detail=f"per_neuron_mean n_neurons={self.means.size}",
        )


def contiguous_segments(indices: Sequence[int]) -> tuple[np.ndarray, ...]:
    """Split sorted unique indices into contiguous runs."""
    arr = np.asarray(indices, dtype=int)
    if arr.size == 0:
        return ()
    arr = np.unique(arr)
    split_points = np.flatnonzero(np.diff(arr) > 1) + 1
    return tuple(np.asarray(seg, dtype=int) for seg in np.split(arr, split_points))


def fit_per_neuron_imputer(
    traces: np.ndarray,
    train_idx: Sequence[int],
    *,
    fold: int = -1,
    strategy: str = "mean",
) -> PerNeuronImputer:
    """Fit a per-neuron imputer using training frames only.

    Args:
        traces: Raw traces (may contain non-finite values), time x neurons.
        train_idx: Outer-training frame indices used to fit the means.
        fold: Outer fold id recorded in the audit.
        strategy: Only ``"mean"`` is currently supported.

    Returns:
        A fitted :class:`PerNeuronImputer`.
    """
    if strategy != "mean":
        raise ValueError(f"Unsupported impute strategy: {strategy!r}.")
    values = np.asarray(traces, dtype=float)
    if values.ndim != 2:
        raise ValueError("traces must be 2D (time x neurons).")
    idx = np.asarray(train_idx, dtype=int)
    if idx.size == 0:
        raise ValueError("train_idx must contain at least one frame.")
    if idx.min() < 0 or idx.max() >= values.shape[0]:
        raise ValueError("train_idx is out of range.")
    train = values[idx]
    means = np.empty(values.shape[1], dtype=float)
    for neuron in range(values.shape[1]):
        column = train[:, neuron]
        finite = column[np.isfinite(column)]
        means[neuron] = float(finite.mean()) if finite.size else 0.0
    return PerNeuronImputer(means=means, fold=int(fold), n_fit_frames=int(idx.size))


def causal_moving_average(
    values: np.ndarray,
    window: int,
    *,
    segments: Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    """Apply a one-sided (trailing) moving average that never reads the future.

    For window ``w`` the output at frame ``i`` is the mean of frames
    ``[max(seg_start, i - w + 1) .. i]`` within the same contiguous segment, so
    the filter state resets at block boundaries and uses only past + current
    samples.

    Args:
        values: 1D signal.
        window: Trailing window length (``<= 1`` returns a copy).
        segments: Optional iterable of index groups defining block boundaries.
            When ``None`` the whole signal is treated as one block.

    Returns:
        The causally smoothed signal, same shape as ``values``.
    """
    signal = np.asarray(values, dtype=float).reshape(-1)
    out = signal.copy()
    if window <= 1:
        return out
    window = int(window)
    if segments is None:
        groups: tuple[np.ndarray, ...] = (np.arange(signal.size, dtype=int),)
    else:
        groups = tuple(np.asarray(seg, dtype=int) for seg in segments if len(seg))
    for group in groups:
        block = signal[group]
        smoothed = np.empty_like(block)
        cumulative = np.cumsum(block)
        for position in range(block.size):
            start = max(0, position - window + 1)
            total = cumulative[position] - (cumulative[start - 1] if start > 0 else 0.0)
            smoothed[position] = total / (position - start + 1)
        out[group] = smoothed
    return out


def fit_bout_threshold(
    vigor: np.ndarray,
    train_idx: Sequence[int],
    *,
    bout_quantile: float,
) -> float:
    """Fit a bout threshold from training vigor only.

    Args:
        vigor: Per-frame vigor signal.
        train_idx: Outer-training frame indices.
        bout_quantile: Quantile in ``[0, 1]``.

    Returns:
        The fitted threshold value.
    """
    if not 0.0 <= float(bout_quantile) <= 1.0:
        raise ValueError("bout_quantile must be between 0 and 1.")
    values = np.asarray(vigor, dtype=float).reshape(-1)
    idx = np.asarray(train_idx, dtype=int)
    if idx.size == 0:
        raise ValueError("train_idx must contain at least one frame.")
    return float(np.quantile(values[idx], float(bout_quantile)))


def build_input_local_targets(
    tail_angle: np.ndarray,
    *,
    n_frames: int,
    train_idx: Sequence[int],
    config: InputLocalConfig,
    fold: int = -1,
    block_segments: Sequence[Sequence[int]] | None = None,
) -> tuple[BehaviorTargets, list[InputTransformAudit]]:
    """Build behavior targets from a raw tail-angle signal, input-locally.

    The frame-binning uses only each frame's own time interval (never future
    frames), vigor smoothing is one-sided causal with block-boundary resets, and
    the bout threshold is fit on training frames only.

    Args:
        tail_angle: Raw high-rate tail-angle signal.
        n_frames: Number of calcium frames to align to.
        train_idx: Outer-training frame indices used to fit the bout threshold.
        config: Input-local configuration.
        fold: Outer fold id recorded in audits.
        block_segments: Optional contiguous index groups for smoother resets.

    Returns:
        A tuple ``(targets, audits)`` where ``audits`` lists the fitted-transform
        audit records (smoother state and bout threshold).
    """
    tail_angle = np.asarray(tail_angle, dtype=float).reshape(-1)
    if tail_angle.size < n_frames:
        raise ValueError(
            f"Tail angle length {tail_angle.size} is shorter than {n_frames} frames."
        )
    angle = bin_signal_to_frames(tail_angle, n_frames, reducer="mean")
    tail_velocity = np.diff(tail_angle, prepend=tail_angle[0])
    raw_vigor = bin_signal_to_frames(tail_velocity, n_frames, reducer="rms")
    vigor = causal_moving_average(
        raw_vigor,
        config.vigor_smooth_window,
        segments=block_segments,
    )
    threshold = fit_bout_threshold(
        vigor,
        train_idx,
        bout_quantile=config.bout_quantile,
    )
    bout_state = (vigor >= threshold).astype(int)
    idx = np.asarray(train_idx, dtype=int)
    audits = [
        InputTransformAudit(
            transform="causal_vigor_smoother",
            fit_scope="stateless_causal",
            fold=int(fold),
            n_fit_frames=int(n_frames),
            uses_future_samples=False,
            detail=f"trailing_window={int(config.vigor_smooth_window)}",
        ),
        InputTransformAudit(
            transform="bout_threshold",
            fit_scope="training_frames_only",
            fold=int(fold),
            n_fit_frames=int(idx.size),
            uses_future_samples=False,
            detail=f"quantile={float(config.bout_quantile)} threshold={threshold:.6g}",
        ),
    ]
    targets = BehaviorTargets(
        angle=angle,
        vigor=vigor,
        bout_state=bout_state,
        bout_threshold=threshold,
        bout_quantile=float(config.bout_quantile),
        smooth_window=int(config.vigor_smooth_window),
        target_variant="input_local",
        raw_vigor=raw_vigor,
    )
    return targets, audits

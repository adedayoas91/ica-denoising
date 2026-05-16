from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.signal import welch


@dataclass(frozen=True)
class ICFeatureConfig:
    """Parameters for IC-level artifact and protection evidence."""

    sample_rate_hz: float
    nperseg: int = 250
    noverlap: int = 125
    low_freq_max_hz: float = 0.5
    high_freq_min_hz: float = 3.0
    robust_peak_z: float = 6.0
    strong_loading_z: float = 3.0
    window_count: int = 4


def compute_ic_features(
    ic_comps: np.ndarray,
    mixing: np.ndarray | None,
    config: ICFeatureConfig,
    *,
    behavior_targets: Mapping[str, np.ndarray] | object | None = None,
    roi_positions: np.ndarray | None = None,
    method: str | None = None,
) -> pd.DataFrame:
    """Return one evidence row per independent component.

    ``ic_comps`` must be frames x components. ``mixing`` is optional, but when
    present it should be neurons x components and enables spatial/loading
    evidence. Behavior targets may be a mapping or an object with attributes
    such as ``angle``, ``vigor``, and ``bout_state``.
    """

    ics = _as_2d_float(ic_comps, "ic_comps")
    n_frames, n_components = ics.shape
    mixing_arr = None if mixing is None else _validate_mixing(mixing, n_components)
    positions = None if roi_positions is None else _validate_positions(roi_positions, mixing_arr)
    targets = _normalize_behavior_targets(behavior_targets, n_frames)

    rows: list[dict[str, float | int | str | None]] = []
    for component in range(n_components):
        trace = ics[:, component]
        row: dict[str, float | int | str | None] = {
            "method": method,
            "component": int(component),
        }
        row.update(_temporal_features(trace, config))
        row.update(_spectral_features(trace, config))
        if mixing_arr is not None:
            row.update(_mixing_features(mixing_arr[:, component], positions, config))
        if targets:
            row.update(_behavior_features(trace, targets))
        rows.append(row)

    return pd.DataFrame(rows)


def _temporal_features(trace: np.ndarray, config: ICFeatureConfig) -> dict[str, float]:
    trace = np.asarray(trace, dtype=float)
    centered = trace - np.mean(trace)
    std = _safe_std(centered)
    robust_z = _robust_zscore(trace)
    abs_robust_z = np.abs(robust_z)
    lag1 = _autocorr(centered, 1)
    lag5 = _autocorr(centered, 5)
    lag10 = _autocorr(centered, 10)
    excess_kurtosis = _excess_kurtosis(centered)

    return {
        "ic_variance": float(np.var(trace)),
        "ic_std": float(std),
        "excess_kurtosis": float(excess_kurtosis),
        "abs_excess_kurtosis": float(abs(excess_kurtosis)),
        "robust_peak_density": float(np.mean(abs_robust_z >= config.robust_peak_z)),
        "max_abs_robust_z": float(np.max(abs_robust_z)) if trace.size else 0.0,
        "lag1_autocorr": float(lag1),
        "lag5_autocorr": float(lag5),
        "lag10_autocorr": float(lag10),
        "window_variance_ratio": _window_variance_ratio(trace, config.window_count),
        "zero_crossing_rate": _zero_crossing_rate(centered),
    }


def _spectral_features(trace: np.ndarray, config: ICFeatureConfig) -> dict[str, float]:
    nperseg = min(max(int(config.nperseg), 2), trace.size)
    noverlap = min(max(int(config.noverlap), 0), nperseg - 1)
    freqs, power = welch(
        trace,
        fs=float(config.sample_rate_hz),
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=True,
    )
    total_power = float(np.sum(power))
    if total_power <= 0 or not np.isfinite(total_power):
        total_power = np.finfo(float).eps
    density = power / total_power
    low_mask = freqs <= config.low_freq_max_hz
    high_mask = freqs >= config.high_freq_min_hz
    peak_idx = int(np.argmax(power)) if power.size else 0
    entropy = -np.sum(density * np.log2(np.clip(density, np.finfo(float).eps, None)))
    entropy_norm = entropy / np.log2(max(power.size, 2))

    return {
        "spectral_total_power": float(total_power),
        "peak_freq_hz": float(freqs[peak_idx]) if freqs.size else 0.0,
        "low_freq_power_ratio": float(np.sum(power[low_mask]) / total_power),
        "high_freq_power_ratio": float(np.sum(power[high_mask]) / total_power),
        "spectral_entropy": float(entropy_norm),
        "narrowband_ratio": float(np.max(power) / max(np.median(power), np.finfo(float).eps)),
    }


def _mixing_features(
    loadings: np.ndarray,
    roi_positions: np.ndarray | None,
    config: ICFeatureConfig,
) -> dict[str, float]:
    abs_loadings = np.abs(np.asarray(loadings, dtype=float))
    n = abs_loadings.size
    l1 = float(np.sum(abs_loadings))
    l2 = float(np.linalg.norm(abs_loadings))
    median = float(np.median(abs_loadings))
    mad = _mad(abs_loadings)
    strong_threshold = median + config.strong_loading_z * mad
    result = {
        "loading_l1": l1,
        "loading_l2": l2,
        "loading_hoyer_sparsity": _hoyer_sparsity(abs_loadings),
        "max_median_loading_ratio": float(np.max(abs_loadings) / max(median, np.finfo(float).eps)),
        "strong_loading_fraction": float(np.mean(abs_loadings >= strong_threshold)) if n else 0.0,
    }
    if roi_positions is not None and l1 > 0:
        weights = abs_loadings / l1
        centroid = weights @ roi_positions
        distances = np.linalg.norm(roi_positions - centroid, axis=1)
        result["weighted_spatial_extent"] = float(np.sqrt(np.sum(weights * distances**2)))
    return result


def _behavior_features(trace: np.ndarray, targets: Mapping[str, np.ndarray]) -> dict[str, float]:
    result: dict[str, float] = {}
    correlations: list[float] = []
    for name, target in targets.items():
        corr = _pearson(trace, target)
        result[f"behavior_corr_{name}"] = float(corr)
        if np.isfinite(corr):
            correlations.append(abs(float(corr)))
    result["max_abs_behavior_corr"] = float(max(correlations)) if correlations else 0.0
    return result


def _normalize_behavior_targets(
    behavior_targets: Mapping[str, np.ndarray] | object | None,
    n_frames: int,
) -> dict[str, np.ndarray]:
    if behavior_targets is None:
        return {}
    if isinstance(behavior_targets, Mapping):
        raw = behavior_targets
    else:
        raw = {
            name: getattr(behavior_targets, name)
            for name in ("angle", "vigor", "bout_state")
            if hasattr(behavior_targets, name)
        }
    return {
        str(name): _align_1d_target(np.asarray(values, dtype=float).reshape(-1), n_frames)
        for name, values in raw.items()
    }


def _align_1d_target(values: np.ndarray, n_frames: int) -> np.ndarray:
    if values.size == n_frames:
        return values
    if values.size == 0:
        return np.zeros(n_frames, dtype=float)
    old_x = np.linspace(0.0, 1.0, values.size)
    new_x = np.linspace(0.0, 1.0, n_frames)
    return np.interp(new_x, old_x, values)


def _as_2d_float(arr: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {out.shape}.")
    return out


def _validate_mixing(mixing: np.ndarray, n_components: int) -> np.ndarray:
    out = _as_2d_float(mixing, "mixing")
    if out.shape[1] != n_components:
        raise ValueError(
            f"mixing must have {n_components} columns to match ICs, got {out.shape}."
        )
    return out


def _validate_positions(positions: np.ndarray, mixing: np.ndarray | None) -> np.ndarray:
    out = _as_2d_float(positions, "roi_positions")
    if mixing is not None and out.shape[0] != mixing.shape[0]:
        raise ValueError(
            "roi_positions must have one row per neuron/loading; "
            f"got {out.shape[0]} positions for {mixing.shape[0]} loadings."
        )
    return out


def _safe_std(values: np.ndarray) -> float:
    std = float(np.std(values))
    return std if std > 0 and np.isfinite(std) else np.finfo(float).eps


def _mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = float(np.median(np.abs(values - median)))
    return 1.4826 * mad if mad > 0 and np.isfinite(mad) else np.finfo(float).eps


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - np.median(values)) / _mad(values)


def _autocorr(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or values.size <= lag:
        return 0.0
    left = values[:-lag]
    right = values[lag:]
    return _pearson(left, right)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"Pearson inputs must match lengths, got {x.size} and {y.size}.")
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std == 0 or y_std == 0 or not np.isfinite(x_std) or not np.isfinite(y_std):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _excess_kurtosis(values: np.ndarray) -> float:
    std = _safe_std(values)
    z = values / std
    return float(np.mean(z**4) - 3.0)


def _window_variance_ratio(values: np.ndarray, window_count: int) -> float:
    windows = np.array_split(values, max(int(window_count), 1))
    variances = np.array([np.var(window) for window in windows if window.size], dtype=float)
    if variances.size == 0:
        return 1.0
    return float(np.max(variances) / max(np.min(variances), np.finfo(float).eps))


def _zero_crossing_rate(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    signs = np.signbit(values)
    return float(np.mean(signs[1:] != signs[:-1]))


def _hoyer_sparsity(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    n = values.size
    if n <= 1:
        return 0.0
    l1 = np.sum(np.abs(values))
    l2 = np.linalg.norm(values)
    if l2 == 0:
        return 0.0
    return float((np.sqrt(n) - (l1 / l2)) / (np.sqrt(n) - 1.0))

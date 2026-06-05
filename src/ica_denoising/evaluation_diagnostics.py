from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from ica_denoising.bss_notebook import DatasetSpec


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def array_file_summary(path: Path | None) -> dict[str, object]:
    if path is None:
        return {
            "path": None,
            "exists": False,
            "sha256": None,
            "shape": None,
            "dtype": None,
            "nonfinite_count": None,
        }
    path = Path(path)
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "sha256": None,
            "shape": None,
            "dtype": None,
            "nonfinite_count": None,
        }
    arr = np.load(path, allow_pickle=False, mmap_mode="r")
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "nonfinite_count": int(np.size(arr) - np.count_nonzero(np.isfinite(arr))),
    }


def build_provenance_manifest(specs: Iterable[DatasetSpec]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs:
        trace = array_file_summary(spec.trace_path)
        behavior = array_file_summary(spec.tail_angle_path)
        rows.append(
            {
                "dataset_key": spec.key,
                "recording_id": spec.recording_id,
                "fish_id": spec.fish_id,
                "run_id": spec.run_id,
                "modality": spec.modality,
                "sample_rate_hz": spec.sample_rate_hz,
                "trace_path": trace["path"],
                "trace_exists": trace["exists"],
                "trace_sha256": trace["sha256"],
                "trace_shape": trace["shape"],
                "trace_dtype": trace["dtype"],
                "trace_nonfinite_count": trace["nonfinite_count"],
                "behavior_path": behavior["path"],
                "behavior_exists": behavior["exists"],
                "behavior_sha256": behavior["sha256"],
                "behavior_shape": behavior["shape"],
                "behavior_dtype": behavior["dtype"],
                "behavior_nonfinite_count": behavior["nonfinite_count"],
            }
        )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        return manifest
    manifest["trace_duplicate_count"] = 0
    trace_valid = manifest["trace_sha256"].notna()
    manifest.loc[trace_valid, "trace_duplicate_count"] = (
        manifest.loc[trace_valid].groupby("trace_sha256")["dataset_key"].transform("size")
    )
    manifest["behavior_duplicate_count"] = 0
    behavior_valid = manifest["behavior_sha256"].notna()
    manifest.loc[behavior_valid, "behavior_duplicate_count"] = (
        manifest.loc[behavior_valid]
        .groupby("behavior_sha256")["dataset_key"]
        .transform("size")
    )
    manifest["provenance_warning"] = ""
    manifest.loc[manifest["trace_duplicate_count"] > 1, "provenance_warning"] = (
        "duplicate trace hash"
    )
    both = (manifest["trace_duplicate_count"] > 1) & (
        manifest["behavior_duplicate_count"] > 1
    )
    manifest.loc[both, "provenance_warning"] = "duplicate trace and behavior hashes"
    return manifest


def write_json_manifest(path: Path, payload: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _json_ready(value: object) -> object:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def compute_bpi_ablation(
    component_scores: pd.DataFrame,
    *,
    denominator_epsilon: float = 1e-8,
    custom_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Compute BPI variants from explicit component, raw, and null scores.

    Required columns are ``recording``, ``method``, ``component``, ``scope``,
    ``score``, ``raw_score``, and ``null_score``. ``scope`` should distinguish
    within-representation and transfer components.
    """
    required = {
        "recording",
        "method",
        "component",
        "scope",
        "score",
        "raw_score",
        "null_score",
    }
    missing = sorted(required - set(component_scores.columns))
    if missing:
        raise ValueError(f"component_scores is missing required columns: {missing}")

    rows: list[dict[str, object]] = []
    for (recording, method), group in component_scores.groupby(
        ["recording", "method"], dropna=False
    ):
        for subset in ("combined", "within", "transfer"):
            selected = group if subset == "combined" else group[group["scope"] == subset]
            if selected.empty:
                continue
            for normalization in ("raw_null", "raw_only", "none"):
                values = _normalize_bpi_components(
                    selected,
                    normalization=normalization,
                    denominator_epsilon=denominator_epsilon,
                )
                for weighting in ("equal", "custom"):
                    if weighting == "custom" and not custom_weights:
                        continue
                    weights = np.ones(len(selected), dtype=float)
                    if weighting == "custom":
                        weights = np.array(
                            [float(custom_weights.get(str(name), 0.0)) for name in selected["component"]],
                            dtype=float,
                        )
                    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
                    score = np.nan
                    if np.any(valid):
                        score = float(np.average(values[valid], weights=weights[valid]))
                    rows.append(
                        {
                            "recording": recording,
                            "method": method,
                            "component_subset": subset,
                            "normalization": normalization,
                            "weighting": weighting,
                            "bpi": score,
                            "n_valid_components": int(np.count_nonzero(valid)),
                        }
                    )
    return pd.DataFrame(rows)


def _normalize_bpi_components(
    scores: pd.DataFrame,
    *,
    normalization: str,
    denominator_epsilon: float,
) -> np.ndarray:
    score = scores["score"].to_numpy(dtype=float)
    raw = scores["raw_score"].to_numpy(dtype=float)
    null = scores["null_score"].to_numpy(dtype=float)
    if normalization == "none":
        return score
    if normalization == "raw_only":
        denominator = raw
        numerator = score
    elif normalization == "raw_null":
        denominator = raw - null
        numerator = score - null
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    values = np.full(score.shape, np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (
        np.abs(denominator) > denominator_epsilon
    )
    values[valid] = 100.0 * numerator[valid] / denominator[valid]
    return values


def cluster_stability_table(
    assignments: Mapping[str, Sequence[int]],
    *,
    retained_sets: Mapping[str, Sequence[int]] | None = None,
    component_ranks: Mapping[str, Sequence[float]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left, right in combinations(sorted(assignments), 2):
        left_labels = np.asarray(assignments[left])
        right_labels = np.asarray(assignments[right])
        if left_labels.shape != right_labels.shape:
            raise ValueError("All cluster assignments must have matching shapes.")
        row: dict[str, object] = {
            "left": left,
            "right": right,
            "adjusted_rand_index": float(adjusted_rand_score(left_labels, right_labels)),
            "retained_jaccard": np.nan,
            "rank_spearman": np.nan,
        }
        if retained_sets is not None and left in retained_sets and right in retained_sets:
            left_set = set(int(value) for value in retained_sets[left])
            right_set = set(int(value) for value in retained_sets[right])
            union = left_set | right_set
            row["retained_jaccard"] = (
                float(len(left_set & right_set) / len(union)) if union else 1.0
            )
        if component_ranks is not None and left in component_ranks and right in component_ranks:
            corr = spearmanr(component_ranks[left], component_ranks[right]).correlation
            row["rank_spearman"] = float(corr) if np.isfinite(corr) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def temporal_dependence_diagnostics(
    raw: np.ndarray,
    variants: Mapping[str, np.ndarray],
    *,
    lags: Iterable[int],
    sample_rate_hz: float,
) -> pd.DataFrame:
    raw = _validate_time_by_features(raw, "raw")
    raw_standardized = _standardize_columns(raw)
    rows: list[dict[str, object]] = []
    for name, traces in variants.items():
        traces = _validate_time_by_features(traces, name)
        if traces.shape != raw.shape:
            raise ValueError(f"{name} has shape {traces.shape}; expected {raw.shape}.")
        standardized = _standardize_columns(traces)
        for lag_value in lags:
            lag = int(lag_value)
            if lag < 1 or lag >= raw.shape[0]:
                raise ValueError(f"lag must be in [1, {raw.shape[0] - 1}], got {lag}.")
            raw_acf = _column_lag_correlation(raw_standardized, lag)
            variant_acf = _column_lag_correlation(standardized, lag)
            raw_xcf = _lagged_cross_correlation(raw_standardized, lag)
            variant_xcf = _lagged_cross_correlation(standardized, lag)
            denominator = np.linalg.norm(raw_xcf, ord="fro")
            rows.append(
                {
                    "variant": name,
                    "lag_frames": lag,
                    "lag_seconds": float(lag / sample_rate_hz),
                    "acf_median_abs_change": float(
                        np.nanmedian(np.abs(variant_acf - raw_acf))
                    ),
                    "xcf_relative_frobenius_change": (
                        float(np.linalg.norm(variant_xcf - raw_xcf, ord="fro") / denominator)
                        if denominator > 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    if values.size > 20:
        raise ValueError("Exact sign-flip enumeration is limited to 20 paired units.")
    observed = abs(float(np.mean(values)))
    permuted = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in product((-1.0, 1.0), repeat=values.size)
    ]
    return float(np.mean(np.asarray(permuted) >= observed - np.finfo(float).eps))


def paired_block_bootstrap_interval(
    differences: Sequence[float],
    *,
    block_size: int,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 0,
) -> dict[str, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    blocks = _segmented_blocks(values, np.zeros(values.size, dtype=int), block_size)
    if not blocks:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    rng = np.random.default_rng(random_state)
    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        resampled = np.concatenate([blocks[item] for item in chosen])[: values.size]
        samples[index] = np.mean(resampled)
    alpha = 1.0 - confidence
    return {
        "estimate": float(np.mean(values)),
        "ci_low": float(np.quantile(samples, alpha / 2.0)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha / 2.0)),
    }


def paired_block_sign_permutation_pvalue(
    differences: Sequence[float],
    *,
    block_size: int,
    n_permutations: int = 10000,
    random_state: int = 0,
) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    blocks = _segmented_blocks(values, np.zeros(values.size, dtype=int), block_size)
    if not blocks:
        return np.nan
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")
    observed = abs(float(np.mean(values)))
    rng = np.random.default_rng(random_state)
    exceedances = 0
    for _ in range(n_permutations):
        signs = rng.choice((-1.0, 1.0), size=len(blocks))
        permuted = np.concatenate(
            [sign * block for sign, block in zip(signs, blocks)]
        )
        exceedances += abs(float(np.mean(permuted))) >= observed
    return float((exceedances + 1) / (n_permutations + 1))


def paired_segmented_block_bootstrap_interval(
    differences: Sequence[float],
    segment_ids: Sequence[object],
    *,
    block_size: int,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 0,
) -> dict[str, float]:
    values = np.asarray(differences, dtype=float)
    segments = np.asarray(segment_ids)
    valid = np.isfinite(values)
    values = values[valid]
    segments = segments[valid]
    blocks = _segmented_blocks(values, segments, block_size)
    if not blocks:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    rng = np.random.default_rng(random_state)
    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        samples[index] = np.mean(np.concatenate([blocks[item] for item in chosen]))
    alpha = 1.0 - confidence
    return {
        "estimate": float(np.mean(values)),
        "ci_low": float(np.quantile(samples, alpha / 2.0)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha / 2.0)),
    }


def paired_segmented_block_sign_permutation_pvalue(
    differences: Sequence[float],
    segment_ids: Sequence[object],
    *,
    block_size: int,
    n_permutations: int = 10000,
    random_state: int = 0,
) -> float:
    values = np.asarray(differences, dtype=float)
    segments = np.asarray(segment_ids)
    valid = np.isfinite(values)
    values = values[valid]
    segments = segments[valid]
    blocks = _segmented_blocks(values, segments, block_size)
    if not blocks:
        return np.nan
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")
    observed = abs(float(np.mean(values)))
    rng = np.random.default_rng(random_state)
    exceedances = 0
    for _ in range(n_permutations):
        signs = rng.choice((-1.0, 1.0), size=len(blocks))
        permuted = np.concatenate(
            [sign * block for sign, block in zip(signs, blocks)]
        )
        exceedances += abs(float(np.mean(permuted))) >= observed
    return float((exceedances + 1) / (n_permutations + 1))


def summarize_recording_effects(
    results: pd.DataFrame,
    *,
    value_col: str,
    reference_method: str = "raw",
) -> pd.DataFrame:
    required = {"recording", "method", value_col}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"results is missing required columns: {missing}")
    pivot = results.pivot_table(
        index="recording",
        columns="method",
        values=value_col,
        aggfunc="first",
    )
    if reference_method not in pivot:
        raise ValueError(f"Reference method {reference_method!r} is absent.")
    rows = []
    for method in pivot.columns:
        if method == reference_method:
            continue
        differences = (pivot[method] - pivot[reference_method]).dropna().to_numpy(dtype=float)
        if differences.size == 0:
            continue
        rows.append(
            {
                "method": method,
                "reference_method": reference_method,
                "n_recordings": int(differences.size),
                "median_difference": float(np.median(differences)),
                "q25_difference": float(np.quantile(differences, 0.25)),
                "q75_difference": float(np.quantile(differences, 0.75)),
                "min_difference": float(np.min(differences)),
                "max_difference": float(np.max(differences)),
                "n_positive": int(np.count_nonzero(differences > 0)),
                "sign_flip_pvalue": exact_sign_flip_pvalue(differences),
            }
        )
    return pd.DataFrame(rows)


def quantify_motor_neuron_artifact(
    raw: np.ndarray,
    variants: Mapping[str, np.ndarray],
    *,
    artifact_center: int,
    half_width: int,
    guard_width: int,
) -> pd.DataFrame:
    raw = _validate_time_by_features(raw, "raw")
    if half_width < 1 or guard_width < half_width:
        raise ValueError("Require guard_width >= half_width >= 1.")
    near = _bounded_window(raw.shape[0], artifact_center, half_width)
    guard = _bounded_window(raw.shape[0], artifact_center, guard_width)
    far_mask = np.ones(raw.shape[0], dtype=bool)
    far_mask[guard] = False
    interpolation = interpolate_window(raw, near)
    rows: list[dict[str, object]] = []
    for name, traces in {"raw": raw, **variants}.items():
        traces = _validate_time_by_features(traces, name)
        if traces.shape != raw.shape:
            raise ValueError(f"{name} has shape {traces.shape}; expected {raw.shape}.")
        for neuron in range(raw.shape[1]):
            scale = float(np.std(raw[far_mask, neuron]))
            rows.append(
                {
                    "variant": name,
                    "neuron": neuron,
                    "artifact_center": int(artifact_center),
                    "half_width": int(half_width),
                    "guard_width": int(guard_width),
                    "far_pearson": _safe_pearson(raw[far_mask, neuron], traces[far_mask, neuron]),
                    "far_nrmse": _normalized_rmse(
                        raw[far_mask, neuron], traces[far_mask, neuron], scale
                    ),
                    "near_interpolation_nrmse": _normalized_rmse(
                        interpolation[near, neuron], traces[near, neuron], scale
                    ),
                }
            )
    return pd.DataFrame(rows)


def interpolate_window(traces: np.ndarray, window: np.ndarray) -> np.ndarray:
    traces = _validate_time_by_features(traces, "traces")
    window = np.asarray(window, dtype=int)
    if window.size == 0:
        return traces.copy()
    start, stop = int(window.min()), int(window.max()) + 1
    if start < 1 or stop >= traces.shape[0]:
        raise ValueError("Interpolation window must have observed frames on both sides.")
    corrected = traces.copy()
    weights = np.linspace(0.0, 1.0, stop - start + 2)[1:-1, np.newaxis]
    corrected[start:stop] = (
        (1.0 - weights) * traces[start - 1] + weights * traces[stop]
    )
    return corrected


def pseudo_artifact_interpolation_test(
    traces: np.ndarray,
    *,
    centers: Iterable[int],
    half_width: int,
) -> pd.DataFrame:
    traces = _validate_time_by_features(traces, "traces")
    rows = []
    for center_value in centers:
        center = int(center_value)
        window = _bounded_window(traces.shape[0], center, half_width)
        corrected = interpolate_window(traces, window)
        rows.append(
            {
                "center": center,
                "half_width": int(half_width),
                "rmse": float(np.sqrt(np.mean((corrected[window] - traces[window]) ** 2))),
            }
        )
    return pd.DataFrame(rows)


def pseudo_artifact_reconstruction_test(
    raw: np.ndarray,
    variants: Mapping[str, np.ndarray],
    *,
    centers: Iterable[int],
    half_width: int,
) -> pd.DataFrame:
    """Compare candidate corrections on apparently clean held-out windows."""
    raw = _validate_time_by_features(raw, "raw")
    rows = []
    for center_value in centers:
        center = int(center_value)
        window = _bounded_window(raw.shape[0], center, half_width)
        candidates = {"local_interpolation": interpolate_window(raw, window), **variants}
        for name, corrected in candidates.items():
            corrected = _validate_time_by_features(corrected, name)
            if corrected.shape != raw.shape:
                raise ValueError(f"{name} has shape {corrected.shape}; expected {raw.shape}.")
            rows.append(
                {
                    "center": center,
                    "half_width": int(half_width),
                    "variant": name,
                    "rmse": float(
                        np.sqrt(np.mean((corrected[window] - raw[window]) ** 2))
                    ),
                }
            )
    return pd.DataFrame(rows)


def _bounded_window(n_samples: int, center: int, half_width: int) -> np.ndarray:
    start = max(0, int(center) - int(half_width))
    stop = min(n_samples, int(center) + int(half_width) + 1)
    return np.arange(start, stop, dtype=int)


def _validate_time_by_features(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2D (time x features), got {values.shape}.")
    return values


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return (values - mean) / std


def _column_lag_correlation(values: np.ndarray, lag: int) -> np.ndarray:
    return np.mean(values[lag:] * values[:-lag], axis=0)


def _lagged_cross_correlation(values: np.ndarray, lag: int) -> np.ndarray:
    return values[lag:].T @ values[:-lag] / (values.shape[0] - lag)


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _normalized_rmse(reference: np.ndarray, estimate: np.ndarray, scale: float) -> float:
    if not np.isfinite(scale) or scale <= 0:
        return np.nan
    return float(np.sqrt(np.mean((reference - estimate) ** 2)) / scale)


def _segmented_blocks(
    values: np.ndarray,
    segment_ids: np.ndarray,
    block_size: int,
) -> list[np.ndarray]:
    if block_size < 1:
        raise ValueError("block_size must be at least 1.")
    if values.shape[0] != segment_ids.shape[0]:
        raise ValueError("values and segment_ids must have matching lengths.")
    split_points = np.flatnonzero(segment_ids[1:] != segment_ids[:-1]) + 1
    segments = np.split(values, split_points)
    return [
        segment[start : start + block_size]
        for segment in segments
        for start in range(0, segment.size, block_size)
        if segment[start : start + block_size].size
    ]

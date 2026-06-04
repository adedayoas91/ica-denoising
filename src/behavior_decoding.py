from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import warnings

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class TraceVariant:
    name: str
    path: Path
    traces: np.ndarray  # shape: time x neurons


@dataclass(frozen=True)
class BehaviorTargets:
    angle: np.ndarray
    vigor: np.ndarray
    bout_state: np.ndarray
    bout_threshold: float


@dataclass(frozen=True)
class DecodingResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class AnalysisScorecard:
    trace_summary: pd.DataFrame
    decoding_summary: pd.DataFrame
    scorecard: pd.DataFrame


def load_trace_variants(
    dataset_dir: Path,
    raw_name: str = "fluo_signals_no_NaN.npy",
    cleaned_glob: str = "cleaned_*.npy",
    cleaned_root: Path | None = None,
    dataset_name: str | None = None,
    method_glob: str = "*",
) -> list[TraceVariant]:
    """Load raw and cleaned traces with a common time x neurons orientation.

    When ``cleaned_root`` is provided, cleaned traces are discovered using the
    method hierarchy ``<cleaned_root>/<method>/cleaned/<cleaned_glob>``, the
    incremental hierarchy ``<cleaned_root>/<method>/incremental/<selection_id>/cleaned/<cleaned_glob>``,
    and fallbacks for earlier method-root and method/dataset layouts.
    """
    dataset_dir = Path(dataset_dir)
    raw_path = dataset_dir / raw_name
    raw = _load_trace(raw_path)
    raw = orient_time_by_neurons(raw)
    raw = replace_nonfinite(raw)

    variants = [TraceVariant("raw", raw_path, raw)]

    if cleaned_root is None:
        cleaned_paths = sorted(dataset_dir.glob(cleaned_glob))
    else:
        cleaned_root = Path(cleaned_root)
        resolved_dataset_name = dataset_name or dataset_dir.name
        cleaned_paths_set: set[Path] = set()
        for pattern in (
            f"{method_glob}/incremental/*/cleaned/{cleaned_glob}",
            f"{method_glob}/cleaned/{cleaned_glob}",
            f"{method_glob}/{cleaned_glob}",
            f"{method_glob}/{resolved_dataset_name}/{cleaned_glob}",
        ):
            for path in cleaned_root.glob(pattern):
                cleaned_paths_set.add(path)
        cleaned_paths = sorted(cleaned_paths_set)

    for path in cleaned_paths:
        traces = orient_time_by_neurons(_load_trace(path), expected_frames=raw.shape[0])
        traces = replace_nonfinite(traces)
        if traces.shape != raw.shape:
            if cleaned_root is None:
                raise ValueError(
                    f"{path.name} has shape {traces.shape}; expected {raw.shape} to match raw traces."
                )
            warnings.warn(
                f"Skipping {path.name}: shape {traces.shape} does not match raw shape {raw.shape}.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if cleaned_root is None:
            variant_name = path.stem
        else:
            variant_name = _infer_variant_name_from_cleaned_path(
                path=path,
                cleaned_root=cleaned_root,
                resolved_dataset_name=resolved_dataset_name,
            )
        variants.append(TraceVariant(variant_name, path, traces))
    return variants


def _infer_variant_name_from_cleaned_path(
    path: Path,
    cleaned_root: Path,
    resolved_dataset_name: str,
) -> str:
    """Return a stable variant name across canonical, incremental, and legacy layouts."""
    stem = path.stem
    try:
        rel_parts = path.relative_to(cleaned_root).parts
    except ValueError:
        rel_parts = path.parts

    # Incremental layout:
    # <method>/incremental/<selection_id>/cleaned/<cleaned_file>.npy
    if len(rel_parts) >= 5 and rel_parts[1] == "incremental" and rel_parts[-2] == "cleaned":
        method_name = rel_parts[0]
        selection_id = rel_parts[2]
        return f"{method_name}/{selection_id}/{stem}"

    # Canonical layout:
    # <method>/cleaned/<cleaned_file>.npy
    if len(rel_parts) >= 3 and rel_parts[-2] == "cleaned":
        method_name = rel_parts[-3]
        return f"{method_name}/{stem}"

    # Legacy layout:
    # <method>/<dataset_name>/<cleaned_file>.npy
    if len(rel_parts) >= 3 and rel_parts[-2] == resolved_dataset_name:
        method_name = rel_parts[-3]
        return f"{method_name}/{stem}"

    # Legacy layout:
    # <method>/<cleaned_file>.npy
    if len(rel_parts) >= 2:
        method_name = rel_parts[-2]
    else:
        method_name = path.parent.name
    return f"{method_name}/{stem}"


def _load_trace(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2:
        raise ValueError(f"{path} must be a 2D trace array, got shape {arr.shape}.")
    return np.asarray(arr, dtype=float)


def orient_time_by_neurons(arr: np.ndarray, expected_frames: int | None = None) -> np.ndarray:
    """Return traces as time x neurons.

    The v2a files are mostly neurons x frames, while some reconstructed arrays
    are frames x neurons. The expected frame count removes ambiguity.
    """
    if expected_frames is not None:
        if arr.shape[0] == expected_frames:
            return arr
        if arr.shape[1] == expected_frames:
            return arr.T
        raise ValueError(f"Cannot orient shape {arr.shape} to {expected_frames} frames.")
    if arr.shape[0] < arr.shape[1]:
        return arr.T
    return arr


def replace_nonfinite(traces: np.ndarray) -> np.ndarray:
    traces = np.asarray(traces, dtype=float).copy()
    if np.isfinite(traces).all():
        return traces
    col_medians = np.nanmedian(np.where(np.isfinite(traces), traces, np.nan), axis=0)
    col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
    bad_rows, bad_cols = np.where(~np.isfinite(traces))
    traces[bad_rows, bad_cols] = col_medians[bad_cols]
    return traces


def make_behavior_targets(
    tail_angle: np.ndarray,
    n_frames: int,
    bout_quantile: float = 0.75,
    smooth_window: int = 3,
) -> BehaviorTargets:
    tail_angle = np.asarray(tail_angle, dtype=float).reshape(-1)
    if tail_angle.size < n_frames:
        raise ValueError(
            f"Tail angle length {tail_angle.size} is shorter than calcium frames {n_frames}."
        )
    angle = bin_signal_to_frames(tail_angle, n_frames, reducer="mean")
    tail_velocity = np.diff(tail_angle, prepend=tail_angle[0])
    vigor = bin_signal_to_frames(tail_velocity, n_frames, reducer="rms")
    if smooth_window > 1:
        vigor = moving_average(vigor, smooth_window)
    threshold = float(np.quantile(vigor, bout_quantile))
    bout_state = (vigor >= threshold).astype(int)
    return BehaviorTargets(angle=angle, vigor=vigor, bout_state=bout_state, bout_threshold=threshold)


def bin_signal_to_frames(signal: np.ndarray, n_frames: int, reducer: str = "mean") -> np.ndarray:
    signal = np.asarray(signal, dtype=float).reshape(-1)
    edges = np.linspace(0, signal.size, n_frames + 1)
    edges = np.rint(edges).astype(int)
    binned = np.empty(n_frames, dtype=float)
    for i in range(n_frames):
        start, stop = edges[i], edges[i + 1]
        if stop <= start:
            stop = min(start + 1, signal.size)
        chunk = signal[start:stop]
        if reducer == "mean":
            binned[i] = float(np.mean(chunk))
        elif reducer == "median":
            binned[i] = float(np.median(chunk))
        elif reducer == "rms":
            binned[i] = float(np.sqrt(np.mean(chunk**2)))
        else:
            raise ValueError(f"Unknown reducer: {reducer}")
    return binned


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(x, dtype=float)
    kernel = np.ones(window, dtype=float) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(np.asarray(x, dtype=float), (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def make_lagged_design(
    traces: np.ndarray,
    target: np.ndarray,
    lags: Iterable[int] = (0, 1, 2),
    target_shift: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    traces = np.asarray(traces, dtype=float)
    target = np.asarray(target)
    lags = tuple(sorted(set(int(lag) for lag in lags)))
    if not lags or min(lags) < 0:
        raise ValueError("lags must contain non-negative integers.")
    if target_shift < 0:
        raise ValueError("target_shift must be non-negative.")
    if traces.shape[0] != target.shape[0]:
        raise ValueError(f"Trace length {traces.shape[0]} and target length {target.shape[0]} differ.")

    max_lag = max(lags)
    times = np.arange(max_lag, traces.shape[0] - target_shift)
    if times.size == 0:
        raise ValueError("No samples remain after applying lags and target_shift.")
    design = np.hstack([traces[times - lag] for lag in lags])
    aligned_target = target[times + target_shift]
    return design, aligned_target, times


def run_decoding_experiment(
    variants: list[TraceVariant],
    target: np.ndarray,
    target_name: str,
    task: str,
    lags: Iterable[int] = (0, 1, 2),
    target_shift: int = 0,
    n_splits: int = 5,
    gap: int = 3,
    ridge_alpha: float = 10.0,
    random_state: int = 0,
    include_transfer: bool = True,
    include_null: bool = True,
    null_block_size: int = 60,
    n_jobs: int = 1,
) -> DecodingResult:
    if not variants:
        raise ValueError("At least one trace variant is required.")
    feature_map: dict[str, np.ndarray] = {}
    aligned_y: np.ndarray | None = None
    time_index: np.ndarray | None = None
    for variant in variants:
        design, y, times = make_lagged_design(variant.traces, target, lags, target_shift)
        feature_map[variant.name] = design
        if aligned_y is None:
            aligned_y = y
            time_index = times
        elif not np.array_equal(aligned_y, y):
            raise ValueError("Target alignment changed across variants.")
    assert aligned_y is not None
    assert time_index is not None

    folds = list(blocked_folds(aligned_y.size, n_splits=n_splits, gap=gap))
    jobs: list[dict[str, object]] = []
    for variant in variants:
        jobs.append(
            {
                "train_features": feature_map[variant.name],
                "test_features": feature_map[variant.name],
                "y": aligned_y,
                "time_index": time_index,
                "folds": folds,
                "task": task,
                "target_name": target_name,
                "comparison": "within",
                "train_version": variant.name,
                "test_version": variant.name,
                "ridge_alpha": ridge_alpha,
            }
        )

    if include_transfer:
        reference = variants[0].name
        for variant in variants[1:]:
            jobs.append(
                {
                    "train_features": feature_map[reference],
                    "test_features": feature_map[variant.name],
                    "y": aligned_y,
                    "time_index": time_index,
                    "folds": folds,
                    "task": task,
                    "target_name": target_name,
                    "comparison": "transfer_raw_to_clean",
                    "train_version": reference,
                    "test_version": variant.name,
                    "ridge_alpha": ridge_alpha,
                }
            )

            jobs.append(
                {
                    "train_features": feature_map[variant.name],
                    "test_features": feature_map[reference],
                    "y": aligned_y,
                    "time_index": time_index,
                    "folds": folds,
                    "task": task,
                    "target_name": target_name,
                    "comparison": "transfer_clean_to_raw",
                    "train_version": variant.name,
                    "test_version": reference,
                    "ridge_alpha": ridge_alpha,
                }
            )

    if include_null:
        rng = np.random.default_rng(random_state)
        null_y = block_shuffle(aligned_y, block_size=null_block_size, rng=rng)
        for variant in variants:
            jobs.append(
                {
                    "train_features": feature_map[variant.name],
                    "test_features": feature_map[variant.name],
                    "y": null_y,
                    "time_index": time_index,
                    "folds": folds,
                    "task": task,
                    "target_name": target_name,
                    "comparison": "null_within",
                    "train_version": variant.name,
                    "test_version": variant.name,
                    "ridge_alpha": ridge_alpha,
                }
            )

    results = _run_evaluation_jobs(jobs, n_jobs=n_jobs)
    metric_rows: list[dict[str, object]] = []
    pred_rows: list[pd.DataFrame] = []
    for rows, preds in results:
        metric_rows.extend(rows)
        pred_rows.extend(preds)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    summary = summarize_metrics(metrics)
    return DecodingResult(metrics=metrics, predictions=predictions, summary=summary)


def _run_evaluation_jobs(
    jobs: list[dict[str, object]],
    n_jobs: int,
) -> list[tuple[list[dict[str, object]], list[pd.DataFrame]]]:
    if n_jobs == 1 or len(jobs) <= 1:
        return [_evaluate_pair(**job) for job in jobs]
    return Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_evaluate_pair)(**job) for job in jobs
    )


def blocked_folds(
    n_samples: int,
    n_splits: int = 5,
    gap: int = 0,
) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    edges = np.linspace(0, n_samples, n_splits + 1).round().astype(int)
    all_idx = np.arange(n_samples)
    for fold in range(n_splits):
        test_start, test_stop = edges[fold], edges[fold + 1]
        test_idx = all_idx[test_start:test_stop]
        gap_start = max(0, test_start - gap)
        gap_stop = min(n_samples, test_stop + gap)
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[gap_start:gap_stop] = False
        train_idx = all_idx[train_mask]
        if train_idx.size == 0 or test_idx.size == 0:
            raise ValueError("Empty train or test block. Reduce n_splits or gap.")
        yield fold, train_idx, test_idx


def block_shuffle(y: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y)
    if block_size <= 1:
        shuffled = y.copy()
        rng.shuffle(shuffled)
        return shuffled
    blocks = [y[start : start + block_size] for start in range(0, y.size, block_size)]
    order = np.arange(len(blocks))
    rng.shuffle(order)
    return np.concatenate([blocks[i] for i in order])


def _evaluate_pair(
    train_features: np.ndarray,
    test_features: np.ndarray,
    y: np.ndarray,
    time_index: np.ndarray,
    folds: list[tuple[int, np.ndarray, np.ndarray]],
    task: str,
    target_name: str,
    comparison: str,
    train_version: str,
    test_version: str,
    ridge_alpha: float,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    metric_rows: list[dict[str, object]] = []
    pred_rows: list[pd.DataFrame] = []
    for fold, train_idx, test_idx in folds:
        x_train, x_test = standardize_train_test(train_features[train_idx], test_features[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]
        if task == "regression":
            pred, score = _fit_predict_regression(x_train, y_train, x_test, ridge_alpha)
            metrics = regression_metrics(y_test, pred)
        elif task == "classification":
            pred, score = _fit_predict_classification(x_train, y_train, x_test)
            metrics = classification_metrics(y_test, pred, score)
        else:
            raise ValueError(f"Unknown task: {task}")

        row = {
            "target": target_name,
            "task": task,
            "comparison": comparison,
            "train_version": train_version,
            "test_version": test_version,
            "fold": fold,
            "n_train": int(train_idx.size),
            "n_test": int(test_idx.size),
        }
        row.update(metrics)
        metric_rows.append(row)

        fold_preds = pd.DataFrame(
            {
                "target": target_name,
                "task": task,
                "comparison": comparison,
                "train_version": train_version,
                "test_version": test_version,
                "fold": fold,
                "time_index": time_index[test_idx],
                "y_true": y_test,
                "y_pred": pred,
            }
        )
        if score is not None:
            fold_preds["y_score"] = score
        pred_rows.append(fold_preds)
    return metric_rows, pred_rows


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return (x_train - mean) / std, (x_test - mean) / std


def _fit_predict_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    ridge_alpha: float,
) -> tuple[np.ndarray, None]:
    model = Ridge(alpha=ridge_alpha)
    model.fit(x_train, y_train)
    return model.predict(x_test), None


def _fit_predict_classification(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    if np.unique(y_train).size < 2:
        pred = np.full(x_test.shape[0], int(y_train[0]))
        return pred, None
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=0,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(x_test)[:, 1]
    else:
        score = None
    return pred, score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = float(np.std(y_true))
    pearson = np.nan
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    spearman = np.nan
    if np.unique(y_true).size > 1 and np.unique(y_pred).size > 1:
        spearman = float(spearmanr(y_true, y_pred).correlation)
    return {
        "r2": float(r2_score(y_true, y_pred)) if np.unique(y_true).size > 1 else np.nan,
        "pearson": pearson,
        "spearman": spearman,
        "rmse": rmse,
        "nrmse": rmse / denom if denom > 0 else np.nan,
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
) -> dict[str, float]:
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": np.nan,
    }
    if y_score is not None and np.unique(y_true).size == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["target", "task", "comparison", "train_version", "test_version"]
    metric_cols = [
        col
        for col in metrics.columns
        if col not in id_cols + ["fold", "n_train", "n_test"]
    ]
    summary = (
        metrics.groupby(id_cols, dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns
    ]
    return summary


def save_result(result: DecodingResult, output_dir: Path, prefix: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(output_dir / f"{prefix}_fold_metrics.csv", index=False)
    result.predictions.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)
    result.summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)


def summarize_trace_preservation(
    variants: list[TraceVariant],
    reference_name: str = "raw",
) -> pd.DataFrame:
    """Summarize how closely each trace variant preserves a reference trace.

    The returned table is intended to sit next to decoding summaries. It
    reports global and per-neuron similarity metrics between each variant and
    the reference, usually raw extracted traces.
    """
    reference = _get_variant(variants, reference_name)
    rows: list[dict[str, object]] = []
    for variant in variants:
        _validate_matching_trace_shape(reference, variant)
        delta = variant.traces - reference.traces
        neuron_corr = columnwise_pearson(reference.traces, variant.traces)
        variance_ratio = _safe_divide(
            np.var(variant.traces, axis=0),
            np.var(reference.traces, axis=0),
        )
        rmse_by_neuron = np.sqrt(np.mean(delta**2, axis=0))
        nrmse_by_neuron = _safe_divide(rmse_by_neuron, np.std(reference.traces, axis=0))
        rows.append(
            {
                "variant": variant.name,
                "path": str(variant.path),
                "reference": reference.name,
                "n_frames": int(variant.traces.shape[0]),
                "n_neurons": int(variant.traces.shape[1]),
                "global_pearson": pearson_1d(reference.traces.ravel(), variant.traces.ravel()),
                "neuron_pearson_mean": _nan_stat(neuron_corr, np.nanmean),
                "neuron_pearson_median": _nan_stat(neuron_corr, np.nanmedian),
                "neuron_pearson_q25": _nan_stat(neuron_corr, lambda x: np.nanquantile(x, 0.25)),
                "neuron_pearson_q75": _nan_stat(neuron_corr, lambda x: np.nanquantile(x, 0.75)),
                "rmse_mean": float(np.mean(rmse_by_neuron)),
                "nrmse_mean": _nan_stat(nrmse_by_neuron, np.nanmean),
                "mean_abs_delta": float(np.mean(np.abs(delta))),
                "variance_ratio_median": _nan_stat(variance_ratio, np.nanmedian),
                "variance_ratio_q25": _nan_stat(variance_ratio, lambda x: np.nanquantile(x, 0.25)),
                "variance_ratio_q75": _nan_stat(variance_ratio, lambda x: np.nanquantile(x, 0.75)),
            }
        )
    return pd.DataFrame(rows)


def rank_representative_neurons(
    variants: list[TraceVariant],
    reference_name: str = "raw",
    n: int = 12,
) -> pd.DataFrame:
    """Rank neurons that are useful for raw-vs-cleaned visual inspection.

    High-ranked neurons have high raw variance and a large average difference
    between cleaned variants and the reference. They are good candidates for
    time-series panels because the cleaning effect is visible without choosing
    a neuron by hand.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    reference = _get_variant(variants, reference_name)
    candidates = [variant for variant in variants if variant.name != reference.name]
    if not candidates:
        candidates = [reference]

    reference_std = np.std(reference.traces, axis=0)
    delta_stack = []
    corr_stack = []
    for variant in candidates:
        _validate_matching_trace_shape(reference, variant)
        delta_stack.append(np.mean(np.abs(variant.traces - reference.traces), axis=0))
        corr_stack.append(columnwise_pearson(reference.traces, variant.traces))

    mean_abs_delta = np.mean(np.vstack(delta_stack), axis=0)
    mean_corr = _nanmean_axis0(np.vstack(corr_stack))
    score = _safe_zscore(reference_std) + _safe_zscore(mean_abs_delta)
    rows = pd.DataFrame(
        {
            "neuron": np.arange(reference.traces.shape[1], dtype=int),
            "reference_std": reference_std,
            "mean_abs_delta": mean_abs_delta,
            "mean_variant_pearson": mean_corr,
            "inspection_score": score,
        }
    )
    return rows.sort_values("inspection_score", ascending=False).head(n).reset_index(drop=True)


def tidy_decoding_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Convert ``summarize_metrics`` output into one row per metric."""
    id_cols = ["target", "task", "comparison", "train_version", "test_version"]
    missing = [col for col in id_cols if col not in summary.columns]
    if missing:
        raise ValueError(f"summary is missing required columns: {missing}")

    metric_stems = sorted(
        {
            col.removesuffix("_mean").removesuffix("_std")
            for col in summary.columns
            if col.endswith(("_mean", "_std"))
        }
    )
    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        for metric in metric_stems:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col not in summary.columns:
                continue
            rows.append(
                {
                    **{col: row[col] for col in id_cols},
                    "metric": metric,
                    "mean": row[mean_col],
                    "std": row[std_col] if std_col in summary.columns else np.nan,
                    "higher_is_better": metric not in {"rmse", "nrmse"},
                }
            )
    return pd.DataFrame(rows)


def make_analysis_scorecard(
    variants: list[TraceVariant],
    decoding_summary: pd.DataFrame,
    reference_name: str = "raw",
) -> AnalysisScorecard:
    """Build a compact table joining trace preservation and within-decoding metrics."""
    trace_summary = summarize_trace_preservation(variants, reference_name=reference_name)
    tidy_summary = tidy_decoding_summary(decoding_summary)
    within = tidy_summary[
        (tidy_summary["comparison"] == "within")
        & (tidy_summary["train_version"] == tidy_summary["test_version"])
    ].copy()
    within = within.rename(columns={"test_version": "variant"})
    metric_table = within.pivot_table(
        index="variant",
        columns=["target", "metric"],
        values="mean",
        aggfunc="first",
    )
    metric_table.columns = [f"{target}_{metric}" for target, metric in metric_table.columns]
    metric_table = metric_table.reset_index()
    scorecard = trace_summary.merge(metric_table, on="variant", how="left")
    return AnalysisScorecard(
        trace_summary=trace_summary,
        decoding_summary=tidy_summary,
        scorecard=scorecard,
    )


def make_behavior_preservation_scorecard(
    summary: pd.DataFrame,
    variants: list[TraceVariant] | None = None,
    reference_name: str = "raw",
) -> pd.DataFrame:
    """Rank trace variants by raw-normalized behavioral information retention.

    The scorecard is intended for paper-facing reporting. It treats raw traces as
    the behavior-information reference and normalizes each method against the
    raw null/shuffle score for the same target and metric. Transfer metrics are
    included because they test whether denoised traces preserve a raw-compatible
    behavior representation, not merely whether they remain decodable in their
    own altered feature space.
    """
    required = ["target", "comparison", "train_version", "test_version"]
    missing = [col for col in required if col not in summary.columns]
    if missing:
        raise ValueError(f"summary is missing required columns: {missing}")

    versions = _ordered_versions(summary["test_version"].dropna().unique(), reference_name)
    rows = []
    for version in versions:
        row: dict[str, object] = {"method": version}
        preservation_scores: list[float] = []

        for component in _preservation_components():
            raw_score = _summary_metric_value(
                summary,
                target=component["target"],
                comparison="within",
                train_version=reference_name,
                test_version=reference_name,
                metric=component["metric"],
            )
            null_score = _summary_metric_value(
                summary,
                target=component["target"],
                comparison="null_within",
                train_version=reference_name,
                test_version=reference_name,
                metric=component["metric"],
            )
            method_score = _component_method_score(
                summary,
                component=component,
                method=version,
                reference_name=reference_name,
                raw_score=raw_score,
            )

            score_col = f"{component['label']}_{component['metric']}"
            preservation_col = f"{score_col}_preservation_pct"
            row[score_col] = method_score
            row[preservation_col] = np.nan

            if (
                np.isfinite(method_score)
                and np.isfinite(raw_score)
                and np.isfinite(null_score)
                and raw_score > null_score
            ):
                preservation = 100.0 * (method_score - null_score) / (raw_score - null_score)
                row[preservation_col] = preservation
                preservation_scores.append(float(preservation))

        row["behavioral_preservation_index"] = (
            float(np.mean(preservation_scores)) if preservation_scores else np.nan
        )
        rows.append(row)

    scorecard = pd.DataFrame(rows)
    if variants is not None:
        trace_summary = summarize_trace_preservation(variants, reference_name=reference_name)
        trace_cols = [
            "variant",
            "neuron_pearson_median",
            "nrmse_mean",
            "variance_ratio_median",
            "mean_abs_delta",
        ]
        available_trace_cols = [col for col in trace_cols if col in trace_summary.columns]
        scorecard = scorecard.merge(
            trace_summary[available_trace_cols].rename(columns={"variant": "method"}),
            on="method",
            how="left",
        )

    scorecard = _label_preservation_recommendations(scorecard, reference_name)
    return scorecard.sort_values(
        ["method_is_reference", "behavioral_preservation_index"],
        ascending=[False, False],
        na_position="last",
    ).drop(columns=["method_is_reference"]).reset_index(drop=True)


def plot_behavior_trace_evidence(
    variants: list[TraceVariant],
    targets: BehaviorTargets,
    neuron: int | None = None,
    start: int | None = None,
    stop: int | None = None,
    window_size: int = 600,
    reference_name: str = "raw",
    method_order: Sequence[str] = ("raw", "fastica", "infomax", "jade", "sobi"),
):
    """Plot raw/denoised traces aligned to tail vigor and bout state.

    This is the qualitative evidence panel for the paper figure. It deliberately
    z-scores and offsets traces so the reader compares timing and preservation
    of structure rather than absolute fluorescence scale.
    """
    if not variants:
        raise ValueError("At least one trace variant is required.")

    reference = _get_variant(variants, reference_name)
    if neuron is None:
        neuron = int(rank_representative_neurons(variants, reference_name=reference_name, n=1)["neuron"].iloc[0])
    if neuron < 0 or neuron >= reference.traces.shape[1]:
        raise ValueError(f"neuron must be in [0, {reference.traces.shape[1] - 1}], got {neuron}.")

    start, stop = _resolve_behavior_window(targets, reference.traces.shape[0], start, stop, window_size)
    selected = _order_variants(variants, method_order=method_order, reference_name=reference_name)
    frames = np.arange(start, stop)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [4.0, 1.1, 0.8]},
    )
    trace_ax, vigor_ax, bout_ax = axes
    colors = _method_color_map([variant.name for variant in selected], reference_name=reference_name)
    offset_step = 3.0

    for offset, variant in enumerate(reversed(selected)):
        trace = _zscore_1d(variant.traces[start:stop, neuron])
        y_offset = offset * offset_step
        trace_ax.plot(frames, trace + y_offset, lw=0.9, color=colors[variant.name])
        trace_ax.text(
            frames[0],
            y_offset,
            _display_method_name(variant.name),
            ha="right",
            va="center",
            fontsize=9,
            color=colors[variant.name],
        )

    trace_ax.set_title(f"Raw and denoised traces aligned to behavior, neuron {neuron}")
    trace_ax.set_ylabel("z-scored trace + offset")
    trace_ax.set_yticks([])

    vigor_ax.plot(frames, targets.vigor[start:stop], color="black", lw=1.0)
    vigor_ax.axhline(targets.bout_threshold, color="0.5", lw=0.8, ls="--")
    vigor_ax.set_ylabel("tail vigor")

    bout_ax.fill_between(frames, 0, targets.bout_state[start:stop], step="mid", color="0.25", alpha=0.35)
    bout_ax.set_ylim(-0.05, 1.05)
    bout_ax.set_yticks([0, 1])
    bout_ax.set_ylabel("bout")
    bout_ax.set_xlabel("calcium frame")

    fig.tight_layout()
    return fig, axes


def plot_behavior_preservation_summary(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    variants: list[TraceVariant] | None = None,
    reference_name: str = "raw",
):
    """Plot paper-facing decoding and preservation panels.

    The figure shows fold-level metric distributions for within and transfer
    decoding, plus a Behavioral Preservation Index scorecard. It is designed to
    be called after the behavior-decoding experiment has produced ``metrics``
    and ``summary``.
    """
    scorecard = make_behavior_preservation_scorecard(summary, variants=variants, reference_name=reference_name)
    method_order = scorecard["method"].tolist()
    colors = _method_color_map(method_order, reference_name=reference_name)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), gridspec_kw={"height_ratios": [1.0, 0.9]})
    _plot_fold_metric_panel(
        metrics,
        summary,
        axes[0, 0],
        target="tail_vigor",
        metric="pearson",
        comparison="within",
        title="Within-representation tail-vigor decoding",
        ylabel="Pearson r",
        method_order=method_order,
        colors=colors,
        reference_name=reference_name,
    )
    _plot_fold_metric_panel(
        metrics,
        summary,
        axes[0, 1],
        target="tail_vigor",
        metric="pearson",
        comparison="transfer_raw_to_clean",
        title="Raw-to-denoised tail-vigor transfer",
        ylabel="Pearson r",
        method_order=method_order,
        colors=colors,
        reference_name=reference_name,
    )
    _plot_preservation_index_panel(scorecard, axes[0, 2], colors, reference_name)

    _plot_fold_metric_panel(
        metrics,
        summary,
        axes[1, 0],
        target="bout_state",
        metric="balanced_accuracy",
        comparison="within",
        title="Within-representation bout decoding",
        ylabel="Balanced accuracy",
        method_order=method_order,
        colors=colors,
        reference_name=reference_name,
        null_line=0.5,
    )
    _plot_fold_metric_panel(
        metrics,
        summary,
        axes[1, 1],
        target="bout_state",
        metric="balanced_accuracy",
        comparison="transfer_raw_to_clean",
        title="Raw-to-denoised bout transfer",
        ylabel="Balanced accuracy",
        method_order=method_order,
        colors=colors,
        reference_name=reference_name,
        null_line=0.5,
    )
    _plot_scorecard_table(scorecard, axes[1, 2])

    fig.suptitle("Behavioral information preservation after denoising", y=1.02)
    fig.tight_layout()
    return fig, axes, scorecard


def _preservation_components() -> list[dict[str, str]]:
    return [
        {
            "target": "tail_vigor",
            "comparison": "within",
            "metric": "pearson",
            "label": "tail_vigor_within",
        },
        {
            "target": "tail_vigor",
            "comparison": "transfer_raw_to_clean",
            "metric": "pearson",
            "label": "tail_vigor_transfer",
        },
        {
            "target": "bout_state",
            "comparison": "within",
            "metric": "balanced_accuracy",
            "label": "bout_state_within",
        },
        {
            "target": "bout_state",
            "comparison": "transfer_raw_to_clean",
            "metric": "balanced_accuracy",
            "label": "bout_state_transfer",
        },
    ]


def _component_method_score(
    summary: pd.DataFrame,
    component: dict[str, str],
    method: str,
    reference_name: str,
    raw_score: float,
) -> float:
    if method == reference_name:
        return raw_score
    train_version = method if component["comparison"] == "within" else reference_name
    return _summary_metric_value(
        summary,
        target=component["target"],
        comparison=component["comparison"],
        train_version=train_version,
        test_version=method,
        metric=component["metric"],
    )


def _summary_metric_value(
    summary: pd.DataFrame,
    target: str,
    comparison: str,
    train_version: str,
    test_version: str,
    metric: str,
) -> float:
    metric_col = f"{metric}_mean"
    if metric_col not in summary.columns:
        return np.nan
    rows = summary[
        (summary["target"] == target)
        & (summary["comparison"] == comparison)
        & (summary["train_version"] == train_version)
        & (summary["test_version"] == test_version)
    ]
    if rows.empty:
        return np.nan
    value = rows.iloc[0][metric_col]
    return float(value) if pd.notna(value) else np.nan


def _summary_metric_std(
    summary: pd.DataFrame,
    target: str,
    comparison: str,
    train_version: str,
    test_version: str,
    metric: str,
) -> float:
    metric_col = f"{metric}_std"
    if metric_col not in summary.columns:
        return np.nan
    rows = summary[
        (summary["target"] == target)
        & (summary["comparison"] == comparison)
        & (summary["train_version"] == train_version)
        & (summary["test_version"] == test_version)
    ]
    if rows.empty:
        return np.nan
    value = rows.iloc[0][metric_col]
    return float(value) if pd.notna(value) else np.nan


def _label_preservation_recommendations(
    scorecard: pd.DataFrame,
    reference_name: str,
) -> pd.DataFrame:
    scorecard = scorecard.copy()
    scorecard["method_is_reference"] = scorecard["method"] == reference_name
    scorecard["recommendation_label"] = "insufficient data"
    scorecard.loc[scorecard["method_is_reference"], "recommendation_label"] = "raw reference"

    method_scores = scorecard.loc[
        ~scorecard["method_is_reference"] & scorecard["behavioral_preservation_index"].notna(),
        "behavioral_preservation_index",
    ]
    best_index = method_scores.idxmax() if not method_scores.empty else None

    for idx, value in scorecard["behavioral_preservation_index"].items():
        if scorecard.at[idx, "method_is_reference"] or not np.isfinite(value):
            continue
        if idx == best_index:
            scorecard.at[idx, "recommendation_label"] = "best preserved"
        elif value >= 80:
            scorecard.at[idx, "recommendation_label"] = "acceptable"
        elif value >= 50:
            scorecard.at[idx, "recommendation_label"] = "information-loss risk"
        else:
            scorecard.at[idx, "recommendation_label"] = "not recommended"
    return scorecard


def _plot_fold_metric_panel(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    ax,
    target: str,
    metric: str,
    comparison: str,
    title: str,
    ylabel: str,
    method_order: Sequence[str],
    colors: dict[str, str],
    reference_name: str,
    null_line: float | None = None,
) -> None:
    x_positions = np.arange(len(method_order))
    for x_pos, method in zip(x_positions, method_order):
        row_comparison = "within" if method == reference_name else comparison
        train_version = method if row_comparison == "within" else reference_name
        values = _fold_metric_values(metrics, target, row_comparison, train_version, method, metric)
        if values.size:
            jitter = np.linspace(-0.11, 0.11, values.size) if values.size > 1 else np.array([0.0])
            ax.scatter(
                np.full(values.size, x_pos) + jitter,
                values,
                s=22,
                color=colors.get(method, "0.4"),
                alpha=0.65,
                linewidths=0,
            )

        mean = _summary_metric_value(summary, target, row_comparison, train_version, method, metric)
        std = _summary_metric_std(summary, target, row_comparison, train_version, method, metric)
        if np.isfinite(mean):
            yerr = std if np.isfinite(std) else None
            ax.errorbar(
                x_pos,
                mean,
                yerr=yerr,
                marker="o",
                color="black",
                markerfacecolor=colors.get(method, "0.4"),
                markeredgecolor="black",
                capsize=3,
                lw=1.0,
            )

    raw_mean = _summary_metric_value(summary, target, "within", reference_name, reference_name, metric)
    if np.isfinite(raw_mean):
        ax.axhline(raw_mean, color="black", lw=0.8, alpha=0.45)

    raw_null = _summary_metric_value(summary, target, "null_within", reference_name, reference_name, metric)
    if np.isfinite(raw_null):
        ax.axhline(raw_null, color="0.55", lw=0.8, ls="--")
    elif null_line is not None:
        ax.axhline(null_line, color="0.55", lw=0.8, ls="--")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_display_method_name(method) for method in method_order], rotation=90, ha="right")
    ax.grid(axis="y", color="0.9", lw=0.8)


def _fold_metric_values(
    metrics: pd.DataFrame,
    target: str,
    comparison: str,
    train_version: str,
    test_version: str,
    metric: str,
) -> np.ndarray:
    if metric not in metrics.columns:
        return np.array([], dtype=float)
    rows = metrics[
        (metrics["target"] == target)
        & (metrics["comparison"] == comparison)
        & (metrics["train_version"] == train_version)
        & (metrics["test_version"] == test_version)
    ]
    return rows[metric].dropna().to_numpy(dtype=float)


def _plot_preservation_index_panel(
    scorecard: pd.DataFrame,
    ax,
    colors: dict[str, str],
    reference_name: str,
) -> None:
    plot_df = scorecard[["method", "behavioral_preservation_index"]].dropna().copy()
    if plot_df.empty:
        ax.text(0.5, 0.5, "No preservation scores available", ha="center", va="center")
        ax.set_axis_off()
        return

    plot_df = plot_df.sort_values("behavioral_preservation_index", ascending=True)
    y_pos = np.arange(len(plot_df))
    ax.barh(
        y_pos,
        plot_df["behavioral_preservation_index"],
        color=[colors.get(method, "0.4") for method in plot_df["method"]],
        alpha=0.85,
    )
    ax.axvline(100, color="black", lw=0.8, alpha=0.6)
    ax.axvline(0, color="0.65", lw=0.8, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_display_method_name(method) for method in plot_df["method"]])
    ax.set_xlabel("Behavioral Preservation Index (%)")
    ax.set_title("Method recommendation score")
    ax.grid(axis="x", color="0.9", lw=0.8)

    if reference_name in set(plot_df["method"]):
        ax.text(100, y_pos[plot_df["method"].tolist().index(reference_name)], " raw", va="center", fontsize=8)


def _plot_scorecard_table(scorecard: pd.DataFrame, ax) -> None:
    ax.set_axis_off()
    columns = ["method", "behavioral_preservation_index", "recommendation_label"]
    if "neuron_pearson_median" in scorecard.columns:
        columns.append("neuron_pearson_median")
    table_df = scorecard[columns].copy()
    table_df["method"] = table_df["method"].map(_display_method_name)
    if "behavioral_preservation_index" in table_df:
        table_df["behavioral_preservation_index"] = table_df["behavioral_preservation_index"].map(
            lambda value: "" if pd.isna(value) else f"{value:.1f}"
        )
    if "neuron_pearson_median" in table_df:
        table_df["neuron_pearson_median"] = table_df["neuron_pearson_median"].map(
            lambda value: "" if pd.isna(value) else f"{value:.2f}"
        )
    table_df = table_df.rename(
        columns={
            "method": "method",
            "behavioral_preservation_index": "BPI",
            "recommendation_label": "recommendation",
            "neuron_pearson_median": "trace r",
        }
    )
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    ax.set_title("Scorecard")


def _order_variants(
    variants: list[TraceVariant],
    method_order: Sequence[str],
    reference_name: str,
) -> list[TraceVariant]:
    names = _ordered_versions([variant.name for variant in variants], reference_name, method_order)
    lookup = {variant.name: variant for variant in variants}
    return [lookup[name] for name in names]


def _ordered_versions(
    versions: Iterable[str],
    reference_name: str = "raw",
    method_order: Sequence[str] = ("raw", "fastica", "infomax", "jade", "sobi"),
) -> list[str]:
    version_list = [str(version) for version in versions]
    return sorted(
        version_list,
        key=lambda name: (
            0 if name == reference_name else 1,
            _method_order_index(name, method_order),
            _display_method_name(name).lower(),
            name,
        ),
    )


def _method_order_index(name: str, method_order: Sequence[str]) -> int:
    lower_name = name.lower()
    for index, method in enumerate(method_order):
        if method.lower() == lower_name or method.lower() in lower_name:
            return index
    return len(method_order)


def _method_color_map(methods: Iterable[str], reference_name: str = "raw") -> dict[str, str]:
    canonical_colors = {
        "raw": "black",
        "fastica": "tab:blue",
        "infomax": "tab:orange",
        "jade": "tab:green",
        "sobi": "tab:red",
    }
    fallback = ["tab:purple", "tab:brown", "tab:pink", "tab:cyan", "tab:olive"]
    colors: dict[str, str] = {}
    fallback_index = 0
    for method in methods:
        display = _display_method_name(method).lower()
        if method == reference_name:
            colors[method] = canonical_colors["raw"]
        elif display in canonical_colors:
            colors[method] = canonical_colors[display]
        else:
            colors[method] = fallback[fallback_index % len(fallback)]
            fallback_index += 1
    return colors


def _display_method_name(name: str) -> str:
    lower_name = name.lower()
    for method in ("raw", "fastica", "infomax", "jade", "sobi"):
        if lower_name == method or method in lower_name:
            return method
    return name.split("/")[0] if "/" in name else name


def _resolve_behavior_window(
    targets: BehaviorTargets,
    n_frames: int,
    start: int | None,
    stop: int | None,
    window_size: int,
) -> tuple[int, int]:
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    window_size = min(window_size, n_frames)
    if start is None and stop is None:
        signal = np.asarray(targets.bout_state[:n_frames], dtype=float)
        if not np.any(signal):
            signal = np.asarray(targets.vigor[:n_frames], dtype=float)
        score = np.convolve(signal, np.ones(window_size, dtype=float), mode="valid")
        start = int(np.argmax(score))
        stop = start + window_size
    elif start is None:
        stop = min(int(stop), n_frames)
        start = max(0, stop - window_size)
    elif stop is None:
        start = max(0, int(start))
        stop = min(n_frames, start + window_size)
    else:
        start = max(0, int(start))
        stop = min(n_frames, int(stop))

    if stop <= start:
        raise ValueError(f"Invalid behavior window: start={start}, stop={stop}.")
    return start, stop


def _zscore_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = np.nanstd(values)
    if std == 0 or not np.isfinite(std):
        return np.zeros_like(values, dtype=float)
    return (values - np.nanmean(values)) / std


def _get_variant(variants: list[TraceVariant], name: str) -> TraceVariant:
    for variant in variants:
        if variant.name == name:
            return variant
    available = ", ".join(variant.name for variant in variants) or "<none>"
    raise ValueError(f"Unknown trace variant {name!r}. Available variants: {available}.")


def _validate_matching_trace_shape(reference: TraceVariant, variant: TraceVariant) -> None:
    if reference.traces.shape != variant.traces.shape:
        raise ValueError(
            f"{variant.name} has shape {variant.traces.shape}; "
            f"expected {reference.traces.shape} to match {reference.name}."
        )


def columnwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Shapes differ: {a.shape} != {b.shape}.")
    a_centered = a - np.mean(a, axis=0)
    b_centered = b - np.mean(b, axis=0)
    numerator = np.sum(a_centered * b_centered, axis=0)
    denominator = np.sqrt(np.sum(a_centered**2, axis=0) * np.sum(b_centered**2, axis=0))
    return _safe_divide(numerator, denominator)


def pearson_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"Shapes differ: {a.shape} != {b.shape}.")
    if a.size == 0 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan, dtype=float)
    return np.divide(numerator, denominator, out=out, where=denominator != 0)


def _safe_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    std = np.nanstd(x)
    if std == 0 or not np.isfinite(std):
        return np.zeros_like(x, dtype=float)
    return (x - np.nanmean(x)) / std


def _nanmean_axis0(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    counts = np.sum(~np.isnan(values), axis=0)
    totals = np.nansum(values, axis=0)
    return _safe_divide(totals, counts)


def _nan_stat(values: np.ndarray, fn) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.isnan(values).all():
        return np.nan
    return float(fn(values))

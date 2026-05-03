from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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


def load_trace_variants(
    dataset_dir: Path,
    raw_name: str = "fluo_signals_no_NaN.npy",
    cleaned_glob: str = "cleanedNew_*.npy",
) -> list[TraceVariant]:
    """Load raw and cleaned traces with a common time x neurons orientation."""
    dataset_dir = Path(dataset_dir)
    raw_path = dataset_dir / raw_name
    raw = _load_trace(raw_path)
    raw = orient_time_by_neurons(raw)
    raw = replace_nonfinite(raw)

    variants = [TraceVariant("raw", raw_path, raw)]
    for path in sorted(dataset_dir.glob(cleaned_glob)):
        traces = orient_time_by_neurons(_load_trace(path), expected_frames=raw.shape[0])
        traces = replace_nonfinite(traces)
        if traces.shape != raw.shape:
            raise ValueError(
                f"{path.name} has shape {traces.shape}; expected {raw.shape} to match raw traces."
            )
        variants.append(TraceVariant(path.stem, path, traces))
    return variants


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
    metric_rows: list[dict[str, object]] = []
    pred_rows: list[pd.DataFrame] = []

    for variant in variants:
        rows, preds = _evaluate_pair(
            feature_map[variant.name],
            feature_map[variant.name],
            aligned_y,
            time_index,
            folds,
            task,
            target_name,
            "within",
            variant.name,
            variant.name,
            ridge_alpha,
        )
        metric_rows.extend(rows)
        pred_rows.extend(preds)

    if include_transfer:
        reference = variants[0].name
        for variant in variants[1:]:
            rows, preds = _evaluate_pair(
                feature_map[reference],
                feature_map[variant.name],
                aligned_y,
                time_index,
                folds,
                task,
                target_name,
                "transfer_raw_to_clean",
                reference,
                variant.name,
                ridge_alpha,
            )
            metric_rows.extend(rows)
            pred_rows.extend(preds)

            rows, preds = _evaluate_pair(
                feature_map[variant.name],
                feature_map[reference],
                aligned_y,
                time_index,
                folds,
                task,
                target_name,
                "transfer_clean_to_raw",
                variant.name,
                reference,
                ridge_alpha,
            )
            metric_rows.extend(rows)
            pred_rows.extend(preds)

    if include_null:
        rng = np.random.default_rng(random_state)
        null_y = block_shuffle(aligned_y, block_size=null_block_size, rng=rng)
        for variant in variants:
            rows, preds = _evaluate_pair(
                feature_map[variant.name],
                feature_map[variant.name],
                null_y,
                time_index,
                folds,
                task,
                target_name,
                "null_within",
                variant.name,
                variant.name,
                ridge_alpha,
            )
            metric_rows.extend(rows)
            pred_rows.extend(preds)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    summary = summarize_metrics(metrics)
    return DecodingResult(metrics=metrics, predictions=predictions, summary=summary)


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

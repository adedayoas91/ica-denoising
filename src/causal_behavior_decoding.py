from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from behavior_decoding import BehaviorTargets, TraceVariant, blocked_folds


@dataclass(frozen=True)
class CausalStateConfig:
    window: int = 15
    target_shifts: tuple[int, ...] = (0, 1, 2, 4)
    latent_dim: int = 3
    n_splits: int = 5
    gap: int = 10
    ridge_alpha: float = 5.0
    random_state: int = 0


@dataclass(frozen=True)
class CausalStateResult:
    embeddings: pd.DataFrame
    fold_metrics: pd.DataFrame
    graph_metrics: pd.DataFrame
    residual_tests: pd.DataFrame
    config: CausalStateConfig


def make_paired_windows(
    traces: np.ndarray,
    targets: BehaviorTargets,
    config: CausalStateConfig,
    target_shift: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    traces = np.asarray(traces, dtype=float)
    if traces.ndim != 2:
        raise ValueError(f"traces must be 2D (time x neurons), got shape {traces.shape}.")
    if config.window < 2:
        raise ValueError("config.window must be at least 2.")
    if target_shift < 0:
        raise ValueError("target_shift must be non-negative.")

    n_frames = traces.shape[0]
    horizon = config.window + target_shift
    n_samples = n_frames - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Not enough frames ({n_frames}) for window={config.window} and shift={target_shift}."
        )

    x0 = np.empty((n_samples, config.window - 1, traces.shape[1]), dtype=float)
    x1 = np.empty_like(x0)
    times = np.empty(n_samples, dtype=int)
    angle = np.empty(n_samples, dtype=float)
    vigor = np.empty(n_samples, dtype=float)
    bout = np.empty(n_samples, dtype=int)

    for i in range(n_samples):
        start = i
        stop = i + config.window
        history = traces[start:stop]
        x0[i] = history[:-1]
        x1[i] = history[1:]
        # Anchor time at the end of x0 window.
        t_anchor = stop - 2
        t_target = t_anchor + target_shift
        times[i] = t_anchor
        angle[i] = float(targets.angle[t_target])
        vigor[i] = float(targets.vigor[t_target])
        bout[i] = int(targets.bout_state[t_target])

    target_map = {
        "angle": angle,
        "vigor": vigor,
        "bout_state": bout,
    }
    return x0, x1, target_map, times


def fit_causal_state_model(
    variants: list[TraceVariant],
    targets: BehaviorTargets,
    config: CausalStateConfig,
) -> CausalStateResult:
    if not variants:
        raise ValueError("At least one TraceVariant is required.")

    fold_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    embedding_rows: list[dict[str, object]] = []

    for variant in variants:
        for shift in config.target_shifts:
            x0, x1, target_map, times = make_paired_windows(variant.traces, targets, config, shift)
            x0_flat = x0.reshape(x0.shape[0], -1)
            x1_flat = x1.reshape(x1.shape[0], -1)
            folds = list(blocked_folds(x0.shape[0], n_splits=config.n_splits, gap=config.gap))

            for fold_id, train_idx, test_idx in folds:
                x0_train, x0_test = _standardize_train_test(x0_flat[train_idx], x0_flat[test_idx])
                x1_train, x1_test = _standardize_train_test(x1_flat[train_idx], x1_flat[test_idx])

                pca = PCA(n_components=min(config.latent_dim, x0_train.shape[1]), random_state=config.random_state)
                z0_train = pca.fit_transform(x0_train)
                z0_test = pca.transform(x0_test)
                z1_obs_train = pca.transform(x1_train)
                z1_obs_test = pca.transform(x1_test)

                # Transition model in latent space: z_{t+1} - z_t = f(z_t)
                transition = Ridge(alpha=config.ridge_alpha)
                transition.fit(z0_train, z1_obs_train - z0_train)
                z1_pred_test = z0_test + transition.predict(z0_test)
                dynamic_mse = float(mean_squared_error(z1_obs_test, z1_pred_test))

                y_angle_train = target_map["angle"][train_idx]
                y_angle_test = target_map["angle"][test_idx]
                y_vigor_train = target_map["vigor"][train_idx]
                y_vigor_test = target_map["vigor"][test_idx]
                y_bout_train = target_map["bout_state"][train_idx]
                y_bout_test = target_map["bout_state"][test_idx]

                angle_model = Ridge(alpha=config.ridge_alpha)
                angle_model.fit(z0_train, y_angle_train)
                angle_pred = angle_model.predict(z0_test)
                angle_pearson = _safe_pearson(y_angle_test, angle_pred)

                vigor_model = Ridge(alpha=config.ridge_alpha)
                vigor_model.fit(z0_train, y_vigor_train)
                vigor_pred = vigor_model.predict(z0_test)
                vigor_pearson = _safe_pearson(y_vigor_test, vigor_pred)

                if np.unique(y_bout_train).size < 2:
                    bout_pred = np.full(y_bout_test.shape[0], int(y_bout_train[0]))
                    bout_score = None
                else:
                    bout_model = LogisticRegression(
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=1000,
                        random_state=config.random_state,
                    )
                    bout_model.fit(z0_train, y_bout_train)
                    bout_pred = bout_model.predict(z0_test)
                    bout_score = bout_model.predict_proba(z0_test)[:, 1]

                bout_bal_acc = float(balanced_accuracy_score(y_bout_test, bout_pred))
                bout_auc = np.nan
                if bout_score is not None and np.unique(y_bout_test).size == 2:
                    bout_auc = float(roc_auc_score(y_bout_test, bout_score))

                fold_rows.append(
                    {
                        "variant": variant.name,
                        "target_shift": shift,
                        "fold": int(fold_id),
                        "n_train": int(train_idx.size),
                        "n_test": int(test_idx.size),
                        "dynamic_mse": dynamic_mse,
                        "angle_r2": _safe_r2(y_angle_test, angle_pred),
                        "angle_pearson": angle_pearson,
                        "vigor_r2": _safe_r2(y_vigor_test, vigor_pred),
                        "vigor_pearson": vigor_pearson,
                        "bout_balanced_accuracy": bout_bal_acc,
                        "bout_roc_auc": bout_auc,
                    }
                )

                residual_norm = np.linalg.norm(z1_obs_test - z1_pred_test, axis=1)
                residual_rows.append(
                    {
                        "variant": variant.name,
                        "target_shift": shift,
                        "fold": int(fold_id),
                        "residual_vigor_corr": _safe_pearson(residual_norm, y_vigor_test),
                        "residual_bout_corr": _safe_pearson(residual_norm, y_bout_test.astype(float)),
                    }
                )

            # Save one set of embeddings per variant/shift for downstream graph checks.
            x0_std_all, _ = _standardize_train_test(x0_flat, x0_flat)
            pca_all = PCA(n_components=min(config.latent_dim, x0_std_all.shape[1]), random_state=config.random_state)
            z_all = pca_all.fit_transform(x0_std_all)
            for i, t in enumerate(times):
                row = {
                    "variant": variant.name,
                    "target_shift": shift,
                    "time_index": int(t),
                    "angle": float(target_map["angle"][i]),
                    "vigor": float(target_map["vigor"][i]),
                    "bout_state": int(target_map["bout_state"][i]),
                }
                for d in range(z_all.shape[1]):
                    row[f"z{d + 1}"] = float(z_all[i, d])
                embedding_rows.append(row)

    embeddings = pd.DataFrame(embedding_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    residual_tests = pd.DataFrame(residual_rows)
    graph_metrics = compare_latent_graphs(embeddings, lag=1)
    return CausalStateResult(
        embeddings=embeddings,
        fold_metrics=fold_metrics,
        graph_metrics=graph_metrics,
        residual_tests=residual_tests,
        config=config,
    )


def evaluate_markov_sufficiency(
    traces: np.ndarray,
    z_t: np.ndarray,
    y: np.ndarray,
    folds: Iterable[tuple[int, np.ndarray, np.ndarray]],
    ridge_alpha: float = 5.0,
) -> pd.DataFrame:
    traces = np.asarray(traces, dtype=float)
    z_t = np.asarray(z_t, dtype=float)
    y = np.asarray(y, dtype=float)
    if traces.shape[0] != z_t.shape[0] or traces.shape[0] != y.shape[0]:
        raise ValueError("traces, z_t, and y must share sample dimension.")

    rows: list[dict[str, object]] = []
    for fold_id, train_idx, test_idx in folds:
        z_train, z_test = _standardize_train_test(z_t[train_idx], z_t[test_idx])
        x_train, x_test = _standardize_train_test(traces[train_idx], traces[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        base = Ridge(alpha=ridge_alpha)
        base.fit(z_train, y_train)
        y_base = base.predict(z_test)
        base_rmse = float(np.sqrt(mean_squared_error(y_test, y_base)))

        augmented_train = np.hstack([z_train, x_train])
        augmented_test = np.hstack([z_test, x_test])
        aug = Ridge(alpha=ridge_alpha)
        aug.fit(augmented_train, y_train)
        y_aug = aug.predict(augmented_test)
        aug_rmse = float(np.sqrt(mean_squared_error(y_test, y_aug)))

        rows.append(
            {
                "fold": int(fold_id),
                "base_rmse": base_rmse,
                "augmented_rmse": aug_rmse,
                "rmse_delta_aug_minus_base": aug_rmse - base_rmse,
            }
        )
    return pd.DataFrame(rows)


def compare_latent_graphs(
    embeddings: pd.DataFrame,
    lag: int = 1,
) -> pd.DataFrame:
    required = {"variant", "target_shift", "time_index", "angle", "vigor", "bout_state"}
    missing = sorted(required - set(embeddings.columns))
    if missing:
        raise ValueError(f"embeddings is missing required columns: {missing}")
    if lag < 1:
        raise ValueError("lag must be at least 1.")

    z_cols = sorted([col for col in embeddings.columns if col.startswith("z")])
    if not z_cols:
        raise ValueError("No latent columns found (expected z1, z2, ...).")

    rows: list[dict[str, object]] = []
    group_cols = ["variant", "target_shift"]
    for (variant, shift), group in embeddings.groupby(group_cols, dropna=False):
        group = group.sort_values("time_index").reset_index(drop=True)
        if len(group) <= lag:
            continue
        for z_col in z_cols:
            z_prev = group[z_col].to_numpy()[:-lag]
            angle_next = group["angle"].to_numpy()[lag:]
            vigor_next = group["vigor"].to_numpy()[lag:]
            bout_next = group["bout_state"].to_numpy()[lag:].astype(float)
            rows.append(
                {
                    "variant": variant,
                    "target_shift": int(shift),
                    "latent_dim": z_col,
                    "lag": int(lag),
                    "corr_to_angle_next": _safe_pearson(z_prev, angle_next),
                    "corr_to_vigor_next": _safe_pearson(z_prev, vigor_next),
                    "corr_to_bout_next": _safe_pearson(z_prev, bout_next),
                }
            )
    return pd.DataFrame(rows)


def _standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return (x_train - mean) / std, (x_test - mean) / std


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.unique(y_true).size < 2:
        return np.nan
    return float(r2_score(y_true, y_pred))

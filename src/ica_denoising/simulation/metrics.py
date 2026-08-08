"""Trace, behavior, and state metrics for the simulation benchmark (Section 5.1).

Graph-recovery metrics live in :mod:`csl.experiments.graph_metrics`; this module
covers the trace-domain, behavior-preservation, and state-diagnostic families.
"""

from __future__ import annotations

import numpy as np

__all__ = ["trace_metrics", "behavior_metrics", "state_metrics"]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    scale = float(np.std(target)) or 1.0
    return rmse / scale


def _effective_rank(matrix: np.ndarray) -> float:
    s = np.linalg.svd(matrix, compute_uv=False)
    s = s[s > 0]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def trace_metrics(
    variant: np.ndarray, clean: np.ndarray, artifact: np.ndarray
) -> dict[str, float]:
    """Trace-recovery metrics versus the clean fluorescence oracle."""

    per_neuron = [
        _pearson(variant[i], clean[i]) for i in range(variant.shape[0])
    ]
    residual = variant - clean
    s = np.linalg.svd(variant, compute_uv=False)
    s = s[s > 1e-12]
    cond = float(s[0] / s[-1]) if s.size else float("inf")
    artifact_resid_corr = _pearson(residual, artifact) if artifact.any() else 0.0
    return {
        "trace_pearson_global": _pearson(variant, clean),
        "trace_pearson_mean": float(np.mean(per_neuron)),
        "trace_rmse": float(np.sqrt(np.mean(residual**2))),
        "trace_nrmse": _nrmse(variant, clean),
        "retained_variance": float(np.var(variant) / (np.var(clean) or 1.0)),
        "artifact_residual_corr": artifact_resid_corr,
        "numerical_rank": float(np.linalg.matrix_rank(variant)),
        "stable_rank": float(np.sum(s**2) / (s[0] ** 2)) if s.size else 0.0,
        "effective_rank": _effective_rank(variant),
        "condition_number": cond,
    }


def _ridge_predict(x: np.ndarray, y: np.ndarray, split: float = 0.7, alpha: float = 1.0):
    t = x.shape[0]
    cut = int(t * split)
    xtr, xte = x[:cut], x[cut:]
    ytr, yte = y[:cut], y[cut:]
    mu = xtr.mean(axis=0)
    xtr_c = xtr - mu
    xte_c = xte - mu
    d = xtr_c.shape[1]
    beta = np.linalg.solve(xtr_c.T @ xtr_c + alpha * np.eye(d), xtr_c.T @ (ytr - ytr.mean()))
    pred = xte_c @ beta + ytr.mean()
    return pred, yte


def behavior_metrics(
    variant: np.ndarray, vigor: np.ndarray, bout: np.ndarray
) -> dict[str, float]:
    """Behavior-preservation metrics from a held-out ridge/logistic decode."""

    x = variant.T  # (T, n)
    pred, yte = _ridge_predict(x, vigor)
    out = {
        "behavior_pearson": _pearson(pred, yte),
        "behavior_nrmse": _nrmse(pred, yte),
    }
    # bout classification via thresholded continuous decode
    bout_pred, bout_te = _ridge_predict(x, bout.astype(float))
    threshold = np.median(bout_pred)
    pred_label = (bout_pred >= threshold).astype(int)
    out.update(_classification_scores(bout_te.astype(int), pred_label, bout_pred))
    return out


def _classification_scores(
    y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    auroc = float("nan")
    if 0 < y_true.sum() < y_true.size:
        try:
            from sklearn.metrics import roc_auc_score

            auroc = float(roc_auc_score(y_true, scores))
        except Exception:  # pragma: no cover
            pass
    return {
        "bout_balanced_accuracy": (recall + specificity) / 2,
        "bout_macro_f1": f1,
        "bout_auroc": auroc,
    }


def state_metrics(variant: np.ndarray, artifact: np.ndarray) -> dict[str, float]:
    """State diagnostics: persistence improvement and forward/reverse gap."""

    x = variant.T
    t, d = x.shape
    if t < 4:
        return {"persistence_improvement": 0.0, "forward_reverse_gap": 0.0}
    # one-step VAR(1) vs persistence
    z, y = x[:-1], x[1:]
    beta, *_ = np.linalg.lstsq(z - z.mean(0), y - y.mean(0), rcond=None)
    pred = (z - z.mean(0)) @ beta + y.mean(0)
    var_mse = float(np.mean((pred - y) ** 2))
    persistence_mse = float(np.mean((z - y) ** 2)) or 1.0
    forward = var_mse
    # reverse-time prediction
    xr = x[::-1]
    zr, yr = xr[:-1], xr[1:]
    betar, *_ = np.linalg.lstsq(zr - zr.mean(0), yr - yr.mean(0), rcond=None)
    predr = (zr - zr.mean(0)) @ betar + yr.mean(0)
    reverse = float(np.mean((predr - yr) ** 2))
    residual = variant - np.mean(variant)
    return {
        "dynamic_mse": var_mse,
        "persistence_improvement": float(1.0 - var_mse / persistence_mse),
        "forward_reverse_gap": float(reverse - forward),
        "residual_artifact_assoc": _pearson(residual, artifact) if artifact.any() else 0.0,
    }

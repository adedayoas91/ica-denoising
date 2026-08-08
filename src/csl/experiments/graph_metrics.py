"""Graph instability and recovery metrics for causal inference.

This module centralizes the experiment-facing metrics helpers so the package
does not need both ``metrics.py`` and ``graph_metrics.py``.
Recovery metrics still come from ``GcStar.compute_metrics()`` in
``causalised-GC.py`` when ground truth is available.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def zero_diagonal(adjacency: np.ndarray) -> np.ndarray:
    """Return a copy of adjacency with diagonal set to zero."""
    out = np.array(adjacency, copy=True)
    np.fill_diagonal(out, 0)
    return out


def compute_edge_counts(adjacencies: dict[int, np.ndarray]) -> dict[int, int]:
    """Count directed off-diagonal edges at each conditioning depth.

    Parameters
    ----------
    adjacencies : dict[int, np.ndarray]
        Dictionary mapping conditioning depth (n_past) to adjacency matrix.

    Returns
    -------
    dict[int, int]
        Number of edges at each conditioning depth.
    """
    return {
        n_past: int(zero_diagonal(adjacency).sum())
        for n_past, adjacency in adjacencies.items()
    }


def compute_graph_instability(adjacencies: dict[int, np.ndarray]) -> dict[int, float]:
    """Compute D_p (instability) between successive conditioning depths.

    Parameters
    ----------
    adjacencies : dict[int, np.ndarray]
        Dictionary mapping conditioning depth (n_past) to adjacency matrix.

    Returns
    -------
    dict[int, float]
        D_p values for each n_past > min (normalized adjacency change).
        D_p[n_past] = ||A(n_past) - A(n_past-1)|| / (d*(d-1))
    """
    n_pasts = sorted(adjacencies)
    if len(n_pasts) < 2:
        return {}

    d = adjacencies[n_pasts[0]].shape[0]
    m = d * (d - 1)
    output: dict[int, float] = {}

    for index in range(1, len(n_pasts)):
        previous_n = n_pasts[index - 1]
        current_n = n_pasts[index]
        previous = zero_diagonal(adjacencies[previous_n]).astype(int)
        current = zero_diagonal(adjacencies[current_n]).astype(int)
        output[current_n] = float(np.abs(current - previous).sum() / m)

    return output


def compute_instability_decomposition(
    adjacencies: dict[int, np.ndarray],
) -> dict[int, dict[str, float]]:
    """Decompose instability into edge deletions and additions.

    Parameters
    ----------
    adjacencies : dict[int, np.ndarray]
        Dictionary mapping conditioning depth (n_past) to adjacency matrix.

    Returns
    -------
    dict[int, dict[str, float]]
        For each n_past, {"D_minus": deletions, "D_plus": additions}.
    """
    n_pasts = sorted(adjacencies)
    if len(n_pasts) < 2:
        return {}

    d = adjacencies[n_pasts[0]].shape[0]
    m = d * (d - 1)
    output: dict[int, dict[str, float]] = {}

    for index in range(1, len(n_pasts)):
        previous_n = n_pasts[index - 1]
        current_n = n_pasts[index]
        previous = zero_diagonal(adjacencies[previous_n]).astype(int)
        current = zero_diagonal(adjacencies[current_n]).astype(int)
        deletions = np.logical_and(previous == 1, current == 0).sum() / m
        additions = np.logical_and(previous == 0, current == 1).sum() / m
        output[current_n] = {
            "D_minus": float(deletions),
            "D_plus": float(additions),
        }

    return output


def compute_stability_test_statistic(d_p: dict[int, float]) -> float:
    """Compute T_obs as the maximum instability across all transitions.

    Parameters
    ----------
    d_p : dict[int, float]
        Dictionary of D_p values from compute_graph_instability().

    Returns
    -------
    float
        T_obs = max(D_p). Low values suggest Markovian system.
    """
    return float(max(d_p.values())) if d_p else 0.0


def recovery_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Compute accuracy, precision, recall, and FPR for binary graphs."""

    predicted = zero_diagonal(predicted).astype(int)
    truth = zero_diagonal(truth).astype(int)

    tp = int(np.logical_and(predicted == 1, truth == 1).sum())
    tn = int(np.logical_and(predicted == 0, truth == 0).sum())
    fp = int(np.logical_and(predicted == 1, truth == 0).sum())
    fn = int(np.logical_and(predicted == 0, truth == 1).sum())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def summarize_run(
    adjacencies: dict[int, np.ndarray],
    truth: np.ndarray | None,
) -> dict[str, Any]:
    """Summarize one experiment run into manuscript-facing statistics."""

    d_stats = compute_graph_instability(adjacencies)
    d_parts = compute_instability_decomposition(adjacencies)
    counts = compute_edge_counts(adjacencies)
    summary: dict[str, Any] = {
        "D_p": d_stats,
        "D_parts": d_parts,
        "edge_counts": counts,
        "T_obs": compute_stability_test_statistic(d_stats),
    }
    if truth is not None:
        summary["metrics_by_p"] = {
            p_value: recovery_metrics(adjacency, truth)
            for p_value, adjacency in adjacencies.items()
        }
    return summary


def compute_graph_stability_metrics(
    adjacencies: dict[int, np.ndarray],
) -> dict[str, Any]:
    """Compute graph stability metrics across conditioning depths.

    Computes:
    - edge_counts: Number of inferred edges at each conditioning depth
    - D_p: Graph instability between consecutive conditioning depths
    - D_parts: Decomposition into deletions (D_minus) and additions (D_plus)
    - T_obs: Test statistic for Markovianity (max D_p)

    Recovery metrics (accuracy, precision, recall, F1, etc.) should be computed
    separately using GcStar.compute_metrics() for each conditioning depth.

    Parameters
    ----------
    adjacencies : dict[int, np.ndarray]
        Dictionary mapping conditioning depth (n_past) to adjacency matrix.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys: edge_counts, D_p, D_parts, T_obs.
    """
    d_p = compute_graph_instability(adjacencies)
    d_parts = compute_instability_decomposition(adjacencies)
    counts = compute_edge_counts(adjacencies)

    return {
        "edge_counts": counts,
        "D_p": d_p,
        "D_parts": d_parts,
        "T_obs": compute_stability_test_statistic(d_p),
    }


# Backwards-compatible aliases used by older experiment code.
compact_graph_instability = compute_graph_instability
edge_counts = compute_edge_counts


def summarize_gcstar_branch(
    adjacencies: dict[int, np.ndarray],
    *,
    ground_truth_available: bool,
    metrics_by_n_past: dict[int, dict[str, float]] | None = None,
    shd_by_n_past: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Route to recovery or stability outputs depending on ground-truth availability.

    When ground truth is available, this function returns the recovery-side
    payload produced elsewhere by GcStar / causalised-GC.py.
    When ground truth is unavailable, it returns graph stability metrics.
    """

    if ground_truth_available:
        if metrics_by_n_past is None:
            raise ValueError(
                "metrics_by_n_past is required when ground_truth_available is True."
            )

        payload: dict[str, Any] = {
            "metrics_by_n_past": metrics_by_n_past,
        }
        if shd_by_n_past is not None:
            payload["shd_by_n_past"] = shd_by_n_past
        return payload

    return compute_graph_stability_metrics(adjacencies)


# ---------------------------------------------------------------------------
# Directed graph-recovery metrics (Section 5.1)
# ---------------------------------------------------------------------------
def _offdiag_mask(n: int) -> np.ndarray:
    """Boolean mask selecting off-diagonal entries of an ``(n, n)`` matrix."""

    mask = ~np.eye(n, dtype=bool)
    return mask


def confusion_counts(predicted: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    """Return TP/FP/TN/FN for binary directed graphs (diagonal excluded)."""

    n = truth.shape[0]
    mask = _offdiag_mask(n)
    pred = (zero_diagonal(predicted) != 0)[mask]
    true = (zero_diagonal(truth) != 0)[mask]
    tp = int(np.logical_and(pred, true).sum())
    fp = int(np.logical_and(pred, ~true).sum())
    tn = int(np.logical_and(~pred, ~true).sum())
    fn = int(np.logical_and(~pred, true).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def structural_hamming_distance(predicted: np.ndarray, truth: np.ndarray) -> int:
    """Number of differing directed off-diagonal edges."""

    n = truth.shape[0]
    mask = _offdiag_mask(n)
    pred = (zero_diagonal(predicted) != 0)[mask]
    true = (zero_diagonal(truth) != 0)[mask]
    return int(np.sum(pred != true))


def edge_density_bias(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Predicted minus true off-diagonal edge density."""

    n = truth.shape[0]
    denom = n * (n - 1)
    if denom == 0:
        return 0.0
    pred = int((zero_diagonal(predicted) != 0).sum())
    true = int((zero_diagonal(truth) != 0).sum())
    return float((pred - true) / denom)


def direction_reversal_count(predicted: np.ndarray, truth: np.ndarray) -> int:
    """Edges present in the wrong direction (``t->s`` predicted, ``s->t`` true)."""

    pred = (zero_diagonal(predicted) != 0).astype(int)
    true = (zero_diagonal(truth) != 0).astype(int)
    reversed_edges = np.logical_and(
        np.logical_and(true == 1, pred == 0),
        pred.T == 1,
    )
    return int(reversed_edges.sum())


def jaccard_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard overlap of edge sets of two directed graphs (diagonal excluded)."""

    ea = zero_diagonal(a) != 0
    eb = zero_diagonal(b) != 0
    union = int(np.logical_or(ea, eb).sum())
    if union == 0:
        return 1.0
    inter = int(np.logical_and(ea, eb).sum())
    return float(inter / union)


def matthews_corrcoef(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Matthews correlation coefficient for binary directed graphs."""

    counts = confusion_counts(predicted, truth)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    numerator = tp * tn - fp * fn
    denominator = np.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def score_auroc_ap(scores: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Weighted-score AUROC and average precision (diagonal excluded)."""

    n = truth.shape[0]
    mask = _offdiag_mask(n)
    y = (zero_diagonal(truth) != 0)[mask].astype(int)
    s = np.abs(zero_diagonal(scores))[mask].astype(float)
    out = {"auroc": float("nan"), "average_precision": float("nan")}
    if y.sum() == 0 or y.sum() == y.size:
        return out
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        out["auroc"] = float(roc_auc_score(y, s))
        out["average_precision"] = float(average_precision_score(y, s))
    except Exception:  # pragma: no cover - sklearn always present here
        pass
    return out


def graph_recovery_metrics(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    scores: np.ndarray | None = None,
    reference: np.ndarray | None = None,
) -> dict[str, float]:
    """Full directed graph-recovery metric bundle (Section 5.1).

    Args:
        predicted: Predicted (weighted or binary) adjacency ``[source, target]``.
        truth: Ground-truth binary adjacency ``[source, target]``.
        scores: Optional continuous edge scores for AUROC/AP.
        reference: Optional reference graph (e.g. clean-oracle estimate) for
            Jaccard overlap.

    Returns:
        Dictionary of directed precision, recall, F1, specificity, FPR, MCC,
        SHD, edge-density bias, direction-reversal count, and optional
        score-based and reference-overlap metrics. Diagonal entries excluded.
    """

    counts = confusion_counts(predicted, truth)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    metrics: dict[str, float] = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "mcc": matthews_corrcoef(predicted, truth),
        "shd": float(structural_hamming_distance(predicted, truth)),
        "edge_density_bias": edge_density_bias(predicted, truth),
        "direction_reversals": float(direction_reversal_count(predicted, truth)),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }
    if scores is not None:
        metrics.update(score_auroc_ap(scores, truth))
    if reference is not None:
        metrics["jaccard_vs_reference"] = jaccard_overlap(predicted, reference)
    return metrics

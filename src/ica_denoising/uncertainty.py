"""Block-level metric contributions and uncertainty aggregation (Section 5.6).

Behavior-level predictions already support paired block-bootstrap intervals
(see :mod:`ica_denoising.evaluation_runner`). Other metrics (trace
correlation/RMSE/NRMSE, dynamic MSE & persistence improvement, extra-history
improvement, residual-artifact association, BPI components) need their
*resampling units* saved so they can be recomputed and resampled later.

This module saves, per fixed temporal block within a recording, the **sufficient
statistics** needed to recompute each global metric exactly. Aggregation across
fish reports the three estimates directly plus the median and range. Cluster
stability intervals reuse existing stability seed pairs but are explicitly
labelled algorithmic (not biological) replicates.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "assign_temporal_blocks",
    "ratio_metric_contributions",
    "recompute_ratio_metric",
    "correlation_contributions",
    "recompute_correlation",
    "rmse_contributions",
    "recompute_rmse",
    "recompute_nrmse",
    "aggregate_across_units",
    "label_algorithmic_replicates",
]


def assign_temporal_blocks(
    time_index: Sequence[int],
    *,
    block_size: int,
) -> np.ndarray:
    """Assign each frame to a fixed temporal block within a recording.

    Args:
        time_index: Per-sample frame indices.
        block_size: Number of frames per block.

    Returns:
        Integer block ids (``time_index // block_size``).
    """
    if block_size < 1:
        raise ValueError("block_size must be at least 1.")
    times = np.asarray(time_index, dtype=int)
    return times // int(block_size)


def _grouped(blocks: np.ndarray) -> list[tuple[int, np.ndarray]]:
    order = np.argsort(blocks, kind="stable")
    sorted_blocks = blocks[order]
    boundaries = np.flatnonzero(np.diff(sorted_blocks)) + 1
    groups = []
    for piece in np.split(order, boundaries):
        groups.append((int(blocks[piece[0]]), piece))
    return groups


def rmse_contributions(
    reference: np.ndarray,
    estimate: np.ndarray,
    blocks: Sequence[int],
    *,
    scale: float | None = None,
) -> pd.DataFrame:
    """Per-block sufficient statistics for (N)RMSE.

    Args:
        reference: Reference values (e.g. raw traces flattened per sample).
        estimate: Estimated values aligned with ``reference``.
        blocks: Block id per sample.
        scale: Optional fixed normalization scale for NRMSE.

    Returns:
        DataFrame with one row per block: ``block``, ``n``, ``sse``, ``scale``.
    """
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    block_ids = np.asarray(blocks, dtype=int)
    if not (ref.shape[0] == est.shape[0] == block_ids.shape[0]):
        raise ValueError("reference, estimate, and blocks must share length.")
    squared_error = (ref - est) ** 2
    if squared_error.ndim > 1:
        squared_error = squared_error.sum(axis=tuple(range(1, squared_error.ndim)))
        per_sample = ref[0].size if ref.ndim > 1 else 1
    else:
        per_sample = 1
    rows = []
    for block_id, idx in _grouped(block_ids):
        rows.append(
            {
                "block": block_id,
                "n": int(idx.size * per_sample),
                "sse": float(squared_error[idx].sum()),
                "scale": float("nan") if scale is None else float(scale),
            }
        )
    return pd.DataFrame(rows)


def recompute_rmse(contributions: pd.DataFrame) -> float:
    """Recompute global RMSE from per-block sufficient statistics."""
    total_n = float(contributions["n"].sum())
    if total_n <= 0:
        return float("nan")
    return float(np.sqrt(contributions["sse"].sum() / total_n))


def recompute_nrmse(contributions: pd.DataFrame) -> float:
    """Recompute global NRMSE from per-block stats using the stored scale."""
    rmse = recompute_rmse(contributions)
    scale = (
        float(contributions["scale"].iloc[0]) if len(contributions) else float("nan")
    )
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float(rmse / scale)


def ratio_metric_contributions(
    numerator_se: np.ndarray,
    denominator_se: np.ndarray,
    blocks: Sequence[int],
) -> pd.DataFrame:
    """Per-block stats for ratio-improvement metrics (e.g. dynamic vs persistence).

    The global metric is ``1 - sum(numerator_se) / sum(denominator_se)`` which is
    the form used for dynamic-MSE persistence improvement and extra-history
    improvement.

    Args:
        numerator_se: Per-sample squared error of the model under test.
        denominator_se: Per-sample squared error of the reference model.
        blocks: Block id per sample.

    Returns:
        DataFrame with ``block``, ``n``, ``numerator``, ``denominator``.
    """
    num = np.asarray(numerator_se, dtype=float)
    den = np.asarray(denominator_se, dtype=float)
    block_ids = np.asarray(blocks, dtype=int)
    if not (num.shape[0] == den.shape[0] == block_ids.shape[0]):
        raise ValueError("numerator_se, denominator_se, and blocks must share length.")
    rows = []
    for block_id, idx in _grouped(block_ids):
        rows.append(
            {
                "block": block_id,
                "n": int(idx.size),
                "numerator": float(num[idx].sum()),
                "denominator": float(den[idx].sum()),
            }
        )
    return pd.DataFrame(rows)


def recompute_ratio_metric(contributions: pd.DataFrame) -> float:
    """Recompute ``1 - sum(numerator)/sum(denominator)`` from block stats."""
    denominator = float(contributions["denominator"].sum())
    if denominator <= 0:
        return float("nan")
    return float(1.0 - contributions["numerator"].sum() / denominator)


def correlation_contributions(
    left: np.ndarray,
    right: np.ndarray,
    blocks: Sequence[int],
) -> pd.DataFrame:
    """Per-block sufficient statistics for a Pearson correlation.

    Suitable for trace correlation and residual-artifact association. The global
    Pearson r is recoverable from the aggregated sums.

    Args:
        left: First signal.
        right: Second signal aligned with ``left``.
        blocks: Block id per sample.

    Returns:
        DataFrame with ``block``, ``n``, ``sum_x``, ``sum_y``, ``sum_xx``,
        ``sum_yy``, ``sum_xy``.
    """
    x = np.asarray(left, dtype=float).reshape(-1)
    y = np.asarray(right, dtype=float).reshape(-1)
    block_ids = np.asarray(blocks, dtype=int).reshape(-1)
    if not (x.shape[0] == y.shape[0] == block_ids.shape[0]):
        raise ValueError("left, right, and blocks must share length.")
    rows = []
    for block_id, idx in _grouped(block_ids):
        bx, by = x[idx], y[idx]
        rows.append(
            {
                "block": block_id,
                "n": int(idx.size),
                "sum_x": float(bx.sum()),
                "sum_y": float(by.sum()),
                "sum_xx": float((bx * bx).sum()),
                "sum_yy": float((by * by).sum()),
                "sum_xy": float((bx * by).sum()),
            }
        )
    return pd.DataFrame(rows)


def recompute_correlation(contributions: pd.DataFrame) -> float:
    """Recompute global Pearson correlation from per-block sums."""
    n = float(contributions["n"].sum())
    if n <= 1:
        return float("nan")
    sum_x = float(contributions["sum_x"].sum())
    sum_y = float(contributions["sum_y"].sum())
    sum_xx = float(contributions["sum_xx"].sum())
    sum_yy = float(contributions["sum_yy"].sum())
    sum_xy = float(contributions["sum_xy"].sum())
    cov = sum_xy - sum_x * sum_y / n
    var_x = sum_xx - sum_x * sum_x / n
    var_y = sum_yy - sum_y * sum_y / n
    denominator = np.sqrt(var_x * var_y)
    if denominator <= 0:
        return float("nan")
    return float(cov / denominator)


def aggregate_across_units(
    values: Mapping[str, float] | Sequence[float],
) -> dict[str, object]:
    """Aggregate per-unit (per-fish) estimates for an across-fish report.

    Reports the individual estimates directly plus their median and range, as
    required when only a few biological replicates are available.

    Args:
        values: Mapping of unit label to estimate, or a sequence of estimates.

    Returns:
        Dict with ``estimates`` (list), ``unit_labels`` (list or ``None``),
        ``n_units``, ``median``, ``min``, ``max``, and ``range``.
    """
    if isinstance(values, Mapping):
        labels = list(values.keys())
        estimates = np.asarray(list(values.values()), dtype=float)
    else:
        labels = None
        estimates = np.asarray(list(values), dtype=float)
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return {
            "estimates": estimates.tolist(),
            "unit_labels": labels,
            "n_units": 0,
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "range": float("nan"),
        }
    return {
        "estimates": estimates.tolist(),
        "unit_labels": labels,
        "n_units": int(finite.size),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "range": float(np.max(finite) - np.min(finite)),
    }


def label_algorithmic_replicates(stability: pd.DataFrame) -> pd.DataFrame:
    """Tag a cluster-stability table as algorithmic (not biological) replicates.

    Args:
        stability: Cluster-stability table (e.g. seed-pair agreements).

    Returns:
        A copy with a ``replicate_type`` column set to ``"algorithmic"``.
    """
    tagged = stability.copy()
    tagged["replicate_type"] = "algorithmic"
    return tagged

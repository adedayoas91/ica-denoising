"""Tidy per-variant summaries derived from saved connectivity artifacts.

These helpers stitch the pure metric and enrichment functions into tables that
can be dropped straight into the paper without rerunning any estimator.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .config import ConnectivityAnalysisConfig, NodeAnnotations
from .enrichment import (
    EnrichmentResult,
    directional_enrichment_test,
    group_enrichment_test,
    ipsilateral_enrichment_test,
)
from .loaders import load_variant_matrices
from .metrics import (
    condition_number,
    degree_changes,
    edge_count,
    edge_density,
    effective_rank,
    jaccard_overlap,
    normalized_hamming,
    numerical_rank,
    reciprocal_fraction,
    stable_rank,
)

logger = logging.getLogger(__name__)

__all__ = ["summarize_recording", "compare_estimators"]

_RAW_VARIANT = "raw"


def _per_variant_rows(
    matrices: Mapping[str, np.ndarray],
    traces: Mapping[str, np.ndarray] | None,
    cfg: ConnectivityAnalysisConfig,
) -> list[dict[str, object]]:
    """Build per-variant metric rows.

    Args:
        matrices: Mapping of variant name to weighted matrix.
        traces: Optional mapping of variant name to trace array for rank metrics.
        cfg: Analysis configuration.

    Returns:
        A list of row dictionaries (one per variant).
    """

    raw = matrices.get(_RAW_VARIANT)
    raw_density = edge_density(raw, cfg.edge_threshold) if raw is not None else None
    rows: list[dict[str, object]] = []
    for variant, matrix in matrices.items():
        density = edge_density(matrix, cfg.edge_threshold)
        row: dict[str, object] = {
            "variant": variant,
            "n_nodes": int(matrix.shape[0]),
            "edge_count": edge_count(matrix, cfg.edge_threshold),
            "edge_density": density,
            "reciprocal_fraction": reciprocal_fraction(matrix, cfg.edge_threshold),
            "is_raw": variant == _RAW_VARIANT,
        }
        if raw is not None and matrix.shape == raw.shape:
            row["density_change_vs_raw"] = density - float(raw_density)
            row["jaccard_vs_raw"] = jaccard_overlap(matrix, raw, cfg.edge_threshold)
            row["hamming_vs_raw"] = normalized_hamming(matrix, raw, cfg.edge_threshold)
            change = degree_changes(matrix, raw, cfg.edge_threshold)
            row["mean_in_degree_delta"] = float(np.mean(change.in_degree_delta))
            row["mean_out_degree_delta"] = float(np.mean(change.out_degree_delta))
            row["max_abs_in_degree_delta"] = float(
                np.max(np.abs(change.in_degree_delta))
            )
            row["max_abs_out_degree_delta"] = float(
                np.max(np.abs(change.out_degree_delta))
            )
        if traces is not None and variant in traces:
            trace = np.asarray(traces[variant], dtype=float)
            row["numerical_rank"] = numerical_rank(trace, cfg.rank_tolerance)
            row["effective_rank"] = effective_rank(trace)
            row["stable_rank"] = stable_rank(trace)
            row["condition_number"] = condition_number(trace)
        rows.append(row)
    return rows


def _rank_density_association(per_variant: pd.DataFrame) -> pd.DataFrame:
    """Associate rank loss with density increase across variants.

    Args:
        per_variant: Per-variant metric table.

    Returns:
        A one-row table with Pearson and Spearman correlations between rank loss
        (raw rank minus variant rank) and density increase. Empty when the
        required columns are missing or fewer than three points exist.
    """

    if "numerical_rank" not in per_variant or "density_change_vs_raw" not in per_variant:
        return pd.DataFrame()
    raw_rows = per_variant[per_variant["is_raw"]]
    if raw_rows.empty:
        return pd.DataFrame()
    raw_rank = float(raw_rows["numerical_rank"].iloc[0])
    raw_eff = float(raw_rows["effective_rank"].iloc[0])
    bss = per_variant[~per_variant["is_raw"]].dropna(
        subset=["numerical_rank", "density_change_vs_raw"]
    )
    if len(bss) < 3:
        return pd.DataFrame()
    rank_loss = raw_rank - bss["numerical_rank"].to_numpy(dtype=float)
    eff_loss = raw_eff - bss["effective_rank"].to_numpy(dtype=float)
    density_increase = bss["density_change_vs_raw"].to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "n_variants": int(len(bss)),
                "pearson_numrank_loss_vs_density": _safe_corr(rank_loss, density_increase),
                "pearson_effrank_loss_vs_density": _safe_corr(eff_loss, density_increase),
            }
        ]
    )


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation guarded against zero-variance inputs.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        The Pearson correlation, or ``nan`` when either vector is constant.
    """

    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _enrichment_rows(
    matrices: Mapping[str, np.ndarray],
    annotations: NodeAnnotations,
    cfg: ConnectivityAnalysisConfig,
) -> list[dict[str, object]]:
    """Run available enrichment tests for every variant.

    Args:
        matrices: Mapping of variant name to weighted matrix.
        annotations: Node annotations describing which tests can run.
        cfg: Analysis configuration.

    Returns:
        A list of enrichment result rows (possibly empty).
    """

    rows: list[dict[str, object]] = []
    for variant, matrix in matrices.items():
        n_nodes = matrix.shape[0]
        tests: dict[str, EnrichmentResult] = {}
        if (
            annotations.group_labels is not None
            and len(annotations.group_labels) == n_nodes
        ):
            tests["group"] = group_enrichment_test(
                matrix,
                annotations.group_labels,
                annotations.source_group,
                annotations.target_group,
                threshold=cfg.edge_threshold,
                n_permutations=cfg.n_permutations,
                rng=cfg.rng(),
            )
        if annotations.side is not None and len(annotations.side) == n_nodes:
            tests["ipsilateral"] = ipsilateral_enrichment_test(
                matrix,
                annotations.side,
                threshold=cfg.edge_threshold,
                n_permutations=cfg.n_permutations,
                rng=cfg.rng(),
            )
        if (
            annotations.rostro_caudal is not None
            and len(annotations.rostro_caudal) == n_nodes
        ):
            tests["directional"] = directional_enrichment_test(
                matrix,
                annotations.rostro_caudal,
                threshold=cfg.edge_threshold,
                n_permutations=cfg.n_permutations,
                rng=cfg.rng(),
            )
        for test_name, result in tests.items():
            rows.append(
                {
                    "variant": variant,
                    "test": test_name,
                    "statistic_name": result.statistic_name,
                    "observed": result.observed,
                    "p_value": result.p_value,
                    "null_mean": result.null_mean,
                    "null_std": result.null_std,
                    "null_q05": result.null_q05,
                    "null_q95": result.null_q95,
                    "n_permutations": result.n_permutations,
                    "n_edges": result.n_edges,
                }
            )
    return rows


def summarize_recording(
    estimator_dir: str | Path,
    traces: Mapping[str, np.ndarray] | None = None,
    node_metadata: NodeAnnotations | None = None,
    config: ConnectivityAnalysisConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Summarise a recording's saved connectivity matrices.

    Args:
        estimator_dir: Path to an ``.../<estimator>/`` directory.
        traces: Optional mapping of variant name to trace array (shape
            ``(n_neurons, n_frames)``) used for rank metrics. Callers must
            supply these because the stored ``source_path`` values are stale.
        node_metadata: Optional node annotations enabling enrichment tests.
        config: Analysis configuration; defaults are used when ``None``.

    Returns:
        A dict of tidy DataFrames with keys ``"per_variant"``,
        ``"rank_density_association"``, and ``"enrichment"``.
    """

    cfg = config or ConnectivityAnalysisConfig()
    matrices = load_variant_matrices(estimator_dir, cfg)
    per_variant = pd.DataFrame(_per_variant_rows(matrices, traces, cfg))
    association = _rank_density_association(per_variant)
    annotations = node_metadata or NodeAnnotations()
    enrichment = pd.DataFrame(_enrichment_rows(matrices, annotations, cfg))
    return {
        "per_variant": per_variant,
        "rank_density_association": association,
        "enrichment": enrichment,
    }


def compare_estimators(
    cgc_dir: str | Path,
    cgc_star_dir: str | Path,
    config: ConnectivityAnalysisConfig | None = None,
) -> pd.DataFrame:
    """Compare matched variants across the c-GC and c-GC* estimator folders.

    Args:
        cgc_dir: Path to the ``c-GC`` estimator directory.
        cgc_star_dir: Path to the ``c-GC-star`` estimator directory.
        config: Analysis configuration; defaults are used when ``None``.

    Returns:
        A tidy DataFrame with per-variant Jaccard overlap and normalized Hamming
        distance between the two estimators (variants present in both folders).
    """

    cfg = config or ConnectivityAnalysisConfig()
    cgc = load_variant_matrices(cgc_dir, cfg)
    cgc_star = load_variant_matrices(cgc_star_dir, cfg)
    shared = [v for v in cgc if v in cgc_star]
    rows: list[dict[str, object]] = []
    for variant in shared:
        mat_a = cgc[variant]
        mat_b = cgc_star[variant]
        if mat_a.shape != mat_b.shape:
            logger.warning(
                "Skipping variant %s: shape mismatch %s vs %s.",
                variant,
                mat_a.shape,
                mat_b.shape,
            )
            continue
        rows.append(
            {
                "variant": variant,
                "jaccard_cgc_vs_cgcstar": jaccard_overlap(
                    mat_a, mat_b, cfg.edge_threshold
                ),
                "hamming_cgc_vs_cgcstar": normalized_hamming(
                    mat_a, mat_b, cfg.edge_threshold
                ),
                "edge_count_cgc": edge_count(mat_a, cfg.edge_threshold),
                "edge_count_cgcstar": edge_count(mat_b, cfg.edge_threshold),
            }
        )
    return pd.DataFrame(rows)

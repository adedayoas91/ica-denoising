"""Aggregation and reporting for the simulation benchmark (Section 5.1).

Every figure/table must be regeneratable from the per-replicate aggregate CSVs;
these helpers load those CSVs and produce benchmark-level summaries.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = [
    "collect_metric_table",
    "summarize_graph_recovery",
    "paired_vs_raw",
]

_METRIC_FILES = {
    "trace": "trace_metrics.csv",
    "behavior": "behavior_metrics.csv",
    "state": "state_metrics.csv",
    "graph": "graph_metrics.csv",
}


def collect_metric_table(benchmark_root: str | Path, family: str) -> pd.DataFrame:
    """Concatenate per-replicate metric CSVs of one ``family`` across the tree.

    Args:
        benchmark_root: ``outputs/simulation/<benchmark_version>`` directory.
        family: One of ``trace``, ``behavior``, ``state``, ``graph``.
    """

    if family not in _METRIC_FILES:
        raise ValueError(f"Unknown metric family: {family}")
    root = Path(benchmark_root)
    frames = []
    for csv_path in sorted(root.glob(f"*/seed_*/{_METRIC_FILES[family]}")):
        scenario = csv_path.parent.parent.name
        seed = int(csv_path.parent.name.replace("seed_", ""))
        frame = pd.read_csv(csv_path)
        frame.insert(0, "scenario", scenario)
        frame.insert(1, "seed", seed)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_graph_recovery(benchmark_root: str | Path) -> pd.DataFrame:
    """Median/IQR of graph F1 and SHD per scenario, estimator, and variant."""

    table = collect_metric_table(benchmark_root, "graph")
    if table.empty:
        return table
    grouped = table.groupby(["scenario", "estimator", "variant_id"])
    return grouped.agg(
        f1_median=("f1", "median"),
        f1_q25=("f1", lambda s: s.quantile(0.25)),
        f1_q75=("f1", lambda s: s.quantile(0.75)),
        shd_median=("shd", "median"),
        n=("f1", "size"),
    ).reset_index()


def paired_vs_raw(benchmark_root: str | Path, metric: str = "f1") -> pd.DataFrame:
    """Paired difference of a graph metric against the raw variant per replicate.

    The primary simulation outcome is the paired graph metric relative to the
    corrupted raw traces (Section 5.1).
    """

    table = collect_metric_table(benchmark_root, "graph")
    if table.empty:
        return table
    keys = ["scenario", "seed", "estimator"]
    raw = table[table["variant_id"] == "raw"][keys + [metric]].rename(
        columns={metric: f"{metric}_raw"}
    )
    merged = table.merge(raw, on=keys, how="left")
    merged[f"{metric}_minus_raw"] = merged[metric] - merged[f"{metric}_raw"]
    return merged

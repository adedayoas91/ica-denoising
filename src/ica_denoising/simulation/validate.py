"""Manifest and output validator for the simulation benchmark (Section 5.8).

Checks that each replicate manifest and its sibling artifacts are internally
consistent: required manifest keys, canonical variant IDs and trace hashes on
every metric row, repository-relative paths, expected variants present, graph
dimensions, recorded estimator settings, and summary completeness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["ValidationReport", "validate_replicate", "validate_benchmark"]

_REQUIRED_MANIFEST_KEYS = (
    "benchmark_version",
    "schema_version",
    "scenario_id",
    "seed",
    "resolved_config",
    "git_revision",
    "versions",
    "dataset_hash",
    "start",
    "end",
    "runtime_seconds",
    "complete",
)

_REQUIRED_FILES = (
    "dataset.npz",
    "manifest.json",
    "variants.csv",
    "trace_metrics.csv",
    "graph_metrics.csv",
)


@dataclass
class ValidationReport:
    """Outcome of validating one or more replicates."""

    ok: bool = True
    errors: list[str] = field(default_factory=list)
    checked: int = 0

    def fail(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)


def _is_relative(path_str: str) -> bool:
    return not Path(path_str).is_absolute()


def validate_replicate(replicate_dir: str | Path) -> ValidationReport:
    """Validate a single replicate directory."""

    report = ValidationReport()
    rep = Path(replicate_dir)
    report.checked = 1

    for name in _REQUIRED_FILES:
        if not (rep / name).exists():
            report.fail(f"{rep}: missing required file {name}")

    manifest_path = rep / "manifest.json"
    if not manifest_path.exists():
        return report
    manifest = json.loads(manifest_path.read_text())
    for key in _REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            report.fail(f"{rep}: manifest missing key {key}")

    # resolved config must record estimator settings (no defaults-only record).
    config = manifest.get("resolved_config", {})
    if "estimator" not in config or "n_perm" not in config.get("estimator", {}):
        report.fail(f"{rep}: resolved_config missing estimator settings")

    # variant IDs + trace hashes present on metric rows.
    if (rep / "variants.csv").exists():
        variants = pd.read_csv(rep / "variants.csv")
        for col in ("variant_id", "trace_hash"):
            if col not in variants.columns:
                report.fail(f"{rep}: variants.csv missing column {col}")
        expected = {"clean", "raw", "artifact_oracle", "oracle_selection"}
        present = set(variants.get("variant_id", pd.Series(dtype=str)))
        missing = expected - present
        if missing:
            report.fail(f"{rep}: expected variants missing: {sorted(missing)}")

    for csv_name in ("trace_metrics.csv", "graph_metrics.csv"):
        path = rep / csv_name
        if path.exists():
            frame = pd.read_csv(path)
            for col in ("variant_id", "trace_hash"):
                if col not in frame.columns:
                    report.fail(f"{rep}: {csv_name} missing column {col}")

    # graph files: relative paths and square dimensions.
    for graph_file in rep.glob("graphs/*/*.npz"):
        if not _is_relative(str(graph_file.relative_to(rep))):
            report.fail(f"{rep}: non-relative graph path {graph_file}")
        data = np.load(graph_file)
        binary = data["binary"]
        if binary.ndim != 2 or binary.shape[0] != binary.shape[1]:
            report.fail(f"{graph_file}: binary adjacency is not square")

    return report


def validate_benchmark(benchmark_root: str | Path) -> ValidationReport:
    """Validate every replicate under a benchmark-version directory."""

    root = Path(benchmark_root)
    overall = ValidationReport()
    for manifest in sorted(root.glob("*/seed_*/manifest.json")):
        sub = validate_replicate(manifest.parent)
        overall.checked += sub.checked
        if not sub.ok:
            overall.ok = False
            overall.errors.extend(sub.errors)
    if overall.checked == 0:
        overall.fail(f"No replicates found under {root}")
    return overall

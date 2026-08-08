"""Replace selected graph-estimator outputs between simulation benchmark roots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import pandas as pd

__all__ = ["ReplacementReport", "replace_graph_estimator_outputs"]


@dataclass(frozen=True)
class ReplacementReport:
    """Summary of graph-estimator replacement across replicate directories."""

    checked: int
    replaced: int
    estimators: tuple[str, ...]
    source_root: str
    target_root: str


def replace_graph_estimator_outputs(
    source_root: str | Path,
    target_root: str | Path,
    *,
    estimators: tuple[str, ...] = ("cgc", "cgc_star"),
    dry_run: bool = False,
) -> ReplacementReport:
    """Replace selected graph-estimator artifacts in an existing benchmark root.

    The source benchmark should contain replacement graph outputs for the same
    scenario/seed replicate layout as the target benchmark. Only
    ``graph_metrics.csv``, ``failures.csv``, matching ``graphs/<estimator>``
    files, and manifest replacement metadata are touched in the target.
    """

    source_root = Path(source_root)
    target_root = Path(target_root)
    estimator_set = {str(estimator) for estimator in estimators}
    source_manifests = sorted(source_root.glob("*/seed_*/manifest.json"))
    if not source_manifests:
        raise FileNotFoundError(f"No source replicates found under {source_root}")

    replaced = 0
    for source_manifest_path in source_manifests:
        source_dir = source_manifest_path.parent
        relative = source_dir.relative_to(source_root)
        target_dir = target_root / relative
        if not (target_dir / "manifest.json").exists():
            raise FileNotFoundError(f"Missing target replicate: {target_dir}")

        replacement_rows = _replacement_rows(source_dir, estimator_set)
        copy_pairs = _graph_copy_pairs(source_dir, target_dir, replacement_rows)
        if dry_run:
            replaced += 1
            continue

        for source_graph, target_graph in copy_pairs:
            target_graph.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_graph, target_graph)
        _replace_graph_metric_rows(target_dir, replacement_rows, estimator_set)
        _replace_failure_rows(source_dir, target_dir, estimator_set)
        _record_manifest_replacement(
            source_dir,
            target_dir,
            estimators=tuple(sorted(estimator_set)),
        )
        replaced += 1

    return ReplacementReport(
        checked=len(source_manifests),
        replaced=replaced,
        estimators=tuple(sorted(estimator_set)),
        source_root=str(source_root),
        target_root=str(target_root),
    )


def _replacement_rows(source_dir: Path, estimator_set: set[str]) -> pd.DataFrame:
    source_metrics = pd.read_csv(source_dir / "graph_metrics.csv")
    replacement = source_metrics[
        source_metrics["estimator"].astype(str).isin(estimator_set)
    ].copy()
    if replacement.empty:
        raise ValueError(
            f"{source_dir}: no graph_metrics rows for estimators {sorted(estimator_set)}"
        )
    return replacement


def _graph_copy_pairs(
    source_dir: Path,
    target_dir: Path,
    replacement_rows: pd.DataFrame,
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for row in replacement_rows[["estimator", "variant_id"]].drop_duplicates().itertuples(
        index=False
    ):
        estimator = str(row.estimator)
        variant_id = str(row.variant_id)
        source_graph = source_dir / "graphs" / estimator / f"{_safe(variant_id)}.npz"
        if not source_graph.exists():
            raise FileNotFoundError(source_graph)
        target_graph = target_dir / "graphs" / estimator / f"{_safe(variant_id)}.npz"
        pairs.append((source_graph, target_graph))
    return pairs


def _replace_graph_metric_rows(
    target_dir: Path,
    replacement_rows: pd.DataFrame,
    estimator_set: set[str],
) -> None:
    target_path = target_dir / "graph_metrics.csv"
    target_metrics = pd.read_csv(target_path)
    keep = ~target_metrics["estimator"].astype(str).isin(estimator_set)
    merged = pd.concat([target_metrics.loc[keep], replacement_rows], ignore_index=True)
    merged.to_csv(target_path, index=False)


def _replace_failure_rows(
    source_dir: Path,
    target_dir: Path,
    estimator_set: set[str],
) -> None:
    target_failures = _read_csv_or_empty(target_dir / "failures.csv")
    source_failures = _read_csv_or_empty(source_dir / "failures.csv")
    target_keep = _without_estimators(target_failures, estimator_set)
    source_replacements = _only_estimators(source_failures, estimator_set)
    merged = pd.concat([target_keep, source_replacements], ignore_index=True)
    merged.to_csv(target_dir / "failures.csv", index=False)


def _record_manifest_replacement(
    source_dir: Path,
    target_dir: Path,
    *,
    estimators: tuple[str, ...],
) -> None:
    source_manifest = json.loads((source_dir / "manifest.json").read_text())
    target_manifest_path = target_dir / "manifest.json"
    target_manifest = json.loads(target_manifest_path.read_text())
    now = datetime.now(timezone.utc).isoformat()
    source_estimator_cfg = (
        source_manifest.get("resolved_config", {}).get("estimator", {})
    )
    history = list(target_manifest.get("graph_replacement_history", []))
    history.append(
        {
            "replaced_at": now,
            "source_replicate": str(source_dir),
            "estimators": list(estimators),
            "source_git_revision": source_manifest.get("git_revision"),
            "source_estimator_config": source_estimator_cfg,
        }
    )
    failures = _read_csv_or_empty(target_dir / "failures.csv")
    target_manifest["graph_replacement_history"] = history
    target_manifest["failures"] = failures.to_dict(orient="records")
    target_manifest["complete"] = failures.empty
    target_manifest["end"] = now
    target_manifest_path.write_text(json.dumps(target_manifest, indent=2))


def _without_estimators(frame: pd.DataFrame, estimator_set: set[str]) -> pd.DataFrame:
    if frame.empty or "estimator" not in frame.columns:
        return frame.copy()
    return frame.loc[~frame["estimator"].astype(str).isin(estimator_set)].copy()


def _only_estimators(frame: pd.DataFrame, estimator_set: set[str]) -> pd.DataFrame:
    if frame.empty or "estimator" not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame.loc[frame["estimator"].astype(str).isin(estimator_set)].copy()


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _safe(name: str) -> str:
    return name.replace("/", "__")

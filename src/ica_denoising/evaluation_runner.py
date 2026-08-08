from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from ica_denoising.behavior_decoding import BehaviorTargets, make_behavior_targets
from ica_denoising.bss_notebook import (
    DatasetSpec,
    add_project_imports,
    available_datasets,
    dataset_registry,
    get_dataset,
    load_traces,
    resolve_project_root,
)
from ica_denoising.causal_behavior_decoding import (
    CausalStateConfig,
    compare_latent_graphs,
)
from ica_denoising.evaluation_diagnostics import (
    build_provenance_manifest,
    compute_bpi_ablation,
    paired_segmented_block_bootstrap_interval,
    paired_segmented_block_sign_permutation_pvalue,
    summarize_recording_effects,
    write_json_manifest,
)
from ica_denoising.evaluation_pipeline import (
    EvaluationConfig,
    EvaluationResult,
    iter_evaluation_folds,
    merge_evaluation_results,
    run_evaluation,
)
from ica_denoising.uncertainty import label_algorithmic_replicates


STRICT_TUPLE_FIELDS = {
    "bss_methods",
    "sobi_lags",
    "keep_cluster_counts",
    "selection_strategies",
    "selection_random_seeds",
    "cluster_stability_seeds",
    "cluster_stability_counts",
    "cluster_stability_feature_transforms",
    "baseline_ranks",
    "benchmark_latent_methods",
    "benchmark_latent_ranks",
    "ic_quality_selection_strategies",
    "bss_component_counts",
    "bss_pca_variance_thresholds",
    "lowpass_cutoffs_hz",
    "decoder_lags",
    "target_bout_quantiles",
    "target_smooth_windows",
    "null_block_sizes",
    "null_seeds",
    "null_strategies",
    "causal_transition_models",
    "causal_sufficiency_targets",
    "artifact_probe_centers",
    "bss_random_states",
    "bss_rank_modes",
    "nested_selection_policies",
}
NESTED_TUPLE_FIELDS = {"sobi_lag_sets"}
CAUSAL_TUPLE_FIELDS = {"target_shifts"}
ProgressCallback = Callable[[Mapping[str, object]], None]


def load_evaluation_config(
    path: Path | None,
    *,
    sample_rate_hz: float,
) -> EvaluationConfig:
    payload: dict[str, object] = {}
    if path is not None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Evaluation config JSON must contain an object.")
    payload["sample_rate_hz"] = float(sample_rate_hz)
    causal_payload = dict(payload.pop("causal", {}))
    for name in CAUSAL_TUPLE_FIELDS:
        if name in causal_payload:
            causal_payload[name] = tuple(causal_payload[name])
    payload["causal"] = CausalStateConfig(**causal_payload)
    for name in STRICT_TUPLE_FIELDS:
        if name in payload:
            payload[name] = tuple(payload[name])
    for name in NESTED_TUPLE_FIELDS:
        if name in payload:
            payload[name] = tuple(tuple(item) for item in payload[name])
    return EvaluationConfig(**payload)


def dataset_evaluation_output_paths(
    dataset_key: str,
    *,
    project_root: Path | None = None,
    output_root: Path | None = None,
    output_data_name_override: str | None = None,
) -> dict[str, Path]:
    """Return the standard per-recording output paths without running evaluation."""

    project_root = add_project_imports(
        resolve_project_root() if project_root is None else Path(project_root)
    )
    spec = get_dataset(dataset_key, project_root)
    output_root = (
        project_root / "outputs" / "evaluation"
        if output_root is None
        else Path(output_root)
    )
    output_dir = output_root / spec.group / _dataset_output_name(
        spec,
        output_data_name_override=output_data_name_override,
    )
    return _evaluation_output_paths(output_dir)


def run_dataset_evaluation(
    dataset_key: str,
    *,
    project_root: Path | None = None,
    config_path: Path | None = None,
    output_root: Path | None = None,
    output_data_name_override: str | None = None,
    resume: bool = False,
    checkpoint_root: Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Path]:
    project_root = add_project_imports(
        resolve_project_root() if project_root is None else Path(project_root)
    )
    spec, traces_neurons_by_frames = load_traces(dataset_key, project_root)
    if spec.sample_rate_hz is None:
        raise ValueError(f"{dataset_key} has no sample rate.")
    if spec.tail_angle_path is None or not spec.tail_angle_path.exists():
        raise FileNotFoundError(f"{dataset_key} has no available tail-angle file.")
    config = load_evaluation_config(config_path, sample_rate_hz=spec.sample_rate_hz)
    traces = traces_neurons_by_frames.T
    tail_angle = np.asarray(
        np.load(spec.tail_angle_path, allow_pickle=False), dtype=float
    )
    targets = make_behavior_targets(
        tail_angle,
        n_frames=traces.shape[0],
        bout_quantile=config.bout_quantile,
    )
    output_root = (
        project_root / "outputs" / "evaluation"
        if output_root is None
        else Path(output_root)
    )
    output_name = _dataset_output_name(
        spec,
        output_data_name_override=output_data_name_override,
    )
    output_dir = output_root / spec.group / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _evaluation_output_paths(output_dir)
    if resume:
        result = _run_evaluation_with_fold_checkpoints(
            dataset_key,
            traces,
            targets,
            config,
            tail_angle=tail_angle,
            output_dir=output_dir,
            checkpoint_root=checkpoint_root,
            progress_callback=progress_callback,
        )
    else:
        result = run_evaluation(traces, targets, config, tail_angle=tail_angle)
    provenance = build_provenance_manifest([spec]).iloc[0].to_dict()
    result.trace_metrics.to_csv(paths["trace_metrics"], index=False)
    result.behavior_metrics.to_csv(paths["behavior_metrics"], index=False)
    result.behavior_predictions.to_csv(paths["behavior_predictions"], index=False)
    behavior_uncertainty = behavior_prediction_uncertainty(
        result.behavior_predictions,
        block_size=config.uncertainty_block_size,
        n_bootstrap=config.uncertainty_n_bootstrap,
        n_permutations=config.uncertainty_n_permutations,
        random_state=config.random_state,
    )
    behavior_uncertainty.to_csv(paths["behavior_uncertainty"], index=False)
    result.trace_uncertainty.to_csv(paths["trace_uncertainty"], index=False)
    result.causal_metrics.to_csv(paths["causal_metrics"], index=False)
    result.causal_embeddings.to_csv(paths["causal_embeddings"], index=False)
    result.causal_uncertainty.to_csv(paths["causal_uncertainty"], index=False)
    result.causal_sufficiency.to_csv(paths["causal_sufficiency"], index=False)
    result.causal_sufficiency_uncertainty.to_csv(
        paths["causal_sufficiency_uncertainty"], index=False
    )
    result.artifact_probe_metrics.to_csv(paths["artifact_probe_metrics"], index=False)
    latent_correlations = compare_latent_graphs(result.causal_embeddings, lag=1)
    latent_correlations.to_csv(paths["causal_latent_correlations"], index=False)
    result.leakage_audit.to_csv(paths["leakage_audit"], index=False)
    result.component_selections.to_csv(paths["component_selections"], index=False)
    result.cluster_stability.to_csv(paths["cluster_stability"], index=False)
    cluster_stability_algorithmic = (
        label_algorithmic_replicates(result.cluster_stability)
        if not result.cluster_stability.empty
        else result.cluster_stability.copy()
    )
    cluster_stability_algorithmic.to_csv(
        paths["cluster_stability_algorithmic"], index=False
    )
    result.cluster_stability_assignments.to_csv(
        paths["cluster_stability_assignments"], index=False
    )
    result.variant_metadata.to_csv(paths["variant_metadata"], index=False)
    result.nested_selection.to_csv(paths["nested_selection"], index=False)
    result.temporal_diagnostics.to_csv(paths["temporal_diagnostics"], index=False)
    bpi_component_scores = bpi_component_scores_from_behavior_metrics(
        result.behavior_metrics,
        recording=spec.recording_id or spec.data_name,
    )
    bpi_component_scores.to_csv(paths["bpi_component_scores"], index=False)
    bpi_ablation = bpi_ablation_from_component_scores(bpi_component_scores)
    bpi_ablation.to_csv(paths["bpi_ablation"], index=False)
    warning_counts = (
        result.leakage_audit["fit_warning_count"]
        if "fit_warning_count" in result.leakage_audit
        else pd.Series(0, index=result.leakage_audit.index)
    )
    convergence = (
        result.leakage_audit["converged"]
        if "converged" in result.leakage_audit
        else pd.Series(True, index=result.leakage_audit.index)
    )
    fit_warning_rows = result.leakage_audit[warning_counts.fillna(0) > 0]
    nonconverged_rows = result.leakage_audit[~convergence.fillna(True).astype(bool)]
    causal_warning_rows = (
        result.causal_metrics[result.causal_metrics["fit_warning_count"].fillna(0) > 0]
        if "fit_warning_count" in result.causal_metrics
        else pd.DataFrame()
    )
    causal_nonconverged_rows = (
        result.causal_metrics[
            ~result.causal_metrics["converged"].fillna(True).astype(bool)
        ]
        if "converged" in result.causal_metrics
        else pd.DataFrame()
    )
    write_json_manifest(
        paths["run_manifest"],
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": _dataset_payload(spec),
            "provenance": provenance,
            "config": asdict(config),
            "fit_scope": "strict_fold_local",
            "all_leakage_checks_passed": bool(
                result.leakage_audit["leakage_free"].all()
            ),
            "all_bss_fits_converged": nonconverged_rows.empty,
            "all_causal_fits_converged": causal_nonconverged_rows.empty,
            "fit_warnings": fit_warning_rows.to_dict(orient="records"),
            "causal_fit_warnings": causal_warning_rows.to_dict(orient="records"),
            "outputs": paths,
            "row_counts": {
                "trace_metrics": len(result.trace_metrics),
                "behavior_metrics": len(result.behavior_metrics),
                "behavior_predictions": len(result.behavior_predictions),
                "behavior_uncertainty": len(behavior_uncertainty),
                "trace_uncertainty": len(result.trace_uncertainty),
                "causal_metrics": len(result.causal_metrics),
                "causal_embeddings": len(result.causal_embeddings),
                "causal_uncertainty": len(result.causal_uncertainty),
                "causal_sufficiency": len(result.causal_sufficiency),
                "causal_sufficiency_uncertainty": len(
                    result.causal_sufficiency_uncertainty
                ),
                "artifact_probe_metrics": len(result.artifact_probe_metrics),
                "causal_latent_correlations": len(latent_correlations),
                "leakage_audit": len(result.leakage_audit),
                "component_selections": len(result.component_selections),
                "cluster_stability": len(result.cluster_stability),
                "cluster_stability_algorithmic": len(cluster_stability_algorithmic),
                "cluster_stability_assignments": len(
                    result.cluster_stability_assignments
                ),
                "variant_metadata": len(result.variant_metadata),
                "nested_selection": len(result.nested_selection),
                "temporal_diagnostics": len(result.temporal_diagnostics),
                "bpi_component_scores": len(bpi_component_scores),
                "bpi_ablation": len(bpi_ablation),
            },
        },
    )
    return paths


def _dataset_output_name(
    spec: DatasetSpec,
    *,
    output_data_name_override: str | None,
) -> str:
    return output_data_name_override or spec.recording_id or spec.data_name


def _evaluation_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "trace_metrics": output_dir / "strict_trace_preservation_metrics.csv",
        "behavior_metrics": output_dir / "strict_behavior_fold_metrics.csv",
        "behavior_predictions": output_dir / "strict_behavior_predictions.csv",
        "behavior_uncertainty": output_dir / "strict_behavior_block_uncertainty.csv",
        "trace_uncertainty": output_dir / "strict_trace_block_contributions.csv",
        "causal_metrics": output_dir / "strict_causal_fold_metrics.csv",
        "causal_embeddings": output_dir / "strict_causal_oof_embeddings.csv",
        "causal_uncertainty": output_dir / "strict_causal_block_contributions.csv",
        "causal_sufficiency": output_dir / "strict_causal_sufficiency.csv",
        "causal_sufficiency_uncertainty": output_dir
        / "strict_causal_sufficiency_block_contributions.csv",
        "artifact_probe_metrics": output_dir / "strict_artifact_probe_metrics.csv",
        "causal_latent_correlations": output_dir
        / "strict_causal_oof_latent_behavior_correlations.csv",
        "leakage_audit": output_dir / "leakage_audit.csv",
        "component_selections": output_dir / "component_selections.csv",
        "cluster_stability": output_dir / "cluster_stability.csv",
        "cluster_stability_algorithmic": output_dir
        / "cluster_stability_algorithmic_replicates.csv",
        "cluster_stability_assignments": output_dir
        / "cluster_stability_assignments.csv",
        "variant_metadata": output_dir / "variant_metadata.csv",
        "nested_selection": output_dir / "nested_selection.csv",
        "temporal_diagnostics": output_dir / "temporal_dependence_diagnostics.csv",
        "bpi_component_scores": output_dir / "bpi_component_scores.csv",
        "bpi_ablation": output_dir / "bpi_ablation.csv",
        "run_manifest": output_dir / "run_manifest.json",
    }


def _run_evaluation_with_fold_checkpoints(
    dataset_key: str,
    traces: np.ndarray,
    targets: BehaviorTargets | None,
    config: EvaluationConfig,
    *,
    tail_angle: np.ndarray | None,
    output_dir: Path,
    checkpoint_root: Path | None,
    progress_callback: ProgressCallback | None,
) -> EvaluationResult:
    checkpoint_dir = _evaluation_checkpoint_dir(
        dataset_key,
        config,
        output_dir=output_dir,
        checkpoint_root=checkpoint_root,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fold_results: dict[int, EvaluationResult] = {}
    total = int(config.n_splits)
    for fold_id in range(total):
        fold_dir = checkpoint_dir / f"fold_{fold_id}"
        if not _fold_checkpoint_complete(fold_dir):
            continue
        fold_results[fold_id] = _read_fold_checkpoint(fold_dir)
        _notify_progress(
            progress_callback,
            event="loaded",
            dataset_key=dataset_key,
            fold=fold_id,
            completed=len(fold_results),
            total=total,
            checkpoint_dir=checkpoint_dir,
        )

    missing_folds = [
        fold_id for fold_id in range(total) if fold_id not in fold_results
    ]
    for fold_id, result in iter_evaluation_folds(
        traces,
        targets,
        config,
        tail_angle=tail_angle,
        fold_ids=missing_folds,
    ):
        _notify_progress(
            progress_callback,
            event="start",
            dataset_key=dataset_key,
            fold=fold_id,
            completed=len(fold_results),
            total=total,
            checkpoint_dir=checkpoint_dir,
        )
        _write_fold_checkpoint(checkpoint_dir / f"fold_{fold_id}", fold_id, result)
        fold_results[fold_id] = result
        _notify_progress(
            progress_callback,
            event="done",
            dataset_key=dataset_key,
            fold=fold_id,
            completed=len(fold_results),
            total=total,
            checkpoint_dir=checkpoint_dir,
        )

    if len(fold_results) != total:
        raise RuntimeError(
            f"Expected {total} completed folds, found {len(fold_results)}."
        )
    _notify_progress(
        progress_callback,
        event="complete",
        dataset_key=dataset_key,
        fold=None,
        completed=len(fold_results),
        total=total,
        checkpoint_dir=checkpoint_dir,
    )
    return merge_evaluation_results(
        [fold_results[fold_id] for fold_id in sorted(fold_results)]
    )


def _evaluation_checkpoint_dir(
    dataset_key: str,
    config: EvaluationConfig,
    *,
    output_dir: Path,
    checkpoint_root: Path | None,
) -> Path:
    root = (
        output_dir / ".strict_fold_checkpoints"
        if checkpoint_root is None
        else Path(checkpoint_root)
    )
    return root / _evaluation_checkpoint_fingerprint(dataset_key, config)


def _evaluation_checkpoint_fingerprint(
    dataset_key: str,
    config: EvaluationConfig,
) -> str:
    payload = {
        "dataset_key": dataset_key,
        "config": asdict(config),
        "checkpoint_schema": 1,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _fold_checkpoint_complete(fold_dir: Path) -> bool:
    return (fold_dir / "done.json").exists()


def _write_fold_checkpoint(
    fold_dir: Path,
    fold_id: int,
    result: EvaluationResult,
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    table_rows = {}
    for result_field in fields(EvaluationResult):
        name = result_field.name
        frame = getattr(result, name)
        table_rows[name] = int(len(frame))
        if frame.empty:
            continue
        frame.to_csv(fold_dir / f"{name}.csv", index=False)
    write_json_manifest(
        fold_dir / "done.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fold": int(fold_id),
            "tables": table_rows,
        },
    )


def _read_fold_checkpoint(fold_dir: Path) -> EvaluationResult:
    tables = {}
    for result_field in fields(EvaluationResult):
        name = result_field.name
        path = fold_dir / f"{name}.csv"
        tables[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return EvaluationResult(**tables)


def _notify_progress(
    progress_callback: ProgressCallback | None,
    **payload: object,
) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def write_provenance(
    *,
    project_root: Path | None = None,
    output_root: Path | None = None,
) -> Path:
    project_root = (
        resolve_project_root() if project_root is None else Path(project_root)
    )
    specs = [
        spec
        for spec in dataset_registry(project_root).values()
        if spec.group == "v2a-RSNs"
    ]
    manifest = build_provenance_manifest(specs)
    output_root = (
        project_root / "outputs" / "evaluation"
        if output_root is None
        else Path(output_root)
    )
    path = output_root / "v2a_provenance_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    return path


def run_all_v2a_evaluations(
    *,
    modality: str = "fluorescence",
    project_root: Path | None = None,
    config_path: Path | None = None,
    output_root: Path | None = None,
    resume: bool = False,
    checkpoint_root: Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, dict[str, Path]]:
    project_root = (
        resolve_project_root() if project_root is None else Path(project_root)
    )
    specs = [
        spec
        for spec in dataset_registry(project_root).values()
        if spec.group == "v2a-RSNs" and spec.modality == modality
    ]
    if not specs:
        raise ValueError(f"No V2a datasets found for modality {modality!r}.")
    completed = {}
    for spec in specs:
        completed[spec.key] = run_dataset_evaluation(
            spec.key,
            project_root=project_root,
            config_path=config_path,
            output_root=output_root,
            resume=resume,
            checkpoint_root=checkpoint_root,
            progress_callback=progress_callback,
        )
    write_recording_aggregate(
        completed,
        project_root=project_root,
        output_root=output_root,
    )
    return completed


def write_recording_aggregate(
    completed: Mapping[str, Mapping[str, Path]],
    *,
    project_root: Path,
    output_root: Path | None = None,
) -> dict[str, Path]:
    output_root = (
        Path(project_root) / "outputs" / "evaluation"
        if output_root is None
        else Path(output_root)
    )
    aggregate_dir = output_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    registry = dataset_registry(project_root)
    behavior_frames = []
    uncertainty_frames = []
    trace_uncertainty_frames = []
    causal_frames = []
    causal_uncertainty_frames = []
    causal_sufficiency_frames = []
    causal_sufficiency_uncertainty_frames = []
    artifact_probe_frames = []
    bpi_frames = []
    trace_frames = []
    nested_selection_frames = []
    for dataset_key, paths in completed.items():
        spec = registry[dataset_key]
        metadata = {
            "dataset_key": dataset_key,
            "recording": spec.recording_id or spec.data_name,
            "fish_id": spec.fish_id,
            "run_id": spec.run_id,
            "modality": spec.modality,
        }
        for label, frames in (
            ("trace_metrics", trace_frames),
            ("behavior_metrics", behavior_frames),
            ("behavior_uncertainty", uncertainty_frames),
            ("trace_uncertainty", trace_uncertainty_frames),
            ("causal_metrics", causal_frames),
            ("causal_uncertainty", causal_uncertainty_frames),
            ("causal_sufficiency", causal_sufficiency_frames),
            ("causal_sufficiency_uncertainty", causal_sufficiency_uncertainty_frames),
            ("artifact_probe_metrics", artifact_probe_frames),
            ("bpi_ablation", bpi_frames),
            ("nested_selection", nested_selection_frames),
        ):
            if label not in paths:
                continue
            table = _read_recording_aggregate_input(paths[label])
            if table is None:
                continue
            for column, value in reversed(metadata.items()):
                if column in table:
                    table[column] = value
                else:
                    table.insert(0, column, value)
            frames.append(table)
    trace = pd.concat(trace_frames, ignore_index=True)
    behavior = pd.concat(behavior_frames, ignore_index=True)
    uncertainty = pd.concat(uncertainty_frames, ignore_index=True)
    trace_uncertainty = (
        pd.concat(trace_uncertainty_frames, ignore_index=True)
        if trace_uncertainty_frames
        else pd.DataFrame()
    )
    causal = pd.concat(causal_frames, ignore_index=True)
    causal_uncertainty = (
        pd.concat(causal_uncertainty_frames, ignore_index=True)
        if causal_uncertainty_frames
        else pd.DataFrame()
    )
    causal_sufficiency = (
        pd.concat(causal_sufficiency_frames, ignore_index=True)
        if causal_sufficiency_frames
        else pd.DataFrame()
    )
    causal_sufficiency_uncertainty = (
        pd.concat(causal_sufficiency_uncertainty_frames, ignore_index=True)
        if causal_sufficiency_uncertainty_frames
        else pd.DataFrame()
    )
    artifact_probe = (
        pd.concat(artifact_probe_frames, ignore_index=True)
        if artifact_probe_frames
        else pd.DataFrame()
    )
    bpi = pd.concat(bpi_frames, ignore_index=True)
    nested_selection = (
        pd.concat(nested_selection_frames, ignore_index=True)
        if nested_selection_frames
        else pd.DataFrame()
    )
    paths = {
        "trace": aggregate_dir / "recording_trace_preservation_metrics.csv",
        "behavior": aggregate_dir / "recording_behavior_fold_metrics.csv",
        "behavior_uncertainty": aggregate_dir
        / "recording_behavior_block_uncertainty.csv",
        "trace_uncertainty": aggregate_dir / "recording_trace_block_contributions.csv",
        "causal": aggregate_dir / "recording_causal_fold_metrics.csv",
        "causal_uncertainty": aggregate_dir / "recording_causal_block_contributions.csv",
        "causal_sufficiency": aggregate_dir / "recording_causal_sufficiency.csv",
        "causal_sufficiency_uncertainty": aggregate_dir
        / "recording_causal_sufficiency_block_contributions.csv",
        "artifact_probe": aggregate_dir / "recording_artifact_probe_metrics.csv",
        "bpi": aggregate_dir / "recording_bpi_ablation.csv",
        "nested_selection": aggregate_dir / "recording_nested_selection.csv",
        "primary_effects": aggregate_dir / "recording_primary_effects.csv",
        "fish_primary_effects": aggregate_dir / "fish_primary_effects.csv",
    }
    trace.to_csv(paths["trace"], index=False)
    behavior.to_csv(paths["behavior"], index=False)
    uncertainty.to_csv(paths["behavior_uncertainty"], index=False)
    trace_uncertainty.to_csv(paths["trace_uncertainty"], index=False)
    causal.to_csv(paths["causal"], index=False)
    causal_uncertainty.to_csv(paths["causal_uncertainty"], index=False)
    causal_sufficiency.to_csv(paths["causal_sufficiency"], index=False)
    causal_sufficiency_uncertainty.to_csv(
        paths["causal_sufficiency_uncertainty"], index=False
    )
    artifact_probe.to_csv(paths["artifact_probe"], index=False)
    bpi.to_csv(paths["bpi"], index=False)
    nested_selection.to_csv(paths["nested_selection"], index=False)
    _primary_unit_effects(behavior, causal, unit_col="recording").to_csv(
        paths["primary_effects"], index=False
    )
    _primary_unit_effects(behavior, causal, unit_col="fish_id").to_csv(
        paths["fish_primary_effects"], index=False
    )
    return paths


def _read_recording_aggregate_input(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _primary_unit_effects(
    behavior: pd.DataFrame,
    causal: pd.DataFrame,
    *,
    unit_col: str,
) -> pd.DataFrame:
    frames = []
    behavior_specs = (
        ("tail_vigor_within_pearson", "tail_vigor", "within", "pearson"),
        (
            "bout_state_within_balanced_accuracy",
            "bout_state",
            "within",
            "balanced_accuracy",
        ),
    )
    for outcome, target, comparison, metric in behavior_specs:
        selected = behavior[
            (behavior["target"] == target) & (behavior["comparison"] == comparison)
        ]
        per_recording = (
            selected.dropna(subset=[unit_col])
            .groupby([unit_col, "test_version"], dropna=False)[metric]
            .mean()
            .reset_index()
            .rename(
                columns={
                    unit_col: "recording",
                    "test_version": "method",
                    metric: "value",
                }
            )
        )
        if per_recording.empty:
            continue
        summary = summarize_recording_effects(per_recording, value_col="value")
        summary.insert(0, "outcome", outcome)
        frames.append(summary)
    causal_primary = (
        causal[causal["transition_model"] == "linear"]
        if "transition_model" in causal
        else causal
    )
    causal_per_recording = (
        causal_primary.dropna(subset=[unit_col])
        .groupby([unit_col, "variant"], dropna=False)["dynamic_mse_normalized"]
        .mean()
        .reset_index()
        .rename(
            columns={
                unit_col: "recording",
                "variant": "method",
                "dynamic_mse_normalized": "value",
            }
        )
    )
    if not causal_per_recording.empty:
        summary = summarize_recording_effects(
            causal_per_recording,
            value_col="value",
        )
        summary.insert(0, "outcome", "causal_dynamic_mse_normalized")
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def bpi_component_scores_from_behavior_metrics(
    metrics: pd.DataFrame,
    *,
    recording: str = "recording",
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    if "target_variant" in metrics:
        primary_mask = metrics["target_variant"].isin(["primary", "input_local"])
        metrics = metrics[primary_mask] if primary_mask.any() else metrics
    components = (
        (
            "tail_vigor_within",
            "within",
            "tail_vigor",
            "within",
            "pearson",
        ),
        (
            "tail_vigor_transfer",
            "transfer",
            "tail_vigor",
            "transfer_raw_to_clean",
            "pearson",
        ),
        (
            "bout_state_within",
            "within",
            "bout_state",
            "within",
            "balanced_accuracy",
        ),
        (
            "bout_state_transfer",
            "transfer",
            "bout_state",
            "transfer_raw_to_clean",
            "balanced_accuracy",
        ),
    )
    null_columns = [
        column for column in ("null_block_size", "null_seed") if column in metrics
    ]
    null_rows = metrics[metrics["comparison"] == "null_within"]
    null_configs = (
        null_rows[null_columns].drop_duplicates().to_dict(orient="records")
        if null_columns
        else [{}]
    )
    rows = []
    for null_config in null_configs:
        selected_null = null_rows
        for column, value in null_config.items():
            selected_null = selected_null[selected_null[column] == value]
        for component, scope, target, comparison, metric in components:
            raw_score = _metric_mean(
                metrics,
                target=target,
                comparison="within",
                train_version="raw",
                test_version="raw",
                metric=metric,
            )
            null_score = _metric_mean(
                selected_null,
                target=target,
                comparison="null_within",
                train_version="raw",
                test_version="raw",
                metric=metric,
            )
            methods = sorted(
                metrics.loc[
                    (metrics["target"] == target)
                    & (metrics["comparison"] == comparison),
                    "test_version",
                ].unique()
            )
            if "raw" not in methods:
                methods.insert(0, "raw")
            for method in methods:
                if method == "raw":
                    score = raw_score
                else:
                    score = _metric_mean(
                        metrics,
                        target=target,
                        comparison=comparison,
                        train_version=method if comparison == "within" else "raw",
                        test_version=method,
                        metric=metric,
                    )
                rows.append(
                    {
                        "recording": recording,
                        "method": method,
                        "component": component,
                        "scope": scope,
                        "score": score,
                        "raw_score": raw_score,
                        "null_score": null_score,
                        **null_config,
                    }
                )
    return pd.DataFrame(rows)


def behavior_prediction_uncertainty(
    predictions: pd.DataFrame,
    *,
    block_size: int,
    n_bootstrap: int,
    n_permutations: int,
    random_state: int,
) -> pd.DataFrame:
    required = {
        "target",
        "task",
        "comparison",
        "test_version",
        "fold",
        "time_index",
        "y_true",
        "y_pred",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions is missing required columns: {missing}")
    context_columns = [
        column
        for column in (
            "target_variant",
            "bout_quantile",
            "smooth_window",
            "null_strategy",
        )
        if column in predictions.columns
    ]
    join_columns = ["target", "task", *context_columns, "fold", "time_index"]
    reference = predictions[
        (predictions["comparison"] == "within") & (predictions["test_version"] == "raw")
    ][[*join_columns, "y_true", "y_pred"]].rename(
        columns={"y_pred": "reference_prediction"}
    )
    candidates = predictions[
        predictions["comparison"].isin(["within", "transfer_raw_to_clean"])
        & (predictions["test_version"] != "raw")
    ]
    rows = []
    for (target, task, comparison, method), group in candidates.groupby(
        ["target", "task", "comparison", "test_version"], dropna=False
    ):
        paired = group.merge(
            reference,
            on=join_columns,
            suffixes=("", "_reference"),
            validate="one_to_one",
        )
        if paired.empty:
            continue
        if task == "regression":
            differences = (
                (paired["y_true"] - paired["reference_prediction"]) ** 2
                - (paired["y_true"] - paired["y_pred"]) ** 2
            ).to_numpy(dtype=float)
            effect = "raw_squared_error_minus_method"
        else:
            differences = (
                (paired["y_true"] == paired["y_pred"]).astype(float)
                - (paired["y_true"] == paired["reference_prediction"]).astype(float)
            ).to_numpy(dtype=float)
            effect = "method_correct_minus_raw"
        interval = paired_segmented_block_bootstrap_interval(
            differences,
            paired["fold"].to_numpy(),
            block_size=block_size,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        rows.append(
            {
                "target": target,
                "task": task,
                "comparison": comparison,
                "method": method,
                "effect_definition": effect,
                "n_predictions": len(paired),
                "block_size": int(block_size),
                "effect": interval["estimate"],
                "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
                "block_sign_permutation_pvalue": paired_segmented_block_sign_permutation_pvalue(
                    differences,
                    paired["fold"].to_numpy(),
                    block_size=block_size,
                    n_permutations=n_permutations,
                    random_state=random_state,
                ),
            }
        )
    return pd.DataFrame(rows)


def bpi_ablation_from_component_scores(component_scores: pd.DataFrame) -> pd.DataFrame:
    if component_scores.empty:
        return pd.DataFrame()
    null_columns = [
        column
        for column in ("null_block_size", "null_seed")
        if column in component_scores
    ]
    if not null_columns:
        return compute_bpi_ablation(component_scores)
    frames = []
    for key, group in component_scores.groupby(null_columns, dropna=False):
        result = compute_bpi_ablation(group)
        values = key if isinstance(key, tuple) else (key,)
        for column, value in zip(null_columns, values):
            result[column] = value
        frames.append(result)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _metric_mean(
    metrics: pd.DataFrame,
    *,
    target: str,
    comparison: str,
    train_version: str,
    test_version: str,
    metric: str,
) -> float:
    selected = metrics[
        (metrics["target"] == target)
        & (metrics["comparison"] == comparison)
        & (metrics["train_version"] == train_version)
        & (metrics["test_version"] == test_version)
    ]
    if selected.empty or metric not in selected:
        return np.nan
    return float(selected[metric].mean())


def _dataset_payload(spec: DatasetSpec) -> Mapping[str, object]:
    return {
        "key": spec.key,
        "data_name": spec.data_name,
        "group": spec.group,
        "trace_path": spec.trace_path,
        "tail_angle_path": spec.tail_angle_path,
        "sample_rate_hz": spec.sample_rate_hz,
        "recording_id": spec.recording_id,
        "fish_id": spec.fish_id,
        "run_id": spec.run_id,
        "modality": spec.modality,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe evaluation analyses."
    )
    parser.add_argument("--dataset-key")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--provenance-only", action="store_true")
    parser.add_argument("--all-v2a", action="store_true")
    parser.add_argument("--modality", default="fluorescence")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed strict-fold checkpoints when running an evaluation.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Optional root directory for strict-fold checkpoints.",
    )
    return parser


def _stdout_progress(event: Mapping[str, object]) -> None:
    label = str(event.get("event"))
    fold = event.get("fold")
    completed = int(event.get("completed", 0))
    total = int(event.get("total", 0))
    fold_label = "" if fold is None else f" fold={fold}"
    print(f"{label}{fold_label} ({completed}/{total})", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    project_root = (
        resolve_project_root()
        if args.project_root is None
        else resolve_project_root(args.project_root)
    )
    if args.list_datasets:
        print(available_datasets(project_root).to_string(index=False))
        return
    provenance_path = write_provenance(
        project_root=project_root,
        output_root=args.output_root,
    )
    if args.provenance_only:
        print(provenance_path)
        return
    if args.all_v2a:
        completed = run_all_v2a_evaluations(
            modality=args.modality,
            project_root=project_root,
            config_path=args.config,
            output_root=args.output_root,
            resume=args.resume,
            checkpoint_root=args.checkpoint_root,
            progress_callback=_stdout_progress if args.resume else None,
        )
        print(f"Completed {len(completed)} recordings.")
        return
    if not args.dataset_key:
        raise SystemExit(
            "--dataset-key is required unless --list-datasets, --provenance-only, or --all-v2a is used."
        )
    get_dataset(args.dataset_key, project_root)
    paths = run_dataset_evaluation(
        args.dataset_key,
        project_root=project_root,
        config_path=args.config,
        output_root=args.output_root,
        resume=args.resume,
        checkpoint_root=args.checkpoint_root,
        progress_callback=_stdout_progress if args.resume else None,
    )
    print(paths["run_manifest"])


if __name__ == "__main__":
    main()

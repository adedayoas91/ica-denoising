from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd


BSS_METHODS = ("fastica", "infomax", "sobi", "jade")
PCA_REDUCED_BSS_METHODS = {"sobi", "jade"}
DEFAULT_PCA_VARIANCE_THRESHOLD = 0.95


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    data_name: str
    group: str
    trace_path: Path
    sample_rate_hz: float | None = None
    default_n_components: int | None = None
    tail_angle_path: Path | None = None
    recording_id: str | None = None
    fish_id: str | None = None
    run_id: str | None = None
    modality: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class BSSRunResult:
    dataset: DatasetSpec
    method: str
    traces: np.ndarray
    ic_comps: np.ndarray
    IC_ft: np.ndarray
    A: np.ndarray
    mean: np.ndarray
    cleaned: np.ndarray
    output_dir: Path
    saved_paths: dict[str, Path]
    n_components: int | None = None
    pca_components: int | None = None
    pca_variance_threshold: float | None = None
    pca_explained_variance_ratio: float | None = None
    component_selection_mode: str | None = None


@dataclass(frozen=True)
class BSSDecompositionResult:
    dataset: DatasetSpec
    method: str
    traces: np.ndarray
    ic_comps: np.ndarray
    IC_ft: np.ndarray
    A: np.ndarray
    mean: np.ndarray
    output_dir: Path
    saved_paths: dict[str, Path]
    n_components: int | None = None
    pca_components: int | None = None
    pca_variance_threshold: float | None = None
    pca_explained_variance_ratio: float | None = None
    component_selection_mode: str | None = None


@dataclass(frozen=True)
class BSSComponentSelection:
    method: str
    n_components: int
    pca_components: int | None
    pca_variance_threshold: float | None
    pca_explained_variance_ratio: float | None
    component_selection_mode: str


def resolve_project_root(start: Path | None = None) -> Path:
    """Resolve the repository root from a notebook or script working directory."""
    start = Path.cwd() if start is None else Path(start)
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing pyproject.toml and src/.")


def add_project_imports(project_root: Path | None = None) -> Path:
    """Add project source directories to ``sys.path`` for notebooks."""
    project_root = resolve_project_root() if project_root is None else Path(project_root)
    # Keep src/core for external legacy notebooks that still import ``ica_utils`` directly.
    for path in (project_root / "src", project_root / "src" / "core"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return project_root


def dataset_registry(project_root: Path | None = None) -> dict[str, DatasetSpec]:
    project_root = resolve_project_root() if project_root is None else Path(project_root)
    registry = {
        "motorneurons/fish3_trace2_dff": DatasetSpec(
            key="motorneurons/fish3_trace2_dff",
            data_name="fish3_trace2_dff",
            group="motorneurons",
            trace_path=project_root / "data" / "motorneurons" / "fish3_trace2_dff.npy",
            sample_rate_hz=4.0,
            default_n_components=11,
            recording_id="fish3_trace2",
            fish_id="fish3",
            run_id="trace2",
            modality="dff",
            notes="Motorneuron dF/F traces.",
        ),
        "motorneurons/gcM_restored": DatasetSpec(
            key="motorneurons/gcM_restored",
            data_name="gcM_restored",
            group="motorneurons",
            trace_path=project_root / "data" / "motorneurons" / "gcM_restored.npy",
            sample_rate_hz=4.0,
            default_n_components=11,
            modality="restored",
            notes="Restored motorneuron traces.",
        ),
    }
    registry.update(_discover_v2a_datasets(project_root))
    return registry


def _discover_v2a_datasets(project_root: Path) -> dict[str, DatasetSpec]:
    """Discover V2a recordings without relying on legacy copied filenames."""
    data_root = project_root / "data" / "v2a-RSNs"
    legacy_data_root = data_root / "new_data_09112022"
    if not data_root.exists():
        return {}
    if not any(path.is_dir() and _looks_like_recording_dir(path) for path in data_root.iterdir()):
        data_root = legacy_data_root
    registry: dict[str, DatasetSpec] = {}
    if not data_root.exists():
        return registry

    for run_dir in sorted(
        path
        for path in data_root.iterdir()
        if path.is_dir() and _looks_like_recording_dir(path)
    ):
        recording_id = run_dir.name
        tokens = recording_id.split("_")
        fish_id = "_".join(tokens[:-1]) if len(tokens) > 1 else recording_id
        run_id = tokens[-1] if tokens else recording_id
        sample_rate_hz = _frame_rate_from_analysis_info(run_dir, default_hz=5.2962)
        tail_angle_path = _best_recording_file(run_dir, recording_id, "tail_angle")

        for modality, token in (
            ("fluorescence", "cells_fluorescence_signals"),
            ("spike_rate", "cells_spike_rate_signals"),
        ):
            trace_path = _best_recording_file(run_dir, recording_id, token)
            if trace_path is None:
                continue
            data_name = f"{recording_id}_{modality}"
            key = f"v2a-RSNs/{data_name}"
            registry[key] = DatasetSpec(
                key=key,
                data_name=data_name,
                group="v2a-RSNs",
                trace_path=trace_path,
                sample_rate_hz=sample_rate_hz,
                default_n_components=None,
                tail_angle_path=tail_angle_path,
                recording_id=recording_id,
                fish_id=fish_id,
                run_id=run_id,
                modality=modality,
                notes="Discovered V2a RSN recording.",
            )
    return registry


def _looks_like_recording_dir(path: Path) -> bool:
    return any(path.glob("*cells_fluorescence_signals*.npy")) or any(
        path.glob("*cells_spike_rate_signals*.npy")
    )


def _best_recording_file(run_dir: Path, recording_id: str, token: str) -> Path | None:
    """Prefer files whose names agree with the enclosing recording directory."""
    candidates = sorted(run_dir.glob(f"*{token}*.npy"))
    if not candidates:
        return None
    recording_tokens = tuple(part.lower() for part in recording_id.split("_"))

    def score(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        token_matches = sum(part in name for part in recording_tokens)
        return token_matches, -len(path.name), path.name

    return max(candidates, key=score)


def _frame_rate_from_analysis_info(run_dir: Path, default_hz: float) -> float:
    """Load frameRateSCAPE from run analysis_info metadata, with fallback."""
    pattern = "*_analysis_info.json"
    for info_path in sorted(run_dir.glob(pattern)):
        try:
            payload = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        value = payload.get("frameRateSCAPE")
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return float(default_hz)


def available_datasets(project_root: Path | None = None) -> pd.DataFrame:
    rows = []
    for spec in dataset_registry(project_root).values():
        rows.append(
            {
                "key": spec.key,
                "group": spec.group,
                "data_name": spec.data_name,
                "trace_path": str(spec.trace_path),
                "exists": spec.trace_path.exists(),
                "default_n_components": spec.default_n_components,
                "sample_rate_hz": spec.sample_rate_hz,
                "recording_id": spec.recording_id,
                "fish_id": spec.fish_id,
                "run_id": spec.run_id,
                "modality": spec.modality,
                "tail_angle_exists": bool(spec.tail_angle_path and spec.tail_angle_path.exists()),
                "notes": spec.notes,
            }
        )
    return pd.DataFrame(rows)


def get_dataset(dataset_key: str, project_root: Path | None = None) -> DatasetSpec:
    registry = dataset_registry(project_root)
    if dataset_key not in registry:
        available = ", ".join(registry)
        raise ValueError(f"Unknown DATASET_KEY {dataset_key!r}. Available: {available}.")
    return registry[dataset_key]


def load_traces(dataset_key: str, project_root: Path | None = None) -> tuple[DatasetSpec, np.ndarray]:
    spec = get_dataset(dataset_key, project_root)
    if not spec.trace_path.exists():
        raise FileNotFoundError(spec.trace_path)
    traces = np.asarray(np.load(spec.trace_path, allow_pickle=False), dtype=float)
    if traces.ndim != 2:
        raise ValueError(f"{spec.trace_path} must be 2D, got shape {traces.shape}.")
    return spec, replace_nonfinite_by_neuron_median(traces)


def replace_nonfinite_by_neuron_median(traces: np.ndarray) -> np.ndarray:
    """Replace NaN/inf values in neurons x frames traces before decomposition."""
    traces = np.asarray(traces, dtype=float).copy()
    if np.isfinite(traces).all():
        return traces
    finite_or_nan = np.where(np.isfinite(traces), traces, np.nan)
    has_finite = np.any(np.isfinite(finite_or_nan), axis=1)
    neuron_medians = np.zeros(traces.shape[0], dtype=float)
    neuron_medians[has_finite] = np.nanmedian(finite_or_nan[has_finite], axis=1)
    bad_neurons, bad_frames = np.where(~np.isfinite(traces))
    traces[bad_neurons, bad_frames] = neuron_medians[bad_neurons]
    return traces


def output_directory(
    method: str,
    data_name: str,
    project_root: Path | None = None,
    analysis_kind: str = "linear",
    dataset_group: str | None = None,
) -> Path:
    method = _validate_method(method)
    project_root = resolve_project_root() if project_root is None else Path(project_root)
    analysis_parts = [
        segment.strip()
        for segment in str(analysis_kind).replace("\\", "/").split("/")
        if segment.strip()
    ]
    if not analysis_parts:
        analysis_parts = ["linear"]
    if any(segment in {".", ".."} for segment in analysis_parts):
        raise ValueError("analysis_kind cannot contain '.' or '..' path segments.")
    parts = [project_root / "outputs"]
    parts.extend(sanitize_name(segment) for segment in analysis_parts)
    if dataset_group:
        parts.append(sanitize_name(dataset_group))
    parts.extend([sanitize_name(data_name), method])
    output_dir = parts[0]
    for part in parts[1:]:
        output_dir = output_dir / part
    return output_dir


def bss_decomposition_output_paths(
    spec: DatasetSpec,
    method: str,
    output_dir: Path,
) -> dict[str, Path]:
    method = _validate_method(method)
    stem = f"{sanitize_name(method)}_{sanitize_name(spec.data_name)}"
    output_dir = Path(output_dir)
    return {
        "components": output_dir / f"components_{stem}.npy",
        "spectra": output_dir / f"spectra_{stem}.npy",
        "mixing": output_dir / f"mixing_{stem}.npy",
        "mean": output_dir / f"mean_{stem}.npy",
        "metadata": output_dir / f"metadata_{stem}.json",
    }


def bss_output_paths(spec: DatasetSpec, method: str, output_dir: Path) -> dict[str, Path]:
    method = _validate_method(method)
    output_dir = Path(output_dir)
    return {
        **cleaned_trace_output_paths(spec, method, output_dir),
        **bss_decomposition_output_paths(spec, method, output_dir),
    }


def cleaned_trace_output_paths(
    spec: DatasetSpec,
    method: str,
    output_dir: Path,
) -> dict[str, Path]:
    method = _validate_method(method)
    stem = f"{sanitize_name(method)}_{sanitize_name(spec.data_name)}"
    output_dir = Path(output_dir) / "cleaned"
    return {
        "cleaned": output_dir / f"cleaned_{stem}.npy",
        "cleaned_metadata": output_dir / f"metadata_cleaned_{stem}.json",
    }


def cluster_selection_output_path(spec: DatasetSpec, method: str, output_dir: Path) -> Path:
    method = _validate_method(method)
    stem = f"{sanitize_name(method)}_{sanitize_name(spec.data_name)}"
    return Path(output_dir) / "clusters" / f"cluster_selection_{stem}.json"


def _legacy_cleaned_output_path(spec: DatasetSpec, method: str, output_dir: Path) -> Path:
    method = _validate_method(method)
    stem = f"{sanitize_name(method)}_{sanitize_name(spec.data_name)}"
    return Path(output_dir) / f"cleaned_{stem}.npy"


def load_bss_decomposition_outputs(
    dataset_key: str,
    method: str,
    project_root: Path | None = None,
    analysis_kind: str = "linear",
    output_dir: Path | None = None,
) -> BSSDecompositionResult:
    """Load saved BSS decomposition artifacts produced by the linear notebook."""
    project_root = add_project_imports(project_root)
    method = _validate_method(method)
    spec, traces = load_traces(dataset_key, project_root)
    output_dir = (
        output_directory(
            method,
            spec.data_name,
            project_root,
            analysis_kind=analysis_kind,
            dataset_group=spec.group,
        )
        if output_dir is None
        else Path(output_dir)
    )
    paths = bss_decomposition_output_paths(spec, method, output_dir)
    missing = [label for label, path in paths.items() if not path.exists()]
    if missing:
        expected = "\n".join(f"  {label}: {path}" for label, path in paths.items())
        raise FileNotFoundError(
            f"Missing saved BSS decomposition outputs for {method!r} in {output_dir}: {missing}.\n"
            "Run the recording's notebooks/v2a-RSNs/<recording>/decompositions.ipynb "
            "with SAVE_DECOMPOSITION_OUTPUTS = True first.\n"
            f"Expected files:\n{expected}"
        )
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

    return BSSDecompositionResult(
        dataset=spec,
        method=method,
        traces=traces,
        ic_comps=np.asarray(np.load(paths["components"], allow_pickle=False), dtype=float),
        IC_ft=np.asarray(np.load(paths["spectra"], allow_pickle=False), dtype=float),
        A=np.asarray(np.load(paths["mixing"], allow_pickle=False), dtype=float),
        mean=np.asarray(np.load(paths["mean"], allow_pickle=False), dtype=float),
        output_dir=output_dir,
        saved_paths=paths,
        n_components=_optional_int(metadata.get("n_components")),
        pca_components=_optional_int(metadata.get("pca_components")),
        pca_variance_threshold=_optional_float(metadata.get("pca_variance_threshold")),
        pca_explained_variance_ratio=_optional_float(
            metadata.get("pca_explained_variance_ratio")
        ),
        component_selection_mode=metadata.get("component_selection_mode"),
    )


def load_bss_outputs(
    dataset_key: str,
    method: str,
    project_root: Path | None = None,
    analysis_kind: str = "linear",
    output_dir: Path | None = None,
) -> BSSRunResult:
    """Load saved BSS artifacts, including a cleaned trace when present."""
    result = load_bss_decomposition_outputs(
        dataset_key=dataset_key,
        method=method,
        project_root=project_root,
        analysis_kind=analysis_kind,
        output_dir=output_dir,
    )
    paths = bss_output_paths(result.dataset, method, result.output_dir)
    saved_paths = dict(result.saved_paths)
    cleaned_path = paths["cleaned"]
    legacy_cleaned_path = _legacy_cleaned_output_path(result.dataset, method, result.output_dir)
    if not cleaned_path.exists() and legacy_cleaned_path.exists():
        cleaned_path = legacy_cleaned_path
    if not cleaned_path.exists():
        cleaned = np.dot(result.ic_comps, result.A.T) + result.mean
    else:
        cleaned_saved = np.asarray(np.load(cleaned_path, allow_pickle=False), dtype=float)
        saved_paths["cleaned"] = cleaned_path
        traces = result.traces
        if cleaned_saved.shape == traces.shape:
            cleaned = cleaned_saved.T
        elif cleaned_saved.shape == traces.T.shape:
            cleaned = cleaned_saved
        else:
            raise ValueError(
                f"{cleaned_path} has shape {cleaned_saved.shape}; expected "
                f"{traces.shape} or {traces.T.shape}."
            )

    return BSSRunResult(
        dataset=result.dataset,
        method=result.method,
        traces=result.traces,
        ic_comps=result.ic_comps,
        IC_ft=result.IC_ft,
        A=result.A,
        mean=result.mean,
        cleaned=cleaned,
        output_dir=result.output_dir,
        saved_paths=saved_paths,
        n_components=result.n_components,
        pca_components=result.pca_components,
        pca_variance_threshold=result.pca_variance_threshold,
        pca_explained_variance_ratio=result.pca_explained_variance_ratio,
        component_selection_mode=result.component_selection_mode,
    )


def run_bss_decomposition(
    dataset_key: str,
    method: str,
    n_components: int | None = None,
    pca_components: int | None = None,
    pca_variance_threshold: float | None = DEFAULT_PCA_VARIANCE_THRESHOLD,
    save_outputs: bool = False,
    project_root: Path | None = None,
    tol: float = 0.0001,
    max_iter: int = 500,
    random_state: int = 0,
    analysis_kind: str = "linear",
) -> BSSDecompositionResult:
    """Run one BSS method and optionally save IC decomposition artifacts only."""
    project_root = add_project_imports(project_root)
    from ica_denoising.core.ica_utils import bss_dec

    method = _validate_method(method)
    spec, traces = load_traces(dataset_key, project_root)
    component_selection = resolve_bss_component_selection(
        traces,
        method,
        n_components=n_components,
        pca_components=pca_components,
        pca_variance_threshold=pca_variance_threshold,
    )
    ic_comps, IC_ft, A, mean = bss_dec(
        traces,
        n_comps=component_selection.n_components,
        t=tol,
        max_=max_iter,
        method=method,
        random_state=random_state,
        pca_components=component_selection.pca_components,
    )
    out_dir = output_directory(
        method,
        spec.data_name,
        project_root,
        analysis_kind=analysis_kind,
        dataset_group=spec.group,
    )
    saved_paths: dict[str, Path] = {}
    if save_outputs:
        saved_paths = save_bss_decomposition_outputs(
            spec=spec,
            method=method,
            traces=traces,
            ic_comps=ic_comps,
            IC_ft=IC_ft,
            A=A,
            mean=mean,
            output_dir=out_dir,
            n_components=component_selection.n_components,
            pca_components=component_selection.pca_components,
            pca_variance_threshold=component_selection.pca_variance_threshold,
            pca_explained_variance_ratio=component_selection.pca_explained_variance_ratio,
            component_selection_mode=component_selection.component_selection_mode,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
        )
    return BSSDecompositionResult(
        dataset=spec,
        method=method,
        traces=traces,
        ic_comps=ic_comps,
        IC_ft=IC_ft,
        A=A,
        mean=mean,
        output_dir=out_dir,
        saved_paths=saved_paths,
        n_components=component_selection.n_components,
        pca_components=component_selection.pca_components,
        pca_variance_threshold=component_selection.pca_variance_threshold,
        pca_explained_variance_ratio=component_selection.pca_explained_variance_ratio,
        component_selection_mode=component_selection.component_selection_mode,
    )


def run_bss_method(
    dataset_key: str,
    method: str,
    n_components: int | None = None,
    pca_components: int | None = None,
    pca_variance_threshold: float | None = DEFAULT_PCA_VARIANCE_THRESHOLD,
    reject_components: Iterable[int] = (),
    save_outputs: bool = False,
    project_root: Path | None = None,
    tol: float = 0.0001,
    max_iter: int = 500,
    random_state: int = 0,
    analysis_kind: str = "linear",
) -> BSSRunResult:
    """Run one BSS method and optionally save cleaned traces plus run artifacts."""
    project_root = add_project_imports(project_root)
    from ica_denoising.core.ica_utils import bss_dec, reconstruct_bss

    method = _validate_method(method)
    spec, traces = load_traces(dataset_key, project_root)
    component_selection = resolve_bss_component_selection(
        traces,
        method,
        n_components=n_components,
        pca_components=pca_components,
        pca_variance_threshold=pca_variance_threshold,
    )
    ic_comps, IC_ft, A, mean = bss_dec(
        traces,
        n_comps=component_selection.n_components,
        t=tol,
        max_=max_iter,
        method=method,
        random_state=random_state,
        pca_components=component_selection.pca_components,
    )
    cleaned = reconstruct_bss(ic_comps, A, mean, reject=list(reject_components))
    out_dir = output_directory(
        method,
        spec.data_name,
        project_root,
        analysis_kind=analysis_kind,
        dataset_group=spec.group,
    )
    saved_paths: dict[str, Path] = {}
    if save_outputs:
        saved_paths = save_bss_outputs(
            spec=spec,
            method=method,
            traces=traces,
            ic_comps=ic_comps,
            IC_ft=IC_ft,
            A=A,
            mean=mean,
            cleaned=cleaned,
            reject_components=reject_components,
            output_dir=out_dir,
            n_components=component_selection.n_components,
            pca_components=component_selection.pca_components,
            pca_variance_threshold=component_selection.pca_variance_threshold,
            pca_explained_variance_ratio=component_selection.pca_explained_variance_ratio,
            component_selection_mode=component_selection.component_selection_mode,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
        )
    return BSSRunResult(
        dataset=spec,
        method=method,
        traces=traces,
        ic_comps=ic_comps,
        IC_ft=IC_ft,
        A=A,
        mean=mean,
        cleaned=cleaned,
        output_dir=out_dir,
        saved_paths=saved_paths,
        n_components=component_selection.n_components,
        pca_components=component_selection.pca_components,
        pca_variance_threshold=component_selection.pca_variance_threshold,
        pca_explained_variance_ratio=component_selection.pca_explained_variance_ratio,
        component_selection_mode=component_selection.component_selection_mode,
    )


def resolve_bss_component_selection(
    traces: np.ndarray,
    method: str,
    *,
    n_components: int | None = None,
    pca_components: int | None = None,
    pca_variance_threshold: float | None = DEFAULT_PCA_VARIANCE_THRESHOLD,
) -> BSSComponentSelection:
    """Resolve method-specific BSS component settings for notebook and runner use."""
    method = _validate_method(method)
    n_components = choose_n_components(traces, n_components)
    if method not in PCA_REDUCED_BSS_METHODS:
        return BSSComponentSelection(
            method=method,
            n_components=n_components,
            pca_components=None,
            pca_variance_threshold=None,
            pca_explained_variance_ratio=None,
            component_selection_mode="full_trace_count",
        )

    if pca_components is not None:
        resolved_pca_components = choose_n_components(traces, pca_components)
        explained = pca_explained_variance_for_rank(traces, resolved_pca_components)
        return BSSComponentSelection(
            method=method,
            n_components=n_components,
            pca_components=resolved_pca_components,
            pca_variance_threshold=None,
            pca_explained_variance_ratio=explained,
            component_selection_mode="explicit_pca_components",
        )

    if pca_variance_threshold is None:
        return BSSComponentSelection(
            method=method,
            n_components=n_components,
            pca_components=None,
            pca_variance_threshold=None,
            pca_explained_variance_ratio=None,
            component_selection_mode="full_trace_count",
        )

    resolved_pca_components, explained = choose_pca_components_for_variance(
        traces,
        pca_variance_threshold,
    )
    return BSSComponentSelection(
        method=method,
        n_components=n_components,
        pca_components=resolved_pca_components,
        pca_variance_threshold=float(pca_variance_threshold),
        pca_explained_variance_ratio=explained,
        component_selection_mode="pca_variance_threshold",
    )


def choose_n_components(traces: np.ndarray, requested: int | None) -> int:
    limit = min(traces.shape)
    if requested is None:
        return limit
    requested = int(requested)
    if requested < 1:
        raise ValueError("n_components must be at least 1.")
    return min(requested, limit)


def choose_pca_components_for_variance(
    traces: np.ndarray,
    variance_threshold: float = DEFAULT_PCA_VARIANCE_THRESHOLD,
) -> tuple[int, float]:
    """Return the smallest PCA rank explaining at least ``variance_threshold``."""
    ratios = pca_explained_variance_ratios(traces)
    if not 0.0 < float(variance_threshold) <= 1.0:
        raise ValueError("variance_threshold must be in the interval (0, 1].")
    if ratios.size == 0:
        return 1, 0.0
    cumulative = np.cumsum(ratios)
    rank = int(np.searchsorted(cumulative, float(variance_threshold), side="left") + 1)
    rank = min(max(rank, 1), ratios.size)
    return rank, float(cumulative[rank - 1])


def pca_explained_variance_for_rank(traces: np.ndarray, rank: int) -> float:
    ratios = pca_explained_variance_ratios(traces)
    if ratios.size == 0:
        return 0.0
    rank = min(max(int(rank), 1), ratios.size)
    return float(np.cumsum(ratios)[rank - 1])


def pca_explained_variance_ratios(traces: np.ndarray) -> np.ndarray:
    traces = np.asarray(traces, dtype=float)
    if traces.ndim != 2:
        raise ValueError(f"traces must be 2D, got shape {traces.shape}.")
    samples_by_features = traces.T
    limit = min(samples_by_features.shape)
    if limit < 1:
        return np.array([], dtype=float)
    centered = samples_by_features - samples_by_features.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    variances = singular_values[:limit] ** 2
    total = float(variances.sum())
    if total <= np.finfo(float).eps:
        ratios = np.zeros(limit, dtype=float)
        ratios[0] = 1.0
        return ratios
    return variances / total


def _dataset_metadata(spec: DatasetSpec) -> dict:
    metadata = asdict(spec)
    metadata["trace_path"] = str(spec.trace_path)
    metadata["tail_angle_path"] = str(spec.tail_angle_path) if spec.tail_angle_path else None
    return metadata


def save_bss_decomposition_outputs(
    spec: DatasetSpec,
    method: str,
    traces: np.ndarray,
    ic_comps: np.ndarray,
    IC_ft: np.ndarray,
    A: np.ndarray,
    mean: np.ndarray,
    output_dir: Path,
    n_components: int,
    tol: float,
    max_iter: int,
    random_state: int,
    pca_components: int | None = None,
    pca_variance_threshold: float | None = None,
    pca_explained_variance_ratio: float | None = None,
    component_selection_mode: str | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = bss_decomposition_output_paths(spec, method, output_dir)
    np.save(paths["components"], ic_comps)
    np.save(paths["spectra"], IC_ft)
    np.save(paths["mixing"], A)
    np.save(paths["mean"], mean)

    metadata = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "artifact_kind": "bss_decomposition",
        "method": method,
        "dataset": _dataset_metadata(spec),
        "input_shape_neurons_by_frames": list(traces.shape),
        "component_shape_frames_by_components": list(ic_comps.shape),
        "spectra_shape_components_by_bins": list(IC_ft.shape),
        "mixing_shape_neurons_by_components": list(A.shape),
        "mean_shape_neurons": list(mean.shape),
        "n_components": int(n_components),
        "pca_components": None if pca_components is None else int(pca_components),
        "pca_variance_threshold": (
            None if pca_variance_threshold is None else float(pca_variance_threshold)
        ),
        "pca_explained_variance_ratio": (
            None
            if pca_explained_variance_ratio is None
            else float(pca_explained_variance_ratio)
        ),
        "component_selection_mode": component_selection_mode,
        "tol": float(tol),
        "max_iter": int(max_iter),
        "random_state": int(random_state),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths


def save_cleaned_trace_output(
    spec: DatasetSpec,
    method: str,
    traces: np.ndarray,
    cleaned: np.ndarray,
    output_dir: Path,
    reject_components: Iterable[int] = (),
    metadata: dict | None = None,
) -> dict[str, Path]:
    paths = cleaned_trace_output_paths(spec, method, output_dir)
    paths["cleaned"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["cleaned"], cleaned.T)

    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "artifact_kind": "cleaned_trace",
        "method": method,
        "dataset": _dataset_metadata(spec),
        "input_shape_neurons_by_frames": list(traces.shape),
        "cleaned_saved_shape_neurons_by_frames": list(cleaned.T.shape),
        "reject_components": [int(component) for component in reject_components],
    }
    if metadata:
        payload.update(metadata)
    paths["cleaned_metadata"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return paths


def save_cluster_selection_output(
    spec: DatasetSpec,
    method: str,
    output_dir: Path,
    selection: dict,
) -> Path:
    path = cluster_selection_output_path(spec, method, output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "artifact_kind": "cluster_selection",
        "method": method,
        "dataset": _dataset_metadata(spec),
    }
    payload.update(selection)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_bss_outputs(
    spec: DatasetSpec,
    method: str,
    traces: np.ndarray,
    ic_comps: np.ndarray,
    IC_ft: np.ndarray,
    A: np.ndarray,
    mean: np.ndarray,
    cleaned: np.ndarray,
    reject_components: Iterable[int],
    output_dir: Path,
    n_components: int,
    tol: float,
    max_iter: int,
    random_state: int,
    pca_components: int | None = None,
    pca_variance_threshold: float | None = None,
    pca_explained_variance_ratio: float | None = None,
    component_selection_mode: str | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = bss_output_paths(spec, method, output_dir)
    paths["cleaned"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["cleaned"], cleaned.T)
    np.save(paths["components"], ic_comps)
    np.save(paths["spectra"], IC_ft)
    np.save(paths["mixing"], A)
    np.save(paths["mean"], mean)

    metadata = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "dataset": _dataset_metadata(spec),
        "input_shape_neurons_by_frames": list(traces.shape),
        "cleaned_saved_shape_neurons_by_frames": list(cleaned.T.shape),
        "component_shape_frames_by_components": list(ic_comps.shape),
        "n_components": int(n_components),
        "pca_components": None if pca_components is None else int(pca_components),
        "pca_variance_threshold": (
            None if pca_variance_threshold is None else float(pca_variance_threshold)
        ),
        "pca_explained_variance_ratio": (
            None
            if pca_explained_variance_ratio is None
            else float(pca_explained_variance_ratio)
        ),
        "component_selection_mode": component_selection_mode,
        "reject_components": [int(component) for component in reject_components],
        "tol": float(tol),
        "max_iter": int(max_iter),
        "random_state": int(random_state),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    cleaned_metadata = {
        "saved_at": metadata["saved_at"],
        "artifact_kind": "cleaned_trace",
        "method": method,
        "dataset": metadata["dataset"],
        "input_shape_neurons_by_frames": metadata["input_shape_neurons_by_frames"],
        "cleaned_saved_shape_neurons_by_frames": metadata["cleaned_saved_shape_neurons_by_frames"],
        "reject_components": metadata["reject_components"],
    }
    paths["cleaned_metadata"].write_text(json.dumps(cleaned_metadata, indent=2), encoding="utf-8")
    return paths


def run_many_methods(
    dataset_key: str,
    methods: Iterable[str],
    n_components: int | None = None,
    pca_components: int | None = None,
    pca_variance_threshold: float | None = DEFAULT_PCA_VARIANCE_THRESHOLD,
    reject_components_by_method: dict[str, Iterable[int]] | None = None,
    save_outputs: bool = False,
    project_root: Path | None = None,
    tol: float = 0.0001,
    max_iter: int = 500,
    random_state: int = 0,
    analysis_kind: str = "linear",
) -> dict[str, BSSRunResult]:
    reject_components_by_method = reject_components_by_method or {}
    results = {}
    for method in methods:
        results[method] = run_bss_method(
            dataset_key=dataset_key,
            method=method,
            n_components=n_components,
            pca_components=pca_components,
            pca_variance_threshold=pca_variance_threshold,
            reject_components=reject_components_by_method.get(method, ()),
            save_outputs=save_outputs,
            project_root=project_root,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            analysis_kind=analysis_kind,
        )
    return results


def summarize_results(results: dict[str, BSSRunResult]) -> pd.DataFrame:
    rows = []
    for method, result in results.items():
        rows.append(
            {
                "method": method,
                "dataset_key": result.dataset.key,
                "data_name": result.dataset.data_name,
                "dataset_group": result.dataset.group,
                "input_shape": tuple(result.traces.shape),
                "components_shape": tuple(result.ic_comps.shape),
                "cleaned_shape_frames_by_neurons": tuple(result.cleaned.shape),
                "n_components": result.n_components,
                "pca_components": result.pca_components,
                "pca_variance_threshold": result.pca_variance_threshold,
                "pca_explained_variance_ratio": result.pca_explained_variance_ratio,
                "component_selection_mode": result.component_selection_mode,
                "output_dir": str(result.output_dir),
                "saved": bool(result.saved_paths),
            }
        )
    return pd.DataFrame(rows)


def summarize_decomposition_results(
    results: dict[str, BSSDecompositionResult],
) -> pd.DataFrame:
    rows = []
    for method, result in results.items():
        rows.append(
            {
                "method": method,
                "dataset_key": result.dataset.key,
                "data_name": result.dataset.data_name,
                "dataset_group": result.dataset.group,
                "input_shape": tuple(result.traces.shape),
                "components_shape": tuple(result.ic_comps.shape),
                "spectra_shape": tuple(result.IC_ft.shape),
                "mixing_shape": tuple(result.A.shape),
                "mean_shape": tuple(result.mean.shape),
                "n_components": result.n_components,
                "pca_components": result.pca_components,
                "pca_variance_threshold": result.pca_variance_threshold,
                "pca_explained_variance_ratio": result.pca_explained_variance_ratio,
                "component_selection_mode": result.component_selection_mode,
                "output_dir": str(result.output_dir),
                "saved": bool(result.saved_paths),
            }
        )
    return pd.DataFrame(rows)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value.strip("_") or "unnamed"


def _validate_method(method: str) -> str:
    method = str(method).lower()
    if method not in BSS_METHODS:
        raise ValueError(f"Unknown BSS method {method!r}. Expected one of {BSS_METHODS}.")
    return method

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


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    data_name: str
    group: str
    trace_path: Path
    sample_rate_hz: float | None = None
    default_n_components: int | None = None
    tail_angle_path: Path | None = None
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
    for path in (project_root / "src", project_root / "src" / "core"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return project_root


def dataset_registry(project_root: Path | None = None) -> dict[str, DatasetSpec]:
    project_root = resolve_project_root() if project_root is None else Path(project_root)
    run2_dir = (
        project_root
        / "data"
        / "v2a-RSNs"
        / "new_data_09112022"
        / "220127_F4_run2"
    )
    run6_dir = (
        project_root
        / "data"
        / "v2a-RSNs"
        / "new_data_09112022"
        / "220210_F1_run6"
    )
    run2_frame_rate = _frame_rate_from_analysis_info(run2_dir, default_hz=5.2962)
    run6_frame_rate = _frame_rate_from_analysis_info(run6_dir, default_hz=5.2962)
    return {
        "motorneurons/fish3_trace2_dff": DatasetSpec(
            key="motorneurons/fish3_trace2_dff",
            data_name="fish3_trace2_dff",
            group="motorneurons",
            trace_path=project_root / "data" / "motorneurons" / "fish3_trace2_dff.npy",
            sample_rate_hz=4.0,
            default_n_components=11,
            notes="Motorneuron dF/F traces.",
        ),
        "motorneurons/gcM_restored": DatasetSpec(
            key="motorneurons/gcM_restored",
            data_name="gcM_restored",
            group="motorneurons",
            trace_path=project_root / "data" / "motorneurons" / "gcM_restored.npy",
            sample_rate_hz=4.0,
            default_n_components=11,
            notes="Restored motorneuron traces.",
        ),
        "v2a-RSNs/220127_F4_run2_fluorescence": DatasetSpec(
            key="v2a-RSNs/220127_F4_run2_fluorescence",
            data_name="220127_F4_run2_fluorescence",
            group="v2a-RSNs",
            trace_path=project_root
            / "data"
            / "v2a-RSNs"
            / "new_data_09112022"
            / "220127_F4_run2"
            / "220127_F4_F4_run2_after_dec_cells_fluorescence_signals.npy",
            sample_rate_hz=run2_frame_rate,
            default_n_components=40,
            tail_angle_path=project_root
            / "data"
            / "v2a-RSNs"
            / "new_data_09112022"
            / "220127_F4_run2"
            / "220127_F4_F4_run2_after_dec_tail_angle.npy",
            notes="V2a RSN fluorescence traces.",
        ),
        "v2a-RSNs/220127_F4_run2_spike_rate": DatasetSpec(
            key="v2a-RSNs/220127_F4_run2_spike_rate",
            data_name="220127_F4_run2_spike_rate",
            group="v2a-RSNs",
            trace_path=project_root
            / "data"
            / "v2a-RSNs"
            / "new_data_09112022"
            / "220127_F4_run2"
            / "220127_F4_F4_run2_after_dec_cells_spike_rate_signals.npy",
            sample_rate_hz=run2_frame_rate,
            default_n_components=40,
            tail_angle_path=project_root
            / "data"
            / "v2a-RSNs"
            / "new_data_09112022"
            / "220127_F4_run2"
            / "220127_F4_F4_run2_after_dec_tail_angle.npy",
            notes="V2a RSN spike-rate traces.",
        ),
        "v2a-RSNs/220210_F1_run6_fluorescence": DatasetSpec(
            key="v2a-RSNs/220210_F1_run6_fluorescence",
            data_name="220210_F1_run6_fluorescence",
            group="v2a-RSNs",
            trace_path=project_root
            / "data"
            / "v2a-RSNs"
            / "new_data_09112022"
            / "220210_F1_run6"
            / "220127_F4_F4_run2_after_dec_cells_fluorescence_signals.npy",
            sample_rate_hz=run6_frame_rate,
            default_n_components=40,
            tail_angle_path=project_root
            / "data"
            / "v2a-RSNs"
            / "new_data_09112022"
            / "220210_F1_run6"
            / "220127_F4_F4_run2_after_dec_tail_angle.npy",
            notes="V2a RSN fluorescence traces. Uses frameRateSCAPE from analysis_info when available.",
        ),
        "v2a-RSNs/220210_F1_run6_spike_rate": DatasetSpec(
            key="v2a-RSNs/220210_F1_run6_spike_rate",
            data_name="220210_F1_run6_spike_rate",
            group="v2a-RSNs",
            trace_path=project_root
            / "data"
            / "v2a-RSNs"
            / "new_data_09112022"
            / "220210_F1_run6"
            / "220127_F4_F4_run2_after_dec_cells_spike_rate_signals.npy",
            sample_rate_hz=run6_frame_rate,
            default_n_components=40,
            tail_angle_path=project_root
            / "data"
            / "v2a-RSNs"
            / "new_data_09112022"
            / "220210_F1_run6"
            / "220127_F4_F4_run2_after_dec_tail_angle.npy",
            notes="V2a RSN spike-rate traces. Uses frameRateSCAPE from analysis_info when available.",
        ),
    }


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


def bss_output_paths(spec: DatasetSpec, method: str, output_dir: Path) -> dict[str, Path]:
    method = _validate_method(method)
    stem = f"{sanitize_name(method)}_{sanitize_name(spec.data_name)}"
    output_dir = Path(output_dir)
    return {
        "cleaned": output_dir / f"cleaned_{stem}.npy",
        "components": output_dir / f"components_{stem}.npy",
        "spectra": output_dir / f"spectra_{stem}.npy",
        "mixing": output_dir / f"mixing_{stem}.npy",
        "mean": output_dir / f"mean_{stem}.npy",
        "metadata": output_dir / f"metadata_{stem}.json",
    }


def load_bss_outputs(
    dataset_key: str,
    method: str,
    project_root: Path | None = None,
    analysis_kind: str = "linear",
    output_dir: Path | None = None,
) -> BSSRunResult:
    """Load saved BSS artifacts produced by ``save_bss_outputs``."""
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
    paths = bss_output_paths(spec, method, output_dir)
    missing = [label for label, path in paths.items() if not path.exists()]
    if missing:
        expected = "\n".join(f"  {label}: {path}" for label, path in paths.items())
        raise FileNotFoundError(
            f"Missing saved BSS outputs for {method!r} in {output_dir}: {missing}.\n"
            "Run notebooks/v2a-RSNs/linear_methods.ipynb with SAVE_OUTPUTS = True first.\n"
            f"Expected files:\n{expected}"
        )

    cleaned_saved = np.asarray(np.load(paths["cleaned"], allow_pickle=False), dtype=float)
    if cleaned_saved.shape == traces.shape:
        cleaned = cleaned_saved.T
    elif cleaned_saved.shape == traces.T.shape:
        cleaned = cleaned_saved
    else:
        raise ValueError(
            f"{paths['cleaned']} has shape {cleaned_saved.shape}; expected "
            f"{traces.shape} or {traces.T.shape}."
        )

    return BSSRunResult(
        dataset=spec,
        method=method,
        traces=traces,
        ic_comps=np.asarray(np.load(paths["components"], allow_pickle=False), dtype=float),
        IC_ft=np.asarray(np.load(paths["spectra"], allow_pickle=False), dtype=float),
        A=np.asarray(np.load(paths["mixing"], allow_pickle=False), dtype=float),
        mean=np.asarray(np.load(paths["mean"], allow_pickle=False), dtype=float),
        cleaned=cleaned,
        output_dir=output_dir,
        saved_paths=paths,
    )


def run_bss_method(
    dataset_key: str,
    method: str,
    n_components: int | None = None,
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
    from ica_utils import bss_dec, reconstruct_bss

    method = _validate_method(method)
    spec, traces = load_traces(dataset_key, project_root)
    n_components = choose_n_components(traces, n_components or spec.default_n_components)
    ic_comps, IC_ft, A, mean = bss_dec(
        traces,
        n_comps=n_components,
        t=tol,
        max_=max_iter,
        method=method,
        random_state=random_state,
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
            n_components=n_components,
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
    )


def choose_n_components(traces: np.ndarray, requested: int | None) -> int:
    limit = min(traces.shape)
    if requested is None:
        return limit
    requested = int(requested)
    if requested < 1:
        raise ValueError("n_components must be at least 1.")
    return min(requested, limit)


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
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = bss_output_paths(spec, method, output_dir)
    np.save(paths["cleaned"], cleaned.T)
    np.save(paths["components"], ic_comps)
    np.save(paths["spectra"], IC_ft)
    np.save(paths["mixing"], A)
    np.save(paths["mean"], mean)

    metadata = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "dataset": asdict(spec),
        "input_shape_neurons_by_frames": list(traces.shape),
        "cleaned_saved_shape_neurons_by_frames": list(cleaned.T.shape),
        "component_shape_frames_by_components": list(ic_comps.shape),
        "n_components": int(n_components),
        "reject_components": [int(component) for component in reject_components],
        "tol": float(tol),
        "max_iter": int(max_iter),
        "random_state": int(random_state),
    }
    metadata["dataset"]["trace_path"] = str(spec.trace_path)
    metadata["dataset"]["tail_angle_path"] = str(spec.tail_angle_path) if spec.tail_angle_path else None
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths


def run_many_methods(
    dataset_key: str,
    methods: Iterable[str],
    n_components: int | None = None,
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
                "output_dir": str(result.output_dir),
                "saved": bool(result.saved_paths),
            }
        )
    return pd.DataFrame(rows)


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value.strip("_") or "unnamed"


def _validate_method(method: str) -> str:
    method = str(method).lower()
    if method not in BSS_METHODS:
        raise ValueError(f"Unknown BSS method {method!r}. Expected one of {BSS_METHODS}.")
    return method

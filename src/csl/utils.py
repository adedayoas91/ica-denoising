from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_square_matrix(matrix: np.ndarray) -> np.ndarray:
    checked = np.asarray(matrix)
    if checked.ndim != 2 or checked.shape[0] != checked.shape[1]:
        raise ValueError(f"Expected a square matrix, got {checked.shape}.")
    return checked


def save_adjacency_dict(
    path: Path,
    p_value: int,
    adjacency: np.ndarray,
    *,
    replace_existing_p: bool = True,
) -> dict[int, np.ndarray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = pd.read_pickle(path)
        if not isinstance(payload, dict):
            raise TypeError(f"{path} exists but does not contain a dict keyed by P values.")
        payload = {int(key): ensure_square_matrix(value) for key, value in payload.items()}
    else:
        payload = {}

    p_value = int(p_value)
    if p_value in payload and not replace_existing_p:
        raise FileExistsError(
            f"{path.name} already contains P={p_value}; "
            "set replace_existing_p=True to replace it."
        )
    payload[p_value] = ensure_square_matrix(adjacency)
    return {key: payload[key] for key in sorted(payload)}


def write_adjacency_dict(path: Path, payload: dict[int, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(payload, path)


def save_variant_adjacency_dict(
    path: Path,
    variant_name: str,
    adjacency: np.ndarray,
    *,
    replace_existing_variant: bool = True,
) -> dict[str, np.ndarray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = pd.read_pickle(path)
        if not isinstance(payload, dict):
            raise TypeError(
                f"{path} exists but does not contain a dict keyed by trace variants."
            )
        payload = {
            str(stored_variant): ensure_square_matrix(value)
            for stored_variant, value in payload.items()
        }
    else:
        payload = {}

    variant_name = str(variant_name)
    if variant_name in payload and not replace_existing_variant:
        raise FileExistsError(
            f"{path.name} already contains variant={variant_name!r}; "
            "set replace_existing_variant=True to replace it."
        )
    payload[variant_name] = ensure_square_matrix(adjacency)

    return {variant: payload[variant] for variant in sorted(payload)}


def write_variant_adjacency_dict(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(payload, path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def graph_to_adjacency(causal_graph: np.ndarray) -> np.ndarray:
    n_vars = causal_graph.shape[0]
    adjacency = np.zeros((n_vars, n_vars), dtype=int)
    for source in range(causal_graph.shape[0]):
        for target in range(causal_graph.shape[1]):
            for lag in range(causal_graph.shape[2]):
                edge_type = str(causal_graph[source, target, lag])
                if ">" in edge_type and "<" not in edge_type:
                    adjacency[target, source] = 1
    np.fill_diagonal(adjacency, 0)
    return adjacency


LEFT_COLOR = (57 / 255, 87 / 255, 225 / 255)
RIGHT_COLOR = (255 / 255, 138 / 255, 0.0)
DRIVER_COLOR = "firebrick"
RECEIVER_COLOR = "navy"
BALANCED_COLOR = "#6a3d9a"


def bilateral_labels(mid: int, total: int) -> tuple[list[int], list[tuple[float, ...]]]:
    if mid < 0 or mid > total:
        raise ValueError("mid must divide the node count into left and right sides")
    labels = [2 * index + 1 for index in range(mid)]
    labels.extend(2 * index + 2 for index in range(total - mid))
    colors = [LEFT_COLOR] * mid + [RIGHT_COLOR] * (total - mid)
    return labels, colors


def _semicircle_coordinates(n_points: int, side: int) -> np.ndarray:
    if n_points == 1:
        return np.array([[1.1 * side, 0.0]])
    half = n_points // 2
    inner = side / max(half, 1)
    outer = 1.0 * side
    x_values = np.linspace(inner, outer, half)
    if n_points % 2 == 1:
        x_values = np.concatenate((x_values, [1.1 * side]))
    x_values = np.concatenate((x_values, np.linspace(outer, inner, half)))
    y_values = np.linspace(1.0, -1.0, n_points)
    return np.column_stack((x_values, y_values))


def bilateral_coordinates(mid: int, total: int) -> np.ndarray:
    if mid < 1 or total - mid < 1:
        raise ValueError("at least one node is required on each side")
    return np.vstack(
        [
            _semicircle_coordinates(mid, side=-1),
            _semicircle_coordinates(total - mid, side=1),
        ]
    )


def plot_matrix(
    scores: np.ndarray,
    mid: int,
    *,
    ax: plt.Axes | None = None,
    cmap: str = "YlOrRd",
) -> plt.Axes:
    matrix = ensure_square_matrix(scores).astype(float, copy=True)
    displayed = matrix.copy()
    displayed[displayed == 0] = np.nan
    axis = plt.gca() if ax is None else ax
    axis.imshow(displayed, cmap=cmap)
    labels, colors = bilateral_labels(mid, matrix.shape[0])
    axis.set_xticks(np.arange(matrix.shape[0]), labels=labels)
    axis.set_yticks(np.arange(matrix.shape[0]), labels=labels)
    for tick, color in zip(axis.get_xticklabels(), colors):
        tick.set_color(color)
    for tick, color in zip(axis.get_yticklabels(), colors):
        tick.set_color(color)
    axis.set_xlabel("to neuron")
    axis.set_ylabel("from neuron")
    return axis


def plot_directed_graph(
    scores: np.ndarray,
    mid: int,
    *,
    ax: plt.Axes | None = None,
    hide_digits: bool = False,
) -> plt.Axes:
    matrix = ensure_square_matrix(scores).astype(float, copy=True)
    axis = plt.gca() if ax is None else ax
    coords = bilateral_coordinates(mid, matrix.shape[0])
    ipsilateral = matrix.copy()
    ipsilateral[:mid, mid:] = 0.0
    ipsilateral[mid:, :mid] = 0.0
    drive = ipsilateral.sum(axis=1) - ipsilateral.sum(axis=0)
    maximum = np.max(np.abs(drive))
    scaled = drive if maximum == 0 else drive / maximum
    labels, colors = bilateral_labels(mid, matrix.shape[0])
    for node, center in enumerate(coords):
        color = (
            DRIVER_COLOR if scaled[node] > 0
            else RECEIVER_COLOR if scaled[node] < 0
            else BALANCED_COLOR
        )
        axis.scatter(*center, s=100 + 400 * abs(scaled[node]), c=color, zorder=2)
        if not hide_digits:
            axis.text(
                center[0] + np.sign(center[0]) * 0.15,
                center[1],
                str(labels[node]),
                color=colors[node],
                ha="center",
                va="center",
                size=15,
                fontweight="bold",
            )
    positive = matrix[matrix > 0]
    scale = float(positive.max()) if positive.size else 1.0
    for source, target in zip(*np.nonzero(matrix > 0)):
        if source == target:
            continue
        color = "black" if (source < mid) == (target < mid) else "gray"
        axis.annotate(
            "",
            xy=coords[target],
            xytext=coords[source],
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": 0.8 + 2.2 * matrix[source, target] / scale,
                "alpha": 0.8,
            },
        )
    axis.set_aspect("equal")
    axis.axis("off")
    return axis


def load_motorneuron_mid(dataset_key: str, project_root: Path) -> int:
    from ica_denoising.bss_notebook import get_dataset

    spec = get_dataset(dataset_key, project_root)
    payload = pd.read_pickle(spec.trace_path)
    if not isinstance(payload, pd.DataFrame):
        raise TypeError(f"{spec.trace_path} must contain a pandas DataFrame.")
    required_columns = {"Fish", "Trace", "fluo_type", "mid"}
    missing = required_columns.difference(payload.columns)
    if missing:
        raise ValueError(f"{spec.trace_path} is missing required columns: {sorted(missing)}.")

    fish_token = (spec.fish_id or "").lower()
    run_token = (spec.run_id or "").lower()
    fish_value = float("".join(ch for ch in fish_token if ch.isdigit()))
    trace_value = float("".join(ch for ch in run_token if ch.isdigit()))
    modality = (spec.modality or "").lower()
    matches = payload[
        (payload["Fish"].astype(float) == fish_value)
        & (payload["Trace"].astype(float) == trace_value)
        & (payload["fluo_type"].astype(str).str.lower() == modality)
    ]
    if matches.empty:
        raise ValueError(f"No mid value found in {spec.trace_path} for {dataset_key}.")
    return int(float(matches.iloc[0]["mid"]))


def plot_motoneuron_connectivity_grid(
    connectivity_by_variant: dict[str, np.ndarray],
    *,
    mid: int,
    variant_order: Sequence[str] | None = None,
    variant_titles: dict[str, str] | None = None,
    figure_title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    if variant_order is None:
        variant_order = tuple(connectivity_by_variant)
    items = [(name, connectivity_by_variant[name]) for name in variant_order if name in connectivity_by_variant]
    if not items:
        raise ValueError("No connectivity matrices available to plot.")

    fig, axes = plt.subplots(2, len(items), figsize=(4.2 * len(items), 8.4), squeeze=False)
    for column_index, (variant_name, matrix) in enumerate(items):
        label = variant_titles.get(variant_name, variant_name) if variant_titles else variant_name
        square = ensure_square_matrix(matrix)
        edge_count = int(np.count_nonzero(square > 0) - np.count_nonzero(np.diag(square) > 0))
        title = f"{label} {edge_count} edges"
        plot_matrix(matrix, mid, ax=axes[0, column_index])
        axes[0, column_index].set_title(title)
        plot_directed_graph(matrix, mid, ax=axes[1, column_index])
    if figure_title:
        fig.suptitle(figure_title)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    return fig, axes


def load_bss_trace_variants(
    dataset_key: str,
    project_root: Path,
    *,
    methods: Iterable[str] | None = None,
    require_saved_cleaned: bool = True,
) -> list[dict[str, Any]]:
    from ica_denoising.bss_notebook import BSS_METHODS, load_bss_outputs, load_traces

    if methods is None:
        methods = BSS_METHODS

    spec, raw_traces = load_traces(dataset_key, project_root)
    variants: list[dict[str, Any]] = [
        {
            "variant_name": "raw",
            "variant_label": f"{spec.data_name}: raw",
            "source_method": None,
            "source_path": str(spec.trace_path),
            "dataset": spec,
            "traces": np.asarray(raw_traces, dtype=float),
        }
    ]

    for method in methods:
        result = load_bss_outputs(
            dataset_key,
            method,
            project_root=project_root,
            output_data_name_override=spec.data_name,
        )
        cleaned_path = result.saved_paths.get("cleaned")
        if require_saved_cleaned and cleaned_path is None:
            raise FileNotFoundError(
                f"Saved cleaned trace for {dataset_key!r} and method {method!r} was not found. "
                "Run notebooks/motorneurons/decomposition.ipynb and save the cleaned outputs first."
            )
        variants.append(
            {
                "variant_name": f"{method}_cleaned",
                "variant_label": f"{spec.data_name}: {method}",
                "source_method": str(method),
                "source_path": str(cleaned_path) if cleaned_path is not None else None,
                "dataset": result.dataset,
                "traces": np.asarray(result.cleaned.T, dtype=float),
            }
        )

    return variants

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .features import _align_1d_target


def reconstruct_with_rejected(
    ic_comps: np.ndarray,
    mixing: np.ndarray,
    mean: np.ndarray,
    reject_components: Iterable[int] = (),
) -> np.ndarray:
    """Reconstruct frames x neurons traces after zeroing selected ICs."""

    ics = np.asarray(ic_comps, dtype=float)
    mixing_arr = np.asarray(mixing, dtype=float)
    mean_arr = np.asarray(mean, dtype=float)
    if ics.ndim != 2:
        raise ValueError(f"ic_comps must be 2D, got shape {ics.shape}.")
    if mixing_arr.ndim != 2 or mixing_arr.shape[1] != ics.shape[1]:
        raise ValueError(
            "mixing must be neurons x components and match ic_comps columns; "
            f"got {mixing_arr.shape} for {ics.shape} ICs."
        )
    if mean_arr.shape != (mixing_arr.shape[0],):
        raise ValueError(
            f"mean must have shape ({mixing_arr.shape[0]},), got {mean_arr.shape}."
        )
    comps = ics.copy()
    reject = [int(component) for component in reject_components]
    if reject:
        comps[:, reject] = 0.0
    return comps @ mixing_arr.T + mean_arr


def leave_one_ic_out_validation(
    ic_comps: np.ndarray,
    mixing: np.ndarray,
    mean: np.ndarray,
    reference_traces: np.ndarray,
    *,
    components: Iterable[int] | None = None,
    behavior_targets: Mapping[str, np.ndarray] | object | None = None,
) -> pd.DataFrame:
    """Measure reconstruction impact after removing each selected IC."""

    full = reconstruct_with_rejected(ic_comps, mixing, mean, reject_components=())
    reference = _orient_reference(reference_traces, full.shape)
    targets = _normalize_targets(behavior_targets, full.shape[0])
    component_ids = list(range(np.asarray(ic_comps).shape[1]))
    if components is not None:
        component_ids = [int(component) for component in components]

    full_reference_corr = _pearson_flat(reference, full)
    full_population = np.mean(full, axis=1)
    rows = []
    for component in component_ids:
        removed = reconstruct_with_rejected(
            ic_comps, mixing, mean, reject_components=[component]
        )
        delta = removed - full
        row = {
            "component": int(component),
            "mean_abs_delta": float(np.mean(np.abs(delta))),
            "rms_delta": float(np.sqrt(np.mean(delta**2))),
            "reference_corr_full": float(full_reference_corr),
            "reference_corr_removed": float(_pearson_flat(reference, removed)),
            "variance_ratio_median_removed": float(
                _median_variance_ratio(reference, removed)
            ),
        }
        row["reference_corr_delta"] = (
            row["reference_corr_removed"] - row["reference_corr_full"]
        )
        removed_population = np.mean(removed, axis=1)
        for name, target in targets.items():
            full_corr = _pearson_1d(full_population, target)
            removed_corr = _pearson_1d(removed_population, target)
            row[f"behavior_corr_full_{name}"] = float(full_corr)
            row[f"behavior_corr_removed_{name}"] = float(removed_corr)
            row[f"behavior_corr_delta_{name}"] = float(
                abs(removed_corr) - abs(full_corr)
            )
        rows.append(row)

    return pd.DataFrame(rows)


def grouped_removal_validation(
    ic_comps: np.ndarray,
    mixing: np.ndarray,
    mean: np.ndarray,
    reference_traces: np.ndarray,
    groups: Mapping[str, Iterable[int]],
    *,
    behavior_targets: Mapping[str, np.ndarray] | object | None = None,
) -> pd.DataFrame:
    """Measure reconstruction impact after removing named IC groups."""

    full = reconstruct_with_rejected(ic_comps, mixing, mean, reject_components=())
    reference = _orient_reference(reference_traces, full.shape)
    targets = _normalize_targets(behavior_targets, full.shape[0])
    full_reference_corr = _pearson_flat(reference, full)
    full_population = np.mean(full, axis=1)
    rows = []
    for name, components in groups.items():
        component_list = [int(component) for component in components]
        removed = reconstruct_with_rejected(
            ic_comps, mixing, mean, reject_components=component_list
        )
        delta = removed - full
        row = {
            "group": str(name),
            "components": ",".join(str(component) for component in component_list),
            "n_components": len(component_list),
            "mean_abs_delta": float(np.mean(np.abs(delta))),
            "rms_delta": float(np.sqrt(np.mean(delta**2))),
            "reference_corr_full": float(full_reference_corr),
            "reference_corr_removed": float(_pearson_flat(reference, removed)),
            "variance_ratio_median_removed": float(
                _median_variance_ratio(reference, removed)
            ),
        }
        row["reference_corr_delta"] = (
            row["reference_corr_removed"] - row["reference_corr_full"]
        )
        removed_population = np.mean(removed, axis=1)
        for target_name, target in targets.items():
            full_corr = _pearson_1d(full_population, target)
            removed_corr = _pearson_1d(removed_population, target)
            row[f"behavior_corr_delta_{target_name}"] = float(
                abs(removed_corr) - abs(full_corr)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _orient_reference(
    reference: np.ndarray, target_shape: tuple[int, int]
) -> np.ndarray:
    arr = np.asarray(reference, dtype=float)
    if arr.shape == target_shape:
        return arr
    if arr.T.shape == target_shape:
        return arr.T
    raise ValueError(
        f"reference_traces shape {arr.shape} cannot match reconstruction {target_shape}."
    )


def _normalize_targets(
    behavior_targets: Mapping[str, np.ndarray] | object | None,
    n_frames: int,
) -> dict[str, np.ndarray]:
    if behavior_targets is None:
        return {}
    if isinstance(behavior_targets, Mapping):
        raw = behavior_targets
    else:
        raw = {
            name: getattr(behavior_targets, name)
            for name in ("angle", "vigor", "bout_state")
            if hasattr(behavior_targets, name)
        }
    return {
        str(name): _align_1d_target(
            np.asarray(values, dtype=float).reshape(-1), n_frames
        )
        for name, values in raw.items()
    }


def _pearson_flat(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson_1d(np.asarray(x).ravel(), np.asarray(y).ravel())


def _pearson_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError(
            f"Correlation inputs must match lengths, got {x.size} and {y.size}."
        )
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std == 0 or y_std == 0 or not np.isfinite(x_std) or not np.isfinite(y_std):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _median_variance_ratio(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref_var = np.var(reference, axis=0)
    cand_var = np.var(candidate, axis=0)
    ratio = cand_var / np.maximum(ref_var, np.finfo(float).eps)
    return float(np.median(ratio))

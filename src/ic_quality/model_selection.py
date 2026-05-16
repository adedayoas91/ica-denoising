from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pipeline import run_ic_quality_pipeline


@dataclass(frozen=True)
class NComponentsSelectionResult:
    """Selected component count plus full sweep evidence."""

    best_n_components: int
    scores: pd.DataFrame


def select_n_components_for_method(
    *,
    traces: np.ndarray,
    method: str,
    sample_rate_hz: float,
    candidate_components: Iterable[int],
    behavior_targets: Mapping[str, np.ndarray] | object | None = None,
    tol: float = 0.0001,
    max_iter: int = 500,
    random_state: int = 0,
) -> NComponentsSelectionResult:
    """
    Select ``n_components`` via hybrid IC-quality evidence (not explained variance).

    The objective rewards settings where:
    - artifact/protect scores clearly separate likely bad vs useful ICs,
    - removing keep ICs hurts reconstruction more than removing drop ICs,
    - the recommendation set is decisive (fewer review/ambiguous ICs).
    """

    from ica_utils import bss_dec

    trace_arr = np.asarray(traces, dtype=float)
    if trace_arr.ndim != 2:
        raise ValueError(f"traces must be 2D neurons x frames, got {trace_arr.shape}.")
    limit = int(min(trace_arr.shape))
    candidates = sorted({int(value) for value in candidate_components if 1 <= int(value) <= limit})
    if not candidates:
        raise ValueError("No valid candidate n_components values after bounds filtering.")

    rows: list[dict[str, float | int | str]] = []
    for n_components in candidates:
        ic_comps, _ic_ft, mixing, mean = bss_dec(
            trace_arr,
            n_comps=int(n_components),
            t=float(tol),
            max_=int(max_iter),
            method=str(method),
            random_state=int(random_state),
        )
        quality = run_ic_quality_pipeline(
            ic_comps=ic_comps,
            mixing=mixing,
            mean=mean,
            sample_rate_hz=float(sample_rate_hz),
            reference_traces=trace_arr.T,
            behavior_targets=behavior_targets,
            method=str(method),
            dataset=None,
            output_dir=None,
        )
        rows.append(_score_candidate(quality.recommendations))

    score_table = pd.DataFrame(rows)
    score_table = score_table.sort_values(["overall_score", "n_components"], ascending=[False, True])
    best_n = int(score_table.iloc[0]["n_components"])
    return NComponentsSelectionResult(best_n_components=best_n, scores=score_table.reset_index(drop=True))


def select_n_components_by_method(
    *,
    traces: np.ndarray,
    methods: Iterable[str],
    sample_rate_hz: float,
    candidate_components: Iterable[int],
    behavior_targets: Mapping[str, np.ndarray] | object | None = None,
    tol: float = 0.0001,
    max_iter: int = 500,
    random_state: int = 0,
) -> tuple[dict[str, int], pd.DataFrame]:
    """Run hybrid n-component selection for each method independently."""

    selected: dict[str, int] = {}
    tables: list[pd.DataFrame] = []
    for method in methods:
        result = select_n_components_for_method(
            traces=traces,
            method=method,
            sample_rate_hz=sample_rate_hz,
            candidate_components=candidate_components,
            behavior_targets=behavior_targets,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
        )
        selected[str(method)] = int(result.best_n_components)
        table = result.scores.copy()
        table.insert(0, "method", str(method))
        tables.append(table)
    return selected, pd.concat(tables, ignore_index=True)


def _score_candidate(recommendations: pd.DataFrame) -> dict[str, float | int]:
    if recommendations.empty:
        raise ValueError("recommendations table is empty.")

    table = recommendations.copy()
    n_components = int(table["component"].nunique())
    labels = table["recommendation"].astype(str)
    drop_mask = labels.eq("drop")
    keep_mask = labels.eq("keep")
    review_mask = labels.eq("review")

    artifact_gap = _safe_mean(table.loc[drop_mask, "artifact_score"]) - _safe_mean(
        table.loc[keep_mask, "artifact_score"]
    )
    protect_gap = _safe_mean(table.loc[keep_mask, "protect_score"]) - _safe_mean(
        table.loc[drop_mask, "protect_score"]
    )

    keep_impact = _safe_mean(np.abs(table.loc[keep_mask, "reference_corr_delta"]))
    drop_impact = _safe_mean(np.abs(table.loc[drop_mask, "reference_corr_delta"]))
    validation_gap = keep_impact - drop_impact

    drop_fraction = float(drop_mask.mean())
    keep_fraction = float(keep_mask.mean())
    review_fraction = float(review_mask.mean())

    overall_score = (
        0.40 * artifact_gap
        + 0.30 * protect_gap
        + 0.20 * validation_gap
        - 0.10 * review_fraction
    )

    return {
        "n_components": n_components,
        "overall_score": float(overall_score),
        "artifact_gap": float(artifact_gap),
        "protect_gap": float(protect_gap),
        "validation_gap": float(validation_gap),
        "drop_fraction": drop_fraction,
        "keep_fraction": keep_fraction,
        "review_fraction": review_fraction,
        "drop_count": int(drop_mask.sum()),
        "keep_count": int(keep_mask.sum()),
        "review_count": int(review_mask.sum()),
    }


def _safe_mean(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))

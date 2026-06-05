from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .features import ICFeatureConfig, compute_ic_features
from .reporting import save_ic_quality_outputs
from .scoring import ICScoringConfig, score_ic_candidates
from .validation import leave_one_ic_out_validation


@dataclass(frozen=True)
class ICQualityResult:
    """Outputs from a hybrid IC-quality review run."""

    features: pd.DataFrame
    recommendations: pd.DataFrame
    validation: pd.DataFrame | None
    saved_paths: dict[str, Path]


def run_ic_quality_pipeline(
    *,
    ic_comps: np.ndarray,
    mixing: np.ndarray,
    mean: np.ndarray,
    sample_rate_hz: float,
    reference_traces: np.ndarray | None = None,
    behavior_targets: Mapping[str, np.ndarray] | object | None = None,
    roi_positions: np.ndarray | None = None,
    method: str | None = None,
    dataset: str | None = None,
    output_dir: Path | None = None,
    feature_config: ICFeatureConfig | None = None,
    scoring_config: ICScoringConfig | None = None,
) -> ICQualityResult:
    """Run feature extraction, scoring, optional validation, and optional saving."""

    feature_config = feature_config or ICFeatureConfig(sample_rate_hz=sample_rate_hz)
    features = compute_ic_features(
        ic_comps,
        mixing,
        feature_config,
        behavior_targets=behavior_targets,
        roi_positions=roi_positions,
        method=method,
    )
    recommendations = score_ic_candidates(features, scoring_config)
    validation = None
    if reference_traces is not None:
        validation = leave_one_ic_out_validation(
            ic_comps,
            mixing,
            mean,
            reference_traces,
            behavior_targets=behavior_targets,
        )
        recommendations = recommendations.merge(validation, on="component", how="left")

    saved_paths: dict[str, Path] = {}
    if output_dir is not None:
        saved_paths = save_ic_quality_outputs(
            Path(output_dir),
            features,
            recommendations,
            validation_table=validation,
            metadata={
                "method": method,
                "dataset": dataset,
                "sample_rate_hz": sample_rate_hz,
                "n_components": int(np.asarray(ic_comps).shape[1]),
                "human_in_loop": True,
            },
        )

    return ICQualityResult(
        features=features,
        recommendations=recommendations,
        validation=validation,
        saved_paths=saved_paths,
    )

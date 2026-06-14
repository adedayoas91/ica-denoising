from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ICScoringConfig:
    """Decision thresholds for human-in-the-loop IC recommendations."""

    artifact_score_drop: float = 0.70
    artifact_score_review: float = 0.55
    protect_score_drop_max: float = 0.45
    protect_score_keep: float = 0.60
    high_freq_ratio_flag: float = 0.35
    robust_peak_density_flag: float = 0.02
    abs_kurtosis_flag: float = 5.0
    variance_ratio_flag: float = 5.0
    loading_ratio_flag: float = 10.0
    behavior_corr_flag: float = 0.2
    low_freq_ratio_flag: float = 0.6
    lag1_autocorr_flag: float = 0.4
    artifact_columns: tuple[str, ...] = field(
        default=(
            "high_freq_power_ratio",
            "robust_peak_density",
            "abs_excess_kurtosis",
            "window_variance_ratio",
            "max_median_loading_ratio",
            "narrowband_ratio",
        )
    )
    protect_columns: tuple[str, ...] = field(
        default=(
            "low_freq_power_ratio",
            "lag1_autocorr",
            "max_abs_behavior_corr",
            "stability_score",
        )
    )


def score_ic_candidates(
    feature_table: pd.DataFrame,
    config: ICScoringConfig | None = None,
) -> pd.DataFrame:
    """Add artifact/protect scores and a ``drop``/``keep``/``review`` label."""

    config = config or ICScoringConfig()
    if "component" not in feature_table.columns:
        raise ValueError("feature_table must contain a component column.")

    scored = feature_table.copy()
    scored["artifact_score"] = _score_columns(scored, config.artifact_columns)
    scored["protect_score"] = _score_columns(scored, config.protect_columns)
    recommendations = []
    evidence_flags = []
    protect_flags = []

    for _, row in scored.iterrows():
        evidence = _artifact_flags(row, config)
        protective = _protect_flags(row, config)
        evidence_flags.append(";".join(evidence))
        protect_flags.append(";".join(protective))
        recommendations.append(_recommendation(row, config))

    scored["artifact_flags"] = evidence_flags
    scored["protect_flags"] = protect_flags
    scored["recommendation"] = recommendations
    order = {"drop": 0, "review": 1, "keep": 2}
    scored["_recommendation_order"] = scored["recommendation"].map(order).fillna(3)
    scored = scored.sort_values(
        ["_recommendation_order", "artifact_score", "protect_score"],
        ascending=[True, False, True],
    ).drop(columns=["_recommendation_order"])
    return scored.reset_index(drop=True)


def _score_columns(table: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    present = [column for column in columns if column in table.columns]
    if not present:
        return pd.Series(np.zeros(len(table)), index=table.index, dtype=float)
    ranks = []
    for column in present:
        values = pd.to_numeric(table[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        values = values.fillna(values.median() if values.notna().any() else 0.0)
        ranks.append(values.rank(method="average", pct=True))
    return pd.concat(ranks, axis=1).mean(axis=1).astype(float)


def _recommendation(row: pd.Series, config: ICScoringConfig) -> str:
    artifact_score = float(row.get("artifact_score", 0.0))
    protect_score = float(row.get("protect_score", 0.0))
    if (
        artifact_score >= config.artifact_score_drop
        and protect_score <= config.protect_score_drop_max
    ):
        return "drop"
    if (
        protect_score >= config.protect_score_keep
        and artifact_score < config.artifact_score_drop
    ):
        return "keep"
    if (
        artifact_score >= config.artifact_score_review
        or protect_score >= config.protect_score_keep
    ):
        return "review"
    return "keep"


def _artifact_flags(row: pd.Series, config: ICScoringConfig) -> list[str]:
    flags = []
    if _value(row, "high_freq_power_ratio") >= config.high_freq_ratio_flag:
        flags.append("high_frequency")
    if _value(row, "robust_peak_density") >= config.robust_peak_density_flag:
        flags.append("bursty")
    if _value(row, "abs_excess_kurtosis") >= config.abs_kurtosis_flag:
        flags.append("heavy_tailed")
    if _value(row, "window_variance_ratio") >= config.variance_ratio_flag:
        flags.append("nonstationary")
    if _value(row, "max_median_loading_ratio") >= config.loading_ratio_flag:
        flags.append("concentrated_loading")
    return flags


def _protect_flags(row: pd.Series, config: ICScoringConfig) -> list[str]:
    flags = []
    if _value(row, "max_abs_behavior_corr") >= config.behavior_corr_flag:
        flags.append("behavior_link")
    if _value(row, "low_freq_power_ratio") >= config.low_freq_ratio_flag:
        flags.append("slow_calcium_band")
    if _value(row, "lag1_autocorr") >= config.lag1_autocorr_flag:
        flags.append("autocorrelated")
    if _value(row, "stability_score") >= config.protect_score_keep:
        flags.append("stable_across_runs")
    return flags


def _value(row: pd.Series, column: str) -> float:
    value = row.get(column, 0.0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, welch
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis, NMF, PCA, SparsePCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, roc_auc_score
from sklearn.neural_network import MLPRegressor

from ica_denoising.behavior_decoding import (
    BehaviorTargets,
    TraceVariant,
    classification_metrics,
    make_lagged_design,
    moving_average,
    regression_metrics,
    standardize_train_test,
    summarize_trace_preservation,
)
from ica_denoising.bss_notebook import DEFAULT_PCA_VARIANCE_THRESHOLD, resolve_bss_component_selection
from ica_denoising.causal_behavior_decoding import CausalStateConfig, make_paired_windows, minimum_causal_gap
from ica_denoising.core.ica_utils import bss_dec, rank_clusters_by_mean_log_psd
from ica_denoising.evaluation_diagnostics import (
    cluster_stability_table,
    pseudo_artifact_reconstruction_test,
    temporal_dependence_diagnostics,
)
from ica_denoising.ic_quality.features import ICFeatureConfig, compute_ic_features
from ica_denoising.ic_quality.scoring import score_ic_candidates


@dataclass(frozen=True)
class StrictFold:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    excluded_idx: np.ndarray


@dataclass(frozen=True)
class FittedBSSModel:
    method: str
    mean: np.ndarray
    mixing: np.ndarray
    unmixing: np.ndarray
    train_components: np.ndarray
    train_idx: np.ndarray
    train_segment_lengths: tuple[int, ...]
    n_components: int
    pca_components: int | None
    pca_variance_threshold: float | None
    pca_explained_variance_ratio: float | None
    component_selection_mode: str
    fit_warnings: tuple[str, ...] = ()

    def transform(self, traces: np.ndarray) -> np.ndarray:
        traces = _validate_traces(traces)
        if traces.shape[1] != self.mean.size:
            raise ValueError(
                f"Expected {self.mean.size} features, got {traces.shape[1]}."
            )
        return (traces - self.mean) @ self.unmixing.T

    def reconstruct(
        self,
        traces: np.ndarray,
        *,
        keep_components: Iterable[int] | None = None,
    ) -> np.ndarray:
        components = self.transform(traces)
        if keep_components is not None:
            keep = np.asarray(sorted(set(int(value) for value in keep_components)), dtype=int)
            if keep.size and (keep.min() < 0 or keep.max() >= components.shape[1]):
                raise ValueError("keep_components contains an out-of-range component.")
            mask = np.zeros(components.shape[1], dtype=bool)
            mask[keep] = True
            components[:, ~mask] = 0.0
        return components @ self.mixing.T + self.mean


@dataclass(frozen=True)
class FittedSubspaceModel:
    name: str
    mean: np.ndarray
    basis: np.ndarray
    train_idx: np.ndarray

    def reconstruct(self, traces: np.ndarray) -> np.ndarray:
        traces = _validate_traces(traces)
        centered = traces - self.mean
        return centered @ self.basis @ self.basis.T + self.mean


@dataclass(frozen=True)
class FittedLatentBenchmarkModel:
    name: str
    rank: int
    train_idx: np.ndarray
    model: object
    mean: np.ndarray | None = None
    train_shift: np.ndarray | None = None

    def reconstruct(self, traces: np.ndarray) -> np.ndarray:
        traces = _validate_traces(traces)
        if self.name == "factor_analysis":
            transformed = self.model.transform(traces)
            return transformed @ self.model.components_ + self.model.mean_
        if self.name == "sparse_pca":
            transformed = self.model.transform(traces - self.mean)
            return transformed @ self.model.components_ + self.mean
        if self.name == "nmf":
            shifted = _apply_train_nonnegative_shift(traces, self.train_shift)
            transformed = self.model.transform(shifted)
            reconstructed = self.model.inverse_transform(transformed)
            return reconstructed - self.train_shift
        raise ValueError(f"Unknown latent benchmark model: {self.name}")


@dataclass(frozen=True)
class ClusterSelection:
    labels: np.ndarray
    spectra: np.ndarray
    features: np.ndarray
    cluster_order: tuple[int, ...]
    component_energy: np.ndarray


@dataclass(frozen=True)
class FoldVariant:
    name: str
    family: str
    traces: np.ndarray
    fit_idx: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationConfig:
    sample_rate_hz: float
    n_splits: int = 5
    gap: int = 14
    bss_methods: tuple[str, ...] = ("fastica", "infomax", "sobi", "jade")
    n_components: int | None = None
    bss_pca_components: int | None = None
    bss_pca_variance_threshold: float | None = DEFAULT_PCA_VARIANCE_THRESHOLD
    bss_tolerance: float = 0.0001
    bss_max_iter: int = 500
    sobi_lags: tuple[int, ...] = (1, 2, 3, 5)
    jade_max_cumulant_matrices: int = 200
    n_clusters: int = 7
    keep_cluster_counts: tuple[int, ...] = (1, 2)
    selection_strategies: tuple[str, ...] = (
        "low_frequency",
        "high_frequency",
        "random",
        "energy_matched_random",
        "all",
    )
    selection_random_seeds: tuple[int, ...] = (0,)
    cluster_stability_seeds: tuple[int, ...] = (0,)
    cluster_stability_counts: tuple[int, ...] = ()
    cluster_stability_feature_transforms: tuple[str, ...] = ()
    cluster_stability_keep_count: int = 1
    feature_start_bin: int = 30
    feature_transform: str = "log1p"
    welch_nperseg: int = 250
    welch_noverlap: int = 125
    ranking_fmin_hz: float = 0.0
    ranking_fmax_hz: float = 0.20
    ranking_aggregate: str = "peak"
    baseline_ranks: tuple[int, ...] = ()
    benchmark_latent_methods: tuple[str, ...] = ()
    benchmark_latent_ranks: tuple[int, ...] = ()
    ic_quality_selection_strategies: tuple[str, ...] = ()
    bss_component_counts: tuple[int, ...] = ()
    bss_pca_variance_thresholds: tuple[float, ...] = ()
    include_energy_matched_pca: bool = True
    lowpass_cutoffs_hz: tuple[float, ...] = (0.20,)
    decoder_lags: tuple[int, ...] = (0, 1, 2)
    decoder_target_shift: int = 0
    decoder_ridge_alpha: float = 10.0
    bout_quantile: float = 0.75
    target_bout_quantiles: tuple[float, ...] = ()
    target_smooth_windows: tuple[int, ...] = ()
    null_block_size: int = 60
    null_block_sizes: tuple[int, ...] = ()
    null_seeds: tuple[int, ...] = (0,)
    null_strategies: tuple[str, ...] = ("block_shuffle",)
    null_circular_min_shift: int = 60
    uncertainty_block_size: int = 60
    uncertainty_n_bootstrap: int = 2000
    uncertainty_n_permutations: int = 5000
    random_state: int = 0
    causal: CausalStateConfig = field(default_factory=CausalStateConfig)
    causal_transition_models: tuple[str, ...] = ()
    causal_sufficiency_targets: tuple[str, ...] = ("vigor",)
    artifact_probe_centers: tuple[int, ...] = ()
    artifact_probe_half_width: int = 2


@dataclass(frozen=True)
class EvaluationResult:
    trace_metrics: pd.DataFrame
    behavior_metrics: pd.DataFrame
    behavior_predictions: pd.DataFrame
    causal_metrics: pd.DataFrame
    causal_embeddings: pd.DataFrame
    causal_sufficiency: pd.DataFrame
    artifact_probe_metrics: pd.DataFrame
    leakage_audit: pd.DataFrame
    component_selections: pd.DataFrame
    cluster_stability: pd.DataFrame
    cluster_stability_assignments: pd.DataFrame
    variant_metadata: pd.DataFrame
    temporal_diagnostics: pd.DataFrame


def make_strict_folds(
    n_samples: int,
    *,
    n_splits: int,
    gap: int,
    required_gap: int = 0,
) -> tuple[StrictFold, ...]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if gap < required_gap:
        raise ValueError(
            f"gap={gap} is too small; use at least {required_gap} for the requested histories."
        )
    edges = np.linspace(0, n_samples, n_splits + 1).round().astype(int)
    all_idx = np.arange(n_samples, dtype=int)
    folds = []
    for fold_id in range(n_splits):
        start, stop = int(edges[fold_id]), int(edges[fold_id + 1])
        test_idx = all_idx[start:stop]
        excluded_start = max(0, start - gap)
        excluded_stop = min(n_samples, stop + gap)
        excluded_idx = all_idx[excluded_start:excluded_stop]
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[excluded_idx] = False
        train_idx = all_idx[train_mask]
        if train_idx.size == 0 or test_idx.size == 0:
            raise ValueError("Empty train or test block. Reduce n_splits or gap.")
        folds.append(
            StrictFold(
                fold=fold_id,
                train_idx=train_idx,
                test_idx=test_idx,
                excluded_idx=excluded_idx,
            )
        )
    return tuple(folds)


def contiguous_index_segments(indices: Sequence[int]) -> tuple[np.ndarray, ...]:
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return ()
    indices = np.unique(indices)
    split_points = np.flatnonzero(np.diff(indices) > 1) + 1
    return tuple(np.asarray(segment, dtype=int) for segment in np.split(indices, split_points))


def fit_bss_model(
    traces: np.ndarray,
    train_idx: Sequence[int],
    *,
    method: str,
    n_components: int | None,
    pca_components: int | None = None,
    pca_variance_threshold: float | None = DEFAULT_PCA_VARIANCE_THRESHOLD,
    tolerance: float = 0.0001,
    max_iter: int = 500,
    random_state: int = 0,
    sobi_lags: Sequence[int] = (1, 2, 3, 5),
    jade_max_cumulant_matrices: int = 200,
) -> FittedBSSModel:
    traces = _validate_traces(traces)
    train_idx = np.asarray(train_idx, dtype=int)
    _validate_fit_indices(train_idx, traces.shape[0])
    segments = contiguous_index_segments(train_idx)
    train_traces = np.vstack([traces[segment] for segment in segments])
    segment_lengths = tuple(int(segment.size) for segment in segments)
    component_selection = resolve_bss_component_selection(
        train_traces.T,
        method,
        n_components=n_components,
        pca_components=pca_components,
        pca_variance_threshold=pca_variance_threshold,
    )
    kwargs: dict[str, object] = {"segment_lengths": segment_lengths}
    if method.lower() == "sobi":
        kwargs["lags"] = tuple(int(lag) for lag in sobi_lags)
    if method.lower() == "jade":
        kwargs["max_cumulant_matrices"] = int(jade_max_cumulant_matrices)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        components, _spectra, mixing, mean = bss_dec(
            train_traces.T,
            n_comps=component_selection.n_components,
            t=tolerance,
            max_=max_iter,
            method=method,
            random_state=random_state,
            pca_components=component_selection.pca_components,
            **kwargs,
        )
    unmixing = np.linalg.pinv(np.asarray(mixing, dtype=float))
    return FittedBSSModel(
        method=method.lower(),
        mean=np.asarray(mean, dtype=float),
        mixing=np.asarray(mixing, dtype=float),
        unmixing=unmixing,
        train_components=np.asarray(components, dtype=float),
        train_idx=train_idx.copy(),
        train_segment_lengths=segment_lengths,
        n_components=component_selection.n_components,
        pca_components=component_selection.pca_components,
        pca_variance_threshold=component_selection.pca_variance_threshold,
        pca_explained_variance_ratio=component_selection.pca_explained_variance_ratio,
        component_selection_mode=component_selection.component_selection_mode,
        fit_warnings=tuple(str(item.message) for item in caught),
    )


def fit_subspace_model(
    traces: np.ndarray,
    train_idx: Sequence[int],
    *,
    rank: int,
    kind: str,
    random_state: int = 0,
) -> FittedSubspaceModel:
    traces = _validate_traces(traces)
    train_idx = np.asarray(train_idx, dtype=int)
    _validate_fit_indices(train_idx, traces.shape[0])
    train = traces[train_idx]
    rank = min(int(rank), *train.shape)
    if rank < 1:
        raise ValueError("rank must be at least 1.")
    mean = train.mean(axis=0)
    if kind == "pca":
        pca = PCA(n_components=rank, svd_solver="full", random_state=random_state)
        pca.fit(train - mean)
        basis = pca.components_.T
    elif kind == "random_subspace":
        rng = np.random.default_rng(random_state)
        basis, _ = np.linalg.qr(rng.standard_normal((traces.shape[1], rank)), mode="reduced")
    else:
        raise ValueError("kind must be 'pca' or 'random_subspace'.")
    return FittedSubspaceModel(
        name=kind,
        mean=mean,
        basis=np.asarray(basis, dtype=float),
        train_idx=train_idx.copy(),
    )


def fit_latent_benchmark_model(
    traces: np.ndarray,
    train_idx: Sequence[int],
    *,
    method: str,
    rank: int,
    random_state: int = 0,
) -> FittedLatentBenchmarkModel:
    traces = _validate_traces(traces)
    train_idx = np.asarray(train_idx, dtype=int)
    _validate_fit_indices(train_idx, traces.shape[0])
    train = traces[train_idx]
    rank = min(int(rank), *train.shape)
    if rank < 1:
        raise ValueError("rank must be at least 1.")
    method = method.lower()
    if method == "factor_analysis":
        model = FactorAnalysis(n_components=rank, random_state=random_state)
        model.fit(train)
        return FittedLatentBenchmarkModel(method, rank, train_idx.copy(), model)
    if method == "sparse_pca":
        mean = train.mean(axis=0)
        model = SparsePCA(
            n_components=rank,
            random_state=random_state,
            max_iter=1000,
            tol=1e-6,
        )
        model.fit(train - mean)
        return FittedLatentBenchmarkModel(
            method,
            rank,
            train_idx.copy(),
            model,
            mean=np.asarray(mean, dtype=float),
        )
    if method == "nmf":
        train_shift = np.minimum(train.min(axis=0), 0.0)
        shifted_train = _apply_train_nonnegative_shift(train, train_shift)
        model = NMF(
            n_components=rank,
            init="nndsvda",
            random_state=random_state,
            max_iter=1000,
        )
        model.fit(shifted_train)
        return FittedLatentBenchmarkModel(
            method,
            rank,
            train_idx.copy(),
            model,
            train_shift=np.asarray(train_shift, dtype=float),
        )
    raise ValueError(
        "benchmark latent method must be 'factor_analysis', 'sparse_pca', or 'nmf'."
    )


def _apply_train_nonnegative_shift(
    traces: np.ndarray,
    train_shift: np.ndarray | None,
) -> np.ndarray:
    if train_shift is None:
        raise ValueError("train_shift is required for NMF reconstruction.")
    shifted = _validate_traces(traces) - np.asarray(train_shift, dtype=float)
    return np.clip(shifted, 0.0, None)


def fit_pca_energy_models(
    traces: np.ndarray,
    train_idx: Sequence[int],
    target_fractions: Mapping[str, float],
    *,
    random_state: int = 0,
) -> dict[str, tuple[FittedSubspaceModel, float]]:
    traces = _validate_traces(traces)
    train_idx = np.asarray(train_idx, dtype=int)
    _validate_fit_indices(train_idx, traces.shape[0])
    train = traces[train_idx]
    mean = train.mean(axis=0)
    max_rank = min(train.shape)
    pca = PCA(n_components=max_rank, svd_solver="full", random_state=random_state)
    pca.fit(train - mean)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    models = {}
    for name, fraction in target_fractions.items():
        target = float(np.clip(fraction, 0.0, 1.0))
        rank = min(int(np.searchsorted(cumulative, target, side="left") + 1), max_rank)
        models[name] = (
            FittedSubspaceModel(
                name="pca_energy_match",
                mean=mean,
                basis=np.asarray(pca.components_[:rank].T, dtype=float),
                train_idx=train_idx.copy(),
            ),
            float(cumulative[rank - 1]),
        )
    return models


def segmented_welch_psd(
    components: np.ndarray,
    segment_lengths: Sequence[int],
    *,
    sample_rate_hz: float,
    nperseg: int,
    noverlap: int,
) -> np.ndarray:
    components = _validate_traces(components)
    lengths = tuple(int(length) for length in segment_lengths)
    if sum(lengths) != components.shape[0]:
        raise ValueError("segment_lengths must sum to the number of component samples.")
    nfft = min(max(2, int(nperseg)), max(lengths))
    weighted_spectra = None
    total_weight = 0.0
    start = 0
    for length in lengths:
        segment = components[start : start + length]
        start += length
        if length < 2:
            continue
        segment_nperseg = min(nfft, length)
        segment_noverlap = min(int(noverlap), segment_nperseg - 1)
        _, power = welch(
            segment,
            fs=sample_rate_hz,
            axis=0,
            nperseg=segment_nperseg,
            noverlap=segment_noverlap,
            nfft=nfft,
            return_onesided=True,
        )
        power = power.T
        weight = float(length)
        weighted_spectra = (
            power * weight
            if weighted_spectra is None
            else weighted_spectra + power * weight
        )
        total_weight += weight
    if weighted_spectra is None or total_weight == 0:
        raise ValueError("No segment is long enough for Welch spectral estimation.")
    return weighted_spectra / total_weight


def fit_cluster_selection(
    model: FittedBSSModel,
    *,
    sample_rate_hz: float,
    n_clusters: int,
    feature_start_bin: int,
    feature_transform: str,
    nperseg: int,
    noverlap: int,
    ranking_fmin_hz: float,
    ranking_fmax_hz: float,
    ranking_aggregate: str,
    random_state: int,
) -> ClusterSelection:
    spectra = segmented_welch_psd(
        model.train_components,
        model.train_segment_lengths,
        sample_rate_hz=sample_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    return cluster_selection_from_spectra(
        model,
        spectra,
        sample_rate_hz=sample_rate_hz,
        n_clusters=n_clusters,
        feature_start_bin=feature_start_bin,
        feature_transform=feature_transform,
        ranking_fmin_hz=ranking_fmin_hz,
        ranking_fmax_hz=ranking_fmax_hz,
        ranking_aggregate=ranking_aggregate,
        random_state=random_state,
    )


def cluster_selection_from_spectra(
    model: FittedBSSModel,
    spectra: np.ndarray,
    *,
    sample_rate_hz: float,
    n_clusters: int,
    feature_start_bin: int,
    feature_transform: str,
    ranking_fmin_hz: float,
    ranking_fmax_hz: float,
    ranking_aggregate: str,
    random_state: int,
) -> ClusterSelection:
    spectra = np.asarray(spectra, dtype=float)
    if spectra.ndim != 2 or spectra.shape[0] != model.train_components.shape[1]:
        raise ValueError("spectra must contain one row per fitted component.")
    start_bin = min(max(int(feature_start_bin), 0), max(spectra.shape[1] - 1, 0))
    features = spectra[:, start_bin:]
    if feature_transform == "log1p":
        features = np.log1p(features)
    elif feature_transform == "normalized":
        denominator = features.sum(axis=1, keepdims=True)
        features = features / np.where(denominator > 0, denominator, 1.0)
    elif feature_transform != "raw":
        raise ValueError("feature_transform must be 'raw', 'log1p', or 'normalized'.")
    n_clusters = min(int(n_clusters), model.train_components.shape[1])
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1.")
    labels = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    ).fit_predict(features)
    ranked = rank_clusters_by_mean_log_psd(
        spectra,
        labels,
        sample_rate_hz,
        fmin=ranking_fmin_hz,
        fmax=ranking_fmax_hz,
        aggregate=ranking_aggregate,
    )
    return ClusterSelection(
        labels=np.asarray(labels, dtype=int),
        spectra=spectra,
        features=features,
        cluster_order=tuple(int(row["cluster"]) for row in ranked),
        component_energy=np.mean(model.train_components**2, axis=0),
    )


def select_components(
    selection: ClusterSelection,
    *,
    strategy: str,
    keep_cluster_count: int | None,
    random_state: int,
) -> np.ndarray:
    n_components = selection.labels.size
    if strategy == "all":
        return np.arange(n_components, dtype=int)
    if keep_cluster_count is None:
        raise ValueError(f"keep_cluster_count is required for strategy {strategy!r}.")
    keep_cluster_count = max(1, min(int(keep_cluster_count), len(selection.cluster_order)))
    order = np.asarray(selection.cluster_order, dtype=int)
    if strategy == "high_frequency":
        order = order[::-1]
    elif strategy == "random":
        order = np.random.default_rng(random_state).permutation(order)
    elif strategy == "energy_matched_random":
        reference_clusters = set(selection.cluster_order[:keep_cluster_count])
        reference_components = np.flatnonzero(np.isin(selection.labels, list(reference_clusters)))
        target_energy = float(selection.component_energy[reference_components].sum())
        component_order = np.random.default_rng(random_state).permutation(n_components)
        cumulative = np.cumsum(selection.component_energy[component_order])
        n_keep = int(np.searchsorted(cumulative, target_energy, side="left") + 1)
        return np.sort(component_order[: min(n_keep, n_components)])
    elif strategy != "low_frequency":
        raise ValueError(f"Unknown selection strategy: {strategy}")
    keep_clusters = set(int(value) for value in order[:keep_cluster_count])
    return np.flatnonzero(np.isin(selection.labels, list(keep_clusters)))


def ic_quality_table_for_bss_model(
    model: FittedBSSModel,
    *,
    sample_rate_hz: float,
    nperseg: int,
    noverlap: int,
) -> pd.DataFrame:
    features = compute_ic_features(
        model.train_components,
        model.mixing,
        ICFeatureConfig(
            sample_rate_hz=sample_rate_hz,
            nperseg=nperseg,
            noverlap=noverlap,
        ),
        behavior_targets=None,
        method=model.method,
    )
    return score_ic_candidates(features)


def select_components_by_ic_quality(
    quality_table: pd.DataFrame,
    *,
    strategy: str,
) -> np.ndarray:
    if "component" not in quality_table.columns or "recommendation" not in quality_table.columns:
        raise ValueError("quality_table must contain component and recommendation columns.")
    if strategy == "ic_quality_nonartifact":
        mask = quality_table["recommendation"].astype(str) != "drop"
    elif strategy == "ic_quality_strict_keep":
        mask = quality_table["recommendation"].astype(str) == "keep"
    else:
        raise ValueError(f"Unknown IC-quality selection strategy: {strategy}")
    return np.sort(quality_table.loc[mask, "component"].astype(int).to_numpy())


def run_cluster_stability_sweep(
    model: FittedBSSModel,
    *,
    sample_rate_hz: float,
    seeds: Iterable[int],
    cluster_counts: Iterable[int],
    feature_transforms: Iterable[str],
    keep_cluster_count: int,
    feature_start_bin: int = 30,
    nperseg: int = 250,
    noverlap: int = 125,
    ranking_fmin_hz: float = 0.0,
    ranking_fmax_hz: float = 0.20,
    ranking_aggregate: str = "peak",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments: dict[str, np.ndarray] = {}
    retained_sets: dict[str, np.ndarray] = {}
    component_ranks: dict[str, np.ndarray] = {}
    detail_rows: list[dict[str, object]] = []
    spectra = segmented_welch_psd(
        model.train_components,
        model.train_segment_lengths,
        sample_rate_hz=sample_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    for feature_transform in feature_transforms:
        for n_clusters in cluster_counts:
            for seed in seeds:
                key = f"{feature_transform}/k{int(n_clusters)}/seed{int(seed)}"
                selection = cluster_selection_from_spectra(
                    model,
                    spectra,
                    sample_rate_hz=sample_rate_hz,
                    n_clusters=int(n_clusters),
                    feature_start_bin=feature_start_bin,
                    feature_transform=feature_transform,
                    ranking_fmin_hz=ranking_fmin_hz,
                    ranking_fmax_hz=ranking_fmax_hz,
                    ranking_aggregate=ranking_aggregate,
                    random_state=int(seed),
                )
                assignments[key] = selection.labels
                retained_sets[key] = select_components(
                    selection,
                    strategy="low_frequency",
                    keep_cluster_count=keep_cluster_count,
                    random_state=int(seed),
                )
                rank_by_cluster = {
                    cluster: rank
                    for rank, cluster in enumerate(selection.cluster_order, start=1)
                }
                component_ranks[key] = np.asarray(
                    [rank_by_cluster[int(label)] for label in selection.labels],
                    dtype=float,
                )
                for component, label in enumerate(selection.labels):
                    detail_rows.append(
                        {
                            "configuration": key,
                            "feature_transform": feature_transform,
                            "n_clusters": int(n_clusters),
                            "seed": int(seed),
                            "component": int(component),
                            "cluster": int(label),
                            "cluster_rank": int(rank_by_cluster[int(label)]),
                            "retained": bool(component in retained_sets[key]),
                        }
                    )
    pairwise = cluster_stability_table(
        assignments,
        retained_sets=retained_sets,
        component_ranks=component_ranks,
    )
    return pairwise, pd.DataFrame(detail_rows)


def run_evaluation(
    traces: np.ndarray,
    targets: BehaviorTargets,
    config: EvaluationConfig,
) -> EvaluationResult:
    traces = _validate_traces(traces)
    _validate_targets(targets, traces.shape[0])
    required_gap = max(
        max(config.decoder_lags, default=0),
        minimum_causal_gap(config.causal),
    )
    folds = make_strict_folds(
        traces.shape[0],
        n_splits=config.n_splits,
        gap=config.gap,
        required_gap=required_gap,
    )
    behavior_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    causal_rows: list[dict[str, object]] = []
    embedding_rows: list[dict[str, object]] = []
    sufficiency_frames: list[pd.DataFrame] = []
    artifact_probe_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    stability_frames: list[pd.DataFrame] = []
    stability_assignment_frames: list[pd.DataFrame] = []
    variant_metadata_rows: list[dict[str, object]] = []
    temporal_frames: list[pd.DataFrame] = []
    trace_metric_frames: list[pd.DataFrame] = []

    for fold in folds:
        (
            variants,
            fold_audit,
            fold_selections,
            fold_stability,
            fold_stability_assignments,
        ) = make_fold_variants(traces, fold, config)
        audit_rows.extend(fold_audit)
        audit_rows.extend(_downstream_audit_rows(fold, variants))
        selection_rows.extend(fold_selections)
        variant_metadata_rows.extend(
            {
                "fold": fold.fold,
                "variant": variant.name,
                "family": variant.family,
                "metadata_json": json.dumps(dict(variant.metadata), sort_keys=True),
            }
            for variant in variants
        )
        if not fold_stability.empty:
            stability_frames.append(fold_stability)
        if not fold_stability_assignments.empty:
            stability_assignment_frames.append(fold_stability_assignments)
        trace_metric_frames.append(_held_out_trace_metrics(variants, fold, config))
        fold_behavior, fold_predictions = _evaluate_behavior_variants(
            variants, targets, fold, config
        )
        behavior_rows.extend(fold_behavior)
        prediction_frames.extend(fold_predictions)
        fold_causal, fold_embeddings = _evaluate_causal_variants(
            variants, targets, fold, config
        )
        causal_rows.extend(fold_causal)
        embedding_rows.extend(fold_embeddings)
        sufficiency = _evaluate_causal_sufficiency_variants(variants, targets, fold, config)
        if not sufficiency.empty:
            sufficiency_frames.append(sufficiency)
        artifact_probe = _evaluate_artifact_probe_variants(variants, fold, config)
        if not artifact_probe.empty:
            artifact_probe_frames.append(artifact_probe)

        test_variants = {
            variant.name: variant.traces[fold.test_idx]
            for variant in variants
            if variant.name != "raw"
        }
        valid_lags = tuple(
            lag
            for lag in sorted(set((*config.decoder_lags, *config.sobi_lags)))
            if 0 < lag < fold.test_idx.size
        )
        if valid_lags and test_variants:
            temporal = temporal_dependence_diagnostics(
                traces[fold.test_idx],
                test_variants,
                lags=valid_lags,
                sample_rate_hz=config.sample_rate_hz,
            )
            temporal.insert(0, "fold", fold.fold)
            temporal_frames.append(temporal)

    return EvaluationResult(
        trace_metrics=pd.concat(trace_metric_frames, ignore_index=True),
        behavior_metrics=pd.DataFrame(behavior_rows),
        behavior_predictions=(
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else pd.DataFrame()
        ),
        causal_metrics=pd.DataFrame(causal_rows),
        causal_embeddings=pd.DataFrame(embedding_rows),
        causal_sufficiency=(
            pd.concat(sufficiency_frames, ignore_index=True)
            if sufficiency_frames
            else pd.DataFrame()
        ),
        artifact_probe_metrics=(
            pd.concat(artifact_probe_frames, ignore_index=True)
            if artifact_probe_frames
            else pd.DataFrame()
        ),
        leakage_audit=pd.DataFrame(audit_rows),
        component_selections=pd.DataFrame(selection_rows),
        cluster_stability=(
            pd.concat(stability_frames, ignore_index=True)
            if stability_frames
            else pd.DataFrame()
        ),
        cluster_stability_assignments=(
            pd.concat(stability_assignment_frames, ignore_index=True)
            if stability_assignment_frames
            else pd.DataFrame()
        ),
        variant_metadata=pd.DataFrame(variant_metadata_rows),
        temporal_diagnostics=(
            pd.concat(temporal_frames, ignore_index=True)
            if temporal_frames
            else pd.DataFrame()
        ),
    )


def _held_out_trace_metrics(
    variants: Sequence[FoldVariant],
    fold: StrictFold,
    config: EvaluationConfig,
) -> pd.DataFrame:
    """Compare fold-local reconstructions with raw traces on held-out frames only."""
    held_out = [
        TraceVariant(
            name=variant.name,
            path=Path(variant.name),
            traces=variant.traces[fold.test_idx],
        )
        for variant in variants
    ]
    summary = summarize_trace_preservation(held_out, reference_name="raw").drop(
        columns="path"
    )
    raw = held_out[0].traces
    segment_lengths = (raw.shape[0],)
    raw_spectra = segmented_welch_psd(
        raw,
        segment_lengths,
        sample_rate_hz=config.sample_rate_hz,
        nperseg=config.welch_nperseg,
        noverlap=config.welch_noverlap,
    )
    frequencies = np.linspace(0.0, config.sample_rate_hz / 2.0, raw_spectra.shape[1])
    low_frequency_mask = (
        (frequencies >= config.ranking_fmin_hz)
        & (frequencies <= config.ranking_fmax_hz)
    )
    if not np.any(low_frequency_mask):
        raise ValueError(
            "No Welch frequency bins fall within the configured ranking band."
        )
    raw_total_power = float(raw_spectra.sum())
    raw_low_frequency_fraction = _safe_ratio(
        float(raw_spectra[:, low_frequency_mask].sum()),
        raw_total_power,
    )
    raw_centered = raw - raw.mean(axis=0, keepdims=True)
    raw_energy = float(np.sum(raw_centered**2))

    spectral_rows = []
    by_name = {variant.name: variant for variant in held_out}
    for name in summary["variant"]:
        values = by_name[str(name)].traces
        spectra = segmented_welch_psd(
            values,
            segment_lengths,
            sample_rate_hz=config.sample_rate_hz,
            nperseg=config.welch_nperseg,
            noverlap=config.welch_noverlap,
        )
        total_power = float(spectra.sum())
        low_frequency_fraction = _safe_ratio(
            float(spectra[:, low_frequency_mask].sum()),
            total_power,
        )
        centered = values - values.mean(axis=0, keepdims=True)
        spectral_rows.append(
            {
                "variant": name,
                "retained_energy_fraction": _safe_ratio(
                    float(np.sum(centered**2)),
                    raw_energy,
                ),
                "spectral_power_retention": _safe_ratio(
                    total_power,
                    raw_total_power,
                ),
                "low_frequency_power_fraction": low_frequency_fraction,
                "low_frequency_power_fraction_delta_from_raw": (
                    low_frequency_fraction - raw_low_frequency_fraction
                ),
                "low_frequency_fmin_hz": float(config.ranking_fmin_hz),
                "low_frequency_fmax_hz": float(config.ranking_fmax_hz),
            }
        )
    metrics = summary.merge(pd.DataFrame(spectral_rows), on="variant", validate="one_to_one")
    family_by_name = {variant.name: variant.family for variant in variants}
    metrics.insert(0, "family", metrics["variant"].map(family_by_name))
    metrics.insert(0, "fold", int(fold.fold))
    return metrics


def make_fold_variants(
    traces: np.ndarray,
    fold: StrictFold,
    config: EvaluationConfig,
) -> tuple[
    list[FoldVariant],
    list[dict[str, object]],
    list[dict[str, object]],
    pd.DataFrame,
    pd.DataFrame,
]:
    variants = [
        FoldVariant(
            name="raw",
            family="raw",
            traces=traces,
            fit_idx=np.array([], dtype=int),
            metadata={"operation": "identity"},
        )
    ]
    audit_rows = [
        _audit_row(fold, "raw", "identity", np.array([], dtype=int), uses_labels=False)
    ]
    selection_rows: list[dict[str, object]] = []
    stability_frames: list[pd.DataFrame] = []
    stability_assignment_frames: list[pd.DataFrame] = []
    energy_match_targets: dict[str, float] = {}

    for method, bss_name, n_components, pca_variance_threshold in _bss_fit_specs(config):
        model = fit_bss_model(
            traces,
            fold.train_idx,
            method=method,
            n_components=n_components,
            pca_components=config.bss_pca_components,
            pca_variance_threshold=pca_variance_threshold,
            tolerance=config.bss_tolerance,
            max_iter=config.bss_max_iter,
            random_state=config.random_state,
            sobi_lags=config.sobi_lags,
            jade_max_cumulant_matrices=config.jade_max_cumulant_matrices,
        )
        selection = fit_cluster_selection(
            model,
            sample_rate_hz=config.sample_rate_hz,
            n_clusters=config.n_clusters,
            feature_start_bin=config.feature_start_bin,
            feature_transform=config.feature_transform,
            nperseg=config.welch_nperseg,
            noverlap=config.welch_noverlap,
            ranking_fmin_hz=config.ranking_fmin_hz,
            ranking_fmax_hz=config.ranking_fmax_hz,
            ranking_aggregate=config.ranking_aggregate,
            random_state=config.random_state,
        )
        if config.cluster_stability_seeds:
            pairwise, assignments = run_cluster_stability_sweep(
                model,
                sample_rate_hz=config.sample_rate_hz,
                seeds=config.cluster_stability_seeds,
                cluster_counts=config.cluster_stability_counts or (config.n_clusters,),
                feature_transforms=config.cluster_stability_feature_transforms
                or (config.feature_transform,),
                keep_cluster_count=config.cluster_stability_keep_count,
                feature_start_bin=config.feature_start_bin,
                nperseg=config.welch_nperseg,
                noverlap=config.welch_noverlap,
                ranking_fmin_hz=config.ranking_fmin_hz,
                ranking_fmax_hz=config.ranking_fmax_hz,
                ranking_aggregate=config.ranking_aggregate,
            )
            if not pairwise.empty:
                pairwise.insert(0, "method", method)
                pairwise.insert(0, "fold", fold.fold)
                stability_frames.append(pairwise)
            if not assignments.empty:
                assignments.insert(0, "method", method)
                assignments.insert(0, "fold", fold.fold)
                stability_assignment_frames.append(assignments)
        cluster_rank = {
            cluster: rank for rank, cluster in enumerate(selection.cluster_order, start=1)
        }
        quality_table = (
            ic_quality_table_for_bss_model(
                model,
                sample_rate_hz=config.sample_rate_hz,
                nperseg=config.welch_nperseg,
                noverlap=config.welch_noverlap,
            )
            if config.ic_quality_selection_strategies
            else pd.DataFrame()
        )
        quality_by_component = (
            quality_table.set_index("component")
            if not quality_table.empty
            else pd.DataFrame()
        )
        for component, cluster in enumerate(selection.labels):
            row = {
                "fold": fold.fold,
                "method": bss_name,
                "component": component,
                "cluster": int(cluster),
                "cluster_rank": int(cluster_rank[int(cluster)]),
                "component_energy": float(selection.component_energy[component]),
            }
            if component in quality_by_component.index:
                quality_row = quality_by_component.loc[component]
                row.update(
                    {
                        "artifact_score": float(quality_row["artifact_score"]),
                        "protect_score": float(quality_row["protect_score"]),
                        "recommendation": str(quality_row["recommendation"]),
                        "artifact_flags": str(quality_row["artifact_flags"]),
                        "protect_flags": str(quality_row["protect_flags"]),
                    }
                )
            selection_rows.append(row)
        decomposition_audit = _audit_row(
            fold, bss_name, "bss_decomposition", model.train_idx, False
        )
        decomposition_audit["fit_warning_count"] = len(model.fit_warnings)
        decomposition_audit["fit_warnings"] = " | ".join(model.fit_warnings)
        decomposition_audit["converged"] = not any(
            "did not converge" in message.lower() for message in model.fit_warnings
        )
        decomposition_audit["n_components"] = model.n_components
        decomposition_audit["pca_components"] = model.pca_components
        decomposition_audit["pca_variance_threshold"] = model.pca_variance_threshold
        decomposition_audit["pca_explained_variance_ratio"] = (
            model.pca_explained_variance_ratio
        )
        decomposition_audit["component_selection_mode"] = model.component_selection_mode
        audit_rows.extend(
            [
                decomposition_audit,
                _audit_row(fold, bss_name, "component_psd", model.train_idx, False),
                _audit_row(fold, bss_name, "spectral_feature_clustering", model.train_idx, False),
                _audit_row(fold, bss_name, "component_selection", model.train_idx, False),
            ]
        )
        if config.ic_quality_selection_strategies:
            audit_rows.append(
                _audit_row(fold, bss_name, "ic_quality_scoring", model.train_idx, False)
            )
        for strategy in config.selection_strategies:
            cluster_counts: tuple[int | None, ...] = (
                (None,) if strategy == "all" else tuple(config.keep_cluster_counts)
            )
            seeds = (
                config.selection_random_seeds
                if strategy in {"random", "energy_matched_random"}
                else (config.random_state,)
            )
            for keep_count in cluster_counts:
                for seed in seeds:
                    keep = select_components(
                        selection,
                        strategy=strategy,
                        keep_cluster_count=keep_count,
                        random_state=seed,
                    )
                    suffix = "all" if keep_count is None else f"k{keep_count}"
                    if strategy in {"random", "energy_matched_random"}:
                        suffix = f"{suffix}/seed{seed}"
                    name = f"{bss_name}/{strategy}/{suffix}"
                    variants.append(
                        FoldVariant(
                            name=name,
                            family="bss",
                            traces=model.reconstruct(traces, keep_components=keep),
                            fit_idx=model.train_idx,
                            metadata={
                                "method": method,
                                "n_components": model.n_components,
                                "pca_components": model.pca_components,
                                "pca_variance_threshold": model.pca_variance_threshold,
                                "pca_explained_variance_ratio": model.pca_explained_variance_ratio,
                        "component_selection_mode": model.component_selection_mode,
                        "strategy": strategy,
                        "keep_cluster_count": keep_count,
                        "keep_components": keep.tolist(),
                        "fit_indices": "training_frames_only",
                        "random_state": seed,
                    },
                )
            )
                    audit_rows.append(
                        _audit_row(fold, name, "reconstruction", model.train_idx, False)
                    )
                    if config.include_energy_matched_pca:
                        train_reconstruction = model.reconstruct(
                            traces[model.train_idx], keep_components=keep
                        )
                        numerator = float(
                            np.sum((train_reconstruction - model.mean) ** 2)
                        )
                        denominator = float(np.sum((traces[model.train_idx] - model.mean) ** 2))
                        energy_match_targets[name] = _safe_ratio(numerator, denominator)

        for strategy in config.ic_quality_selection_strategies:
            keep = select_components_by_ic_quality(quality_table, strategy=strategy)
            name = f"{bss_name}/{strategy}"
            variants.append(
                FoldVariant(
                    name=name,
                    family="bss",
                    traces=model.reconstruct(traces, keep_components=keep),
                    fit_idx=model.train_idx,
                    metadata={
                        "method": method,
                        "n_components": model.n_components,
                        "pca_components": model.pca_components,
                        "pca_variance_threshold": model.pca_variance_threshold,
                        "pca_explained_variance_ratio": model.pca_explained_variance_ratio,
                        "component_selection_mode": model.component_selection_mode,
                        "strategy": strategy,
                        "keep_components": keep.tolist(),
                        "selection_source": "ic_quality",
                        "fit_indices": "training_frames_only",
                    },
                )
            )
            audit_rows.append(_audit_row(fold, name, "reconstruction", model.train_idx, False))

    for method in config.benchmark_latent_methods:
        for rank_value in config.benchmark_latent_ranks:
            latent_model = fit_latent_benchmark_model(
                traces,
                fold.train_idx,
                method=method,
                rank=int(rank_value),
                random_state=config.random_state,
            )
            name = f"{latent_model.name}/rank{latent_model.rank}"
            metadata = {
                "method": latent_model.name,
                "rank": latent_model.rank,
                "random_state": config.random_state,
                "fit_indices": "training_frames_only",
            }
            if latent_model.train_shift is not None:
                metadata["train_nonnegative_shift_min"] = float(latent_model.train_shift.min())
                metadata["train_nonnegative_shift_max"] = float(latent_model.train_shift.max())
            variants.append(
                FoldVariant(
                    name=name,
                    family="latent_benchmark",
                    traces=latent_model.reconstruct(traces),
                    fit_idx=latent_model.train_idx,
                    metadata=metadata,
                )
            )
            audit_rows.append(
                _audit_row(fold, name, "latent_fit_reconstruct", latent_model.train_idx, False)
            )

    baseline_ranks = config.baseline_ranks or (min(traces.shape),)
    for rank_value in baseline_ranks:
        rank = min(int(rank_value), traces.shape[1], fold.train_idx.size)
        pca_model = fit_subspace_model(
            traces,
            fold.train_idx,
            rank=rank,
            kind="pca",
            random_state=config.random_state,
        )
        name = f"pca/rank{rank}"
        variants.append(
            FoldVariant(name, "baseline", pca_model.reconstruct(traces), pca_model.train_idx)
        )
        audit_rows.append(_audit_row(fold, name, "pca_fit_reconstruct", fold.train_idx, False))

        for seed in config.selection_random_seeds:
            random_model = fit_subspace_model(
                traces,
                fold.train_idx,
                rank=rank,
                kind="random_subspace",
                random_state=seed,
            )
            name = f"random_subspace/rank{rank}/seed{seed}"
            variants.append(
                FoldVariant(
                    name,
                    "baseline",
                    random_model.reconstruct(traces),
                    random_model.train_idx,
                )
            )
            audit_rows.append(
                _audit_row(fold, name, "random_subspace_reconstruct", fold.train_idx, False)
            )

    finite_energy_targets = {
        name: fraction
        for name, fraction in energy_match_targets.items()
        if np.isfinite(fraction)
    }
    energy_models = fit_pca_energy_models(
        traces,
        fold.train_idx,
        finite_energy_targets,
        random_state=config.random_state,
    ) if finite_energy_targets else {}
    energy_by_rank: dict[int, dict[str, object]] = {}
    for source_name, (pca_model, matched_fraction) in energy_models.items():
        rank = pca_model.basis.shape[1]
        entry = energy_by_rank.setdefault(
            rank,
            {
                "model": pca_model,
                "matched_fraction": matched_fraction,
                "source_variants": [],
                "target_fractions": [],
            },
        )
        entry["source_variants"].append(source_name)
        entry["target_fractions"].append(finite_energy_targets[source_name])
    for rank, entry in sorted(energy_by_rank.items()):
        pca_model = entry["model"]
        name = f"pca_energy_match/rank{rank}"
        variants.append(
            FoldVariant(
                name,
                "baseline",
                pca_model.reconstruct(traces),
                pca_model.train_idx,
                metadata={
                    "source_variants": entry["source_variants"],
                    "target_energy_fractions": entry["target_fractions"],
                    "matched_energy_fraction": entry["matched_fraction"],
                },
            )
        )
        audit_rows.append(
            _audit_row(fold, name, "pca_energy_match_fit_reconstruct", fold.train_idx, False)
        )

    for cutoff in config.lowpass_cutoffs_hz:
        name = f"causal_lowpass/{float(cutoff):g}hz"
        filtered = causal_lowpass_for_fold(
            traces,
            fold,
            sample_rate_hz=config.sample_rate_hz,
            cutoff_hz=float(cutoff),
        )
        variants.append(
            FoldVariant(
                name=name,
                family="baseline",
                traces=filtered,
                fit_idx=np.array([], dtype=int),
                metadata={"cutoff_hz": float(cutoff)},
            )
        )
        audit_rows.append(
            _audit_row(fold, name, "fixed_causal_filter", np.array([], dtype=int), False)
        )
    return (
        variants,
        audit_rows,
        selection_rows,
        pd.concat(stability_frames, ignore_index=True)
        if stability_frames
        else pd.DataFrame(),
        pd.concat(stability_assignment_frames, ignore_index=True)
        if stability_assignment_frames
        else pd.DataFrame(),
    )


def _bss_fit_specs(
    config: EvaluationConfig,
) -> tuple[tuple[str, str, int | None, float | None], ...]:
    specs: list[tuple[str, str, int | None, float | None]] = []
    seen: set[tuple[str, int | None, float | None]] = set()
    for method in config.bss_methods:
        method = method.lower()
        candidates: list[tuple[str, int | None, float | None]] = [
            (method, config.n_components, config.bss_pca_variance_threshold)
        ]
        candidates.extend(
            (
                f"{method}/components{int(count)}",
                int(count),
                config.bss_pca_variance_threshold,
            )
            for count in config.bss_component_counts
        )
        candidates.extend(
            (
                f"{method}/pca_var{float(threshold):g}",
                config.n_components,
                float(threshold),
            )
            for threshold in config.bss_pca_variance_thresholds
        )
        for name, n_components, threshold in candidates:
            key = (method, n_components, threshold)
            if key in seen:
                continue
            seen.add(key)
            specs.append((method, name, n_components, threshold))
    return tuple(specs)


def causal_lowpass_for_fold(
    traces: np.ndarray,
    fold: StrictFold,
    *,
    sample_rate_hz: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    traces = _validate_traces(traces)
    nyquist = float(sample_rate_hz) / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyquist:
        raise ValueError(f"cutoff_hz must be between 0 and Nyquist ({nyquist:g}).")
    sos = butter(order, cutoff_hz, btype="lowpass", fs=sample_rate_hz, output="sos")
    filtered = np.empty_like(traces)
    for indices in (fold.train_idx, fold.excluded_idx):
        for segment in contiguous_index_segments(indices):
            filtered[segment] = sosfilt(sos, traces[segment], axis=0)
    return filtered


def _evaluate_behavior_variants(
    variants: Sequence[FoldVariant],
    targets: BehaviorTargets,
    fold: StrictFold,
    config: EvaluationConfig,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    raw = next(variant for variant in variants if variant.name == "raw")

    for target_variant, bout_quantile, smooth_window, vigor in _behavior_target_variants(
        targets, config
    ):
        fold_threshold = float(np.quantile(vigor[fold.train_idx], bout_quantile))
        fold_targets = {
            "tail_vigor": ("regression", vigor),
            "bout_state": ("classification", (vigor >= fold_threshold).astype(int)),
        }
        designs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for variant in variants:
            for target_name, (_task, target) in fold_targets.items():
                designs[(variant.name, target_name)] = make_lagged_design(
                    variant.traces,
                    target,
                    lags=config.decoder_lags,
                    target_shift=config.decoder_target_shift,
                )

        metadata = {
            "target_variant": target_variant,
            "bout_quantile": float(bout_quantile),
            "smooth_window": int(smooth_window),
        }
        for variant in variants:
            for target_name, (task, _target) in fold_targets.items():
                design, y, times = designs[(variant.name, target_name)]
                train_mask, test_mask = _decoding_masks(times, fold, config)
                metric_row, prediction = _evaluate_decoding_pair(
                    design,
                    design,
                    y,
                    times,
                    train_mask,
                    test_mask,
                    task=task,
                    target_name=target_name,
                    comparison="within",
                    train_version=variant.name,
                    test_version=variant.name,
                    fold=fold.fold,
                    ridge_alpha=config.decoder_ridge_alpha,
                )
                _annotate_behavior_outputs(metric_row, prediction, metadata)
                metric_row["bout_threshold"] = (
                    fold_threshold if task == "classification" else np.nan
                )
                rows.append(metric_row)
                predictions.append(prediction)

                if variant.name != raw.name:
                    raw_design, raw_y, raw_times = designs[(raw.name, target_name)]
                    if not np.array_equal(times, raw_times) or not np.array_equal(y, raw_y):
                        raise ValueError("Raw and variant target alignment differs.")
                    transfer_row, transfer_prediction = _evaluate_decoding_pair(
                        raw_design,
                        design,
                        y,
                        times,
                        train_mask,
                        test_mask,
                        task=task,
                        target_name=target_name,
                        comparison="transfer_raw_to_clean",
                        train_version=raw.name,
                        test_version=variant.name,
                        fold=fold.fold,
                        ridge_alpha=config.decoder_ridge_alpha,
                    )
                    _annotate_behavior_outputs(transfer_row, transfer_prediction, metadata)
                    transfer_row["bout_threshold"] = (
                        fold_threshold if task == "classification" else np.nan
                    )
                    rows.append(transfer_row)
                    predictions.append(transfer_prediction)

        for target_name, (task, _target) in fold_targets.items():
            design, y, times = designs[(raw.name, target_name)]
            train_mask, test_mask = _decoding_masks(times, fold, config)
            for null_strategy in config.null_strategies:
                for block_size in config.null_block_sizes or (config.null_block_size,):
                    for seed in config.null_seeds:
                        rng = np.random.default_rng(seed)
                        null_train = _null_train_values(
                            y[train_mask],
                            times[train_mask],
                            strategy=null_strategy,
                            block_size=int(block_size),
                            min_shift=config.null_circular_min_shift,
                            rng=rng,
                        )
                        row, prediction = _evaluate_decoding_pair(
                            design,
                            design,
                            y,
                            times,
                            train_mask,
                            test_mask,
                            task=task,
                            target_name=target_name,
                            comparison="null_within",
                            train_version=raw.name,
                            test_version=raw.name,
                            fold=fold.fold,
                            ridge_alpha=config.decoder_ridge_alpha,
                            train_y_override=null_train,
                        )
                        _annotate_behavior_outputs(row, prediction, metadata)
                        row["null_seed"] = int(seed)
                        row["null_block_size"] = int(block_size)
                        row["null_strategy"] = null_strategy
                        row["bout_threshold"] = (
                            fold_threshold if task == "classification" else np.nan
                        )
                        prediction["null_seed"] = int(seed)
                        prediction["null_block_size"] = int(block_size)
                        prediction["null_strategy"] = null_strategy
                        rows.append(row)
                        predictions.append(prediction)
    return rows, predictions


def _behavior_target_variants(
    targets: BehaviorTargets,
    config: EvaluationConfig,
) -> tuple[tuple[str, float, int, np.ndarray], ...]:
    """Return the primary target plus configured sensitivity targets."""
    quantiles = (config.bout_quantile, *config.target_bout_quantiles)
    windows = (1, *config.target_smooth_windows)
    variants: list[tuple[str, float, int, np.ndarray]] = []
    seen: set[tuple[float, int]] = set()
    for quantile in quantiles:
        if not 0.0 <= float(quantile) <= 1.0:
            raise ValueError("target bout quantiles must be between 0 and 1.")
        for window in windows:
            window = int(window)
            if window < 1:
                raise ValueError("target smooth windows must be positive.")
            key = (float(quantile), window)
            if key in seen:
                continue
            seen.add(key)
            name = "primary" if key == (float(config.bout_quantile), 1) else (
                f"q{float(quantile):.3g}_smooth{window}"
            )
            vigor = targets.vigor if window == 1 else moving_average(targets.vigor, window)
            variants.append((name, float(quantile), window, vigor))
    return tuple(variants)


def _annotate_behavior_outputs(
    row: dict[str, object],
    prediction: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    for column, value in metadata.items():
        row[column] = value
        prediction[column] = value


def _null_train_values(
    y_train: np.ndarray,
    times_train: np.ndarray,
    *,
    strategy: str,
    block_size: int,
    min_shift: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if strategy == "block_shuffle":
        return _block_shuffle_with_time_gaps(
            y_train,
            times_train,
            block_size=block_size,
            rng=rng,
        )
    if strategy == "circular_shift":
        return _circular_shift_train_values(y_train, min_shift=min_shift, rng=rng)
    raise ValueError(f"Unknown null strategy: {strategy}")


def _circular_shift_train_values(
    y_train: np.ndarray,
    *,
    min_shift: int,
    rng: np.random.Generator,
) -> np.ndarray:
    y_train = np.asarray(y_train)
    if y_train.size < 2:
        return y_train.copy()
    min_shift = max(1, min(int(min_shift), y_train.size - 1))
    if min_shift > y_train.size // 2:
        shift = min_shift
    else:
        candidates = np.r_[min_shift : y_train.size - min_shift + 1]
        shift = int(rng.choice(candidates))
    return np.roll(y_train, shift)


def _decoding_masks(
    times: np.ndarray,
    fold: StrictFold,
    config: EvaluationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    lags = np.asarray(config.decoder_lags, dtype=int)
    feature_times = times[:, np.newaxis] - lags[np.newaxis, :]
    target_times = times + int(config.decoder_target_shift)
    train_mask = np.isin(target_times, fold.train_idx) & np.all(
        np.isin(feature_times, fold.train_idx), axis=1
    )
    test_mask = np.isin(target_times, fold.test_idx) & ~np.any(
        np.isin(feature_times, fold.train_idx), axis=1
    )
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("No strict decoder samples remain for this fold.")
    return train_mask, test_mask


def _evaluate_decoding_pair(
    train_features: np.ndarray,
    test_features: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    task: str,
    target_name: str,
    comparison: str,
    train_version: str,
    test_version: str,
    fold: int,
    ridge_alpha: float,
    train_y_override: np.ndarray | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    x_train, x_test = standardize_train_test(
        train_features[train_mask], test_features[test_mask]
    )
    y_train = y[train_mask] if train_y_override is None else train_y_override
    y_test = y[test_mask]
    if task == "regression":
        model = Ridge(alpha=ridge_alpha)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        score = None
        metrics = regression_metrics(y_test, pred)
    elif task == "classification":
        if np.unique(y_train).size < 2:
            pred = np.full(y_test.shape, int(y_train[0]))
            score = None
        else:
            model = LogisticRegression(
                class_weight="balanced",
                solver="liblinear",
                max_iter=1000,
                random_state=0,
            )
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            score = model.predict_proba(x_test)[:, 1]
        metrics = classification_metrics(y_test, pred, score)
    else:
        raise ValueError(f"Unknown task: {task}")
    row = {
        "target": target_name,
        "task": task,
        "comparison": comparison,
        "train_version": train_version,
        "test_version": test_version,
        "fold": int(fold),
        "n_train": int(np.count_nonzero(train_mask)),
        "n_test": int(np.count_nonzero(test_mask)),
    }
    row.update(metrics)
    prediction = pd.DataFrame(
        {
            "target": target_name,
            "task": task,
            "comparison": comparison,
            "train_version": train_version,
            "test_version": test_version,
            "fold": int(fold),
            "time_index": times[test_mask],
            "y_true": y_test,
            "y_pred": pred,
        }
    )
    if score is not None:
        prediction["y_score"] = score
    return row, prediction


def _evaluate_causal_variants(
    variants: Sequence[FoldVariant],
    targets: BehaviorTargets,
    fold: StrictFold,
    config: EvaluationConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    embeddings: list[dict[str, object]] = []
    causal = config.causal
    fold_threshold = float(np.quantile(targets.vigor[fold.train_idx], config.bout_quantile))
    for variant in variants:
        for shift in causal.target_shifts:
            x0, x1, target_map, times = make_paired_windows(
                variant.traces, targets, causal, shift
            )
            train_mask, test_mask = _causal_masks(times, shift, fold, causal.window)
            x0_flat = x0.reshape(x0.shape[0], -1)
            x1_flat = x1.reshape(x1.shape[0], -1)
            scaler = _fit_scaler(x0_flat[train_mask])
            x0_train = _apply_scaler(x0_flat[train_mask], scaler)
            x0_test = _apply_scaler(x0_flat[test_mask], scaler)
            x1_train = _apply_scaler(x1_flat[train_mask], scaler)
            x1_test = _apply_scaler(x1_flat[test_mask], scaler)
            latent_dim = min(causal.latent_dim, x0_train.shape[0], x0_train.shape[1])
            pca = PCA(n_components=latent_dim, random_state=causal.random_state)
            z0_train = pca.fit_transform(x0_train)
            z0_test = pca.transform(x0_test)
            z1_train = pca.transform(x1_train)
            z1_test = pca.transform(x1_test)
            persistence_mse = float(mean_squared_error(z1_test, z0_test))
            mean_prediction = np.broadcast_to(z1_train.mean(axis=0), z1_test.shape)
            mean_state_mse = float(mean_squared_error(z1_test, mean_prediction))
            latent_variance = float(np.mean(np.var(z1_test, axis=0)))

            behavior_metrics: dict[str, float] = {}
            for target_name in ("angle", "vigor"):
                y_train = target_map[target_name][train_mask]
                y_test = target_map[target_name][test_mask]
                model = Ridge(alpha=causal.ridge_alpha)
                model.fit(z0_train, y_train)
                pred = model.predict(z0_test)
                behavior_metrics[f"{target_name}_r2"] = _safe_r2(y_test, pred)
                behavior_metrics[f"{target_name}_pearson"] = _safe_pearson(y_test, pred)

            bout = (targets.vigor >= fold_threshold).astype(int)
            bout_aligned = bout[times + shift]
            y_bout_train = bout_aligned[train_mask]
            y_bout_test = bout_aligned[test_mask]
            if np.unique(y_bout_train).size < 2:
                bout_pred = np.full(y_bout_test.shape, int(y_bout_train[0]))
                bout_score = None
            else:
                bout_model = LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=causal.random_state,
                )
                bout_model.fit(z0_train, y_bout_train)
                bout_pred = bout_model.predict(z0_test)
                bout_score = bout_model.predict_proba(z0_test)[:, 1]
            behavior_metrics["bout_balanced_accuracy"] = (
                float(balanced_accuracy_score(y_bout_test, bout_pred))
                if np.unique(y_bout_test).size == 2
                else np.nan
            )
            behavior_metrics["bout_roc_auc"] = (
                float(roc_auc_score(y_bout_test, bout_score))
                if bout_score is not None and np.unique(y_bout_test).size == 2
                else np.nan
            )
            behavior_metrics["bout_threshold"] = fold_threshold

            transition_models = config.causal_transition_models or (
                causal.transition_model,
            )
            for transition_name in transition_models:
                transition_config = replace(causal, transition_model=transition_name)
                transition = _make_transition(transition_config)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    transition.fit(z0_train, z1_train - z0_train)
                z1_pred = z0_test + transition.predict(z0_test)
                dynamic_mse = float(mean_squared_error(z1_test, z1_pred))
                fit_warnings = tuple(str(item.message) for item in caught)
                row: dict[str, object] = {
                    "variant": variant.name,
                    "fold": fold.fold,
                    "target_shift": int(shift),
                    "transition_model": transition_name,
                    "n_train": int(np.count_nonzero(train_mask)),
                    "n_test": int(np.count_nonzero(test_mask)),
                    "dynamic_mse": dynamic_mse,
                    "held_out_latent_variance": latent_variance,
                    "dynamic_mse_normalized": _safe_ratio(dynamic_mse, latent_variance),
                    "persistence_mse": persistence_mse,
                    "mean_state_mse": mean_state_mse,
                    "dynamic_improvement_vs_persistence": 1.0
                    - _safe_ratio(dynamic_mse, persistence_mse),
                    "fit_warning_count": len(fit_warnings),
                    "fit_warnings": " | ".join(fit_warnings),
                    "converged": not any(
                        "did not converge" in message.lower()
                        or "maximum iterations" in message.lower()
                        for message in fit_warnings
                    ),
                    **behavior_metrics,
                }
                rows.append(row)

            test_sample_indices = np.flatnonzero(test_mask)
            for local_idx, sample_idx in enumerate(test_sample_indices):
                embedding = {
                    "variant": variant.name,
                    "fold": fold.fold,
                    "target_shift": int(shift),
                    "time_index": int(times[sample_idx]),
                    "angle": float(target_map["angle"][sample_idx]),
                    "vigor": float(target_map["vigor"][sample_idx]),
                    "bout_state": int(bout_aligned[sample_idx]),
                }
                for dim in range(z0_test.shape[1]):
                    embedding[f"z{dim + 1}"] = float(z0_test[local_idx, dim])
                embeddings.append(embedding)
    return rows, embeddings


def _evaluate_causal_sufficiency_variants(
    variants: Sequence[FoldVariant],
    targets: BehaviorTargets,
    fold: StrictFold,
    config: EvaluationConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    causal = config.causal
    target_names = config.causal_sufficiency_targets or ()
    if not target_names:
        return pd.DataFrame()
    for variant in variants:
        for shift in causal.target_shifts:
            x0, _x1, target_map, times = make_paired_windows(
                variant.traces, targets, causal, shift
            )
            train_mask, test_mask = _causal_masks(times, shift, fold, causal.window)
            x0_flat = x0.reshape(x0.shape[0], -1)
            x0_train, x0_test = standardize_train_test(
                x0_flat[train_mask], x0_flat[test_mask]
            )
            latent_dim = min(causal.latent_dim, x0_train.shape[0], x0_train.shape[1])
            pca = PCA(n_components=latent_dim, random_state=causal.random_state)
            z_train = pca.fit_transform(x0_train)
            z_test = pca.transform(x0_test)
            augmented_train = np.hstack([z_train, x0_train])
            augmented_test = np.hstack([z_test, x0_test])
            for target_name in target_names:
                if target_name not in target_map:
                    raise ValueError(f"Unknown causal sufficiency target: {target_name}")
                y_train = target_map[target_name][train_mask]
                y_test = target_map[target_name][test_mask]
                base = Ridge(alpha=causal.ridge_alpha)
                base.fit(z_train, y_train)
                base_pred = base.predict(z_test)
                augmented = Ridge(alpha=causal.ridge_alpha)
                augmented.fit(augmented_train, y_train)
                augmented_pred = augmented.predict(augmented_test)
                base_rmse = float(np.sqrt(mean_squared_error(y_test, base_pred)))
                augmented_rmse = float(np.sqrt(mean_squared_error(y_test, augmented_pred)))
                rows.append(
                    {
                        "variant": variant.name,
                        "fold": int(fold.fold),
                        "target_shift": int(shift),
                        "target": target_name,
                        "n_train": int(np.count_nonzero(train_mask)),
                        "n_test": int(np.count_nonzero(test_mask)),
                        "base_rmse": base_rmse,
                        "augmented_rmse": augmented_rmse,
                        "rmse_delta_aug_minus_base": augmented_rmse - base_rmse,
                    }
                )
    return pd.DataFrame(rows)


def _empty_artifact_probe_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "fold",
            "probe",
            "artifact_center",
            "local_center",
            "half_width",
            "variant",
            "rmse",
        ]
    )


def _evaluate_artifact_probe_variants(
    variants: Sequence[FoldVariant],
    fold: StrictFold,
    config: EvaluationConfig,
) -> pd.DataFrame:
    if not config.artifact_probe_centers:
        return _empty_artifact_probe_metrics()
    half_width = int(config.artifact_probe_half_width)
    if half_width < 1:
        raise ValueError("artifact_probe_half_width must be at least 1.")
    center_lookup = {int(frame): position for position, frame in enumerate(fold.test_idx)}
    center_pairs = [
        (int(center), int(center_lookup[int(center)]))
        for center in config.artifact_probe_centers
        if int(center) in center_lookup
    ]
    center_pairs = [
        (global_center, local_center)
        for global_center, local_center in center_pairs
        if local_center - half_width >= 1
        and local_center + half_width < fold.test_idx.size
    ]
    if not center_pairs:
        return _empty_artifact_probe_metrics()
    raw = next(variant for variant in variants if variant.name == "raw")
    raw_test = raw.traces[fold.test_idx]
    candidate_traces = {
        variant.name: variant.traces[fold.test_idx]
        for variant in variants
        if variant.name != raw.name
    }
    rows = []
    for global_center, local_center in center_pairs:
        table = pseudo_artifact_reconstruction_test(
            raw_test,
            candidate_traces,
            centers=(local_center,),
            half_width=half_width,
        )
        if table.empty:
            continue
        table.insert(0, "fold", int(fold.fold))
        table.insert(1, "probe", "pseudo_reconstruction")
        table.insert(2, "artifact_center", int(global_center))
        table = table.rename(columns={"center": "local_center"})
        rows.append(table)
    return (
        pd.concat(rows, ignore_index=True).reindex(columns=_empty_artifact_probe_metrics().columns)
        if rows
        else _empty_artifact_probe_metrics()
    )


def _causal_masks(
    times: np.ndarray,
    target_shift: int,
    fold: StrictFold,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(-(window - 2), 2, dtype=int)
    neural_times = times[:, np.newaxis] + offsets[np.newaxis, :]
    target_times = times + int(target_shift)
    train_mask = np.isin(target_times, fold.train_idx) & np.all(
        np.isin(neural_times, fold.train_idx), axis=1
    )
    test_mask = np.isin(target_times, fold.test_idx) & np.isin(times, fold.test_idx) & ~np.any(
        np.isin(neural_times, fold.train_idx), axis=1
    )
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("No strict causal-state samples remain for this fold.")
    return train_mask, test_mask


def _make_transition(config: CausalStateConfig):
    if config.transition_model == "linear":
        return Ridge(alpha=config.ridge_alpha)
    if config.transition_model == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(config.mlp_hidden_units,),
            activation="tanh",
            alpha=config.ridge_alpha,
            max_iter=config.mlp_max_iter,
            early_stopping=True,
            random_state=config.random_state,
        )
    raise ValueError("transition_model must be 'linear' or 'mlp'.")


def _audit_row(
    fold: StrictFold,
    variant: str,
    operation: str,
    fit_idx: np.ndarray,
    uses_labels: bool,
) -> dict[str, object]:
    fit_idx = np.asarray(fit_idx, dtype=int)
    overlap = np.intersect1d(fit_idx, fold.test_idx)
    if overlap.size:
        raise AssertionError(
            f"Leakage detected for {variant}/{operation}: {overlap.size} held-out frames in fit set."
        )
    return {
        "fold": fold.fold,
        "variant": variant,
        "operation": operation,
        "fit_scope": "none" if fit_idx.size == 0 else "training_frames_only",
        "n_fit_frames": int(fit_idx.size),
        "n_test_frames_seen_during_fit": int(overlap.size),
        "uses_behavior_labels": bool(uses_labels),
        "test_time_action": "fixed transform",
        "leakage_free": overlap.size == 0,
    }


def _downstream_audit_rows(
    fold: StrictFold,
    variants: Sequence[FoldVariant],
) -> list[dict[str, object]]:
    rows = [
        _audit_row(
            fold,
            "shared",
            "bout_threshold_estimation",
            fold.train_idx,
            uses_labels=True,
        )
    ]
    for variant in variants:
        for operation, uses_labels in (
            ("decoder_standardization", False),
            ("behavior_decoder_fit", True),
            ("causal_history_standardization", False),
            ("causal_pca_encoder_fit", False),
            ("causal_transition_fit", False),
            ("causal_behavior_head_fit", True),
        ):
            rows.append(
                _audit_row(
                    fold,
                    variant.name,
                    operation,
                    fold.train_idx,
                    uses_labels=uses_labels,
                )
            )
    return rows


def _fit_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return mean, np.where(std > 0, std, 1.0)


def _block_shuffle_with_time_gaps(
    values: np.ndarray,
    times: np.ndarray,
    *,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values)
    times = np.asarray(times, dtype=int)
    if values.shape[0] != times.shape[0]:
        raise ValueError("values and times must have the same length.")
    if block_size < 1:
        raise ValueError("block_size must be at least 1.")
    split_points = np.flatnonzero(np.diff(times) > 1) + 1
    value_segments = np.split(values, split_points)
    blocks = [
        segment[start : start + block_size]
        for segment in value_segments
        for start in range(0, segment.size, block_size)
    ]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[index] for index in order])


def _apply_scaler(
    values: np.ndarray,
    scaler: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    mean, std = scaler
    return (values - mean) / std


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denominator <= 0:
        return np.nan
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denominator)


def _validate_fit_indices(indices: np.ndarray, n_samples: int) -> None:
    if indices.size == 0:
        raise ValueError("At least one training index is required.")
    if indices.min() < 0 or indices.max() >= n_samples:
        raise ValueError("Training indices are out of range.")


def _validate_traces(traces: np.ndarray) -> np.ndarray:
    traces = np.asarray(traces, dtype=float)
    if traces.ndim != 2:
        raise ValueError(f"traces must be 2D (time x features), got {traces.shape}.")
    if not np.isfinite(traces).all():
        raise ValueError("traces must be finite before strict evaluation.")
    return traces


def _validate_targets(targets: BehaviorTargets, n_frames: int) -> None:
    for name in ("angle", "vigor", "bout_state"):
        values = np.asarray(getattr(targets, name))
        if values.shape[0] != n_frames:
            raise ValueError(f"targets.{name} has length {values.shape[0]}, expected {n_frames}.")

"""Trace variants evaluated for each simulated dataset (Section 5.1).

Ground truth must not enter ordinary component selection. Only the
``artifact_oracle`` and ``oracle_selection`` variants may use it; those are the
only functions that receive oracle arrays, which keeps the constraint
structural rather than conventional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import butter, lfilter
from sklearn.decomposition import PCA, FastICA

from ica_denoising.core.ica_utils import infomax_dec, reconstruct_bss
from ica_denoising.variant_id import build_variant_id, cluster_keep_selection

__all__ = ["VariantResult", "build_variants", "ORACLE_VARIANTS"]

ORACLE_VARIANTS = frozenset({"clean", "artifact_oracle", "oracle_selection"})


@dataclass(frozen=True)
class VariantResult:
    """A single reconstructed trace variant."""

    variant_id: str
    traces: np.ndarray  # (n_neurons, T)
    reconstruction_rank: int
    explained_variance_ratio: float
    uses_ground_truth: bool
    rank_mode: str = ""
    random_state: int | None = None
    sobi_lags: tuple[int, ...] = ()


def _impute(traces: np.ndarray) -> np.ndarray:
    """Replace non-finite entries with per-neuron finite means."""

    out = np.array(traces, dtype=float, copy=True)
    for i in range(out.shape[0]):
        row = out[i]
        finite = np.isfinite(row)
        if finite.all():
            continue
        fill = float(np.mean(row[finite])) if finite.any() else 0.0
        row[~finite] = fill
    return out


def _numerical_rank(matrix: np.ndarray) -> int:
    if matrix.size == 0:
        return 0
    return int(np.linalg.matrix_rank(matrix))


def _explained_variance(original: np.ndarray, recon: np.ndarray) -> float:
    total = float(np.var(original))
    if total == 0:
        return 1.0
    residual = float(np.var(original - recon))
    return float(max(0.0, 1.0 - residual / total))


def _mask_sources_by_variance(
    sources: np.ndarray, keep_top: Optional[int]
) -> np.ndarray:
    n_components = sources.shape[1]
    if keep_top is None or keep_top >= n_components:
        return sources
    variances = np.var(sources, axis=0)
    keep = np.argsort(variances)[::-1][:keep_top]
    mask = np.zeros(n_components, dtype=bool)
    mask[keep] = True
    return sources * mask[None, :]


def _ica_reconstruct(
    traces: np.ndarray,
    keep_top: Optional[int],
    rng_seed: int,
    fun: str,
    *,
    n_components: int | None = None,
) -> np.ndarray:
    """ICA reconstruction keeping the ``keep_top`` highest-variance components."""

    x = traces.T  # (T, n)
    n_components = x.shape[1] if n_components is None else int(n_components)
    n_components = max(1, min(n_components, x.shape[1], x.shape[0]))
    ica = FastICA(
        n_components=n_components,
        random_state=rng_seed,
        whiten="unit-variance",
        fun=fun,
        max_iter=500,
    )
    sources = ica.fit_transform(x)  # (T, n_components)
    sources = _mask_sources_by_variance(sources, keep_top)
    recon = ica.inverse_transform(sources)
    return recon.T


def _infomax_reconstruct(
    traces: np.ndarray,
    keep_top: Optional[int],
    rng_seed: int,
    *,
    n_components: int | None = None,
) -> np.ndarray:
    """Infomax reconstruction using the project natural-gradient implementation."""

    n_components = traces.shape[0] if n_components is None else int(n_components)
    n_components = max(1, min(n_components, traces.shape[0], traces.shape[1]))
    sources, _ic_ft, mixing, mean = infomax_dec(
        traces,
        n_comps=n_components,
        max_iter=500,
        random_state=rng_seed,
    )
    sources = _mask_sources_by_variance(sources, keep_top)
    return reconstruct_bss(sources, mixing, mean).T


def _pca_reconstruct(traces: np.ndarray, rank: int) -> np.ndarray:
    x = traces.T
    rank = max(1, min(rank, x.shape[1]))
    pca = PCA(n_components=rank)
    recon = pca.inverse_transform(pca.fit_transform(x))
    return recon.T


def _rank_restricted_input(traces: np.ndarray, rank: int) -> np.ndarray:
    rank = int(rank)
    if rank >= min(traces.shape):
        return traces
    return _pca_reconstruct(traces, rank)


def _random_subspace(traces: np.ndarray, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = traces.T
    n = x.shape[1]
    rank = max(1, min(rank, n))
    basis, _ = np.linalg.qr(rng.normal(size=(n, rank)))
    recon = x @ basis @ basis.T
    return recon.T


def _causal_lowpass(traces: np.ndarray, cutoff_hz: float, fs: float) -> np.ndarray:
    nyq = fs / 2.0
    wn = min(max(cutoff_hz / nyq, 1e-3), 0.99)
    b, a = butter(2, wn, btype="low")
    return lfilter(b, a, traces, axis=-1)


def build_variants(
    *,
    clean: np.ndarray,
    corrupted: np.ndarray,
    artifact: np.ndarray,
    methods: tuple[str, ...],
    keep_top: int,
    rank_target: int,
    random_state: int,
    sample_rate_hz: float,
    lowpass_cutoff_hz: float = 0.0,
    rank_targets: tuple[tuple[str, int], ...] | None = None,
    random_states: tuple[int, ...] | None = None,
    sobi_lag_sets: tuple[tuple[int, ...], ...] | None = None,
) -> list[VariantResult]:
    """Build the configured trace variants for one dataset.

    Args:
        clean: Clean fluorescence oracle (ground truth).
        corrupted: Corrupted raw traces (the observation).
        artifact: Injected artifact (ground truth).
        methods: BSS methods to run (subset of fastica/infomax/sobi/jade).
        keep_top: Number of components retained for cluster_keep selection.
        rank_target: Rank for rank-matched controls.
        random_state: Base seed for stochastic decompositions.
        sample_rate_hz: Acquisition rate.
        lowpass_cutoff_hz: Causal low-pass cutoff (0 -> sample_rate / 4).
        rank_targets: Optional ``(rank_mode, rank)`` grid. Defaults to
            ``(("rank", rank_target),)`` for backward compatibility.
        random_states: Optional restart grid. Defaults to ``(random_state,)``.
        sobi_lag_sets: Optional SOBI lag-set grid.
    """

    raw = _impute(corrupted)
    results: list[VariantResult] = []
    if rank_targets is None:
        rank_targets = (("rank", int(rank_target)),)
    if random_states is None:
        random_states = (int(random_state),)
    if sobi_lag_sets is None:
        sobi_lag_sets = ((1, 2, 3, 5),)

    def add(
        vid: str,
        traces: np.ndarray,
        oracle: bool,
        *,
        rank_mode: str = "",
        seed: int | None = None,
        sobi_lags: tuple[int, ...] = (),
    ) -> None:
        results.append(
            VariantResult(
                variant_id=vid,
                traces=traces,
                reconstruction_rank=_numerical_rank(traces),
                explained_variance_ratio=_explained_variance(clean, traces),
                uses_ground_truth=oracle,
                rank_mode=rank_mode,
                random_state=seed,
                sobi_lags=tuple(int(lag) for lag in sobi_lags),
            )
        )

    add("clean", clean.copy(), True)
    add("raw", raw, False)
    add("artifact_oracle", raw - _impute(artifact), True)

    for method in methods:
        method = method.lower()
        if method == "pca":
            continue
        for rank_mode, rank in rank_targets:
            rank = max(1, min(int(rank), raw.shape[0], raw.shape[1]))
            rank_token = _selection_with_rank_mode(
                cluster_keep_selection(keep_top), rank_mode
            )
            for seed in random_states:
                seed = int(seed)
                if method == "fastica":
                    recon = _ica_reconstruct(
                        raw,
                        keep_top,
                        seed,
                        "logcosh",
                        n_components=rank,
                    )
                    add(
                        build_variant_id(
                            method=method,
                            selection=rank_token,
                            rank=rank,
                            seed=seed,
                        ),
                        recon,
                        False,
                        rank_mode=rank_mode,
                        seed=seed,
                    )
                elif method == "infomax":
                    recon = _infomax_reconstruct(
                        raw,
                        keep_top,
                        seed,
                        n_components=rank,
                    )
                    add(
                        build_variant_id(
                            method=method,
                            selection=rank_token,
                            rank=rank,
                            seed=seed,
                        ),
                        recon,
                        False,
                        rank_mode=rank_mode,
                        seed=seed,
                    )
                elif method == "sobi":
                    for lags in sobi_lag_sets:
                        lags = tuple(int(lag) for lag in lags)
                        selection = _selection_with_rank_mode(
                            f"{cluster_keep_selection(keep_top)}_lags_{_lag_token(lags)}",
                            rank_mode,
                        )
                        add(
                            build_variant_id(
                                method="sobi",
                                selection=selection,
                                rank=rank,
                                seed=seed,
                            ),
                            _sobi(raw, keep_top, lags=lags, rank=rank),
                            False,
                            rank_mode=rank_mode,
                            seed=seed,
                            sobi_lags=lags,
                        )
                elif method == "jade":
                    # JADE falls back to a documented FastICA reconstruction.
                    recon = _ica_reconstruct(
                        raw, keep_top, seed, "cube", n_components=rank
                    )
                    add(
                        build_variant_id(
                            method="jade_fastica_fallback",
                            selection=rank_token,
                            rank=rank,
                            seed=seed,
                        ),
                        recon,
                        False,
                        rank_mode=rank_mode,
                        seed=seed,
                    )

    for rank_mode, rank in rank_targets:
        rank = max(1, min(int(rank), raw.shape[0], raw.shape[1]))
        add(
            build_variant_id(
                method="pca",
                selection=_selection_with_rank_mode("rank_matched", rank_mode),
                rank=rank,
                seed=0,
            ),
            _pca_reconstruct(raw, rank),
            False,
            rank_mode=rank_mode,
            seed=0,
        )
        for seed in random_states:
            seed = int(seed)
            add(
                build_variant_id(
                    method="random_subspace",
                    selection=_selection_with_rank_mode("rank_matched", rank_mode),
                    rank=rank,
                    seed=seed,
                ),
                _random_subspace(raw, rank, seed),
                False,
                rank_mode=rank_mode,
                seed=seed,
            )
    full_rank = min(raw.shape)
    add(
        build_variant_id(method="fastica", selection="all", rank=full_rank, seed=random_state),
        _ica_reconstruct(raw, None, random_state, "logcosh"),
        False,
        rank_mode="full",
        seed=random_state,
    )

    cutoff = lowpass_cutoff_hz or sample_rate_hz / 4.0
    add(
        build_variant_id(
            method="causal_lowpass",
            selection=f"cutoff_{cutoff:g}hz",
            rank=min(raw.shape),
            seed=0,
        ),
        _causal_lowpass(raw, cutoff, sample_rate_hz),
        False,
        seed=0,
    )

    add("oracle_selection", _oracle_select(raw, artifact, random_state), True)
    return results


def _oracle_select(traces: np.ndarray, artifact: np.ndarray, seed: int) -> np.ndarray:
    """Upper-bound: drop ICA components most correlated with the known artifact."""

    x = traces.T
    n = x.shape[1]
    ica = FastICA(n_components=n, random_state=seed, whiten="unit-variance", max_iter=500)
    sources = ica.fit_transform(x)
    art = _impute(artifact).T
    art_mean = art.mean(axis=1)
    keep = np.ones(n, dtype=bool)
    for c in range(n):
        corr = abs(np.corrcoef(sources[:, c], art_mean)[0, 1])
        if np.isfinite(corr) and corr > 0.3:
            keep[c] = False
    recon = ica.inverse_transform(sources * keep[None, :])
    return recon.T


def _selection_with_rank_mode(selection: str, rank_mode: str) -> str:
    token = str(rank_mode or "rank").replace(".", "p")
    return f"{selection}_{token}"


def _lag_token(lags: tuple[int, ...]) -> str:
    return "_".join(str(int(lag)) for lag in lags)


def _sobi(
    traces: np.ndarray,
    keep_top: int,
    *,
    lags: tuple[int, ...] = (1, 2, 3, 5),
    rank: int | None = None,
) -> np.ndarray:
    """Second-order blind identification via approximate joint diagonalization."""

    if rank is not None:
        traces = _rank_restricted_input(traces, int(rank))
    x = traces - traces.mean(axis=1, keepdims=True)
    n, t = x.shape
    # whiten
    cov = (x @ x.T) / t
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-9, None)
    whitening = np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    z = whitening @ x
    # one Jacobi sweep over lagged covariances (compact approximation)
    rotation = np.eye(n)
    for lag in sorted(set(int(lag) for lag in lags if int(lag) > 0)):
        if lag >= t:
            continue
        lagged = (z[:, lag:] @ z[:, :-lag].T) / (t - lag)
        lagged = 0.5 * (lagged + lagged.T)
        _, vecs = np.linalg.eigh(lagged)
        rotation = vecs.T @ rotation
    sources = rotation @ z
    variances = np.var(sources, axis=1)
    keep = np.argsort(variances)[::-1][: min(keep_top, n)]
    mask = np.zeros(n)
    mask[keep] = 1.0
    mixing = np.linalg.pinv(rotation @ whitening)
    recon = mixing @ (mask[:, None] * sources)
    return recon

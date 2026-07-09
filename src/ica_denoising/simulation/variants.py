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


def _ica_reconstruct(
    traces: np.ndarray, keep_top: Optional[int], rng_seed: int, fun: str
) -> np.ndarray:
    """ICA reconstruction keeping the ``keep_top`` highest-variance components."""

    x = traces.T  # (T, n)
    n_components = x.shape[1]
    ica = FastICA(
        n_components=n_components,
        random_state=rng_seed,
        whiten="unit-variance",
        fun=fun,
        max_iter=500,
    )
    sources = ica.fit_transform(x)  # (T, n_components)
    if keep_top is not None and keep_top < n_components:
        variances = np.var(sources, axis=0)
        keep = np.argsort(variances)[::-1][:keep_top]
        mask = np.zeros(n_components, dtype=bool)
        mask[keep] = True
        sources = sources * mask[None, :]
    recon = ica.inverse_transform(sources)
    return recon.T


def _pca_reconstruct(traces: np.ndarray, rank: int) -> np.ndarray:
    x = traces.T
    rank = max(1, min(rank, x.shape[1]))
    pca = PCA(n_components=rank)
    recon = pca.inverse_transform(pca.fit_transform(x))
    return recon.T


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
    """

    raw = _impute(corrupted)
    results: list[VariantResult] = []

    def add(vid: str, traces: np.ndarray, oracle: bool) -> None:
        results.append(
            VariantResult(
                variant_id=vid,
                traces=traces,
                reconstruction_rank=_numerical_rank(traces),
                explained_variance_ratio=_explained_variance(clean, traces),
                uses_ground_truth=oracle,
            )
        )

    add("clean", clean.copy(), True)
    add("raw", raw, False)
    add("artifact_oracle", raw - _impute(artifact), True)

    fun_map = {"fastica": "logcosh", "infomax": "exp"}
    for method in methods:
        if method in fun_map:
            recon = _ica_reconstruct(raw, keep_top, random_state, fun_map[method])
            add(f"{method}/cluster_keep_top_{keep_top:02d}", recon, False)
        elif method == "sobi":
            add(f"sobi/cluster_keep_top_{keep_top:02d}", _sobi(raw, keep_top), False)
        elif method == "jade":
            # JADE falls back to a documented FastICA reconstruction.
            recon = _ica_reconstruct(raw, keep_top, random_state, "cube")
            add(f"jade/cluster_keep_top_{keep_top:02d}", recon, False)

    add(f"pca/rank_{rank_target:02d}", _pca_reconstruct(raw, rank_target), False)
    add(
        f"random_subspace/rank_{rank_target:02d}",
        _random_subspace(raw, rank_target, random_state),
        False,
    )
    add("full_rank", _ica_reconstruct(raw, None, random_state, "logcosh"), False)

    cutoff = lowpass_cutoff_hz or sample_rate_hz / 4.0
    add("lowpass", _causal_lowpass(raw, cutoff, sample_rate_hz), False)

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


def _sobi(traces: np.ndarray, keep_top: int, n_lags: int = 4) -> np.ndarray:
    """Second-order blind identification via approximate joint diagonalization."""

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
    for lag in range(1, n_lags + 1):
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

#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
# from utils import *
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from scipy.signal import welch
import pandas as pd
from sklearn.cluster import KMeans
from visualization import (
    plot_clusters,
    plot_FT_spectrals,
    plott_ics,
    plottings_group_spectrals,
    plottings_logSpectral,
    plottings_spectrals,
)


def eig_dec(data,var_to_keep):
    cov = np.cov(data)
    eig_values,eig_vectors = np.linalg.eig(cov)
    var = 0
    i = 1
    while var<var_to_keep:
        var= np.sum(eig_values[:i])/np.sum(eig_values)
        i+=1
    plt.stem(np.arange(len(eig_values)),eig_values)
    plt.vlines(i,0,eig_values.max(),label='{} eig_vals = {}'.format(i,np.sum(eig_values[:i])/np.sum(eig_values)),color='r')
    plt.legend()
    plt.title('Eigen values')
    return i


def ica_dec(data,n_comps,t,max_):
    ica = FastICA(n_components=n_comps,tol=t,max_iter=max_,whiten='unit-variance')
    ic_comps = ica.fit_transform(data.T)
    A = ica.mixing_
    mean = ica.mean_
    IC_ft = np.zeros((n_comps,data.shape[1]))
    for i in range(n_comps):
        IC_ft[i,:] = np.abs(np.fft.fft(ic_comps[:,i]))
    return ic_comps,IC_ft,A,mean


def bss_dec(
    data,
    n_comps,
    t=0.0001,
    max_=500,
    method="fastica",
    random_state=0,
    **kwargs,
):
    """Fit a BSS method and return the same outputs as ``ica_dec``.

    Parameters
    ----------
    data : array, shape (neurons, frames)
        Extracted calcium traces in the orientation used by the existing
        notebooks.
    n_comps : int
        Number of latent components.
    method : {"fastica", "infomax", "sobi", "jade"}
        BSS algorithm to fit.

    Returns
    -------
    ic_comps : array, shape (frames, n_comps)
    IC_ft : array, shape (n_comps, frames)
    A : array, shape (neurons, n_comps)
        Mixing matrix compatible with ``np.dot(ic_comps, A.T) + mean``.
    mean : array, shape (neurons,)
    """
    method = method.lower()
    if method in {"fastica", "ica"}:
        return ica_dec(data, n_comps, t=t, max_=max_)
    if method == "infomax":
        return infomax_dec(
            data,
            n_comps=n_comps,
            tol=t,
            max_iter=max_,
            random_state=random_state,
            **kwargs,
        )
    if method == "sobi":
        return sobi_dec(
            data,
            n_comps=n_comps,
            tol=t,
            max_iter=max_,
            **kwargs,
        )
    if method == "jade":
        return jade_dec(
            data,
            n_comps=n_comps,
            tol=t,
            max_iter=max_,
            **kwargs,
        )
    raise ValueError(f"Unknown BSS method: {method}")


def sobi_dec(
    data,
    n_comps,
    lags=(1, 2, 3, 5),
    tol=0.0001,
    max_iter=500,
):
    """Second-order blind identification for extracted trace matrices."""
    X, mean, X_white, _whitening = _prepare_whitened_data(data, n_comps)
    covs = _lagged_covariances(X_white, lags=lags)
    rotation = _joint_diag_symmetric(covs, tol=tol, max_iter=max_iter)
    ic_comps = X_white @ rotation
    return _format_bss_output(X, mean, ic_comps, data.shape[1])


def jade_dec(
    data,
    n_comps,
    tol=0.0001,
    max_iter=500,
    max_cumulant_matrices=200,
):
    """Approximate JADE using fourth-order cumulant joint diagonalization.

    Full JADE builds O(k^2) cumulant matrices for k components. The
    ``max_cumulant_matrices`` cap keeps this usable for exploratory notebooks.
    """
    X, mean, X_white, _whitening = _prepare_whitened_data(data, n_comps)
    cumulants = _jade_cumulant_matrices(
        X_white,
        max_cumulant_matrices=max_cumulant_matrices,
    )
    rotation = _joint_diag_symmetric(cumulants, tol=tol, max_iter=max_iter)
    ic_comps = X_white @ rotation
    return _format_bss_output(X, mean, ic_comps, data.shape[1])


def infomax_dec(
    data,
    n_comps,
    tol=0.0001,
    max_iter=500,
    learning_rate=0.01,
    random_state=0,
):
    """Infomax-style natural-gradient ICA on PCA-whitened traces.

    The learned ``W`` below is an unmixing operator in PCA-whitened space.
    The public BSS API returns a neuron-space mixing matrix instead, so
    ``reconstruct_bss(ic_comps, A, mean)`` keeps the same contract as FastICA.
    """
    X, mean, X_white, whitening = _prepare_whitened_data(data, n_comps)
    rng = np.random.default_rng(random_state)
    W = np.eye(n_comps) + 0.01 * rng.standard_normal((n_comps, n_comps))
    W = _sym_decorrelate(W)
    previous = W.copy()
    Xw = X_white.T

    for iteration in range(max_iter):
        Y = W @ Xw
        logistic_score = 1.0 - 2.0 / (1.0 + np.exp(-np.clip(Y, -50, 50)))
        update = (np.eye(n_comps) + logistic_score @ Y.T / Xw.shape[1]) @ W
        step = learning_rate / np.sqrt(iteration + 1.0)
        W = W + step * update
        W = _sym_decorrelate(W)
        delta = np.max(np.abs(np.abs(np.diag(W @ previous.T)) - 1.0))
        if delta < tol:
            break
        previous = W.copy()

    ic_comps = (W @ Xw).T
    mixing = _mixing_from_unmixing(W @ whitening)
    return _format_bss_output(X, mean, ic_comps, data.shape[1], mixing=mixing)


def reconstruct_bss(ic_comps, A, mean, reject=None, keep=None):
    """Reconstruct traces after zeroing or keeping selected components.

    Returns traces in frames x neurons orientation, matching the existing
    notebook reconstruction convention.
    """
    comps = np.asarray(ic_comps, dtype=float).copy()
    if keep is not None and reject is not None:
        raise ValueError("Use either keep or reject, not both.")
    if keep is not None:
        mask = np.zeros(comps.shape[1], dtype=bool)
        mask[list(keep)] = True
        comps[:, ~mask] = 0.0
    if reject is not None:
        comps[:, list(reject)] = 0.0
    return np.dot(comps, np.asarray(A).T) + mean


def accepted_to_rejected(accepted_components, n_total_components):
    """Convert accepted component indices to rejected component indices.

    Parameters
    ----------
    accepted_components : iterable[int] | None
        Component indices to keep. ``None`` means keep all components.
    n_total_components : int
        Total number of available components.
    """
    if accepted_components is None:
        return []
    n_total_components = int(n_total_components)
    accepted = sorted({int(component) for component in accepted_components})
    invalid = [component for component in accepted if component < 0 or component >= n_total_components]
    if invalid:
        raise ValueError(f"Accepted component indices out of range: {invalid}")
    return [component for component in range(n_total_components) if component not in accepted]


def reject_components_from_cluster_selection(predictions, *, reject_clusters=(), keep_clusters=()):
    """Convert cluster label selections into component indices to reject."""
    predictions = np.asarray(predictions)
    reject_clusters = list(reject_clusters)
    keep_clusters = list(keep_clusters)
    if reject_clusters and keep_clusters:
        raise ValueError("Use reject_clusters or keep_clusters for a method, not both.")
    if keep_clusters:
        keep_mask = np.isin(predictions, keep_clusters)
        return np.flatnonzero(~keep_mask)
    reject_mask = np.isin(predictions, reject_clusters)
    return np.flatnonzero(reject_mask)


def _prepare_whitened_data(data, n_comps):
    X = np.asarray(data, dtype=float).T
    n_comps = int(n_comps)
    if n_comps < 1 or n_comps > min(X.shape):
        raise ValueError(f"n_comps must be between 1 and {min(X.shape)}, got {n_comps}.")
    mean = X.mean(axis=0)
    X_centered = X - mean
    pca = PCA(n_components=n_comps, whiten=True, svd_solver="full")
    X_white = pca.fit_transform(X_centered)
    whitening = pca.components_ / np.sqrt(pca.explained_variance_)[:, np.newaxis]
    return X_centered, mean, X_white, whitening


def _format_bss_output(X_centered, mean, ic_comps, n_frames, mixing=None):
    """Convert source activations into the shared notebook output contract.

    Custom methods are fit in whitened component space. If the caller does not
    provide a neuron-space mixing matrix, recover that projection explicitly
    from the fitted sources.
    """
    if mixing is None:
        A = _least_squares_mixing(ic_comps, X_centered)
    else:
        A = np.asarray(mixing, dtype=float)
    IC_ft = np.zeros((ic_comps.shape[1], n_frames))
    for i in range(ic_comps.shape[1]):
        IC_ft[i, :] = np.abs(np.fft.fft(ic_comps[:, i], n=n_frames))
    return ic_comps, IC_ft, A, mean


def _mixing_from_unmixing(unmixing):
    """Return the neuron-space mixing matrix for a linear unmixing operator."""
    return np.linalg.pinv(np.asarray(unmixing, dtype=float))


def _least_squares_mixing(ic_comps, X_centered):
    mixing_t, *_ = np.linalg.lstsq(ic_comps, X_centered, rcond=None)
    return mixing_t.T


def _lagged_covariances(X_white, lags):
    covs = []
    n_samples = X_white.shape[0]
    for lag in lags:
        lag = int(lag)
        if lag <= 0 or lag >= n_samples:
            continue
        cov = X_white[lag:].T @ X_white[:-lag] / (n_samples - lag)
        covs.append((cov + cov.T) / 2.0)
    if not covs:
        raise ValueError("No valid SOBI lags for this recording length.")
    return np.stack(covs, axis=0)


def _jade_cumulant_matrices(X_white, max_cumulant_matrices):
    n_samples, n_comps = X_white.shape
    eye = np.eye(n_comps)
    matrices = []
    for i in range(n_comps):
        for j in range(i, n_comps):
            weights = X_white[:, i] * X_white[:, j]
            mat = (X_white.T * weights) @ X_white / n_samples
            if i == j:
                mat = mat - eye
            e_i = np.zeros(n_comps)
            e_j = np.zeros(n_comps)
            e_i[i] = 1.0
            e_j[j] = 1.0
            mat = mat - np.outer(e_i, e_j) - np.outer(e_j, e_i)
            matrices.append((mat + mat.T) / 2.0)
            if len(matrices) >= max_cumulant_matrices:
                return np.stack(matrices, axis=0)
    return np.stack(matrices, axis=0)


def _joint_diag_symmetric(mats, tol=0.0001, max_iter=500):
    mats = np.asarray(mats, dtype=float).copy()
    if mats.ndim != 3 or mats.shape[1] != mats.shape[2]:
        raise ValueError("mats must have shape (n_matrices, n_components, n_components).")
    n_comps = mats.shape[1]
    rotation = np.eye(n_comps)
    for _ in range(max_iter):
        max_sin = 0.0
        for p in range(n_comps - 1):
            for q in range(p + 1, n_comps):
                app = mats[:, p, p]
                aqq = mats[:, q, q]
                apq = mats[:, p, q]
                h1 = apq
                h2 = 0.5 * (aqq - app)
                gram = np.array(
                    [
                        [np.dot(h1, h1), np.dot(h1, h2)],
                        [np.dot(h1, h2), np.dot(h2, h2)],
                    ]
                )
                eigvals, eigvecs = np.linalg.eigh(gram)
                vec = eigvecs[:, np.argmin(eigvals)]
                theta = 0.5 * np.arctan2(vec[1], vec[0])
                if abs(theta) > np.pi / 4:
                    theta -= np.sign(theta) * np.pi / 2
                c = np.cos(theta)
                s = np.sin(theta)
                max_sin = max(max_sin, abs(s))
                if abs(s) <= tol:
                    continue
                _apply_givens_to_mats(mats, p, q, c, s)
                G = np.eye(n_comps)
                G[p, p] = c
                G[q, q] = c
                G[p, q] = -s
                G[q, p] = s
                rotation = rotation @ G
        if max_sin <= tol:
            break
    return rotation


def _apply_givens_to_mats(mats, p, q, c, s):
    Mp = mats[:, :, p].copy()
    Mq = mats[:, :, q].copy()
    mats[:, :, p] = c * Mp + s * Mq
    mats[:, :, q] = -s * Mp + c * Mq
    Mp = mats[:, p, :].copy()
    Mq = mats[:, q, :].copy()
    mats[:, p, :] = c * Mp + s * Mq
    mats[:, q, :] = -s * Mp + c * Mq


def _sym_decorrelate(W):
    eigvals, eigvecs = np.linalg.eigh(W @ W.T)
    eigvals = np.maximum(eigvals, np.finfo(float).eps)
    return (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T) @ W


def cluster(
    ICs,
    n_clusters,
    sample_rate_hz,
    *,
    feature_start_bin=30,
    nperseg=250,
    noverlap=125,
    random_state=0,
):
    """Cluster ICs by truncated Welch spectra and project them to 3D for plotting.

    Parameters
    ----------
    ICs : array, shape (frames, components)
        Time-domain independent components returned by the BSS decomposition.
    n_clusters : int
        Number of KMeans clusters to request. Values above the component count are clipped.
    sample_rate_hz : float
        Recording sample rate used for Welch spectra.
    feature_start_bin : int
        Lower spectral bin removed before KMeans/PCA, matching the reference notebook's ``[:, 30:]``.
    """
    ICs = np.asarray(ICs, dtype=float)
    if ICs.ndim != 2:
        raise ValueError(f"ICs must have shape (frames, components), got {ICs.shape}.")

    n_frames, n_ics = ICs.shape
    if n_ics < 1:
        raise ValueError("ICs must contain at least one component.")

    nperseg = min(int(nperseg), n_frames)
    noverlap = min(int(noverlap), max(nperseg - 1, 0))
    spectra = []
    for ic_idx in range(n_ics):
        _, power = welch(
            ICs[:, ic_idx],
            fs=sample_rate_hz,
            return_onesided=True,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        spectra.append(power)
    spectra = np.asarray(spectra)

    start_bin = min(max(int(feature_start_bin), 0), max(spectra.shape[1] - 1, 0))
    features = np.log1p(spectra[:, start_bin:])
    if features.shape[1] < 3:
        features = np.log1p(spectra)

    n_clusters = min(int(n_clusters), n_ics)
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    predictions = kmeans.fit_predict(features)

    n_pca_components = min(3, features.shape[0], features.shape[1])
    new_mat = PCA(n_components=n_pca_components).fit_transform(features)
    if n_pca_components < 3:
        new_mat = np.pad(new_mat, ((0, 0), (0, 3 - n_pca_components)), constant_values=0.0)

    return new_mat, predictions, spectra, features


def plot_mean_log_psd_by_cluster(
    spectra,
    predictions,
    sample_rate_hz,
    *,
    xlim=(0.0, 0.75),
    per_cluster_ylim=(-5.0, 2.0),
    overlay_ylim=(-3.0, 3.0),
    show_individual=True,
    title_prefix="LogPower clus",
):
    """Plot mean log-PSD per cluster plus one overlay panel.

    Parameters
    ----------
    spectra : array, shape (n_components, n_freq_bins)
        Non-negative PSD values (for example from ``cluster(...)[2]``).
    predictions : array, shape (n_components,)
        Cluster labels for each component.
    sample_rate_hz : float
        Sampling rate used to estimate ``spectra``.
    """
    spectra = np.asarray(spectra, dtype=float)
    predictions = np.asarray(predictions)
    if spectra.ndim != 2:
        raise ValueError(f"spectra must have shape (n_components, n_freq_bins), got {spectra.shape}.")
    if predictions.ndim != 1 or predictions.shape[0] != spectra.shape[0]:
        raise ValueError(
            "predictions must be 1D with one label per component: "
            f"got {predictions.shape} for {spectra.shape[0]} components."
        )

    unique_labels = np.unique(predictions)
    n_clusters = int(unique_labels.size)
    eps = np.finfo(float).eps
    freqs = np.linspace(0.0, float(sample_rate_hz) / 2.0, spectra.shape[1])

    fig, ax = plt.subplots(1, n_clusters + 1, figsize=(4.5 * (n_clusters + 1), 3.2), squeeze=False)
    axes = ax[0]
    for panel_idx, cluster_label in enumerate(unique_labels):
        group = spectra[predictions == cluster_label]
        if show_individual:
            for row in group:
                axes[panel_idx].plot(freqs, np.log(np.maximum(row, eps)), lw=0.8, alpha=0.45)
        mean_log = np.log(np.maximum(group.mean(axis=0), eps))
        axes[panel_idx].plot(freqs, mean_log, color="black", lw=1.5, label="mean log")
        axes[panel_idx].set_xlim(list(xlim))
        axes[panel_idx].set_ylim(list(per_cluster_ylim))
        axes[panel_idx].set_title(f"{title_prefix} {int(cluster_label)}, {group.shape[0]}")
        axes[panel_idx].set_xlabel("Frequency (Hz)")
        axes[panel_idx].set_ylabel("Log PSD")
        axes[panel_idx].legend(loc="best")

        axes[n_clusters].plot(freqs, mean_log, lw=1.2, label=f"clus {int(cluster_label)}")

    axes[n_clusters].set_xlim(list(xlim))
    axes[n_clusters].set_ylim(list(overlay_ylim))
    axes[n_clusters].set_title("Mean Log PSD by Cluster")
    axes[n_clusters].set_xlabel("Frequency (Hz)")
    axes[n_clusters].set_ylabel("Log PSD")
    axes[n_clusters].legend(loc="best")
    fig.tight_layout()
    return fig, axes


def rank_clusters_by_mean_log_psd(
    spectra,
    predictions,
    sample_rate_hz,
    *,
    fmin=0.0,
    fmax=0.75,
    aggregate="mean",
):
    """Rank clusters by mean log-PSD score in a target frequency band.

    Parameters
    ----------
    spectra : array, shape (n_components, n_freq_bins)
        Non-negative PSD values per component.
    predictions : array, shape (n_components,)
        Cluster labels per component.
    sample_rate_hz : float
        Sampling rate used to estimate ``spectra``.
    fmin, fmax : float
        Frequency band (Hz) used for ranking.
    aggregate : {"mean", "sum", "peak"}
        How to collapse log-PSD values within the selected frequency band.

    Returns
    -------
    ranked : list[dict]
        Highest-score-first entries with keys:
        ``cluster``, ``score``, ``n_components``, ``rank``.
    """
    spectra = np.asarray(spectra, dtype=float)
    predictions = np.asarray(predictions)
    if spectra.ndim != 2:
        raise ValueError(f"spectra must have shape (n_components, n_freq_bins), got {spectra.shape}.")
    if predictions.ndim != 1 or predictions.shape[0] != spectra.shape[0]:
        raise ValueError(
            "predictions must be 1D with one label per component: "
            f"got {predictions.shape} for {spectra.shape[0]} components."
        )
    if aggregate not in {"mean", "sum", "peak"}:
        raise ValueError("aggregate must be one of: 'mean', 'sum', 'peak'.")

    fs = float(sample_rate_hz)
    freqs = np.linspace(0.0, fs / 2.0, spectra.shape[1])
    lo = max(0.0, min(float(fmin), fs / 2.0))
    hi = max(lo, min(float(fmax), fs / 2.0))
    band_mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(band_mask):
        raise ValueError(f"No frequency bins fall inside [{lo}, {hi}] Hz.")

    eps = np.finfo(float).eps
    ranked = []
    for label in np.unique(predictions):
        group = spectra[predictions == label]
        mean_log = np.log(np.maximum(group.mean(axis=0), eps))
        band_values = mean_log[band_mask]
        if aggregate == "mean":
            score = float(band_values.mean())
        elif aggregate == "sum":
            score = float(band_values.sum())
        else:
            score = float(band_values.max())
        ranked.append(
            {
                "cluster": int(label),
                "score": score,
                "n_components": int(group.shape[0]),
            }
        )

    ranked.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


# proceed function without returning the individuals results
def compute(traces,n_clus,f_s):
    a = eig_dec(traces)
    ic_comps, IC_ft,A,mean = ica_dec(traces,a,t=0.0001,max_=500)
    new_mat, predictions, spectra, _ = cluster(ic_comps,n_clus,f_s)
    al = np.c_[new_mat.round(1),predictions.round(1)]
    plot_clusters(new_mat,predictions)
    plottings_spectrals(IC_ft, n_clus, al, f_s)
    # Keep existing mean-log spectral behavior, but from PSDs used for clustering.
    plot_mean_log_psd_by_cluster(spectra, predictions, f_s)

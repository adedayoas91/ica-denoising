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
    lags=(1, 2, 3, 5, 10, 20),
    tol=0.0001,
    max_iter=500,
):
    """Second-order blind identification for extracted trace matrices."""
    X, mean, X_white = _prepare_whitened_data(data, n_comps)
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
    X, mean, X_white = _prepare_whitened_data(data, n_comps)
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
    """Infomax-style natural-gradient ICA on PCA-whitened traces."""
    X, mean, X_white = _prepare_whitened_data(data, n_comps)
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
    return _format_bss_output(X, mean, ic_comps, data.shape[1])


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


def _prepare_whitened_data(data, n_comps):
    X = np.asarray(data, dtype=float).T
    n_comps = int(n_comps)
    if n_comps < 1 or n_comps > min(X.shape):
        raise ValueError(f"n_comps must be between 1 and {min(X.shape)}, got {n_comps}.")
    mean = X.mean(axis=0)
    X_centered = X - mean
    pca = PCA(n_components=n_comps, whiten=True, svd_solver="full")
    X_white = pca.fit_transform(X_centered)
    return X_centered, mean, X_white


def _format_bss_output(X_centered, mean, ic_comps, n_frames):
    A = _least_squares_mixing(ic_comps, X_centered)
    IC_ft = np.zeros((ic_comps.shape[1], n_frames))
    for i in range(ic_comps.shape[1]):
        IC_ft[i, :] = np.abs(np.fft.fft(ic_comps[:, i], n=n_frames))
    return ic_comps, IC_ft, A, mean


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


# clusterigs
def cluster(ICs,n_clus,f_s):
    """
    ICs (array ): The matrix of all ICs; shape [n_features, n_ICs].
    l (int): the length to which mat is truncated to cluster on
    n_clus (int): number of cluster (2 or 3)
    """

    # Kmeans on truncated spectra
    IC_welch = np.zeros((ICs.shape[1],126))
    for i in range(ICs.shape[1]):
        _, IC_welch[i,:] = welch(ICs[:,i],f_s,return_onesided=True,nperseg=250,noverlap=125)
    kmeans = KMeans(n_clusters=n_clus)
    predictions = kmeans.fit_predict(IC_welch[:,30:])  # can remove the [:,30:]
    pca = PCA(n_components=3)
    new_mat = pca.fit_transform(IC_welch[:,30:])    # can remove the [:,30:]
    c = kmeans.cluster_centers_
    centers = pca.transform(c)
    return new_mat, predictions #, centers


# proceed function without returning the individuals results      
def compute(traces,n_clus,f_s):
    a = eig_dec(traces)
    ic_comps, IC_ft,A,mean = ica_dec(traces,a,t=0.0001,max_=500)
    new_mat, predictions= cluster(IC_ft,n_clus,f_s)
    al = np.c_[new_mat.round(1),predictions.round(1)]
    plot_clusters(new_mat,predictions)
    plottings_spectrals(n_clus,al)
    plottings_logSpectral(n_clus,al) 

#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


def plott_ics(ics):  # ,found_artifacts
    fig, ax = plt.subplots(ics.shape[1], 1, figsize=(15, 1.3 * ics.shape[1]))
    for i in range(ics.shape[1]):
        ax[i].vlines(
            np.arange(30, 1210, 60),
            ymin=ics.T[i, :].min(),
            ymax=ics.T[i, :].max(),
            ls="--",
            color="g",
            lw=0.6,
        )
        ax[i].plot((ics.T[i, :]), label="{}".format(i), lw=0.6)
        ax[i].set_ylim([-0.2, 0.2])
        ax[i].legend()


def plot_FT_spectrals(ICs, f_s, n_comps):
    """
    Ics: all ICA returns indep components with returned shape [n_features,n_var]
    f_s: sampling frequency
    n_comps is the number of ICs
    """
    fig, ax = plt.subplots(n_comps, 2, figsize=(15, 1.2 * n_comps))
    for i in range(n_comps):
        ax[i, 0].plot(
            np.fft.fftshift(np.linspace(-f_s / 2, f_s / 2, ICs.shape[0])),
            np.abs(np.fft.fft(ICs[:, i])),
        )
        ax[i, 0].set_title("FFT Amplitude Spectrum of IC {}".format(i))
        ax[i, 0].grid()
        ax[i, 0].set_xlim([0, f_s / 2])
        overlaps = [100, 125, 150]
        for overlap in overlaps:
            freqs, power = welch(
                ICs[:, i],
                f_s,
                return_onesided=True,
                nperseg=250,
                noverlap=overlap,
            )
            ax[i, 1].plot(freqs, power, label="overlap = {}".format(overlap), alpha=0.75)
            ax[i, 1].set_title("Welch spectrum {}".format(i))
            ax[i, 1].grid()
            ax[i, 1].legend()
    plt.tight_layout()


def plot_clusters(new_mat, predictions):  # ,centers
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    unique_labels = np.unique(predictions)
    for label in unique_labels:
        cluster_points = new_mat[predictions == label]
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            cluster_points[:, 2],
            label=label,
            cmap="brg",
        )

    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    ax.legend()


def plottings_spectrals(IC_ft, n_clus, alll, f_s):
    """All spectrals overlapped in each group."""
    fig, ax = plt.subplots(1, n_clus, figsize=(15, 3))
    for i in range(n_clus):
        group = IC_ft[np.where(alll[:, 3] == i)]
        for j in range(group.shape[0]):
            ax[i].plot(np.fft.fftshift(np.linspace(-f_s / 2, f_s / 2, IC_ft.shape[1])), group[j, :])
        ax[i].plot(
            np.fft.fftshift(np.linspace(-f_s / 2, f_s / 2, IC_ft.shape[1])),
            group.mean(0),
            color="black",
            lw=2,
            label="mean",
        )
        ax[i].set_xlim([0, 3])
        ax[i].set_title("cluster {} with {} enteries".format(i, group.shape[0]))
        ax[i].legend()


def plottings_logSpectral(IC_ft, n_clus, alll, f_s):
    """Log of all spectrals overlapped in each group."""
    fig, ax = plt.subplots(1, n_clus + 1, figsize=(15, 3))
    for i in range(n_clus):
        group = IC_ft[np.where(alll[:, 3] == i)]
        for j in range(group.shape[0]):
            ax[i].plot(
                np.fft.fftshift(np.linspace(-f_s / 2, f_s / 2, IC_ft.shape[1])),
                np.log(group[j, :]),
            )
        ax[i].plot(
            np.fft.fftshift(np.linspace(-f_s / 2, f_s / 2, IC_ft.shape[1])),
            np.log(group.mean(0)),
            color="black",
            lw=1,
            label="mean log",
        )
        ax[i].set_xlim([0, 0.75])
        ax[i].set_ylim([-5, 2])
        ax[i].set_title("LogPower clus {}, {}".format(i, group.shape[0]))
        ax[i].legend()

        ax[n_clus].plot(
            np.fft.fftshift(np.linspace(-f_s / 2, f_s / 2, IC_ft.shape[1])),
            np.log(group.mean(0)),
            lw=1,
            label="clus {}".format(i),
        )
        ax[n_clus].set_xlim([0, 0.75])
        ax[n_clus].set_ylim([-3, 3])
        ax[n_clus].legend()


def plottings_group_spectrals(IC_ft, n_clus, alll, f_s):
    fig, ax = plt.subplots(1, n_clus, figsize=(10, 3))
    for i in range(n_clus):
        group = IC_ft[np.where(alll[:, 4] == i)]
        for j in range(group.shape[0]):
            ax[i].plot(np.fft.fftshift(np.linspace(-f_s / 2, f_s / 2, IC_ft.shape[1])), group[j, :])
        ax[i].set_xlim([0, 3])
        ax[i].set_title("cl {} with {}".format(i, group.shape[0]))

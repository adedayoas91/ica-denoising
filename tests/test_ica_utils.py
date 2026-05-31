from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

CORE_DIR = Path(__file__).resolve().parents[1] / "src" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from ica_utils import (
    cluster,
    infomax_dec,
    rank_clusters_by_mean_log_psd,
    reconstruct_bss,
    reject_components_from_cluster_selection,
)


class ICAUtilsTests(unittest.TestCase):
    def test_cluster_returns_notebook_outputs(self) -> None:
        rng = np.random.default_rng(0)
        frames = 128
        t = np.arange(frames) / 30.0
        ics = np.column_stack(
            [
                np.sin(2 * np.pi * 0.5 * t),
                np.sin(2 * np.pi * 2.0 * t),
                np.sin(2 * np.pi * 7.0 * t),
                rng.normal(0, 0.2, frames),
            ]
        )

        new_mat, predictions, spectra, features = cluster(
            ics,
            n_clusters=10,
            sample_rate_hz=30.0,
            feature_start_bin=1,
            nperseg=64,
            noverlap=32,
            random_state=0,
        )

        self.assertEqual(new_mat.shape, (4, 3))
        self.assertEqual(predictions.shape, (4,))
        self.assertEqual(spectra.shape[0], 4)
        self.assertEqual(features.shape[0], 4)
        self.assertEqual(features.shape[1], spectra.shape[1] - 1)
        self.assertEqual(len(np.unique(predictions)), 4)

    def test_cluster_rejects_non_matrix_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            cluster(np.arange(10), n_clusters=2, sample_rate_hz=30.0)

    def test_reject_components_from_cluster_selection(self) -> None:
        predictions = np.array([0, 1, 1, 2, 2, 2])
        rejected = reject_components_from_cluster_selection(predictions, reject_clusters=[1])
        self.assertListEqual(rejected.tolist(), [1, 2])

        rejected = reject_components_from_cluster_selection(predictions, keep_clusters=[2])
        self.assertListEqual(rejected.tolist(), [0, 1, 2])

    def test_reject_components_selection_reject_and_keep_is_invalid(self) -> None:
        predictions = np.array([0, 1, 2])
        with self.assertRaisesRegex(ValueError, "reject_clusters or keep_clusters"):
            reject_components_from_cluster_selection(
                predictions,
                reject_clusters=[1],
                keep_clusters=[2],
            )

    def test_rank_clusters_by_mean_log_psd_orders_high_to_low(self) -> None:
        # Cluster 1 has highest power, then cluster 2, then cluster 0.
        spectra = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],   # c0
                [10.0, 8.0, 9.0, 10.0], # c1
                [9.0, 7.0, 8.0, 9.0],   # c1
                [3.0, 2.0, 2.5, 3.0],   # c2
            ]
        )
        predictions = np.array([0, 1, 1, 2])
        ranked = rank_clusters_by_mean_log_psd(
            spectra,
            predictions,
            sample_rate_hz=30.0,
            fmin=0.0,
            fmax=15.0,
            aggregate="mean",
        )
        self.assertListEqual([row["cluster"] for row in ranked], [1, 2, 0])
        self.assertListEqual([row["rank"] for row in ranked], [1, 2, 3])

    def test_rank_clusters_by_peak_log_psd_orders_by_peak(self) -> None:
        # c0 has one strong low-frequency peak and should rank first in peak mode.
        spectra = np.array(
            [
                [100.0, 1.0, 1.0, 1.0],  # c0
                [9.0, 9.0, 9.0, 9.0],    # c1
                [4.0, 4.0, 4.0, 4.0],    # c2
            ]
        )
        predictions = np.array([0, 1, 2])
        ranked = rank_clusters_by_mean_log_psd(
            spectra,
            predictions,
            sample_rate_hz=30.0,
            fmin=0.0,
            fmax=2.0,
            aggregate="peak",
        )
        self.assertListEqual([row["cluster"] for row in ranked], [0, 1, 2])

    def test_infomax_returns_reconstruction_mixing_not_unmixing(self) -> None:
        frames = 240
        t = np.linspace(0.0, 1.0, frames, endpoint=False)
        sources = np.column_stack(
            [
                np.sin(2 * np.pi * 3 * t),
                np.sign(np.sin(2 * np.pi * 5 * t)),
                np.cos(2 * np.pi * 7 * t) ** 3,
            ]
        )
        true_mixing = np.array(
            [
                [1.0, 0.2, -0.1],
                [0.3, 1.1, 0.2],
                [-0.4, 0.1, 1.2],
                [0.2, -0.5, 0.4],
                [0.6, 0.3, -0.7],
            ]
        )
        traces = (sources @ true_mixing.T + np.array([0.5, -0.2, 0.1, 0.3, -0.4])).T

        ic_comps, _ic_ft, mixing, mean = infomax_dec(
            traces,
            n_comps=3,
            max_iter=5,
            random_state=0,
        )

        self.assertEqual(mixing.shape, (traces.shape[0], ic_comps.shape[1]))
        reconstructed = reconstruct_bss(ic_comps, mixing, mean)
        np.testing.assert_allclose(reconstructed, traces.T, atol=1e-10)

        centered = traces.T - mean
        effective_unmixing = np.linalg.lstsq(centered, ic_comps, rcond=None)[0].T
        np.testing.assert_allclose(effective_unmixing @ mixing, np.eye(3), atol=1e-10)


if __name__ == "__main__":
    unittest.main()

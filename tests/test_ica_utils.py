from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

CORE_DIR = Path(__file__).resolve().parents[1] / "src" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from ica_utils import cluster, reject_components_from_cluster_selection


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


if __name__ == "__main__":
    unittest.main()

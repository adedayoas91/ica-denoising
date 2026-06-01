from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ic_quality import (
    ICFeatureConfig,
    compute_ic_features,
    leave_one_ic_out_validation,
    reconstruct_with_rejected,
    run_ic_quality_pipeline,
    score_ic_candidates,
)


class ICQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        frames = 500
        t = np.arange(frames) / 30.0
        slow = np.sin(2 * np.pi * 0.2 * t)
        high = np.sin(2 * np.pi * 8.0 * t)
        burst = rng.normal(0, 0.05, frames)
        burst[100:105] += 8.0
        behavior = np.sin(2 * np.pi * 0.4 * t)
        self.ic_comps = np.column_stack([slow, high, burst, behavior])
        self.mixing = np.array(
            [
                [1.0, 0.1, 5.0, 0.2],
                [0.9, 0.2, 0.1, 0.1],
                [0.8, 0.1, 0.1, 0.3],
                [0.7, 0.2, 0.1, 0.4],
                [0.6, 0.1, 0.1, 0.5],
            ]
        )
        self.mean = np.zeros(self.mixing.shape[0])
        self.reference = self.ic_comps @ self.mixing.T
        self.behavior_targets = {"vigor": behavior}

    def test_compute_features_separates_slow_and_high_frequency(self) -> None:
        table = compute_ic_features(
            self.ic_comps,
            self.mixing,
            ICFeatureConfig(sample_rate_hz=30.0),
            behavior_targets=self.behavior_targets,
            method="synthetic",
        )
        self.assertEqual(len(table), 4)
        high_ratio = table.loc[table["component"] == 1, "high_freq_power_ratio"].iloc[0]
        slow_ratio = table.loc[table["component"] == 0, "high_freq_power_ratio"].iloc[0]
        behavior_corr = table.loc[table["component"] == 3, "max_abs_behavior_corr"].iloc[0]
        self.assertGreater(high_ratio, slow_ratio)
        self.assertGreater(behavior_corr, 0.95)

    def test_score_recommends_drop_only_for_unprotected_artifact(self) -> None:
        features = pd.DataFrame(
            {
                "component": [0, 1, 2],
                "high_freq_power_ratio": [0.8, 0.1, 0.6],
                "robust_peak_density": [0.05, 0.0, 0.04],
                "abs_excess_kurtosis": [9.0, 0.5, 8.0],
                "window_variance_ratio": [8.0, 1.0, 7.0],
                "max_median_loading_ratio": [12.0, 1.0, 12.0],
                "narrowband_ratio": [20.0, 1.0, 18.0],
                "low_freq_power_ratio": [0.0, 0.8, 0.2],
                "lag1_autocorr": [0.0, 0.9, 0.1],
                "max_abs_behavior_corr": [0.0, 0.8, 0.9],
            }
        )
        scored = score_ic_candidates(features)
        by_component = scored.set_index("component")
        self.assertEqual(by_component.loc[0, "recommendation"], "drop")
        self.assertNotEqual(by_component.loc[2, "recommendation"], "drop")

    def test_leave_one_out_validation_and_reconstruction_shape(self) -> None:
        reconstructed = reconstruct_with_rejected(self.ic_comps, self.mixing, self.mean, [1])
        self.assertEqual(reconstructed.shape, self.reference.shape)
        validation = leave_one_ic_out_validation(
            self.ic_comps,
            self.mixing,
            self.mean,
            self.reference,
            behavior_targets=self.behavior_targets,
        )
        self.assertEqual(set(validation["component"]), {0, 1, 2, 3})
        self.assertIn("reference_corr_delta", validation.columns)

    def test_pipeline_writes_review_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_ic_quality_pipeline(
                ic_comps=self.ic_comps,
                mixing=self.mixing,
                mean=self.mean,
                sample_rate_hz=30.0,
                reference_traces=self.reference,
                behavior_targets=self.behavior_targets,
                method="synthetic",
                dataset="unit-test",
                output_dir=Path(tmp),
            )
            self.assertIn("features", result.saved_paths)
            self.assertTrue(result.saved_paths["features"].exists())
            self.assertTrue(result.saved_paths["html_report"].exists())


if __name__ == "__main__":
    unittest.main()

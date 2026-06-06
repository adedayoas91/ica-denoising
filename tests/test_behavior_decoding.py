from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from ica_denoising.behavior_decoding import (
    BehaviorTargetVariant,
    TraceVariant,
    circular_shift,
    classification_metrics,
    run_decoding_experiment,
    make_behavior_targets,
    make_behavior_target_variants,
    load_trace_variants,
)


class BehaviorDecodingLoadTests(unittest.TestCase):
    def test_classification_metrics_marks_single_class_test_fold_as_undefined(self) -> None:
        metrics = classification_metrics(
            np.zeros(5, dtype=int),
            np.zeros(5, dtype=int),
            None,
        )
        self.assertTrue(np.isnan(metrics["balanced_accuracy"]))
        self.assertTrue(np.isnan(metrics["macro_f1"]))

    def test_load_trace_variants_discovers_method_cleaned_subdirectory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "data"
            dataset_dir.mkdir()
            raw = np.arange(12, dtype=float).reshape(3, 4)
            np.save(dataset_dir / "raw.npy", raw)

            cleaned_root = root / "outputs" / "linear" / "demo" / "demo_data"
            cleaned_dir = cleaned_root / "fastica" / "cleaned"
            cleaned_dir.mkdir(parents=True)
            np.save(cleaned_dir / "cleaned_fastica_demo_data.npy", raw * 0.5)

            variants = load_trace_variants(
                dataset_dir=dataset_dir,
                raw_name="raw.npy",
                cleaned_root=cleaned_root,
                dataset_name="demo_data",
            )

            self.assertEqual([variant.name for variant in variants], ["raw", "fastica/cleaned_fastica_demo_data"])
            np.testing.assert_allclose(variants[1].traces, (raw * 0.5).T)

    def test_load_trace_variants_accepts_filtered_raw_override(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "data"
            dataset_dir.mkdir()
            raw_on_disk = np.arange(24, dtype=float).reshape(4, 6)
            np.save(dataset_dir / "raw.npy", raw_on_disk)

            filtered_raw = raw_on_disk[[0, 2, 3]][:, [0, 2, 3, 5]]
            cleaned_root = root / "outputs" / "linear" / "demo" / "demo_data"
            cleaned_dir = cleaned_root / "fastica" / "cleaned"
            cleaned_dir.mkdir(parents=True)
            np.save(cleaned_dir / "cleaned_fastica_demo_data.npy", filtered_raw * 0.5)

            variants = load_trace_variants(
                dataset_dir=dataset_dir,
                raw_name="raw.npy",
                cleaned_root=cleaned_root,
                dataset_name="demo_data",
                raw_traces=filtered_raw.T,
            )

            self.assertEqual(
                [variant.name for variant in variants],
                ["raw", "fastica/cleaned_fastica_demo_data"],
            )
            np.testing.assert_allclose(variants[0].traces, filtered_raw.T)
            np.testing.assert_allclose(variants[1].traces, (filtered_raw * 0.5).T)

    def test_behavior_target_variants_use_configured_quantiles_and_windows(self) -> None:
        tail_angle = np.array([0.0, 1.0, 3.0, 2.0, 6.0, 7.0, 7.5, 9.0])
        targets = make_behavior_targets(
            tail_angle,
            n_frames=4,
            bout_quantile=0.75,
            smooth_window=1,
        )

        variants = make_behavior_target_variants(
            targets,
            bout_quantiles=(0.5, 0.9),
            smooth_windows=(1, 3),
        )

        self.assertEqual(
            [variant.target_variant for variant in variants],
            ["primary", "q0p5_sw1", "q0p5_sw3", "q0p9_sw1", "q0p9_sw3"],
        )
        self.assertEqual(variants[1].bout_quantile, 0.5)
        self.assertEqual(variants[2].smooth_window, 3)
        self.assertNotEqual(
            float(variants[1].bout_threshold),
            float(variants[3].bout_threshold),
        )
        self.assertFalse(np.array_equal(variants[1].vigor, variants[2].vigor))

    def test_circular_shift_preserves_values_and_changes_order(self) -> None:
        values = np.arange(8)
        shifted = circular_shift(values, np.random.default_rng(0))

        np.testing.assert_array_equal(np.sort(shifted), values)
        self.assertFalse(np.array_equal(shifted, values))

    def test_simple_decoding_emits_target_and_null_provenance(self) -> None:
        time = np.arange(60, dtype=float)
        target = np.sin(time / 5.0)
        traces = np.column_stack([target, np.cos(time / 7.0)])

        result = run_decoding_experiment(
            [TraceVariant("raw", Path("raw.npy"), traces)],
            target,
            target_name="tail_vigor",
            task="regression",
            n_splits=3,
            gap=1,
            include_transfer=False,
            include_null=True,
            null_block_size=6,
            null_strategies=("block_shuffle", "circular_shift"),
            target_variants=(
                BehaviorTargetVariant("primary", target),
                BehaviorTargetVariant(
                    "smooth",
                    np.convolve(target, np.ones(3) / 3, mode="same"),
                    0.75,
                    3,
                ),
            ),
        )

        self.assertEqual(set(result.metrics["target_variant"]), {"primary", "smooth"})
        self.assertEqual(
            set(result.metrics["null_strategy"]),
            {"observed", "block_shuffle", "circular_shift"},
        )
        self.assertIn("target_variant", result.predictions.columns)
        self.assertIn("null_strategy", result.predictions.columns)


if __name__ == "__main__":
    unittest.main()

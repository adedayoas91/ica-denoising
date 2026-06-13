from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from ica_denoising.bss_notebook import DatasetSpec
from ica_denoising.evaluation_runner import (
    behavior_prediction_uncertainty,
    load_evaluation_config,
    write_recording_aggregate,
)


class EvaluationRunnerTests(unittest.TestCase):
    def test_example_config_loads_tuple_fields_and_causal_models(self) -> None:
        path = (
            Path(__file__).resolve().parents[1] / "configs" / "evaluation.example.json"
        )
        config = load_evaluation_config(path, sample_rate_hz=7.5)

        self.assertEqual(config.sample_rate_hz, 7.5)
        self.assertIsInstance(config.bss_methods, tuple)
        self.assertIsInstance(config.cluster_stability_seeds, tuple)
        self.assertEqual(config.causal_transition_models, ("linear", "mlp"))
        self.assertEqual(config.target_bout_quantiles, (0.65, 0.85))
        self.assertEqual(config.target_smooth_windows, (3, 5))
        self.assertEqual(config.null_strategies, ("block_shuffle", "circular_shift"))
        self.assertEqual(config.causal_sufficiency_targets, ("vigor",))
        self.assertEqual(
            config.benchmark_latent_methods,
            ("factor_analysis", "sparse_pca", "nmf"),
        )
        self.assertEqual(config.benchmark_latent_ranks, (10, 20, 40))
        self.assertEqual(
            config.ic_quality_selection_strategies,
            ("ic_quality_nonartifact", "ic_quality_strict_keep"),
        )
        self.assertIsInstance(config.bss_component_counts, tuple)
        self.assertIsInstance(config.bss_pca_variance_thresholds, tuple)
        self.assertIsNone(config.n_components)
        self.assertIsNone(config.bss_pca_components)
        self.assertIsNone(config.bss_pca_variance_threshold)

    def test_behavior_uncertainty_uses_paired_out_of_fold_predictions(self) -> None:
        rows = []
        for fold in (0, 1):
            for time_index in range(fold * 5, fold * 5 + 5):
                y_true = time_index % 2
                rows.extend(
                    [
                        {
                            "target": "bout_state",
                            "task": "classification",
                            "comparison": "within",
                            "train_version": "raw",
                            "test_version": "raw",
                            "fold": fold,
                            "time_index": time_index,
                            "y_true": y_true,
                            "y_pred": 1 - y_true,
                        },
                        {
                            "target": "bout_state",
                            "task": "classification",
                            "comparison": "within",
                            "train_version": "clean",
                            "test_version": "clean",
                            "fold": fold,
                            "time_index": time_index,
                            "y_true": y_true,
                            "y_pred": y_true,
                        },
                    ]
                )

        table = behavior_prediction_uncertainty(
            pd.DataFrame(rows),
            block_size=2,
            n_bootstrap=100,
            n_permutations=100,
            random_state=0,
        )

        self.assertEqual(len(table), 1)
        self.assertAlmostEqual(float(table["effect"].iloc[0]), 1.0)
        self.assertGreaterEqual(float(table["ci_low"].iloc[0]), 1.0)

    def test_behavior_uncertainty_preserves_target_variant_pairing(self) -> None:
        rows = []
        for target_variant in ("primary", "q0p65_sw3"):
            for time_index in range(5):
                rows.extend(
                    [
                        {
                            "target": "tail_vigor",
                            "target_variant": target_variant,
                            "bout_quantile": 0.75
                            if target_variant == "primary"
                            else 0.65,
                            "smooth_window": 1 if target_variant == "primary" else 3,
                            "null_strategy": "observed",
                            "task": "regression",
                            "comparison": "within",
                            "train_version": "raw",
                            "test_version": "raw",
                            "fold": 0,
                            "time_index": time_index,
                            "y_true": float(time_index),
                            "y_pred": float(time_index + 1),
                        },
                        {
                            "target": "tail_vigor",
                            "target_variant": target_variant,
                            "bout_quantile": 0.75
                            if target_variant == "primary"
                            else 0.65,
                            "smooth_window": 1 if target_variant == "primary" else 3,
                            "null_strategy": "observed",
                            "task": "regression",
                            "comparison": "transfer_raw_to_clean",
                            "train_version": "raw",
                            "test_version": "clean",
                            "fold": 0,
                            "time_index": time_index,
                            "y_true": float(time_index),
                            "y_pred": float(time_index + 0.5),
                        },
                    ]
                )

        table = behavior_prediction_uncertainty(
            pd.DataFrame(rows),
            block_size=2,
            n_bootstrap=100,
            n_permutations=100,
            random_state=0,
        )

        self.assertEqual(len(table), 1)
        self.assertEqual(table["method"].iloc[0], "clean")
        self.assertEqual(int(table["n_predictions"].iloc[0]), 10)

    def test_recording_aggregate_overwrites_existing_identity_columns(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs"
            run_dir = output / "demo" / "run1"
            run_dir.mkdir(parents=True)
            behavior = pd.DataFrame(
                [
                    {
                        "target": "tail_vigor",
                        "comparison": "within",
                        "test_version": method,
                        "pearson": value,
                        "balanced_accuracy": value,
                    }
                    for method, value in (("raw", 0.5), ("clean", 0.6))
                ]
            )
            causal = pd.DataFrame(
                [
                    {
                        "transition_model": "linear",
                        "variant": method,
                        "dynamic_mse_normalized": value,
                    }
                    for method, value in (("raw", 1.0), ("clean", 0.9))
                ]
            )
            uncertainty = pd.DataFrame([{"method": "clean", "effect": 0.1}])
            causal_sufficiency = pd.DataFrame(
                [
                    {
                        "variant": "clean",
                        "target": "vigor",
                        "rmse_delta_aug_minus_base": -0.1,
                    }
                ]
            )
            artifact = pd.DataFrame(
                [{"variant": "clean", "artifact_center": 40, "rmse": 0.1}]
            )
            bpi = pd.DataFrame(
                [{"recording": "already-present", "method": "clean", "bpi": 90.0}]
            )
            tables = {
                "trace_metrics": (
                    pd.DataFrame([{"variant": "raw", "global_pearson": 1.0}]),
                    "trace.csv",
                ),
                "behavior_metrics": (behavior, "behavior.csv"),
                "behavior_uncertainty": (uncertainty, "uncertainty.csv"),
                "causal_metrics": (causal, "causal.csv"),
                "causal_sufficiency": (causal_sufficiency, "causal_sufficiency.csv"),
                "artifact_probe_metrics": (artifact, "artifact.csv"),
                "bpi_ablation": (bpi, "bpi.csv"),
            }
            paths = {}
            for label, (table, filename) in tables.items():
                path = run_dir / filename
                table.to_csv(path, index=False)
                paths[label] = path
            spec = DatasetSpec(
                key="demo/run1",
                data_name="run1",
                group="demo",
                trace_path=root / "trace.npy",
                recording_id="run1",
                fish_id="fish1",
                run_id="run1",
                modality="fluorescence",
            )

            with patch(
                "ica_denoising.evaluation_runner.dataset_registry",
                return_value={spec.key: spec},
            ):
                aggregate_paths = write_recording_aggregate(
                    {spec.key: paths},
                    project_root=root,
                    output_root=output,
                )

            aggregate_bpi = pd.read_csv(aggregate_paths["bpi"])
            aggregate_trace = pd.read_csv(aggregate_paths["trace"])
            aggregate_artifact = pd.read_csv(aggregate_paths["artifact_probe"])
            aggregate_sufficiency = pd.read_csv(aggregate_paths["causal_sufficiency"])
            self.assertEqual(aggregate_bpi["recording"].iloc[0], "run1")
            self.assertEqual(aggregate_trace["recording"].iloc[0], "run1")
            self.assertEqual(aggregate_artifact["recording"].iloc[0], "run1")
            self.assertEqual(aggregate_sufficiency["recording"].iloc[0], "run1")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from ica_denoising.bss_notebook import DatasetSpec
from ica_denoising.evaluation_diagnostics import (
    build_provenance_manifest,
    compute_bpi_ablation,
    exact_sign_flip_pvalue,
    paired_block_bootstrap_interval,
    paired_block_sign_permutation_pvalue,
    quantify_motor_neuron_artifact,
    temporal_dependence_diagnostics,
)


class EvaluationDiagnosticsTests(unittest.TestCase):
    def test_provenance_manifest_flags_duplicate_trace_hashes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.npy"
            behavior = root / "tail.npy"
            np.save(trace, np.arange(20, dtype=float).reshape(4, 5))
            np.save(behavior, np.arange(30, dtype=float))
            specs = [
                DatasetSpec(
                    key=f"demo/run{index}",
                    data_name=f"run{index}",
                    group="demo",
                    trace_path=trace,
                    tail_angle_path=behavior,
                    recording_id=f"run{index}",
                )
                for index in (1, 2)
            ]

            manifest = build_provenance_manifest(specs)

            self.assertTrue((manifest["trace_duplicate_count"] == 2).all())
            self.assertTrue((manifest["behavior_duplicate_count"] == 2).all())
            self.assertTrue(
                (manifest["provenance_warning"] == "duplicate trace and behavior hashes").all()
            )

    def test_bpi_ablation_normalizes_raw_to_one_hundred(self) -> None:
        scores = pd.DataFrame(
            [
                {
                    "recording": "r1",
                    "method": method,
                    "component": component,
                    "scope": scope,
                    "score": score,
                    "raw_score": 0.8,
                    "null_score": 0.5,
                }
                for method, score in (("raw", 0.8), ("clean", 0.7))
                for component, scope in (("vigor", "within"), ("bout", "transfer"))
            ]
        )

        table = compute_bpi_ablation(scores)
        raw = table[
            (table["method"] == "raw")
            & (table["component_subset"] == "combined")
            & (table["normalization"] == "raw_null")
            & (table["weighting"] == "equal")
        ]
        clean = table[
            (table["method"] == "clean")
            & (table["component_subset"] == "combined")
            & (table["normalization"] == "raw_null")
            & (table["weighting"] == "equal")
        ]
        self.assertAlmostEqual(float(raw["bpi"].iloc[0]), 100.0)
        self.assertAlmostEqual(float(clean["bpi"].iloc[0]), 100.0 * (0.7 - 0.5) / 0.3)

    def test_temporal_diagnostics_are_zero_for_identical_variant(self) -> None:
        rng = np.random.default_rng(2)
        raw = rng.normal(size=(120, 4))

        table = temporal_dependence_diagnostics(
            raw,
            {"copy": raw.copy()},
            lags=(1, 3, 5),
            sample_rate_hz=10.0,
        )

        np.testing.assert_allclose(table["acf_median_abs_change"], 0.0, atol=1e-12)
        np.testing.assert_allclose(
            table["xcf_relative_frobenius_change"], 0.0, atol=1e-12
        )

    def test_exact_sign_flip_uses_paired_units(self) -> None:
        pvalue = exact_sign_flip_pvalue([1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(pvalue, 2 / 16)

    def test_block_uncertainty_preserves_contiguous_units(self) -> None:
        differences = np.r_[np.ones(20), np.full(20, 0.5)]
        interval = paired_block_bootstrap_interval(
            differences,
            block_size=10,
            n_bootstrap=100,
            random_state=1,
        )
        pvalue = paired_block_sign_permutation_pvalue(
            differences,
            block_size=10,
            n_permutations=200,
            random_state=1,
        )
        self.assertAlmostEqual(interval["estimate"], 0.75)
        self.assertLessEqual(interval["ci_low"], interval["estimate"])
        self.assertGreaterEqual(interval["ci_high"], interval["estimate"])
        self.assertGreaterEqual(pvalue, 0.0)
        self.assertLessEqual(pvalue, 1.0)

    def test_motor_neuron_metrics_separate_near_and_far_distortion(self) -> None:
        time = np.arange(80, dtype=float)
        raw = np.column_stack([np.sin(time / 8.0), np.cos(time / 10.0)])
        local = raw.copy()
        local[38:43] = np.linspace(local[37], local[43], 7)[1:-1]
        global_change = raw * 0.5

        table = quantify_motor_neuron_artifact(
            raw,
            {"local": local, "global": global_change},
            artifact_center=40,
            half_width=2,
            guard_width=6,
        )

        local_far = table.loc[table["variant"] == "local", "far_nrmse"].mean()
        global_far = table.loc[table["variant"] == "global", "far_nrmse"].mean()
        self.assertLess(local_far, global_far)


if __name__ == "__main__":
    unittest.main()

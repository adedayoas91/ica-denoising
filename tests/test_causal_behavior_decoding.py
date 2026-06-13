from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from ica_denoising.behavior_decoding import BehaviorTargets, TraceVariant
from ica_denoising.causal_behavior_decoding import (
    CausalStateConfig,
    compare_latent_graphs,
    evaluate_markov_sufficiency,
    fit_causal_state_model,
    make_paired_windows,
    minimum_causal_gap,
)


class CausalBehaviorDecodingTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        frames = 180
        neurons = 6
        t = np.arange(frames, dtype=float)
        traces = np.zeros((frames, neurons), dtype=float)
        traces[:, 0] = np.sin(2 * np.pi * t / 20.0)
        traces[:, 1] = np.cos(2 * np.pi * t / 30.0)
        traces[:, 2] = np.sin(2 * np.pi * t / 12.0)
        traces[:, 3] = (
            0.3 * traces[:, 0] + 0.4 * traces[:, 1] + rng.normal(0, 0.08, frames)
        )
        traces[:, 4] = rng.normal(0, 0.1, frames)
        traces[:, 5] = 0.5 * traces[:, 2] + rng.normal(0, 0.06, frames)

        angle = 0.7 * traces[:, 0] + 0.2 * traces[:, 1] + rng.normal(0, 0.05, frames)
        vigor = np.abs(np.diff(angle, prepend=angle[0]))
        bout_threshold = float(np.quantile(vigor, 0.7))
        bout = (vigor >= bout_threshold).astype(int)

        self.targets = BehaviorTargets(
            angle=angle,
            vigor=vigor,
            bout_state=bout,
            bout_threshold=bout_threshold,
        )
        self.raw_variant = TraceVariant("raw", Path("raw.npy"), traces)
        self.clean_variant = TraceVariant("clean", Path("clean.npy"), traces * 0.98)

    def test_make_paired_windows_shapes(self) -> None:
        cfg = CausalStateConfig(
            window=8, target_shifts=(0,), latent_dim=2, n_splits=3, gap=2
        )
        x0, x1, ymap, times = make_paired_windows(
            self.raw_variant.traces, self.targets, cfg, target_shift=2
        )
        self.assertEqual(x0.shape[1], 7)
        self.assertEqual(x1.shape, x0.shape)
        self.assertEqual(x0.shape[0], len(times))
        self.assertEqual(len(ymap["angle"]), len(times))
        self.assertEqual(len(ymap["vigor"]), len(times))
        self.assertEqual(len(ymap["bout_state"]), len(times))

    def test_fit_causal_state_model_outputs_tables(self) -> None:
        cfg = CausalStateConfig(
            window=10,
            target_shifts=(0, 1),
            latent_dim=3,
            n_splits=4,
            gap=9,
            random_state=1,
        )
        result = fit_causal_state_model(
            [self.raw_variant, self.clean_variant], self.targets, cfg
        )
        self.assertFalse(result.fold_metrics.empty)
        self.assertFalse(result.embeddings.empty)
        self.assertFalse(result.residual_tests.empty)
        self.assertFalse(result.graph_metrics.empty)

        for col in (
            "dynamic_mse",
            "dynamic_mse_normalized",
            "persistence_mse",
            "dynamic_improvement_vs_persistence",
            "angle_r2",
            "vigor_r2",
            "bout_balanced_accuracy",
        ):
            self.assertIn(col, result.fold_metrics.columns)
        for col in ("variant", "fold", "time_index", "z1"):
            self.assertIn(col, result.embeddings.columns)
        self.assertIn("fold", result.graph_metrics.columns)

    def test_fit_causal_state_model_rejects_overlapping_history_gap(self) -> None:
        cfg = CausalStateConfig(
            window=10, target_shifts=(0,), latent_dim=2, n_splits=3, gap=2
        )
        self.assertEqual(minimum_causal_gap(cfg), 9)
        with self.assertRaisesRegex(ValueError, "too small"):
            fit_causal_state_model([self.raw_variant], self.targets, cfg)

    def test_markov_sufficiency_returns_fold_rows(self) -> None:
        cfg = CausalStateConfig(
            window=9, target_shifts=(0,), latent_dim=2, n_splits=3, gap=1
        )
        x0, _x1, ymap, _times = make_paired_windows(
            self.raw_variant.traces, self.targets, cfg, target_shift=0
        )
        x0_flat = x0.reshape(x0.shape[0], -1)
        z = x0_flat[:, :2]
        from ica_denoising.behavior_decoding import blocked_folds

        folds = list(blocked_folds(x0_flat.shape[0], n_splits=3, gap=1))
        table = evaluate_markov_sufficiency(x0_flat, z, ymap["vigor"], folds)
        self.assertEqual(len(table), 3)
        self.assertIn("rmse_delta_aug_minus_base", table.columns)

    def test_compare_latent_graphs_requires_z_columns(self) -> None:
        bad = pd.DataFrame(
            {
                "variant": ["raw"],
                "target_shift": [0],
                "time_index": [0],
                "angle": [0.0],
                "vigor": [0.0],
                "bout_state": [0],
            }
        )
        with self.assertRaises(ValueError):
            compare_latent_graphs(bad)


if __name__ == "__main__":
    unittest.main()

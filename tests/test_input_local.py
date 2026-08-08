from __future__ import annotations

import unittest

import numpy as np

from ica_denoising.input_local import (
    InputLocalConfig,
    build_input_local_targets,
    causal_moving_average,
    fit_bout_threshold,
    fit_per_neuron_imputer,
)


class CausalMovingAverageTests(unittest.TestCase):
    def test_trailing_window_uses_only_past(self) -> None:
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        smoothed = causal_moving_average(signal, window=2)
        # output[i] depends only on i-1, i.
        np.testing.assert_allclose(smoothed, [1.0, 1.5, 2.5, 3.5])

    def test_future_change_does_not_affect_earlier_outputs(self) -> None:
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        baseline = causal_moving_average(signal, window=3)
        mutated = signal.copy()
        mutated[-1] = 100.0
        changed = causal_moving_average(mutated, window=3)
        np.testing.assert_allclose(baseline[:-1], changed[:-1])

    def test_resets_at_block_boundaries(self) -> None:
        signal = np.array([1.0, 2.0, 10.0, 20.0])
        segments = [[0, 1], [2, 3]]
        smoothed = causal_moving_average(signal, window=4, segments=segments)
        # block 2 starts fresh, so frame 2 equals its own value.
        self.assertEqual(smoothed[2], 10.0)


class PerNeuronImputerLeakageTests(unittest.TestCase):
    def test_training_means_unchanged_when_holdout_mutates(self) -> None:
        rng = np.random.default_rng(0)
        traces = rng.normal(size=(40, 5))
        traces[3, 0] = np.nan  # training NaN to impute
        train_idx = np.arange(0, 20)
        test_idx = np.arange(20, 40)

        imputer = fit_per_neuron_imputer(traces, train_idx, fold=0)
        baseline_means = imputer.means.copy()

        mutated = traces.copy()
        mutated[test_idx] += 1000.0
        refit = fit_per_neuron_imputer(mutated, train_idx, fold=0)

        np.testing.assert_allclose(refit.means, baseline_means)

    def test_transform_imputes_nonfinite(self) -> None:
        traces = np.array([[1.0, 2.0], [np.nan, 3.0], [3.0, np.nan]])
        imputer = fit_per_neuron_imputer(traces, [0, 2], fold=0)
        result = imputer.transform(traces)
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(result[1, 0], imputer.means[0])

    def test_audit_records_training_scope(self) -> None:
        traces = np.zeros((10, 3))
        audit = fit_per_neuron_imputer(traces, np.arange(5), fold=2).audit()
        self.assertEqual(audit.fit_scope, "training_frames_only")
        self.assertFalse(audit.uses_future_samples)
        self.assertEqual(audit.fold, 2)


class BoutThresholdLeakageTests(unittest.TestCase):
    def test_threshold_unchanged_when_holdout_mutates(self) -> None:
        vigor = np.linspace(0.0, 1.0, 40)
        train_idx = np.arange(0, 20)
        baseline = fit_bout_threshold(vigor, train_idx, bout_quantile=0.75)

        mutated = vigor.copy()
        mutated[20:] = 999.0
        changed = fit_bout_threshold(mutated, train_idx, bout_quantile=0.75)
        self.assertEqual(baseline, changed)


class BuildInputLocalTargetsTests(unittest.TestCase):
    def test_threshold_fit_on_training_and_audits_present(self) -> None:
        rng = np.random.default_rng(1)
        tail_angle = rng.normal(size=400)
        config = InputLocalConfig(bout_quantile=0.75, vigor_smooth_window=3)
        train_idx = np.arange(0, 50)

        targets, audits = build_input_local_targets(
            tail_angle,
            n_frames=100,
            train_idx=train_idx,
            config=config,
            fold=1,
        )
        self.assertEqual(targets.angle.shape[0], 100)
        self.assertEqual(targets.bout_state.shape[0], 100)
        self.assertEqual(targets.target_variant, "input_local")
        transforms = {audit.transform for audit in audits}
        self.assertIn("bout_threshold", transforms)
        self.assertIn("causal_vigor_smoother", transforms)
        self.assertTrue(all(not audit.uses_future_samples for audit in audits))

    def test_holdout_mutation_does_not_change_threshold(self) -> None:
        rng = np.random.default_rng(2)
        tail_angle = rng.normal(size=400)
        config = InputLocalConfig(bout_quantile=0.8, vigor_smooth_window=1)
        train_idx = np.arange(0, 40)

        targets_a, _ = build_input_local_targets(
            tail_angle, n_frames=100, train_idx=train_idx, config=config
        )
        # Mutating the tail angle only in held-out frame range must not change the
        # fitted threshold (it is a quantile over training frames of vigor).
        mutated = tail_angle.copy()
        mutated[200:] += 50.0
        targets_b, _ = build_input_local_targets(
            mutated, n_frames=100, train_idx=train_idx, config=config
        )
        self.assertAlmostEqual(
            targets_a.bout_threshold, targets_b.bout_threshold, places=6
        )


if __name__ == "__main__":
    unittest.main()

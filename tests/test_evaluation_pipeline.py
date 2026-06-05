from __future__ import annotations

import unittest

import numpy as np

from ica_denoising.behavior_decoding import BehaviorTargets
from ica_denoising.causal_behavior_decoding import CausalStateConfig
from ica_denoising.evaluation_pipeline import (
    EvaluationConfig,
    fit_bss_model,
    make_strict_folds,
    run_cluster_stability_sweep,
    run_evaluation,
    segmented_welch_psd,
)


class EvaluationPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(11)
        frames = 180
        time = np.arange(frames, dtype=float)
        sources = np.column_stack(
            [
                np.sin(2 * np.pi * time / 25.0),
                np.cos(2 * np.pi * time / 38.0),
                np.sign(np.sin(2 * np.pi * time / 16.0)),
            ]
        )
        mixing = np.array(
            [
                [1.0, 0.2, -0.1],
                [0.4, 1.1, 0.3],
                [-0.2, 0.5, 1.2],
                [0.7, -0.3, 0.4],
                [0.1, 0.8, -0.5],
            ]
        )
        self.traces = sources @ mixing.T + rng.normal(0, 0.03, (frames, mixing.shape[0]))
        angle = 0.8 * sources[:, 0] + 0.2 * sources[:, 1]
        vigor = np.abs(np.diff(angle, prepend=angle[0]))
        threshold = float(np.quantile(vigor, 0.75))
        self.targets = BehaviorTargets(
            angle=angle,
            vigor=vigor,
            bout_state=(vigor >= threshold).astype(int),
            bout_threshold=threshold,
        )

    def test_strict_folds_enforce_required_gap_and_exclude_test(self) -> None:
        with self.assertRaisesRegex(ValueError, "too small"):
            make_strict_folds(100, n_splits=5, gap=2, required_gap=4)

        folds = make_strict_folds(100, n_splits=5, gap=4, required_gap=4)
        for fold in folds:
            self.assertEqual(np.intersect1d(fold.train_idx, fold.test_idx).size, 0)
            self.assertTrue(np.isin(fold.test_idx, fold.excluded_idx).all())

    def test_fitted_bss_does_not_change_when_only_held_out_frames_change(self) -> None:
        fold = make_strict_folds(180, n_splits=3, gap=5, required_gap=5)[1]
        changed = self.traces.copy()
        changed[fold.excluded_idx] += 1000.0

        original_model = fit_bss_model(
            self.traces,
            fold.train_idx,
            method="fastica",
            n_components=3,
            max_iter=200,
            random_state=3,
        )
        changed_model = fit_bss_model(
            changed,
            fold.train_idx,
            method="fastica",
            n_components=3,
            max_iter=200,
            random_state=3,
        )

        np.testing.assert_allclose(original_model.mean, changed_model.mean)
        np.testing.assert_allclose(original_model.mixing, changed_model.mixing)
        np.testing.assert_allclose(original_model.train_components, changed_model.train_components)
        self.assertEqual(np.intersect1d(original_model.train_idx, fold.test_idx).size, 0)

    def test_strict_sobi_uses_variance_threshold_pca_rank_by_default(self) -> None:
        fold = make_strict_folds(180, n_splits=3, gap=5, required_gap=5)[1]

        model = fit_bss_model(
            self.traces,
            fold.train_idx,
            method="sobi",
            n_components=None,
            pca_components=None,
            pca_variance_threshold=0.95,
            max_iter=20,
            random_state=3,
        )

        self.assertEqual(model.n_components, min(self.traces[fold.train_idx].shape))
        self.assertLess(model.pca_components, model.n_components)
        self.assertGreaterEqual(model.pca_explained_variance_ratio, 0.95)
        self.assertEqual(model.component_selection_mode, "pca_variance_threshold")
        self.assertEqual(model.train_components.shape[1], model.pca_components)

    def test_segmented_welch_returns_one_spectrum_per_component(self) -> None:
        components = np.column_stack(
            [
                np.sin(np.linspace(0, 8 * np.pi, 100)),
                np.cos(np.linspace(0, 4 * np.pi, 100)),
            ]
        )
        spectra = segmented_welch_psd(
            components,
            (45, 55),
            sample_rate_hz=10.0,
            nperseg=32,
            noverlap=16,
        )
        self.assertEqual(spectra.shape[0], 2)
        self.assertTrue(np.isfinite(spectra).all())

    def test_cluster_stability_sweep_reports_pairwise_seed_agreement(self) -> None:
        fold = make_strict_folds(180, n_splits=3, gap=5, required_gap=5)[0]
        model = fit_bss_model(
            self.traces,
            fold.train_idx,
            method="fastica",
            n_components=3,
            max_iter=100,
            random_state=0,
        )

        pairwise, assignments = run_cluster_stability_sweep(
            model,
            sample_rate_hz=10.0,
            seeds=(0, 1),
            cluster_counts=(2,),
            feature_transforms=("log1p",),
            keep_cluster_count=1,
            feature_start_bin=1,
            nperseg=32,
            noverlap=16,
        )

        self.assertEqual(len(pairwise), 1)
        self.assertEqual(len(assignments), 6)
        self.assertIn("adjusted_rand_index", pairwise.columns)

    def test_evaluation_produces_audited_behavior_and_causal_outputs(self) -> None:
        config = EvaluationConfig(
            sample_rate_hz=10.0,
            n_splits=3,
            gap=5,
            bss_methods=("fastica",),
            n_components=3,
            bss_max_iter=100,
            n_clusters=2,
            keep_cluster_counts=(1,),
            selection_strategies=("low_frequency", "all"),
            selection_random_seeds=(0,),
            feature_start_bin=1,
            welch_nperseg=32,
            welch_noverlap=16,
            baseline_ranks=(3,),
            lowpass_cutoffs_hz=(1.0,),
            decoder_lags=(0, 1, 2),
            null_block_size=12,
            null_seeds=(0, 1),
            causal=CausalStateConfig(
                window=6,
                target_shifts=(0,),
                latent_dim=2,
                n_splits=3,
                gap=5,
                ridge_alpha=2.0,
                random_state=0,
            ),
        )

        result = run_evaluation(self.traces, self.targets, config)

        self.assertFalse(result.behavior_metrics.empty)
        self.assertFalse(result.trace_metrics.empty)
        self.assertFalse(result.behavior_predictions.empty)
        self.assertFalse(result.causal_metrics.empty)
        self.assertFalse(result.causal_embeddings.empty)
        self.assertFalse(result.component_selections.empty)
        self.assertFalse(result.cluster_stability_assignments.empty)
        self.assertFalse(result.variant_metadata.empty)
        self.assertFalse(result.temporal_diagnostics.empty)
        self.assertTrue(result.leakage_audit["leakage_free"].all())
        self.assertEqual(result.leakage_audit["n_test_frames_seen_during_fit"].max(), 0)
        self.assertIn("dynamic_mse_normalized", result.causal_metrics.columns)
        self.assertIn("fold", result.causal_embeddings.columns)
        self.assertIn("spectral_power_retention", result.trace_metrics.columns)
        raw_trace_metrics = result.trace_metrics[result.trace_metrics["variant"] == "raw"]
        np.testing.assert_allclose(raw_trace_metrics["global_pearson"], 1.0)
        np.testing.assert_allclose(raw_trace_metrics["retained_energy_fraction"], 1.0)
        np.testing.assert_allclose(raw_trace_metrics["spectral_power_retention"], 1.0)


if __name__ == "__main__":
    unittest.main()

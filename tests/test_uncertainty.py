from __future__ import annotations

import unittest

import numpy as np

from ica_denoising.uncertainty import (
    aggregate_across_units,
    assign_temporal_blocks,
    correlation_contributions,
    label_algorithmic_replicates,
    ratio_metric_contributions,
    recompute_correlation,
    recompute_nrmse,
    recompute_ratio_metric,
    recompute_rmse,
    rmse_contributions,
)


class BlockAssignmentTests(unittest.TestCase):
    def test_fixed_blocks(self) -> None:
        blocks = assign_temporal_blocks([0, 1, 2, 3, 4, 5], block_size=2)
        np.testing.assert_array_equal(blocks, [0, 0, 1, 1, 2, 2])


class RMSEContributionTests(unittest.TestCase):
    def test_block_contributions_recompute_global_rmse(self) -> None:
        rng = np.random.default_rng(0)
        reference = rng.normal(size=200)
        estimate = reference + rng.normal(scale=0.3, size=200)
        blocks = assign_temporal_blocks(np.arange(200), block_size=25)
        scale = float(np.std(reference))

        contributions = rmse_contributions(reference, estimate, blocks, scale=scale)
        global_rmse = float(np.sqrt(np.mean((reference - estimate) ** 2)))
        self.assertAlmostEqual(recompute_rmse(contributions), global_rmse, places=10)
        self.assertAlmostEqual(
            recompute_nrmse(contributions), global_rmse / scale, places=10
        )

    def test_multivariate_rmse(self) -> None:
        reference = np.arange(24.0).reshape(8, 3)
        estimate = reference + 0.5
        blocks = assign_temporal_blocks(np.arange(8), block_size=4)
        contributions = rmse_contributions(reference, estimate, blocks)
        global_rmse = float(np.sqrt(np.mean((reference - estimate) ** 2)))
        self.assertAlmostEqual(recompute_rmse(contributions), global_rmse, places=10)


class RatioMetricTests(unittest.TestCase):
    def test_block_contributions_recompute_improvement(self) -> None:
        rng = np.random.default_rng(1)
        num = rng.uniform(0.1, 1.0, size=120)
        den = rng.uniform(1.0, 2.0, size=120)
        blocks = assign_temporal_blocks(np.arange(120), block_size=30)
        contributions = ratio_metric_contributions(num, den, blocks)
        expected = 1.0 - num.sum() / den.sum()
        self.assertAlmostEqual(
            recompute_ratio_metric(contributions), expected, places=12
        )


class CorrelationTests(unittest.TestCase):
    def test_block_contributions_recompute_global_pearson(self) -> None:
        rng = np.random.default_rng(2)
        left = rng.normal(size=300)
        right = 0.7 * left + rng.normal(scale=0.5, size=300)
        blocks = assign_temporal_blocks(np.arange(300), block_size=40)
        contributions = correlation_contributions(left, right, blocks)
        expected = float(np.corrcoef(left, right)[0, 1])
        self.assertAlmostEqual(
            recompute_correlation(contributions), expected, places=10
        )


class AggregationTests(unittest.TestCase):
    def test_three_estimates_with_median_and_range(self) -> None:
        result = aggregate_across_units({"fish1": 0.2, "fish2": 0.4, "fish4": 0.9})
        self.assertEqual(result["n_units"], 3)
        self.assertAlmostEqual(result["median"], 0.4)
        self.assertAlmostEqual(result["range"], 0.7)
        self.assertEqual(result["unit_labels"], ["fish1", "fish2", "fish4"])

    def test_handles_nan(self) -> None:
        result = aggregate_across_units([np.nan, np.nan])
        self.assertEqual(result["n_units"], 0)


class ReplicateLabelTests(unittest.TestCase):
    def test_labelled_algorithmic(self) -> None:
        import pandas as pd

        table = pd.DataFrame([{"seed_a": 0, "seed_b": 1, "agreement": 0.8}])
        tagged = label_algorithmic_replicates(table)
        self.assertEqual(tagged["replicate_type"].iloc[0], "algorithmic")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ica_denoising.nested_selection import (
    NestedSelectionConfig,
    make_nested_folds,
    no_selection_baseline,
    select_candidate,
)


class NestedFoldTests(unittest.TestCase):
    def test_inner_folds_never_touch_outer_test_block(self) -> None:
        outer_test = np.arange(60, 80)
        outer_train = np.concatenate([np.arange(0, 46), np.arange(94, 120)])
        folds = make_nested_folds(outer_train, n_inner_splits=3, inner_gap=5)
        self.assertGreaterEqual(len(folds), 2)
        for fold in folds:
            # Outer test indices must never appear in inner selection.
            self.assertEqual(np.intersect1d(fold.inner_train_idx, outer_test).size, 0)
            self.assertEqual(np.intersect1d(fold.inner_val_idx, outer_test).size, 0)
            # All inner indices come from the outer training block.
            self.assertTrue(np.all(np.isin(fold.inner_train_idx, outer_train)))
            self.assertTrue(np.all(np.isin(fold.inner_val_idx, outer_train)))
            # Inner train and inner val are disjoint.
            self.assertEqual(
                np.intersect1d(fold.inner_train_idx, fold.inner_val_idx).size, 0
            )


class SelectCandidateTests(unittest.TestCase):
    def _table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "candidate": "a",
                    "trace_preservation": 0.9,
                    "state_change": 0.1,
                    "artifact_score": 0.2,
                    "behavior_effect": 0.05,
                },
                {
                    "candidate": "b",
                    "trace_preservation": 0.4,  # violates min_trace_preservation
                    "state_change": 0.1,
                    "artifact_score": 0.05,
                    "behavior_effect": 0.5,
                },
                {
                    "candidate": "c",
                    "trace_preservation": 0.8,
                    "state_change": 0.1,
                    "artifact_score": 0.3,
                    "behavior_effect": 0.2,
                },
            ]
        )

    def test_unsupervised_respects_constraints_and_picks_low_artifact(self) -> None:
        config = NestedSelectionConfig(
            policy="unsupervised", min_trace_preservation=0.5
        )
        result = select_candidate(self._table(), outer_fold=0, config=config)
        # b has the lowest artifact score but violates trace preservation, so the
        # eligible minimum-artifact candidate is a.
        self.assertEqual(result.selected_candidate, "a")
        self.assertEqual(result.constraint_status, "satisfied")

    def test_behavior_assisted_uses_absolute_effect(self) -> None:
        config = NestedSelectionConfig(
            policy="behavior_assisted", min_trace_preservation=0.5
        )
        result = select_candidate(self._table(), outer_fold=1, config=config)
        # Among eligible (a, c), c has the larger absolute behavior effect.
        self.assertEqual(result.selected_candidate, "c")

    def test_no_eligible_candidate(self) -> None:
        config = NestedSelectionConfig(
            policy="unsupervised", min_trace_preservation=0.99
        )
        result = select_candidate(self._table(), outer_fold=2, config=config)
        self.assertIsNone(result.selected_candidate)
        self.assertEqual(result.constraint_status, "no_eligible_candidate")

    def test_tie_break_prefers_trace_preservation(self) -> None:
        table = pd.DataFrame(
            [
                {"candidate": "a", "trace_preservation": 0.7, "behavior_effect": 0.3},
                {"candidate": "b", "trace_preservation": 0.9, "behavior_effect": 0.3},
            ]
        )
        config = NestedSelectionConfig(policy="behavior_assisted")
        result = select_candidate(table, outer_fold=0, config=config)
        self.assertEqual(result.selected_candidate, "b")
        self.assertEqual(result.tie_break_reason, "max_trace_preservation_then_name")


class NoSelectionBaselineTests(unittest.TestCase):
    def test_fixed_rule(self) -> None:
        result = no_selection_baseline(outer_fold=3)
        self.assertEqual(result.selected_candidate, "cluster_keep_top_04")
        self.assertEqual(result.constraint_status, "fixed_rule")


if __name__ == "__main__":
    unittest.main()

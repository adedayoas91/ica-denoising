"""Nested outer-evaluation / inner-selection loop (Section 5.4).

The strict runner reports two selection policies:

1. *Unsupervised* - select the candidate using training-only artifact scores
   subject to preregistered trace-preservation constraints.
2. *Behavior-assisted* - select on inner-validation behavior performance subject
   to the same trace and state constraints.

Both policies operate on an *inner* blocked selection loop that lives entirely
inside the outer training block, so the outer test block never enters selection.
The inner objective uses **absolute paired effects** (method minus raw), never a
ratio to a possibly near-zero raw score.

A fixed no-selection baseline (``cluster_keep_top_04``) is also provided.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "NestedSelectionConfig",
    "NestedFold",
    "SelectionResult",
    "make_nested_folds",
    "select_candidate",
    "no_selection_baseline",
    "DEFAULT_BASELINE_RULE",
]

DEFAULT_BASELINE_RULE = "cluster_keep_top_04"


@dataclass(frozen=True)
class NestedSelectionConfig:
    """Configuration for the nested selection loop.

    Attributes:
        policy: ``"unsupervised"`` or ``"behavior_assisted"``.
        n_inner_splits: Number of inner blocked folds inside the training block.
        inner_gap: Purge gap (in frames) between inner train and inner validation.
        min_trace_preservation: Preregistered minimum retained-energy fraction a
            candidate must satisfy to be eligible.
        max_state_change: Preregistered maximum allowed bout-state change rate
            (``inf`` disables the constraint).
        baseline_rule: Selection rule used by the no-selection baseline.
        random_state: Seed for any tie-break randomization (unused by default).
    """

    policy: str = "unsupervised"
    n_inner_splits: int = 3
    inner_gap: int = 14
    min_trace_preservation: float = 0.5
    max_state_change: float = float("inf")
    baseline_rule: str = DEFAULT_BASELINE_RULE
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.policy not in {"unsupervised", "behavior_assisted"}:
            raise ValueError("policy must be 'unsupervised' or 'behavior_assisted'.")
        if self.n_inner_splits < 2:
            raise ValueError("n_inner_splits must be at least 2.")


@dataclass(frozen=True)
class NestedFold:
    """An inner blocked fold contained inside the outer training block.

    Attributes:
        inner_fold: Inner fold id.
        inner_train_idx: Global frame indices used to fit candidates.
        inner_val_idx: Global frame indices used to score candidates.
    """

    inner_fold: int
    inner_train_idx: np.ndarray
    inner_val_idx: np.ndarray


@dataclass(frozen=True)
class SelectionResult:
    """Result of selecting a candidate for one outer fold.

    Attributes:
        outer_fold: Outer fold id.
        policy: Selection policy used.
        selected_candidate: Chosen candidate id (``None`` if none eligible).
        selected_score: Objective score of the chosen candidate.
        constraint_status: ``"satisfied"`` or ``"no_eligible_candidate"``.
        tie_break_reason: Description of how ties were broken.
        candidate_table: Per-candidate scoring/constraint table for this fold.
    """

    outer_fold: int
    policy: str
    selected_candidate: str | None
    selected_score: float
    constraint_status: str
    tie_break_reason: str
    candidate_table: pd.DataFrame


def make_nested_folds(
    outer_train_idx: Sequence[int],
    *,
    n_inner_splits: int,
    inner_gap: int,
) -> tuple[NestedFold, ...]:
    """Build inner blocked folds entirely inside the outer training block.

    Inner folds are carved from contiguous training segments by position, then
    mapped back to global indices. Every returned index is a subset of
    ``outer_train_idx``; the outer test block can therefore never enter
    selection.

    Args:
        outer_train_idx: Global indices of the outer training block.
        n_inner_splits: Number of inner folds.
        inner_gap: Purge gap (frames) excluded around each inner validation block.

    Returns:
        A tuple of :class:`NestedFold`.
    """
    if n_inner_splits < 2:
        raise ValueError("n_inner_splits must be at least 2.")
    if inner_gap < 0:
        raise ValueError("inner_gap must be non-negative.")
    train = np.unique(np.asarray(outer_train_idx, dtype=int))
    if train.size < n_inner_splits:
        raise ValueError("outer_train_idx is too small for the requested inner splits.")
    positions = np.arange(train.size, dtype=int)
    edges = np.linspace(0, train.size, n_inner_splits + 1).round().astype(int)
    folds: list[NestedFold] = []
    for fold_id in range(n_inner_splits):
        start, stop = int(edges[fold_id]), int(edges[fold_id + 1])
        val_positions = positions[start:stop]
        excluded_start = max(0, start - inner_gap)
        excluded_stop = min(train.size, stop + inner_gap)
        train_mask = np.ones(train.size, dtype=bool)
        train_mask[excluded_start:excluded_stop] = False
        train_positions = positions[train_mask]
        if train_positions.size == 0 or val_positions.size == 0:
            continue
        folds.append(
            NestedFold(
                inner_fold=fold_id,
                inner_train_idx=train[train_positions],
                inner_val_idx=train[val_positions],
            )
        )
    if not folds:
        raise ValueError("No usable inner folds; reduce inner_gap or n_inner_splits.")
    return tuple(folds)


def _check_constraints(
    row: pd.Series,
    config: NestedSelectionConfig,
) -> bool:
    preservation = float(row.get("trace_preservation", np.nan))
    if not np.isfinite(preservation) or preservation < config.min_trace_preservation:
        return False
    state_change = float(row.get("state_change", 0.0))
    if np.isfinite(config.max_state_change) and state_change > config.max_state_change:
        return False
    return True


def select_candidate(
    candidate_table: pd.DataFrame,
    *,
    outer_fold: int,
    config: NestedSelectionConfig,
) -> SelectionResult:
    """Select the best candidate for one outer fold under the configured policy.

    The candidate table must contain one row per candidate with columns:

    * ``candidate``: candidate id.
    * ``trace_preservation``: retained-energy fraction (constraint input).
    * ``state_change``: bout-state change rate (constraint input).
    * ``artifact_score`` (unsupervised policy) or ``behavior_effect``
      (behavior-assisted policy): the objective input. ``behavior_effect`` is the
      **absolute paired effect** (method minus raw) on inner validation.

    Args:
        candidate_table: Per-candidate table as described above.
        outer_fold: Outer fold id.
        config: Nested selection configuration.

    Returns:
        A :class:`SelectionResult`.
    """
    objective_column = (
        "artifact_score" if config.policy == "unsupervised" else "behavior_effect"
    )
    if objective_column not in candidate_table.columns:
        raise ValueError(
            f"candidate_table is missing required column {objective_column!r}."
        )
    table = candidate_table.copy().reset_index(drop=True)
    table["constraint_ok"] = table.apply(
        lambda row: _check_constraints(row, config), axis=1
    )
    if config.policy == "unsupervised":
        # Lower artifact score is better; report objective as the negated score so
        # higher objective is always better across policies.
        table["objective"] = -table[objective_column].astype(float)
    else:
        # Absolute paired effect: larger improvement over raw is better.
        table["objective"] = table[objective_column].astype(float)
    table.insert(0, "outer_fold", int(outer_fold))
    table.insert(1, "policy", config.policy)

    eligible = table[table["constraint_ok"] & np.isfinite(table["objective"])]
    if eligible.empty:
        return SelectionResult(
            outer_fold=int(outer_fold),
            policy=config.policy,
            selected_candidate=None,
            selected_score=float("nan"),
            constraint_status="no_eligible_candidate",
            tie_break_reason="none",
            candidate_table=table,
        )
    best_objective = float(eligible["objective"].max())
    tied = eligible[np.isclose(eligible["objective"], best_objective)]
    if len(tied) == 1:
        tie_break_reason = "unique_best_objective"
        chosen = tied.iloc[0]
    else:
        # Deterministic tie-break: higher trace preservation, then lexicographic
        # candidate id.
        ordered = tied.sort_values(
            by=["trace_preservation", "candidate"],
            ascending=[False, True],
        )
        tie_break_reason = "max_trace_preservation_then_name"
        chosen = ordered.iloc[0]
    return SelectionResult(
        outer_fold=int(outer_fold),
        policy=config.policy,
        selected_candidate=str(chosen["candidate"]),
        selected_score=float(chosen["objective"]),
        constraint_status="satisfied",
        tie_break_reason=tie_break_reason,
        candidate_table=table,
    )


def no_selection_baseline(
    *,
    outer_fold: int,
    rule: str = DEFAULT_BASELINE_RULE,
) -> SelectionResult:
    """Return the fixed no-selection baseline result for one outer fold.

    Args:
        outer_fold: Outer fold id.
        rule: Fixed selection rule (defaults to ``cluster_keep_top_04``).

    Returns:
        A :class:`SelectionResult` recording the fixed rule with no scoring.
    """
    table = pd.DataFrame(
        [
            {
                "outer_fold": int(outer_fold),
                "policy": "no_selection",
                "candidate": rule,
                "objective": float("nan"),
                "constraint_ok": True,
            }
        ]
    )
    return SelectionResult(
        outer_fold=int(outer_fold),
        policy="no_selection",
        selected_candidate=rule,
        selected_score=float("nan"),
        constraint_status="fixed_rule",
        tie_break_reason="fixed_rule",
        candidate_table=table,
    )

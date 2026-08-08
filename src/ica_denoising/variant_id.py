"""Canonical variant identity for decomposition, evaluation, and connectivity.

This module provides a single source of truth for the artifact identity used
across the project (Section 4.4 / 5.5). A canonical variant id has the form::

    fastica/cluster_keep_top_04/rank_40/seed_0

The four segments encode, in order, the decomposition ``method``, the component
``selection`` rule, the reconstruction ``rank``, and the random ``seed``. Using
one builder/parser everywhere removes the ambiguity that arises when each
sub-pipeline invents its own naming scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "VariantId",
    "build_variant_id",
    "parse_variant_id",
    "cluster_keep_selection",
    "is_canonical_variant_id",
]

_RANK_PREFIX = "rank_"
_SEED_PREFIX = "seed_"


@dataclass(frozen=True)
class VariantId:
    """Structured representation of a canonical variant identity.

    Attributes:
        method: Decomposition method, e.g. ``fastica``, ``sobi``, ``pca``.
        selection: Component selection rule, e.g. ``cluster_keep_top_04``,
            ``all``, ``ic_quality_nonartifact``.
        rank: Reconstruction rank (number of retained latent dimensions).
        seed: Random seed used by the decomposition / selection.
    """

    method: str
    selection: str
    rank: int
    seed: int

    def to_string(self) -> str:
        """Return the canonical ``/``-joined string form."""
        return build_variant_id(
            method=self.method,
            selection=self.selection,
            rank=self.rank,
            seed=self.seed,
        )


def cluster_keep_selection(keep_cluster_count: int) -> str:
    """Return the canonical selection token for keeping the top clusters.

    Args:
        keep_cluster_count: Number of top-ranked clusters retained.

    Returns:
        A selection token such as ``cluster_keep_top_04``.
    """
    count = int(keep_cluster_count)
    if count < 0:
        raise ValueError("keep_cluster_count must be non-negative.")
    return f"cluster_keep_top_{count:02d}"


def build_variant_id(
    *,
    method: str,
    selection: str,
    rank: int,
    seed: int,
) -> str:
    """Build a canonical variant id string.

    Args:
        method: Decomposition method name (lower-cased on output).
        selection: Component selection rule token (no ``/`` allowed).
        rank: Reconstruction rank (non-negative integer).
        seed: Random seed (non-negative integer).

    Returns:
        Canonical id, e.g. ``fastica/cluster_keep_top_04/rank_40/seed_0``.

    Raises:
        ValueError: If any segment is empty, contains ``/``, or is negative.
    """
    method_token = str(method).strip().lower()
    selection_token = str(selection).strip()
    if not method_token:
        raise ValueError("method must be a non-empty string.")
    if not selection_token:
        raise ValueError("selection must be a non-empty string.")
    for name, token in (("method", method_token), ("selection", selection_token)):
        if "/" in token:
            raise ValueError(f"{name} must not contain '/': {token!r}.")
    rank_value = int(rank)
    seed_value = int(seed)
    if rank_value < 0:
        raise ValueError("rank must be non-negative.")
    if seed_value < 0:
        raise ValueError("seed must be non-negative.")
    return (
        f"{method_token}/{selection_token}/"
        f"{_RANK_PREFIX}{rank_value}/{_SEED_PREFIX}{seed_value}"
    )


def parse_variant_id(variant_id: str) -> VariantId:
    """Parse a canonical variant id back into structured fields.

    Args:
        variant_id: Canonical id string produced by :func:`build_variant_id`.

    Returns:
        The parsed :class:`VariantId`.

    Raises:
        ValueError: If the string is not a canonical variant id.
    """
    parts = str(variant_id).split("/")
    if len(parts) != 4:
        raise ValueError(
            f"Variant id must have 4 '/'-separated segments, got {variant_id!r}."
        )
    method_token, selection_token, rank_token, seed_token = parts
    if not rank_token.startswith(_RANK_PREFIX):
        raise ValueError(
            f"Rank segment must start with {_RANK_PREFIX!r}: {rank_token!r}."
        )
    if not seed_token.startswith(_SEED_PREFIX):
        raise ValueError(
            f"Seed segment must start with {_SEED_PREFIX!r}: {seed_token!r}."
        )
    try:
        rank_value = int(rank_token[len(_RANK_PREFIX) :])
        seed_value = int(seed_token[len(_SEED_PREFIX) :])
    except ValueError as error:
        raise ValueError(f"Could not parse rank/seed from {variant_id!r}.") from error
    return VariantId(
        method=method_token,
        selection=selection_token,
        rank=rank_value,
        seed=seed_value,
    )


def is_canonical_variant_id(variant_id: str) -> bool:
    """Return ``True`` if ``variant_id`` parses as a canonical id."""
    try:
        parse_variant_id(variant_id)
    except ValueError:
        return False
    return True

from __future__ import annotations

import unittest

from ica_denoising.variant_id import (
    VariantId,
    build_variant_id,
    cluster_keep_selection,
    is_canonical_variant_id,
    parse_variant_id,
)


class VariantIdTests(unittest.TestCase):
    def test_build_matches_section_4_4_example(self) -> None:
        variant_id = build_variant_id(
            method="fastica",
            selection=cluster_keep_selection(4),
            rank=40,
            seed=0,
        )
        self.assertEqual(variant_id, "fastica/cluster_keep_top_04/rank_40/seed_0")

    def test_round_trip_parse(self) -> None:
        original = VariantId(
            method="sobi",
            selection="all",
            rank=20,
            seed=3,
        )
        parsed = parse_variant_id(original.to_string())
        self.assertEqual(parsed, original)

    def test_method_is_lowercased(self) -> None:
        variant_id = build_variant_id(
            method="FastICA", selection="all", rank=10, seed=1
        )
        self.assertTrue(variant_id.startswith("fastica/"))

    def test_invalid_segments_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_variant_id(method="a/b", selection="all", rank=1, seed=0)
        with self.assertRaises(ValueError):
            build_variant_id(method="x", selection="all", rank=-1, seed=0)

    def test_parse_rejects_noncanonical(self) -> None:
        self.assertFalse(is_canonical_variant_id("fastica/all/seed_0"))
        self.assertFalse(is_canonical_variant_id("fastica/all/r_40/seed_0"))
        with self.assertRaises(ValueError):
            parse_variant_id("fastica/all/rank_x/seed_0")


if __name__ == "__main__":
    unittest.main()

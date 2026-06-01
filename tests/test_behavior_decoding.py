from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from behavior_decoding import load_trace_variants


class BehaviorDecodingLoadTests(unittest.TestCase):
    def test_load_trace_variants_discovers_method_cleaned_subdirectory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "data"
            dataset_dir.mkdir()
            raw = np.arange(12, dtype=float).reshape(3, 4)
            np.save(dataset_dir / "raw.npy", raw)

            cleaned_root = root / "outputs" / "linear" / "demo" / "demo_data"
            cleaned_dir = cleaned_root / "fastica" / "cleaned"
            cleaned_dir.mkdir(parents=True)
            np.save(cleaned_dir / "cleaned_fastica_demo_data.npy", raw * 0.5)

            variants = load_trace_variants(
                dataset_dir=dataset_dir,
                raw_name="raw.npy",
                cleaned_root=cleaned_root,
                dataset_name="demo_data",
            )

            self.assertEqual([variant.name for variant in variants], ["raw", "fastica/cleaned_fastica_demo_data"])
            np.testing.assert_allclose(variants[1].traces, (raw * 0.5).T)


if __name__ == "__main__":
    unittest.main()

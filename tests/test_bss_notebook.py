from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bss_notebook import (
    DatasetSpec,
    load_bss_outputs,
    output_directory,
    save_bss_outputs,
)


class BSSNotebookTests(unittest.TestCase):
    def test_output_directory_preserves_analysis_kind_path_segments(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)

            output_dir = output_directory(
                "fastica",
                "demo data",
                root,
                analysis_kind="linear/v2a-RSNs/clustered",
                dataset_group="v2a-RSNs",
            )

            self.assertEqual(
                output_dir,
                root
                / "outputs"
                / "linear"
                / "v2a-RSNs"
                / "clustered"
                / "v2a-RSNs"
                / "demo_data"
                / "fastica",
            )

    def test_load_bss_outputs_roundtrips_saved_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = DatasetSpec(
                key="demo/demo_data",
                data_name="demo_data",
                group="demo",
                trace_path=root / "traces.npy",
                sample_rate_hz=10.0,
            )
            traces = np.arange(12, dtype=float).reshape(3, 4)
            np.save(spec.trace_path, traces)

            ic_comps = np.arange(8, dtype=float).reshape(4, 2)
            spectra = np.arange(10, dtype=float).reshape(2, 5)
            mixing = np.array([[1.0, 0.0], [0.5, 0.25], [0.0, 1.0]])
            mean = np.array([0.1, 0.2, 0.3])
            cleaned = ic_comps @ mixing.T + mean
            output_dir = output_directory(
                "fastica",
                spec.data_name,
                root,
                dataset_group=spec.group,
            )
            save_bss_outputs(
                spec=spec,
                method="fastica",
                traces=traces,
                ic_comps=ic_comps,
                IC_ft=spectra,
                A=mixing,
                mean=mean,
                cleaned=cleaned,
                reject_components=[1],
                output_dir=output_dir,
                n_components=2,
                tol=0.0001,
                max_iter=500,
                random_state=0,
            )

            with patch("bss_notebook.dataset_registry", return_value={spec.key: spec}):
                result = load_bss_outputs(spec.key, "fastica", root)

            self.assertEqual(result.output_dir, output_dir)
            np.testing.assert_allclose(result.traces, traces)
            np.testing.assert_allclose(result.ic_comps, ic_comps)
            np.testing.assert_allclose(result.IC_ft, spectra)
            np.testing.assert_allclose(result.A, mixing)
            np.testing.assert_allclose(result.mean, mean)
            np.testing.assert_allclose(result.cleaned, cleaned)


if __name__ == "__main__":
    unittest.main()

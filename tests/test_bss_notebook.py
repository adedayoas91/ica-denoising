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
    cleaned_trace_output_paths,
    cluster_selection_output_path,
    load_bss_decomposition_outputs,
    load_bss_outputs,
    output_directory,
    run_bss_method,
    save_bss_decomposition_outputs,
    save_bss_outputs,
    save_cleaned_trace_output,
    save_cluster_selection_output,
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

    def test_load_bss_decomposition_outputs_roundtrips_ic_artifacts(self) -> None:
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
            output_dir = output_directory(
                "fastica",
                spec.data_name,
                root,
                dataset_group=spec.group,
            )
            saved = save_bss_decomposition_outputs(
                spec=spec,
                method="fastica",
                traces=traces,
                ic_comps=ic_comps,
                IC_ft=spectra,
                A=mixing,
                mean=mean,
                output_dir=output_dir,
                n_components=2,
                tol=0.0001,
                max_iter=500,
                random_state=0,
            )

            with patch("bss_notebook.dataset_registry", return_value={spec.key: spec}):
                result = load_bss_decomposition_outputs(spec.key, "fastica", root)

            self.assertEqual(result.output_dir, output_dir)
            self.assertNotIn("cleaned", saved)
            np.testing.assert_allclose(result.traces, traces)
            np.testing.assert_allclose(result.ic_comps, ic_comps)
            np.testing.assert_allclose(result.IC_ft, spectra)
            np.testing.assert_allclose(result.A, mixing)
            np.testing.assert_allclose(result.mean, mean)

    def test_cleaned_and_cluster_selection_outputs_use_subdirectories(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = DatasetSpec(
                key="demo/demo_data",
                data_name="demo_data",
                group="demo",
                trace_path=root / "traces.npy",
                sample_rate_hz=10.0,
            )
            output_dir = output_directory(
                "fastica",
                spec.data_name,
                root,
                dataset_group=spec.group,
            )

            cleaned_paths = cleaned_trace_output_paths(spec, "fastica", output_dir)
            selection_path = cluster_selection_output_path(spec, "fastica", output_dir)

            self.assertEqual(
                cleaned_paths["cleaned"],
                output_dir / "cleaned" / "cleaned_fastica_demo_data.npy",
            )
            self.assertEqual(
                cleaned_paths["cleaned_metadata"],
                output_dir / "cleaned" / "metadata_cleaned_fastica_demo_data.json",
            )
            self.assertEqual(
                selection_path,
                output_dir / "clusters" / "cluster_selection_fastica_demo_data.json",
            )

    def test_save_cleaned_and_cluster_selection_outputs(self) -> None:
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
            cleaned = traces.T
            output_dir = output_directory(
                "fastica",
                spec.data_name,
                root,
                dataset_group=spec.group,
            )

            cleaned_paths = save_cleaned_trace_output(
                spec=spec,
                method="fastica",
                traces=traces,
                cleaned=cleaned,
                output_dir=output_dir,
                reject_components=[1],
                metadata={"cluster_selection_path": "clusters/demo.json"},
            )
            selection_path = save_cluster_selection_output(
                spec=spec,
                method="fastica",
                output_dir=output_dir,
                selection={
                    "accepted_components": [0],
                    "rejected_components": [1],
                    "clusters": [{"cluster": 0, "ic_indices": [0, 1]}],
                },
            )

            self.assertTrue(cleaned_paths["cleaned"].exists())
            self.assertTrue(cleaned_paths["cleaned_metadata"].exists())
            self.assertTrue(selection_path.exists())
            np.testing.assert_allclose(np.load(cleaned_paths["cleaned"]), cleaned.T)

    def test_load_bss_outputs_can_consume_decomposition_only_artifacts(self) -> None:
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
            output_dir = output_directory(
                "fastica",
                spec.data_name,
                root,
                dataset_group=spec.group,
            )
            save_bss_decomposition_outputs(
                spec=spec,
                method="fastica",
                traces=traces,
                ic_comps=ic_comps,
                IC_ft=spectra,
                A=mixing,
                mean=mean,
                output_dir=output_dir,
                n_components=2,
                tol=0.0001,
                max_iter=500,
                random_state=0,
            )

            with patch("bss_notebook.dataset_registry", return_value={spec.key: spec}):
                result = load_bss_outputs(spec.key, "fastica", root)

            self.assertNotIn("cleaned", result.saved_paths)
            np.testing.assert_allclose(result.cleaned, ic_comps @ mixing.T + mean)

    def test_load_bss_outputs_roundtrips_saved_cleaned_trace(self) -> None:
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

            self.assertIn("cleaned", result.saved_paths)
            self.assertEqual(result.saved_paths["cleaned"].parent.name, "cleaned")
            np.testing.assert_allclose(result.cleaned, cleaned)

    def test_run_bss_method_forwards_pca_components_to_sobi(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            project_root = Path(__file__).resolve().parents[1]
            spec = DatasetSpec(
                key="demo/demo_data",
                data_name="demo_data",
                group="demo",
                trace_path=tmp_root / "traces.npy",
                sample_rate_hz=10.0,
            )
            rng = np.random.default_rng(4)
            traces = rng.normal(size=(5, 80))
            np.save(spec.trace_path, traces)

            with patch("bss_notebook.dataset_registry", return_value={spec.key: spec}):
                result = run_bss_method(
                    spec.key,
                    "sobi",
                    n_components=5,
                    pca_components=2,
                    project_root=project_root,
                    max_iter=1,
                )

            self.assertEqual(result.ic_comps.shape, (80, 2))
            self.assertEqual(result.A.shape, (5, 2))
            self.assertEqual(result.cleaned.shape, traces.T.shape)


if __name__ == "__main__":
    unittest.main()

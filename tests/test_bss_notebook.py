from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from ica_denoising.bss_notebook import (
    DatasetSpec,
    cleaned_trace_output_paths,
    cluster_selection_output_path,
    dataset_registry,
    load_bss_decomposition_outputs,
    load_bss_outputs,
    output_directory,
    resolve_bss_component_selection,
    run_bss_method,
    save_bss_decomposition_outputs,
    save_bss_outputs,
    save_cleaned_trace_output,
    save_cluster_selection_output,
)


class BSSNotebookTests(unittest.TestCase):
    def test_dataset_registry_handles_missing_v2a_root(self) -> None:
        with TemporaryDirectory() as tmp:
            registry = dataset_registry(Path(tmp))

            self.assertFalse(any(key.startswith("v2a-RSNs/") for key in registry))

    def test_dataset_registry_discovers_direct_recording_directories(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "data" / "v2a-RSNs" / "220119_F2_run11"
            run_dir.mkdir(parents=True)
            trace = run_dir / "220119_F2_F2_run11_cells_fluorescence_signals.npy"
            tail = run_dir / "220119_F2_F2_run11_tail_angle.npy"
            np.save(trace, np.ones((3, 20)))
            np.save(tail, np.ones(20))

            registry = dataset_registry(root)
            spec = registry["v2a-RSNs/220119_F2_run11_fluorescence"]

            self.assertEqual(spec.trace_path, trace)
            self.assertEqual(spec.tail_angle_path, tail)

    def test_dataset_registry_falls_back_to_legacy_root_and_prefers_matching_names(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "data" / "v2a-RSNs" / "new_data_09112022" / "220210_F1_run6"
            run_dir.mkdir(parents=True)
            correct_trace = run_dir / "220210_F1_F1_run6_cells_fluorescence_signals.npy"
            legacy_trace = run_dir / "220127_F4_F4_run2_after_dec_cells_fluorescence_signals.npy"
            correct_tail = run_dir / "220210_F1_F1_run6_tail_angle.npy"
            legacy_tail = run_dir / "220127_F4_F4_run2_after_dec_tail_angle.npy"
            np.save(correct_trace, np.ones((3, 20)))
            np.save(legacy_trace, np.zeros((3, 20)))
            np.save(correct_tail, np.ones(100))
            np.save(legacy_tail, np.zeros(100))

            registry = dataset_registry(root)
            spec = registry["v2a-RSNs/220210_F1_run6_fluorescence"]

            self.assertEqual(spec.trace_path, correct_trace)
            self.assertEqual(spec.tail_angle_path, correct_tail)
            self.assertEqual(spec.recording_id, "220210_F1_run6")
            self.assertEqual(spec.fish_id, "220210_F1")
            self.assertEqual(spec.run_id, "run6")
            self.assertEqual(spec.modality, "fluorescence")

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

            with patch("ica_denoising.bss_notebook.dataset_registry", return_value={spec.key: spec}):
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

            with patch("ica_denoising.bss_notebook.dataset_registry", return_value={spec.key: spec}):
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

            with patch("ica_denoising.bss_notebook.dataset_registry", return_value={spec.key: spec}):
                result = load_bss_outputs(spec.key, "fastica", root)

            self.assertIn("cleaned", result.saved_paths)
            self.assertEqual(result.saved_paths["cleaned"].parent.name, "cleaned")
            np.testing.assert_allclose(result.cleaned, cleaned)

    def test_sobi_and_jade_auto_select_pca_rank_from_variance_threshold(self) -> None:
        rng = np.random.default_rng(11)
        latent = rng.normal(size=(2, 120))
        mixing = rng.normal(size=(8, 2))
        traces = mixing @ latent + 0.01 * rng.normal(size=(8, 120))

        fastica = resolve_bss_component_selection(traces, "fastica")
        sobi = resolve_bss_component_selection(
            traces,
            "sobi",
            pca_variance_threshold=0.95,
        )
        jade = resolve_bss_component_selection(
            traces,
            "jade",
            pca_variance_threshold=0.95,
        )

        self.assertEqual(fastica.n_components, min(traces.shape))
        self.assertIsNone(fastica.pca_components)
        self.assertLess(sobi.pca_components, fastica.n_components)
        self.assertLess(jade.pca_components, fastica.n_components)
        self.assertGreaterEqual(sobi.pca_explained_variance_ratio, 0.95)
        self.assertGreaterEqual(jade.pca_explained_variance_ratio, 0.95)
        self.assertEqual(sobi.component_selection_mode, "pca_variance_threshold")

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

            with patch("ica_denoising.bss_notebook.dataset_registry", return_value={spec.key: spec}):
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

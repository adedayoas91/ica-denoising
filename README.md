# ICA/BSS denoising experiments

This project contains post-extraction denoising experiments for calcium-imaging traces.

## Behavior decoding judge

The notebook [notebooks/behavior-decoding/behavior_decoding_pipeline.ipynb](notebooks/behavior-decoding/behavior_decoding_pipeline.ipynb) evaluates whether BSS-cleaned traces preserve behavior-decodable structure from the raw extracted traces.

The canonical V2a template currently targets:

- raw traces: `data/v2a-RSNs/220119_F2_run11/220119_F2_F2_run11_cells_fluorescence_signals.npy`
- tail angle: `data/v2a-RSNs/220119_F2_run11/220119_F2_F2_run11_tail_angle.npy`
- recording notebooks: `notebooks/v2a-RSNs/220119_F2_run11/`

Dataset-specific output paths are derived from the dataset registry. The strict
evaluation notebook builds calcium-frame behavior targets from the high-rate tail
angle, then runs:

- within-version decoding: train and test on the same trace version
- raw-to-clean transfer: train on raw traces and test on cleaned traces
- clean-to-raw transfer: train on cleaned traces and test on raw traces
- block-shuffled null decoding

Outputs are written under `outputs/behavior_decoding/`, which is intentionally not versioned.

### Further analysis helpers

`src/behavior_decoding.py` also exposes notebook-friendly helpers for comparing
cleaned traces before plotting or interpreting decoding results:

```python
from behavior_decoding import (
    make_analysis_scorecard,
    rank_representative_neurons,
    summarize_trace_preservation,
)

trace_summary = summarize_trace_preservation(variants, reference_name="raw")
neurons_to_plot = rank_representative_neurons(variants, reference_name="raw", n=8)
scorecard = make_analysis_scorecard(variants, summary, reference_name="raw").scorecard
```

The trace summary reports global and per-neuron preservation metrics such as
Pearson correlation, normalized RMSE, mean absolute change, and variance ratio.
The neuron ranking selects traces with high raw variance and visible cleaning
effects, which is useful for choosing example panels. The scorecard joins these
trace-preservation metrics to within-version behavior-decoding scores.

## BSS methods in the existing ICA workflow

The existing FastICA workflow can now switch methods through `bss_dec` in `src/core/ica_utils.py`.
The `Fluo*.ipynb` notebooks expose the same switch through the
`DECOMPOSITION_METHOD` variable near the top of each notebook. The
`infomax.ipynb` notebook uses the same interface and defaults to
`DECOMPOSITION_METHOD = "infomax"` on the 844-ROI dataset.

```python
from ica_utils import bss_dec, reconstruct_bss

ic_comps, IC_ft, A, mean = bss_dec(
    traces,
    n_comps=n,
    t=0.0001,
    max_=500,
    method="sobi",  # "fastica", "infomax", "sobi", or "jade"
)

cleaned = reconstruct_bss(ic_comps, A, mean, reject=[0, 3, 6])
```

The returned objects match the original `ica_dec` convention:

- `ic_comps`: frames by components
- `IC_ft`: components by frames
- `A`: neurons by components
- `mean`: neurons

That means existing notebook reconstruction code such as `np.dot(ic_, A.T) + mean` still works. SOBI and JADE can be slower than FastICA when `n_comps` is large, so start with a smaller component count while testing.

## Linear BSS notebooks

Use the canonical all-method notebooks for new runs:

- `notebooks/v2a-RSNs/220119_F2_run11/decompositions.ipynb`
- `notebooks/v2a-RSNs/220119_F2_run11/cluster_clean_reconstruct.ipynb`
- `notebooks/v2a-RSNs/220119_F2_run11/behavior_decoding.ipynb`
- `notebooks/behavior-decoding/ic_quality_pipeline_example.ipynb` (optional IC audit)
- `notebooks/motorneurons/linear_methods.ipynb`

The V2a notebooks use `src/bss_notebook.py` for dataset selection, NaN cleanup,
and output paths. Copy the recording folder and change `DATASET_KEY` for another
dataset. Run the V2a notebooks in this order:

1. `decompositions.ipynb`
2. `cluster_clean_reconstruct.ipynb`
3. `behavior_decoding.ipynb`

The decomposition notebook exposes:

```python
METHODS_TO_RUN = list(BSS_METHODS)  # or ["fastica"], ["infomax"], ["sobi"], ["jade"]
N_COMPONENTS = None  # FastICA/Infomax use the full trace count by default
PCA_COMPONENTS = None  # explicit SOBI/JADE override
SOBI_JADE_PCA_VARIANCE_THRESHOLD = 0.95
SAVE_DECOMPOSITION_OUTPUTS = True
```

With these defaults, FastICA and Infomax fit `min(n_neurons, n_frames)`
components. SOBI and JADE fit a PCA-reduced rank chosen as the smallest number
of PCs whose cumulative explained variance is at least 95%. The selected
`pca_components`, threshold, and achieved explained-variance fraction are written
to each method's decomposition metadata and to `decomposition_summary.csv`.

Saved outputs use:

```text
outputs/linear/<dataset_group>/<data_name>/<method>/
```

with files named `components_<method>_<data_name>.npy`, `spectra_<method>_<data_name>.npy`,
`mixing_<method>_<data_name>.npy`, `mean_<method>_<data_name>.npy`, and
`metadata_<method>_<data_name>.json`. `cluster_clean_reconstruct.ipynb` consumes
those decomposition files, writes cluster selections under each method's
`clusters/` directory, writes cleaned traces under each method's `cleaned/`
directory, and writes dataset-level manuscript candidates under
`outputs/linear/<dataset_group>/<data_name>/figures/`. The cluster-selection JSON
records each cluster's IC membership, accepted/rejected clusters, accepted/rejected
ICs, ranking-band settings, cleaned-output paths, and figure paths.
`behavior_decoding.ipynb` then loads those cleaned traces and cluster-selection
JSON files, runs the chained full-recording supervised and causal-state behavior
decoding analyses, and writes downstream descriptive outputs under
`outputs/behavior_decoding/<dataset_group>/<data_name>/`. Its final strict
evaluation section writes to `outputs/evaluation/<dataset_group>/<data_name>/`
and refits preprocessing within folds for leakage-safe manuscript claims. The
top-level `outputs/linear/` and
`outputs/nonlinear/` directories are kept separate so nonlinear methods can use
the same grouping later without mixing analysis families.

## Leakage-safe evaluation pipeline

`src/evaluation_runner.py` implements the strict evaluation boundary used for the
evaluation analyses. It discovers V2a recordings from their run
directories, writes file-hash provenance, fits every learned preprocessing
operation on training frames only, and transforms held-out blocks with frozen
parameters.

The strict runner includes:

- train-only FastICA, Infomax, SOBI, and JADE transforms;
- segment-aware SOBI lag covariances and Welch PSD features;
- low-frequency, high-frequency, random, energy-matched, and all-component
  selection controls;
- PCA, random-subspace deletion, causal low-pass, and raw baselines;
- fold-local bout thresholds and behavioral decoders;
- optional behavior-target sensitivity grids over bout thresholds and smoothed
  vigor targets;
- block-shuffle and circular-shift behavior nulls that preserve temporal
  structure better than frame-wise permutation;
- out-of-fold causal-state embeddings, normalized dynamic MSE, persistence,
  and mean-state references;
- out-of-fold latent-to-behavior correlation tables for causal-state diagnostics;
- strict causal-sufficiency probes that test whether extra neural history
  improves behavior prediction beyond the learned latent state;
- optional held-out pseudo-artifact probes for configured candidate artifact
  centers;
- matched linear and shallow nonlinear transition sensitivity;
- held-out trace correlation, NRMSE, retained energy, and spectral-power
  preservation;
- leakage-audit, convergence, cluster-stability, BPI-ablation,
  block-uncertainty, temporal-dependence, and provenance tables.

List discovered datasets and audit their provenance before running experiments:

```bash
uv run python src/evaluation_runner.py --list-datasets
uv run python src/evaluation_runner.py --provenance-only
```

Run one recording with the frozen evaluation configuration:

```bash
uv run python src/evaluation_runner.py \
  --dataset-key v2a-RSNs/220119_F2_run11_fluorescence \
  --config configs/evaluation.example.json
```

Use `configs/evaluation.smoke.json` for a quick end-to-end integration
check before launching the full matrix.

Run the frozen configuration across every discovered V2a fluorescence
recording and write recording-level aggregate tables:

```bash
uv run python src/evaluation_runner.py \
  --all-v2a \
  --modality fluorescence \
  --config configs/evaluation.example.json
```

Outputs are written below
`outputs/evaluation/<dataset-group>/<data-name>/`. The run manifest is the
source of truth for the executed settings. Use a reduced configuration while
debugging; the example configuration intentionally expands the full
selection-order and baseline matrix.

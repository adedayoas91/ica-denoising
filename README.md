# ICA/BSS denoising experiments

This project contains post-extraction denoising experiments for calcium-imaging traces.

## Behavior decoding judge

The notebook [notebooks/behavior-decoding/behavior_decoding_pipeline.ipynb](notebooks/behavior-decoding/behavior_decoding_pipeline.ipynb) evaluates whether BSS-cleaned traces preserve behavior-decodable structure from the raw extracted traces.

The first implemented dataset target is:

- raw traces: `data/v2a-RSNs/new_run2_844ROI/fluo_signals_no_NaN.npy`
- cleaned traces: `data/v2a-RSNs/new_run2_844ROI/cleanedNew_*.npy`
- tail angle: `data/v2a-RSNs/220127_F4_run2_tail_angle.npy`

The notebook builds calcium-frame behavior targets from the high-rate tail angle, then runs:

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

- `notebooks/v2a-RSNs/linear_methods.ipynb`
- `notebooks/motorneurons/linear_methods.ipynb`

Both notebooks use `src/bss_notebook.py` for imports, dataset selection, method
selection, NaN cleanup, and output paths. Set `DATASET_KEY` to one of the
registered datasets shown in the notebook, then set:

```python
METHODS_TO_RUN = list(BSS_METHODS)  # or ["fastica"], ["infomax"], ["sobi"], ["jade"]
SAVE_OUTPUTS = True
```

Saved outputs use:

```text
outputs/linear/<dataset_group>/<data_name>/<method>/
```

with files named `cleaned_<method>_<data_name>.npy`,
`components_<method>_<data_name>.npy`, `spectra_<method>_<data_name>.npy`,
`mixing_<method>_<data_name>.npy`, `mean_<method>_<data_name>.npy`, and
`metadata_<method>_<data_name>.json`. The top-level `outputs/linear/` and
`outputs/nonlinear/` directories are kept separate so nonlinear methods can use
the same grouping later without mixing analysis families.

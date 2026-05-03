# ICA/BSS denoising experiments

This project contains post-extraction denoising experiments for calcium-imaging traces.

## Behavior decoding judge

The notebook [notebooks/behavior_decoding_pipeline.ipynb](notebooks/behavior_decoding_pipeline.ipynb) evaluates whether BSS-cleaned traces preserve behavior-decodable structure from the raw extracted traces.

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

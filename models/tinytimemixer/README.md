# Local Tiny Time Mixer

This directory contains the public IBM Granite TTM implementation needed for
component-codebook training.  It is imported from `models.tinytimemixer`; no
runtime checkout or `third_party` source directory is used.

`train_ttm_component_codebooks.py` loads `ibm-research/ttm-research-r2`, freezes
`TinyTimeMixerForPrediction.backbone`, streams the parquet `x` column, and trains
one two-stage RQ-VAE per requested component (`trend`, `seasonal`, `residual`).
It writes portable codebooks under `checkpoints/ttm_component_rqvae/`.

Run from the repository root:

```bash
GPU_ID=0 bash script/train_ttm_component_codebooks.sh
```

For a safe smoke test, add `--max_train_batches 2 --max_val_batches 2` to the
Python command.  The parquet reader streams Arrow record batches, avoiding the
full-corpus memory load used by older codebook trainers.

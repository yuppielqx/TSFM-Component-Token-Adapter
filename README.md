# TimeFKA: Component Tokens for Frozen Chronos-Bolt

This repository contains the **Chronos-Bolt** implementation of TimeFKA.
It learns shared trend and seasonal codebooks from historical contexts, then
trains a dataset-specific adapter while keeping the backbone, codebooks, and
forecasting head frozen.

## Method

1. **Stage I — learn component codebooks.** Decompose each pretraining context,
   encode its components with the frozen backbone, and learn a residual
   quantizer and reconstruction heads for each component.
2. **Stage II — train the forecasting adapter.** Quantize each input component
   using the frozen codebooks. Combine trainable code, component, RQ-level,
   and patch-position embeddings into a condition for each input patch.
   Fuse it with the original encoder states through a gated residual adapter,
   then use the frozen Chronos-Bolt decoder and prediction head.

The default setting uses trend and seasonal components, two RQ levels, 256
entries per level, 512 input observations, and a native 64-step forecast.
Longer horizons use recursive median forecasts with a sliding 512-step context.
Codes are recomputed from the current context at each rollout step. Decoded
codebook curves are interpretation readouts; they are not fed into forecasting.

## Installation

Use Python 3.11 or 3.12. Install a PyTorch build appropriate for your hardware,
then install the remaining requirements in a fresh environment:

```bash
git clone https://github.com/yuppielqx/TSFM-Component-Token-Adapter.git
cd TSFM-Component-Token-Adapter
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The local CPU validation used PyTorch 2.5.1 and Transformers 4.49.0. Keep
Transformers on 4.49.x because this implementation uses T5 internal APIs.
The implementation does not require AutoGluon or GluonTS. Model loading uses
local checkpoint directories or Hugging Face; download S3-hosted files locally
before use.

## Prepare model and data

Download [Chronos-Bolt-base](https://huggingface.co/amazon/chronos-bolt-base):

```bash
python scripts/download_backbone.py --output-dir checkpoints/base
```

Place benchmark CSVs under `datasets/`; each CSV must contain a `date` column
followed by numeric variables, including the target column (`OT` by default).
The four ETT files are available from the
[ETDataset project](https://github.com/zhouhaoyi/ETDataset).

```text
checkpoints/base/{config.json,model.safetensors}
datasets/ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv
datasets/exchange_rate/exchange_rate.csv
datasets/weather/weather.csv
datasets/electricity/electricity.csv
datasets/traffic/traffic.csv
datasets/pretrain/retrieval_database_512.parquet
```

For Stage I, prepare a parquet file with an `x` column: one length-512
historical context per row. The trainers use only `x`, not future values or
stored embeddings. They split rows deterministically into 70%/10%/20% subsets;
the corpus preparation must establish provenance and prevent evaluation-data
overlap. A parquet construction script and a verified public download for this
pretraining corpus are **not included**.

## Stage I: train the codebooks

```bash
GPU_ID=0 bash script/train_component_codebooks.sh
```

This creates `trend_prototypes_rqvae.pt` and
`seasonal_prototypes_rqvae.pt` under `checkpoints/component_rqvae/`.
The launcher specifies two RQ levels and 256 entries per level. Residual
component codebooks remain available as an explicit ablation:
`bash script/train_component_codebooks.sh --components "trend seasonal residual"`.

## Stage II: train and evaluate an adapter

```bash
DATASET=ETTh2 GPU_ID=0 bash script/run_chronos_bolt.sh
```

This trains a native 64-step adapter and evaluates horizons 96/192/336/720.
Use `DATASET=all` to process all eight datasets sequentially. Configuration
overrides include `PYTHON_BIN`, `BACKBONE_DIR`, `CODEBOOK_DIR`, `TRAIN_EPOCHS`,
`BATCH_SIZE`, `NUM_WORKERS`, `SEED`, and `HORIZONS`:

```bash
DATASET=ETTh2 HORIZONS="96" BATCH_SIZE=32 NUM_WORKERS=0 \
  bash script/run_chronos_bolt.sh
```

The new launcher uses the ETT-specific splits (`ett_h` / `ett_m`) for ETT and
the 70%/10%/20% custom split for the remaining datasets. Older Git history
contains launchers that used `--data custom` for ETT; those runs have different
splits and are not interchangeable with this recipe. See
[evaluation details](docs/REPRODUCIBILITY.md) before comparing metrics.

The public repository currently supplies training and evaluation code. Trained
component codebooks, dataset adapters, and exact per-result experiment manifests
do **not** have verified public download links here. Reproducing a reported
checkpoint therefore requires those artifacts in addition to the code and
benchmark CSVs; do not interpret this repository as a complete weight release.

## Code map

| File | Purpose |
|---|---|
| `train_component_prototypes.py` | Stage I component codebook training |
| `train_component_token_adapter.py` | Stage II dataset-specific adapter training |
| `eval_component_token_adapter_rollout.py` | Sliding-context recursive forecasting and MSE/MAE |
| `models/chronos_bolt/ChronosBolt.py` | Chronos-Bolt backbone and native pipeline |
| `models/chronos_bolt/ChronosBoltPrototype.py` | Residual quantizer and Stage I model |
| `models/chronos_bolt/ComponentTemporalTokenAdapter.py` | Token construction, aggregation, and fusion |
| `utils/decomposition.py` | Moving-average trend and phase-average seasonality |
| `data_provider/` | Channel-independent benchmark windows |
| `tests/` | Offline import, data, and model smoke checks |

## Acknowledgments and license

The implementation builds on
[Chronos](https://github.com/amazon-science/chronos-forecasting) and forecasting
utilities inherited from [TS-RAG](https://github.com/UConn-DSIS/TS-RAG).
See [LICENSE](LICENSE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and
the license notices in the corresponding source files. Pretrained models and
datasets retain their respective upstream terms.

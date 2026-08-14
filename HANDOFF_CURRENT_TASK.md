# Current Task Handoff

## Remote project

- SSH: `qingxiangliu@100.114.238.103`
- Project root: `/data/qingxiangliu/TS-RAG`
- Python environment: `/home/qingxiangliu/miniforge3/envs/tsrag` (Python 3.9)
- Do all remote work from the project root.

## Goal and method

This project studies component-token adapters for time-series foundation models (TSFMs).

1. Decompose a 512-point input window into trend, seasonal, and residual components.
2. Freeze the selected TSFM backbone.
3. Train a TSFM-specific component RQ-VAE codebook on pretrained parquet windows.
4. Freeze backbone and codebooks.
5. Train one downstream adapter per dataset and evaluate rollout horizons 96/192/336/720.

Codebooks cannot be shared across TSFMs because hidden-state representation spaces differ.

## Model layout

- ChronosBolt: `models/chronos_bolt/`
- TimesFM 2.5 PyTorch: `models/timesfm/`
- Chronos-2 official source: `models/chronos2/chronos/`
- Chronos-2 RQ-VAE wrapper: `models/chronos2/Chronos2Prototype.py`

Base weights:

- ChronosBolt: `checkpoints/chronosbolt-base/`
  - A compatibility symlink `checkpoints/base -> chronosbolt-base` remains.
- TimesFM 2.5 200M PyTorch: `checkpoints/timesfm-2.5-200m-pytorch/`
- Chronos-2: `checkpoints/chronos-2/`

## Completed: TimesFM codebooks

Training implementation:

- `models/timesfm/TimesFMPrototype.py`
- `train_timesfm_component_prototypes.py`

Codebooks are complete in `checkpoints/timesfm_component_rqvae/`:

| Component | Test loss |
|---|---:|
| trend | 0.787194 |
| seasonal | 1.097521 |
| residual | 1.399087 |

Each checkpoint uses 2 residual-quantization stages, 256 entries per stage, and a TimesFM-specific 1280-dimensional patch representation. No TimesFM codebook training process remains running.

## Current next task: Chronos-2 codebooks

Chronos-2 official checkpoint and source have been installed and verified.

- Weight: `checkpoints/chronos-2/model.safetensors` (~456 MB)
- Official model: 120M parameters, 12 encoder layers, hidden dim 768.
- Maximum context length is 8192, but this experiment intentionally always uses 512 input points.
- Input patch size is 16, so 512 points produce 32 context patches.
- The copied official source has minimal Python 3.9 type-annotation compatibility edits only; model computation is unchanged.
- Local weight loading succeeded:
  `CHRONOS2_LOAD_OK d_model=768 layers=12 context_length=8192 input_patch_size=16`.

Chronos-2 codebook implementation:

- `models/chronos2/Chronos2Prototype.py`
- `train_chronos2_component_prototypes.py`

The wrapper extracts only the 32 context-patch encoder states and excludes Chronos-2 REG/future tokens. It uses a frozen backbone, 2-stage 256-entry RQ-VAE, and 16-point reconstruction heads. A CPU forward/backward smoke test passed:

```
CHRONOS2_RQVAE_SMOKE_OK
codes: (1, 32, 2)
backbone frozen: True
```

### Launch Chronos-2 codebook training

The three components are independent and should run in parallel on GPUs 1, 2, and 3 if available:

```bash
mkdir -p logs checkpoints/chronos2_component_rqvae

nohup /home/qingxiangliu/miniforge3/envs/tsrag/bin/python train_chronos2_component_prototypes.py \
  --gpu_loc 1 --components trend   > logs/chronos2_rqvae_trend_gpu1.log 2>&1 < /dev/null &

nohup /home/qingxiangliu/miniforge3/envs/tsrag/bin/python train_chronos2_component_prototypes.py \
  --gpu_loc 2 --components seasonal   > logs/chronos2_rqvae_seasonal_gpu2.log 2>&1 < /dev/null &

nohup /home/qingxiangliu/miniforge3/envs/tsrag/bin/python train_chronos2_component_prototypes.py \
  --gpu_loc 3 --components residual   > logs/chronos2_rqvae_residual_gpu3.log 2>&1 < /dev/null &
```

Expected outputs:

- `checkpoints/chronos2_component_rqvae/trend_prototypes_rqvae.pt`
- `checkpoints/chronos2_component_rqvae/seasonal_prototypes_rqvae.pt`
- `checkpoints/chronos2_component_rqvae/residual_prototypes_rqvae.pt`

Each has an accompanying `.meta.json` file.

## Important historical status

- Per-dataset ChronosBolt adapter/evaluation is the default experimental approach.
- A universal/general adapter branch was tried and stopped because zero-shot results were poor. Do not restart it unless explicitly requested.
- Existing remote worktree is dirty. Preserve unrelated changes; do not reset/revert broadly.
- `third_party/timesfm/` is a full official TimesFM source reference. The removed `third_party/chronos-forecasting/` clone is not needed because required Chronos-2 source is already in `models/chronos2/`.

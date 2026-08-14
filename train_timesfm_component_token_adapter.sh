#!/usr/bin/env bash
set -euo pipefail

# Example: CUDA_VISIBLE_DEVICES=0 bash train_timesfm_component_token_adapter.sh
PYTHON_BIN=${PYTHON_BIN:-/home/qingxiangliu/miniforge3/envs/tsrag/bin/python}
GPU_LOC=${GPU_LOC:-0}
DATASET=${DATASET:-ETTh1}
PERIOD=${PERIOD:-24}

"${PYTHON_BIN}" train_timesfm_component_token_adapter.py \
  --model_id "${DATASET}_timesfm_component_adapter" \
  --data ett_h \
  --root_path ./datasets/ETT-small/ \
  --data_path "${DATASET}.csv" \
  --features M --freq h --percent 100 \
  --seq_len 512 --pred_len 64 --label_len 48 \
  --pretrained_model_path ./checkpoints/timesfm-2.5-200m-pytorch \
  --component_codebook_dir ./checkpoints/timesfm_component_rqvae \
  --components "trend seasonal residual" \
  --period "${PERIOD}" --trend_kernel 25 \
  --train_epochs 20 --batch_size 64 --num_workers 4 \
  --learning_rate 1e-4 --weight_decay 0.01 --patience 3 \
  --gpu_loc "${GPU_LOC}" \
  --checkpoint_dir "./checkpoints/timesfm_component_token_adapter_${DATASET}" \
  --results_dir "./results/timesfm_component_token_adapter_${DATASET}"

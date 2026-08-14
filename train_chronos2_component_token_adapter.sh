#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/qingxiangliu/miniforge3/envs/tsrag/bin/python}
GPU_LOC=${GPU_LOC:-0}
DATASET=${DATASET:-ETTh1}
PERIOD=${PERIOD:-24}

"${PYTHON_BIN}" train_chronos2_component_token_adapter.py \
  --model_id "${DATASET}_chronos2_component_adapter" \
  --data ett_h --root_path ./datasets/ETT-small/ --data_path "${DATASET}.csv" \
  --features M --freq h --percent 100 \
  --seq_len 512 --pred_len 64 --label_len 48 \
  --pretrained_model_path ./checkpoints/chronos-2 \
  --component_codebook_dir ./checkpoints/chronos2_component_rqvae \
  --components "trend seasonal" --period "${PERIOD}" --trend_kernel 25 \
  --train_epochs 20 --batch_size 64 --num_workers 4 \
  --learning_rate 1e-4 --weight_decay 0.01 --patience 3 \
  --gpu_loc "${GPU_LOC}" \
  --checkpoint_dir "./checkpoints/chronos2_component_token_adapter_${DATASET}" \
  --results_dir "./results/chronos2_component_token_adapter_${DATASET}"

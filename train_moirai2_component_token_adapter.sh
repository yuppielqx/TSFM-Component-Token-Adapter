#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="${PYTHONPATH:-.}:models/moirai2"

PYTHON_BIN="${PYTHON_BIN:-/home/qingxiangliu/miniforge3/envs/timexl/bin/python}"
GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-ETTh1}"
PERIOD="${PERIOD:-24}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_moirai2_component_token_adapter.py \
  --model_id "${DATASET}_moirai2_component_adapter" --data ett_h \
  --root_path ./datasets/ETT-small/ --data_path "${DATASET}.csv" --features M --freq h --percent 100 \
  --seq_len 512 --pred_len 64 --label_len 48 --components "trend seasonal" --period "${PERIOD}" \
  --component_codebook_dir ./checkpoints/moirai2_component_rqvae --train_epochs 20 --batch_size 32 --num_workers 4 \
  --learning_rate 1e-4 --weight_decay 0.01 --patience 3 --gpu_loc 0 \
  --checkpoint_dir "./checkpoints/moirai2_component_token_adapter_${DATASET}" \
  --results_dir "./results/moirai2_component_token_adapter_${DATASET}" "$@"

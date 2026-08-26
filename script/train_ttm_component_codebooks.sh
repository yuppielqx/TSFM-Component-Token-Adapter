#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/qingxiangliu/miniforge3/envs/tsrag/bin/python}"
GPU_ID="${GPU_ID:-0}"
TTM_MODEL_DIR="${TTM_MODEL_DIR:-./checkpoints/ttm-research-r2}"
"${PYTHON_BIN}" train_ttm_component_codebooks.py \
  --gpu_loc "${GPU_ID}" \
  --model_id "${TTM_MODEL_DIR}" \
  --parquet_path ./datasets/pretrain/retrieval_database_512.parquet \
  --output_dir ./checkpoints/ttm_component_rqvae \
  --context_length 512 \
  --components "trend seasonal residual" \
  --period 24 \
  --n_proto 256 --n_rq_stages 2 --beta 0.25 \
  --train_epochs 20 --batch_size 128 --learning_rate 1e-4 --patience 3

#!/usr/bin/env bash
# Train the frozen-backbone component RQ-VAE codebooks used by
# Component-Token Adapter.
#
# Run from the repository root:
#   bash script/train_component_codebooks.sh

set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.

PYTHON_BIN="${PYTHON_BIN:-/home/qingxiangliu/miniforge3/envs/tsrag/bin/python}"
GPU_ID="${GPU_ID:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

"${PYTHON_BIN}" train_component_prototypes.py \
  --components "trend seasonal residual" \
  --pretrained_model_path ./checkpoints/base/ \
  --parquet_path ./datasets/pretrain/retrieval_database_512.parquet \
  --output_dir ./checkpoints/component_rqvae/ \
  --checkpoint_dir ./checkpoints/component_proto_ckpt/ \
  --period 24 \
  --trend_kernel 25 \
  --n_proto 256 \
  --n_rq_stages 2 \
  --beta 0.25 \
  --mask_ratio 0.1 \
  --train_epochs 20 \
  --patience 3 \
  --batch_size 128 \
  --num_workers 4 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  --gpu_loc 0 \
  "$@"


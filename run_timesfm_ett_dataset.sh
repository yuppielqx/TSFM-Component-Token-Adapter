#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:?Usage: run_timesfm_ett_dataset.sh DATASET}
PYTHON_BIN=${PYTHON_BIN:-/home/qingxiangliu/miniforge3/envs/tsrag/bin/python}
GPU_LOC=${GPU_LOC:-0}

case "${DATASET}" in
  ETTh1|ETTh2)
    DATA_KIND=ett_h
    FREQ=h
    ;;
  ETTm1|ETTm2)
    DATA_KIND=ett_m
    FREQ=t
    ;;
  *)
    echo "Unsupported ETT dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

CHECKPOINT_DIR="./checkpoints/timesfm_component_token_adapter_${DATASET}_noresid"
RESULTS_DIR="./results/timesfm_component_token_adapter_${DATASET}_noresid"

"${PYTHON_BIN}" train_timesfm_component_token_adapter.py \
  --model_id "${DATASET}_timesfm_component_adapter_noresid" \
  --data "${DATA_KIND}" \
  --root_path ./datasets/ETT-small/ \
  --data_path "${DATASET}.csv" \
  --features M --freq "${FREQ}" --percent 100 \
  --seq_len 512 --pred_len 64 --label_len 48 \
  --pretrained_model_path ./checkpoints/timesfm-2.5-200m-pytorch \
  --component_codebook_dir ./checkpoints/timesfm_component_rqvae \
  --components "trend seasonal" \
  --period 24 --trend_kernel 25 \
  --token_dropout 0.1 --gate_bias_init -2.0 \
  --train_epochs 20 --batch_size 64 --num_workers 4 \
  --learning_rate 1e-4 --weight_decay 0.01 --grad_clip 1.0 --patience 3 \
  --gpu_loc "${GPU_LOC}" --seed 2021 \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --results_dir "${RESULTS_DIR}"

"${PYTHON_BIN}" eval_timesfm_component_token_adapter.py \
  --model_id "${DATASET}_timesfm_component_adapter_noresid_rollout" \
  --data "${DATA_KIND}" \
  --root_path ./datasets/ETT-small/ \
  --data_path "${DATASET}.csv" \
  --features M --freq "${FREQ}" --percent 100 \
  --seq_len 512 --pred_len 720 --label_len 48 \
  --horizons "96 192 336 720" --rollout_chunk 64 \
  --pretrained_model_path ./checkpoints/timesfm-2.5-200m-pytorch \
  --component_codebook_dir ./checkpoints/timesfm_component_rqvae \
  --adapter_checkpoint "${CHECKPOINT_DIR}/adapter_checkpoint.pt" \
  --components "trend seasonal" \
  --period 24 --trend_kernel 25 \
  --batch_size 64 --num_workers 4 \
  --gpu_loc "${GPU_LOC}" --seed 2021 \
  --results_dir "${RESULTS_DIR}"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.
PYTHON_BIN="${PYTHON_BIN:-/home/qingxiangliu/miniforge3/envs/tsrag/bin/python}"
DATASET="${DATASET:?Set DATASET to ETTh1, ETTh2, ETTm1, or ETTm2}"
HORIZON="${HORIZON:?Set HORIZON to 96, 192, 336, or 720}"
GPU_ID="${GPU_ID:-0}"

case "${DATASET}" in
  ETTh1|ETTh2) DATA=ett_h; FREQ=h; PERIOD=24; FREQUENCY_TOKEN=7 ;;
  ETTm1|ETTm2) DATA=ett_m; FREQ=t; PERIOD=96; FREQUENCY_TOKEN=5 ;;
  *) echo "Unsupported ETT dataset: ${DATASET}" >&2; exit 2 ;;
esac

case "${HORIZON}" in
  96) MODEL_PATH=./checkpoints/ttm-research-r2 ;;
  192|336|720) MODEL_PATH="./checkpoints/ttm-research-r2-${HORIZON}" ;;
  *) echo "Unsupported official TTM horizon: ${HORIZON}" >&2; exit 2 ;;
esac

exec "${PYTHON_BIN}" train_ttm_component_token_adapter.py \
  --model_id "${DATASET}_ttm_component_adapter" \
  --data "${DATA}" --root_path ./datasets/ETT-small/ --data_path "${DATASET}.csv" \
  --features M --freq "${FREQ}" --percent 100 \
  --seq_len 512 --pred_len "${HORIZON}" --label_len 48 \
  --pretrained_model_path "${MODEL_PATH}" \
  --component_codebook_dir ./checkpoints/ttm_component_rqvae \
  --components "trend seasonal" --period "${PERIOD}" --frequency_token "${FREQUENCY_TOKEN}" \
  --trend_kernel 25 --train_epochs "${TRAIN_EPOCHS:-20}" \
  --batch_size "${BATCH_SIZE:-64}" --num_workers 4 \
  --learning_rate 1e-4 --weight_decay 0.01 --patience 3 \
  --gpu_loc "${GPU_ID}" \
  --checkpoint_dir "./checkpoints/ttm_component_token_adapter_${DATASET}_h${HORIZON}_noresid" \
  --results_dir "./results/ttm_component_token_adapter_${DATASET}_h${HORIZON}_noresid"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.
PYTHON_BIN="${PYTHON_BIN:-/home/qingxiangliu/miniforge3/envs/tsrag/bin/python}"
DATASET="${DATASET:?Set DATASET}"
HORIZON="${HORIZON:?Set HORIZON to 96, 192, 336, or 720}"
GPU_ID="${GPU_ID:-0}"

case "${DATASET}" in
  Exchange-rate)
    ROOT=./datasets/exchange_rate/; FILE=exchange_rate.csv
    FREQ=h; PERIOD=24; FREQUENCY_TOKEN=7 ;;
  Weather)
    ROOT=./datasets/weather/; FILE=weather.csv
    FREQ=t; PERIOD=144; FREQUENCY_TOKEN=4 ;;
  Electricity)
    ROOT=./datasets/electricity/; FILE=electricity.csv
    FREQ=t; PERIOD=96; FREQUENCY_TOKEN=5 ;;
  Traffic)
    ROOT=./datasets/traffic/; FILE=traffic.csv
    FREQ=h; PERIOD=24; FREQUENCY_TOKEN=7 ;;
  *) echo "Unsupported dataset: ${DATASET}" >&2; exit 2 ;;
esac

case "${HORIZON}" in
  96) MODEL_PATH=./checkpoints/ttm-research-r2 ;;
  192|336|720) MODEL_PATH="./checkpoints/ttm-research-r2-${HORIZON}" ;;
  *) echo "Unsupported horizon: ${HORIZON}" >&2; exit 2 ;;
esac

SAFE_NAME="${DATASET//-/_}"
exec "${PYTHON_BIN}" train_ttm_component_token_adapter.py \
  --model_id "${SAFE_NAME}_ttm_h${HORIZON}_component_adapter" \
  --data custom --root_path "${ROOT}" --data_path "${FILE}" \
  --features M --freq "${FREQ}" --percent 100 \
  --seq_len 512 --pred_len "${HORIZON}" --label_len 48 \
  --pretrained_model_path "${MODEL_PATH}" \
  --component_codebook_dir ./checkpoints/ttm_component_rqvae \
  --components "trend seasonal" --period "${PERIOD}" \
  --frequency_token "${FREQUENCY_TOKEN}" --trend_kernel 25 \
  --train_epochs "${TRAIN_EPOCHS:-20}" --batch_size "${BATCH_SIZE:-64}" \
  --num_workers 4 --learning_rate 1e-4 --weight_decay 0.01 --patience 3 \
  --gpu_loc "${GPU_ID}" \
  --checkpoint_dir "./checkpoints/ttm_component_token_adapter_${SAFE_NAME}_h${HORIZON}_noresid" \
  --results_dir "./results/ttm_component_token_adapter_${SAFE_NAME}_h${HORIZON}_noresid"

#!/usr/bin/env bash
# Train a 64-step Component-Token Adapter and run recursive rollout
# evaluation for the benchmark datasets.
#
# Run from the repository root:
#   bash script/run_component_token_adapter_all_datasets.sh
#
# This is intentionally sequential. Electricity and Traffic are large and
# can take many hours. Use DATASET=ETTh1 (or another name) to run one dataset.

set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.

PYTHON_BIN="${PYTHON_BIN:-/home/qingxiangliu/miniforge3/envs/tsrag/bin/python}"
GPU_ID="${GPU_ID:-0}"
DATASET_FILTER="${DATASET:-all}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"

run_one() {
  local name="$1" root="$2" file="$3" freq="$4" period="$5" ckpt="$6" result="$7" rollout_result="$8"

  if [[ "${DATASET_FILTER}" != "all" && "${DATASET_FILTER}" != "${name}" ]]; then
    return
  fi

  echo "============================================================"
  echo "Dataset: ${name}"
  echo "Training checkpoint: ${ckpt}"
  echo "============================================================"

  "${PYTHON_BIN}" train_component_token_adapter.py \
    --model_id "${name}_component_token_adapter" \
    --data custom \
    --root_path "${root}" \
    --data_path "${file}" \
    --features M \
    --freq "${freq}" \
    --percent 100 \
    --seq_len 512 \
    --pred_len 64 \
    --label_len 48 \
    --pretrained_model_path ./checkpoints/base/ \
    --component_codebook_dir ./checkpoints/component_rqvae/ \
    --components "trend seasonal" \
    --period "${period}" \
    --train_epochs "${TRAIN_EPOCHS}" \
    --batch_size 64 \
    --num_workers 4 \
    --gpu_loc 0 \
    --seed 2021 \
    --checkpoint_dir "${ckpt}" \
    --results_dir "${result}"

  test -f "${ckpt}/adapter_checkpoint.pt"

  echo "Starting recursive rollout for ${name} ..."
  "${PYTHON_BIN}" eval_component_token_adapter_rollout.py \
    --model_id "${name}_component_token_adapter" \
    --data custom \
    --root_path "${root}" \
    --data_path "${file}" \
    --features M \
    --freq "${freq}" \
    --percent 100 \
    --seq_len 512 \
    --pred_len 720 \
    --label_len 48 \
    --horizons "96 192 336 720" \
    --pretrained_model_path ./checkpoints/base/ \
    --component_codebook_dir ./checkpoints/component_rqvae/ \
    --adapter_checkpoint "${ckpt}/adapter_checkpoint.pt" \
    --components "trend seasonal" \
    --period "${period}" \
    --batch_size 64 \
    --num_workers 4 \
    --gpu_loc 0 \
    --seed 2021 \
    --results_dir "${rollout_result}"
}

run_one ETTh1 ./datasets/ETT-small/ ETTh1.csv h 24 \
  ./checkpoints/component_token_adapter_full_p100_noresid \
  ./results/component_token_adapter_full_p100_noresid \
  ./results/component_token_adapter_rollout_full_noresid

run_one ETTh2 ./datasets/ETT-small/ ETTh2.csv h 24 \
  ./checkpoints/component_token_adapter_ETTh2_p100_noresid \
  ./results/component_token_adapter_ETTh2_p100_noresid \
  ./results/component_token_adapter_rollout_ETTh2_noresid

run_one ETTm1 ./datasets/ETT-small/ ETTm1.csv t 96 \
  ./checkpoints/component_token_adapter_ETTm1_p100_noresid \
  ./results/component_token_adapter_ETTm1_p100_noresid \
  ./results/component_token_adapter_rollout_ETTm1_noresid

run_one ETTm2 ./datasets/ETT-small/ ETTm2.csv t 96 \
  ./checkpoints/component_token_adapter_ETTm2_p100_noresid \
  ./results/component_token_adapter_ETTm2_p100_noresid \
  ./results/component_token_adapter_rollout_ETTm2_noresid

run_one Exchange-rate ./datasets/exchange_rate/ exchange_rate.csv h 24 \
  ./checkpoints/component_token_adapter_exchange_rate_p100_noresid \
  ./results/component_token_adapter_exchange_rate_p100_noresid \
  ./results/component_token_adapter_rollout_exchange_rate_noresid

run_one Weather ./datasets/weather/ weather.csv t 144 \
  ./checkpoints/component_token_adapter_weather_p100_noresid \
  ./results/component_token_adapter_weather_p100_noresid \
  ./results/component_token_adapter_rollout_weather_noresid

run_one Electricity ./datasets/electricity/ electricity.csv t 96 \
  ./checkpoints/component_token_adapter_electricity_p100_noresid \
  ./results/component_token_adapter_electricity_p100_noresid \
  ./results/component_token_adapter_rollout_electricity_noresid

run_one Traffic ./datasets/traffic/ traffic.csv t 96 \
  ./checkpoints/component_token_adapter_traffic_p100_noresid \
  ./results/component_token_adapter_traffic_p100_noresid \
  ./results/component_token_adapter_rollout_traffic_noresid


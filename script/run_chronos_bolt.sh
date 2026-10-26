#!/usr/bin/env bash
# Train a native 64-step adapter, then evaluate recursive median forecasts.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
BACKBONE_DIR="${BACKBONE_DIR:-./checkpoints/base}"
CODEBOOK_DIR="${CODEBOOK_DIR:-./checkpoints/component_rqvae}"
DATASET="${DATASET:-ETTh2}"
HORIZONS="${HORIZONS:-96 192 336 720}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-2021}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

read -r -a requested_horizons <<< "${HORIZONS}"
if [[ ${#requested_horizons[@]} -eq 0 ]]; then
  echo "HORIZONS must contain at least one positive integer" >&2
  exit 2
fi
max_horizon=0
for horizon in "${requested_horizons[@]}"; do
  if [[ ! "${horizon}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid horizon: ${horizon}" >&2
    exit 2
  fi
  if (( horizon > max_horizon )); then max_horizon=${horizon}; fi
done

run_one() {
  local name="$1" kind root file freq period
  case "${name}" in
    ETTh1|ETTh2) kind=ett_h; root=./datasets/ETT-small; file="${name}.csv"; freq=h; period=24 ;;
    ETTm1|ETTm2) kind=ett_m; root=./datasets/ETT-small; file="${name}.csv"; freq=min; period=96 ;;
    Exchange-rate) kind=custom; root=./datasets/exchange_rate; file=exchange_rate.csv; freq=h; period=24 ;;
    Weather) kind=custom; root=./datasets/weather; file=weather.csv; freq=min; period=144 ;;
    Electricity) kind=custom; root=./datasets/electricity; file=electricity.csv; freq=h; period=24 ;;
    Traffic) kind=custom; root=./datasets/traffic; file=traffic.csv; freq=h; period=24 ;;
    *) echo "Unknown dataset: ${name}" >&2; exit 2 ;;
  esac
  local checkpoint="${CHECKPOINT_ROOT:-./checkpoints/chronos_bolt}/${name}"
  local result="${RESULTS_ROOT:-./results/chronos_bolt}/${name}"
  local common=(
    --model_id "${name}_component_token_adapter" --data "${kind}"
    --root_path "${root}" --data_path "${file}" --features M --freq "${freq}"
    --percent 100 --seq_len 512 --label_len 48
    --pretrained_model_path "${BACKBONE_DIR}" --component_codebook_dir "${CODEBOOK_DIR}"
    --components "trend seasonal" --n_proto 256 --n_rq_stages 2
    --period "${period}" --trend_kernel 25 --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}" --seed "${SEED}" --gpu_loc 0
  )
  "${PYTHON_BIN}" train_component_token_adapter.py "${common[@]}" \
    --pred_len 64 --train_epochs "${TRAIN_EPOCHS}" \
    --checkpoint_dir "${checkpoint}" --results_dir "${result}/direct64"
  test -f "${checkpoint}/adapter_checkpoint.pt"
  "${PYTHON_BIN}" eval_component_token_adapter_rollout.py "${common[@]}" \
    --pred_len "${max_horizon}" --horizons "${HORIZONS}" \
    --adapter_checkpoint "${checkpoint}/adapter_checkpoint.pt" \
    --results_dir "${result}/rollout"
}

if [[ "${DATASET}" == all ]]; then
  for dataset in ETTh1 ETTh2 ETTm1 ETTm2 Exchange-rate Weather Electricity Traffic; do
    run_one "${dataset}"
  done
else
  run_one "${DATASET}"
fi

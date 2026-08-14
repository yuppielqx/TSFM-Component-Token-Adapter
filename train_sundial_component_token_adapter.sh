#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/qingxiangliu/miniforge3/envs/tsrag/bin/python}
DATASET=${1:-ETTh1}

case "${DATASET}" in
  ETTh1|ETTh2) DATA=ett_h; FREQ=h ;;
  ETTm1|ETTm2) DATA=ett_m; FREQ=t ;;
  *) echo "Unsupported ETT dataset: ${DATASET}" >&2; exit 2 ;;
esac

CHECKPOINT_DIR="./checkpoints/sundial_component_token_adapter_${DATASET}_noresid"
RESULTS_DIR="./results/sundial_component_token_adapter_${DATASET}_noresid"

PYTHONPATH=. "${PYTHON_BIN}" train_sundial_component_token_adapter.py \
  --model_id "${DATASET}_sundial_component_adapter_noresid" \
  --data "${DATA}" --root_path ./datasets/ETT-small/ --data_path "${DATASET}.csv" \
  --features M --target OT --freq "${FREQ}" --seq_len 512 --pred_len 96 \
  --components "trend seasonal" --checkpoint_dir "${CHECKPOINT_DIR}" \
  --results_dir "${RESULTS_DIR}" "${@:2}"

PYTHONPATH=. "${PYTHON_BIN}" eval_sundial_component_token_adapter.py \
  --model_id "${DATASET}_sundial_component_adapter_noresid_eval" \
  --data "${DATA}" --root_path ./datasets/ETT-small/ --data_path "${DATASET}.csv" \
  --features M --target OT --freq "${FREQ}" --seq_len 512 --pred_len 720 \
  --horizons "96 192 336 720" --components "trend seasonal" \
  --adapter_checkpoint "${CHECKPOINT_DIR}/adapter_checkpoint.pt" \
  --results_dir "${RESULTS_DIR}" "${@:2}"

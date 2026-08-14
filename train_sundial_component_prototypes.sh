#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/qingxiangliu/miniforge3/envs/tsrag/bin/python}
GPU_LIST=${GPU_LIST:-0,1,3}
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
COMPONENTS=(trend seasonal residual)

mkdir -p logs/sundial_component_rqvae checkpoints/sundial_component_rqvae

pids=()
for index in "${!COMPONENTS[@]}"; do
  component=${COMPONENTS[$index]}
  gpu=${GPUS[$((index % ${#GPUS[@]}))]}
  CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=. "${PYTHON_BIN}" train_sundial_component_prototypes.py \
    --components "${component}" \
    --gpu_loc 0 \
    "$@" \
    > "logs/sundial_component_rqvae/${component}.log" 2>&1 &
  pids+=("$!")
  echo "Started ${component} on physical GPU ${gpu}, pid=${pids[-1]}"
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"

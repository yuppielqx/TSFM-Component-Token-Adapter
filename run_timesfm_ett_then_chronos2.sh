#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/data/qingxiangliu/TS-RAG
PYTHON_BIN=/home/qingxiangliu/miniforge3/envs/tsrag/bin/python
cd "${PROJECT_DIR}"

# Keep GPU 2 untouched because it is occupied by another workload.
(
  env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 bash run_timesfm_ett_dataset.sh ETTm1 \
    > logs/timesfm_ETTm1_noresid.log 2>&1
) &
PID_ETTM1=$!

(
  env PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 bash run_timesfm_ett_dataset.sh ETTm2 \
    > logs/timesfm_ETTm2_noresid.log 2>&1
) &
PID_ETTM2=$!

(
  env PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 bash run_timesfm_ett_dataset.sh ETTh1 \
    > logs/timesfm_ETTh1_noresid.log 2>&1
  env PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 bash run_timesfm_ett_dataset.sh ETTh2 \
    > logs/timesfm_ETTh2_noresid.log 2>&1
) &
PID_ETTH=$!

wait "${PID_ETTM1}"
wait "${PID_ETTM2}"
wait "${PID_ETTH}"

COMMON_ARGS=(
  --pretrained_model_path ./checkpoints/chronos-2
  --parquet_path ./datasets/pretrain/retrieval_database_512.parquet
  --output_dir ./checkpoints/chronos2_component_rqvae
  --period 24 --trend_kernel 25
  --n_proto 256 --n_rq_stages 2 --beta 0.25 --mask_ratio 0.1
  --train_epochs 20 --batch_size 64 --num_workers 4
  --learning_rate 1e-4 --weight_decay 0.01 --grad_clip 1.0 --patience 3
  --gpu_loc 0 --seed 2021
)

env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" \
  train_chronos2_component_prototypes.py --components trend "${COMMON_ARGS[@]}" \
  > logs/chronos2_codebook_trend_after_ett.log 2>&1 &
PID_TREND=$!

env PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" \
  train_chronos2_component_prototypes.py --components seasonal "${COMMON_ARGS[@]}" \
  > logs/chronos2_codebook_seasonal_after_ett.log 2>&1 &
PID_SEASONAL=$!

env PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 "${PYTHON_BIN}" \
  train_chronos2_component_prototypes.py --components residual "${COMMON_ARGS[@]}" \
  > logs/chronos2_codebook_residual_after_ett.log 2>&1 &
PID_RESIDUAL=$!

wait "${PID_TREND}"
wait "${PID_SEASONAL}"
wait "${PID_RESIDUAL}"

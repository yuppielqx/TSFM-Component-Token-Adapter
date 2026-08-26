#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
QUEUE="${QUEUE:?Set QUEUE to 0, 1, or 2}"
GPU_ID="${GPU_ID:?Set GPU_ID}"

case "${QUEUE}" in
  0) jobs=("ETTh1 192" "ETTh1 336" "ETTh1 720" "ETTh2 192") ;;
  1) jobs=("ETTh2 336" "ETTh2 720" "ETTm1 192" "ETTm1 336") ;;
  2) jobs=("ETTm1 720" "ETTm2 192" "ETTm2 336" "ETTm2 720") ;;
  *) echo "Unsupported queue: ${QUEUE}" >&2; exit 2 ;;
esac

mkdir -p logs/ttm_official
for job in "${jobs[@]}"; do
  read -r dataset horizon <<< "${job}"
  echo "Starting ${dataset} horizon=${horizon} on GPU ${GPU_ID}"
  DATASET="${dataset}" HORIZON="${horizon}" GPU_ID="${GPU_ID}" \
    bash script/run_ttm_ett_adapter.sh \
    > "logs/ttm_official/${dataset}_h${horizon}.log" 2>&1
done

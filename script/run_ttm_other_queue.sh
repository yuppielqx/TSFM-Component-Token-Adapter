#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
QUEUE="${QUEUE:?Set QUEUE to 0 or 1}"
GPU_ID="${GPU_ID:?Set GPU_ID}"

case "${QUEUE}" in
  0) jobs=("Exchange-rate 96" "Exchange-rate 192" "Exchange-rate 336" "Exchange-rate 720" "Weather 96" "Weather 192" "Weather 336" "Weather 720") ;;
  1) jobs=("Electricity 96" "Electricity 192" "Electricity 336" "Electricity 720" "Traffic 96" "Traffic 192" "Traffic 336" "Traffic 720") ;;
  *) echo "Unsupported queue: ${QUEUE}" >&2; exit 2 ;;
esac

mkdir -p logs/ttm_other
for job in "${jobs[@]}"; do
  read -r dataset horizon <<< "${job}"
  echo "Starting ${dataset} horizon=${horizon} on GPU ${GPU_ID}"
  DATASET="${dataset}" HORIZON="${horizon}" GPU_ID="${GPU_ID}" \
    bash script/run_ttm_other_adapter.sh \
    > "logs/ttm_other/${dataset}_h${horizon}.log" 2>&1
done

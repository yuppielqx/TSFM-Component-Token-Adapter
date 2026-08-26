#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
for dataset in ETTh1 ETTh2 ETTm1 ETTm2; do
  for horizon in 96 192 336 720; do
    DATASET="${dataset}" HORIZON="${horizon}" GPU_ID="${GPU_ID:-0}" \
      bash script/run_ttm_ett_adapter.sh
  done
done

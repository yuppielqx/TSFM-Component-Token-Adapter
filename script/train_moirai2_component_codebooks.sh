#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHONPATH=third_party/uni2ts/src /home/qingxiangliu/miniforge3/envs/timexl/bin/python train_moirai2_component_prototypes.py "$@"

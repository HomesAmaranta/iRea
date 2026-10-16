#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/train_onerec.sh beauty [--gin "..."]
REPO_ROOT="$(dirname -- "$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")")"
cd "$REPO_ROOT"

dataset="${1:-beauty}"
if [[ $# -gt 0 ]]; then shift; fi
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

"$PYTHON" -c 'import torch; import sys; n = torch.cuda.device_count(); sys.exit(0 if n == 4 else f"Expected exactly 4 visible CUDA GPUs, found {n}. Edit CUDA_VISIBLE_DEVICES in this script to select four available GPUs.")'

exec "$PYTHON" -u -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 \
  -m trainers.qwen3_trainer config/onerec.gin \
  --split "$dataset" \
  --gin "train_lcrec.wandb_logging=False" \
  "$@"

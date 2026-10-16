#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/train_onerec.sh beauty [--gpus 4,5,6,7] [--gin "..."]
REPO_ROOT="$(dirname -- "$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")")"
cd "$REPO_ROOT"

dataset="${1:-beauty}"
if [[ $# -gt 0 ]]; then shift; fi
gpu_ids="4,5,6,7"
trainer_args=()
while (( $# > 0 )); do
  case "$1" in
    --gpus)
      if [[ $# -lt 2 ]]; then echo "--gpus requires four comma-separated GPU IDs" >&2; exit 2; fi
      gpu_ids=$2
      shift 2
      ;;
    --gpus=*) gpu_ids=${1#*=}; shift ;;
    *) trainer_args+=("$1"); shift ;;
  esac
done
if [[ ! "$gpu_ids" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]]; then
  echo "--gpus requires exactly four comma-separated GPU IDs: $gpu_ids" >&2
  exit 2
fi
IFS=, read -r -a gpu_list <<< "$gpu_ids"
declare -A seen_gpus=()
for gpu in "${gpu_list[@]}"; do
  gpu=$((10#$gpu))
  if [[ -n "${seen_gpus[$gpu]:-}" ]]; then echo "Duplicate GPU ID: $gpu" >&2; exit 2; fi
  seen_gpus[$gpu]=1
done
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="$gpu_ids"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

"$PYTHON" -c 'import torch; import sys; n = torch.cuda.device_count(); sys.exit(0 if n == 4 else f"Expected exactly 4 visible CUDA GPUs, found {n}. Use --gpus to select four available GPUs.")'

exec "$PYTHON" -u -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 \
  -m trainers.qwen3_trainer config/onerec.gin \
  --split "$dataset" \
  --gin "train_lcrec.wandb_logging=False" \
  "${trainer_args[@]}"

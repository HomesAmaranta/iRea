#!/usr/bin/env bash
set -euo pipefail

cd /opt/tiger/project/iRea

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 {tiger|sasrec|hstu|lcrec|onerec} {beauty|sports|toys} [--gin ...]" >&2
  exit 2
fi
model=$1
dataset=$2
shift 2

case "$model" in
  lcrec|onerec)
    exec bash "scripts/train_${model}.sh" "$dataset" "$@"
    ;;
  tiger|sasrec|hstu)
    trainer=trainers.general_trainer
    binding=train
    ;;
  *)
    echo "Unsupported model: $model" >&2
    exit 2
    ;;
esac

exec python -u -m "$trainer" "config/${model}.gin" \
  --split "$dataset" \
  --gin "${binding}.wandb_logging=False" \
  "$@"

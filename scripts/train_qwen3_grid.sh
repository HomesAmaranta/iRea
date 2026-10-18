#!/usr/bin/env bash
set -euo pipefail

# Edit these arrays to select the experiment grid.
models=(lcrec)
datasets=(beauty sports toys)
align_targets=(item latent quantized)
mse_loss_weights=(5)
align_items=(current)
align_sources=(last_inp)
learning_rates=(5e-4)
align_loss_types=(cos gram cosgram) # gram: bias-free Linear(d,d)
cos_loss_weights=(0.5) # cosgram: alpha*cos + (1-alpha)*gram
gpu_ids="0,1,2,3"

# Pass the four visible GPUs above to each model's bash launcher.
# Runs execute sequentially, without a task scheduler.
cd /opt/tiger/project/iRea

# Usage: bash scripts/train_qwen3_grid.sh [--dry-run] [--gin "..."]
dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi

# Validate the full grid before starting any training.
python - "${cos_loss_weights[@]}" <<'PY'
import sys
try:
    valid = len(sys.argv) > 1 and all(0 <= float(x) <= 1 for x in sys.argv[1:])
except ValueError:
    valid = False
if not valid:
    sys.exit("cos_loss_weights must contain finite numbers in [0, 1]")
PY
for model in "${models[@]}"; do
  case "$model" in lcrec|onerec) ;; *) echo "Unsupported model: $model" >&2; exit 2 ;; esac
done
for dataset in "${datasets[@]}"; do
  case "${dataset,,}" in beauty|sports|toys) ;; *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;; esac
done
for target in "${align_targets[@]}"; do
  case "$target" in item|latent|quantized) ;; *) echo "Unsupported target: $target" >&2; exit 2 ;; esac
done
for direction in "${align_items[@]}"; do
  case "$direction" in current|next) ;; *) echo "Unsupported direction: $direction" >&2; exit 2 ;; esac
done
for source in "${align_sources[@]}"; do
  case "$source" in last_inp|last_out|mean_out) ;; *) echo "Unsupported source: $source" >&2; exit 2 ;; esac
done
for loss in "${align_loss_types[@]}"; do
  case "$loss" in cos|gram|cosgram) ;; *) echo "Unsupported loss: $loss" >&2; exit 2 ;; esac
done
for number in "${mse_loss_weights[@]}" "${learning_rates[@]}"; do
  if [[ ! "$number" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
    echo "Invalid nonnegative numeric value: $number" >&2
    exit 2
  fi
done
for lr in "${learning_rates[@]}"; do
  if [[ "$lr" =~ ^0+([.]0+)?([eE][-+]?[0-9]+)?$ ]]; then
    echo "Learning rate must be positive" >&2
    exit 2
  fi
done

declare -A seen=()
failed_runs=()
success_count=0
count=0
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    dataset=${dataset,,}
    for weight in "${mse_loss_weights[@]}"; do
      printf -v weight '%g' "$weight"
      for target in "${align_targets[@]}"; do
        if [[ "$model" == "onerec" && "$target" == "latent" && "$weight" != 0 ]]; then
          printf 'Skip: onerec %s latent weight=%s (RQ-KMeans has no latent target)\n' "$dataset" "$weight" >&2
          continue
        fi
        for direction in "${align_items[@]}"; do
          for source in "${align_sources[@]}"; do
            for lr in "${learning_rates[@]}"; do
              printf -v lr '%g' "$lr"
              for align_loss_type in "${align_loss_types[@]}"; do
                ratios=(0.5)
                if [[ "$align_loss_type" == cosgram && "$weight" != 0 ]]; then
                  ratios=("${cos_loss_weights[@]}")
                fi
                for cos_loss_weight in "${ratios[@]}"; do
                  run_target=$target
                  run_direction=$direction
                  run_source=$source
                  suffix="${target}_${direction}_${align_loss_type}${weight}_${source}_lr${lr}"
                  if [[ "$weight" == 0 ]]; then
                    # Alignment settings have no effect on a weight-zero baseline.
                    run_target=item
                    run_direction=current
                    run_source=last_inp
                    suffix="base_lr${lr}"
                  fi
                  key="${model}_${dataset}_${suffix}"
                  if [[ "$align_loss_type" == cosgram && "$weight" != 0 ]]; then
                    key+="_cw${cos_loss_weight}"
                  fi
                  if [[ -n "${seen[$key]:-}" ]]; then continue; fi
                  seen[$key]=1
                  # The trainer appends its alignment suffix; only add the LR here.
                  output_root=out
                  if [[ "$weight" != 0 && "$align_loss_type" != cos ]]; then
                    output_root=out_linear2gram
                  fi
                  save_dir="${output_root}/${model}/amazon/${dataset}/sft_lr${lr}"
                  cmd=(
                    bash "./scripts/train_${model}.sh" "$dataset"
                    --gpus "$gpu_ids"
                    "$@"
                    --gin "train_lcrec.align_target='${run_target}'"
                    --gin "train_lcrec.align_item='${run_direction}'"
                    --gin "train_lcrec.mse_loss_weight=${weight}"
                    --gin "train_lcrec.align_source='${run_source}'"
                    --gin "train_lcrec.align_loss_type='${align_loss_type}'"
                    --gin "train_lcrec.cos_loss_weight=${cos_loss_weight}"
                    --gin "train_lcrec.learning_rate=${lr}"
                    --gin "train_lcrec.save_dir_root='${save_dir}'"
                  )
                  if "$dry_run"; then
                    printf '%q ' "${cmd[@]}"
                    printf '\n'
                  else
                    printf 'Run: %s\n' "$key"
                    if "${cmd[@]}"; then
                      success_count=$((success_count + 1))
                    else
                      status=$?
                      # Explicit interruption should stop the grid, not launch another run.
                      if [[ "$status" == 130 || "$status" == 143 ]]; then
                        exit "$status"
                      fi
                      failed_runs+=("${key} (exit=${status})")
                      printf 'Failed: %s (exit=%d); continuing\n' "$key" "$status" >&2
                    fi
                  fi
                  count=$((count + 1))
                done
              done
            done
          done
        done
      done
    done
  done
done
printf 'Runs: %d\n' "$count"
if ! "$dry_run"; then
  printf 'Succeeded: %d; Failed: %d\n' "$success_count" "${#failed_runs[@]}"
  if (( ${#failed_runs[@]} > 0 )); then
    printf 'Failed runs:\n' >&2
    printf '  %s\n' "${failed_runs[@]}" >&2
    exit 1
  fi
fi

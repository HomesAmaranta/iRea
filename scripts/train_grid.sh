#!/usr/bin/env bash
set -euo pipefail

cd /opt/tiger/project/iRea

datasets=( beauty   )
models=(hstu tiger)
align_targets=( item  )
align_items=(current next)
mse_loss_weights=(5 0)
mse_loss_modes=(last)
learning_rates=(5e-4 )
align_loss_types=(cosgram ) # cos: MLP cosine; gram: bias-free Linear(d,d) Gram
cos_loss_weights=(0.5) # cosgram only: alpha*cos + (1-alpha)*gram
nitp_weights=(0) # Independent auxiliary loss weight; 0 disables NITP.
nitp_target_layers=(1) # One-based block index; e.g. (1 2 3).
linear_probe=True # Frozen single-item probe and full-catalogue CKA after every epoch.

count=0
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for align_target in "${align_targets[@]}"; do
      for align_item in "${align_items[@]}"; do
        for mse_loss_weight in "${mse_loss_weights[@]}"; do
          for mse_loss_mode in "${mse_loss_modes[@]}"; do
            for learning_rate in "${learning_rates[@]}"; do
              for align_loss_type in "${align_loss_types[@]}"; do
                suffix="${align_target}_${align_item}_${align_loss_type}${mse_loss_weight}_${mse_loss_mode}_lr${learning_rate}"
                ratios=(0.5)
                if [[ "$align_loss_type" == cosgram && "$mse_loss_weight" != 0 ]]; then
                  ratios=("${cos_loss_weights[@]}")
                fi
                for cos_loss_weight in "${ratios[@]}"; do
                  for nitp_weight in "${nitp_weights[@]}"; do
                    target_layers=(1)
                    if [[ "$nitp_weight" != 0 && "$nitp_weight" != 0.0 ]]; then
                      target_layers=("${nitp_target_layers[@]}")
                    fi
                    for nitp_target_layer in "${target_layers[@]}"; do
                      task_suffix="$suffix"
                      if [[ "$align_loss_type" == cosgram && "$mse_loss_weight" != 0 ]]; then
                        task_suffix+="_cw${cos_loss_weight}"
                      fi
                      if [[ "$nitp_weight" != 0 && "$nitp_weight" != 0.0 ]]; then
                        task_suffix+="_nitp_w${nitp_weight}_layer${nitp_target_layer}"
                      fi
                      # The trainer adds ratio/NITP tags to the output path.
                      cmd=(
                        mytask ./scripts/train.sh "$model" "$dataset"
                        --gin "train.align_target='${align_target}'"
                        --gin "train.align_item='${align_item}'"
                        --gin "train.mse_loss_weight=${mse_loss_weight}"
                        --gin "train.mse_loss_mode='${mse_loss_mode}'"
                        --gin "train.align_loss_type='${align_loss_type}'"
                        --gin "train.cos_loss_weight=${cos_loss_weight}"
                        --gin "train.nitp_weight=${nitp_weight}"
                        --gin "train.nitp_target_layer=${nitp_target_layer}"
                        --gin "train.linear_probe=${linear_probe}"
                        --gin "train.learning_rate=${learning_rate}"
                        --gin "train.save_dir_root='out_linear2gram/${model}/amazon/${dataset}/align_${suffix}'"
                        "$@"
                        -m "irea_${model}_${dataset}_${task_suffix}"
                      )
                      "${cmd[@]}"
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
  done
done
printf 'Tasks: %d\n' "$count"

#!/usr/bin/env bash
set -euo pipefail

cd /opt/tiger/project/iRea

datasets=( beauty   )
models=(hstu)
align_targets=( item  )
align_items=(current next)
mse_loss_weights=(5 0)
mse_loss_modes=(last)
learning_rates=(5e-4 )
align_loss_types=(cos ) # cos: MLP cosine; gram: Linear(d,d) Gram; cosgram: 0.5*cos + 0.5*gram
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
                cmd=(
                  mytask ./scripts/train.sh "$model" "$dataset"
                  --gin "train.align_target='${align_target}'"
                  --gin "train.align_item='${align_item}'"
                  --gin "train.mse_loss_weight=${mse_loss_weight}"
                  --gin "train.mse_loss_mode='${mse_loss_mode}'"
                  --gin "train.align_loss_type='${align_loss_type}'"
                  --gin "train.linear_probe=${linear_probe}"
                  --gin "train.learning_rate=${learning_rate}"
                  --gin "train.save_dir_root='out_linear2gram/${model}/amazon/${dataset}/align_${suffix}'"
                  "$@"
                  -m "irea_${model}_${dataset}_${suffix}"
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
printf 'Tasks: %d\n' "$count"

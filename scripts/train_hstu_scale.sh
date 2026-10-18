#!/usr/bin/env bash
set -euo pipefail

cd /opt/tiger/project/iRea

# Usage: bash scripts/train_hstu_scale.sh [--dry-run] [extra training/task args]
# These are scale settings for the CURRENT FFN-based HSTU, not the official block.
# Each tuple is "num_blocks num_heads head_dim max_seq_len"; no Cartesian product.
# embed_dim = num_heads * head_dim.
# Length counts history items, not SID tokens (3 tokens/item by default).
scales=(
  # "1 1 32 20"
  # "2 2 32 20"
  # "4 4 32 20"
  "2 2 32 20"
  "2 4 32 20"
  "3 2 32 20"
  "3 4 32 20"
  "4 2 32 20"
  # "6 4 32 20"
  # "6 4 32 40"
  # "8 4 32 20"
  # "8 4 32 80"
  # "8 6 32 80"

)

datasets=(beauty)
model=hstu
align_targets=(item )
align_items=(current)
mse_loss_weights=(0)
mse_loss_modes=(last)
learning_rates=(1e-4)
# Beam search expands this batch by beam_size; long histories may need a smaller batch.
infer_batch_size=64
align_loss_types=(cos)
cos_loss_weights=(0.2 )

dry_run=false
extra_args=()
for arg in "$@"; do
  if [[ "$arg" == --dry-run ]]; then
    dry_run=true
  else
    extra_args+=("$arg")
  fi
done

# Validate all fixed configurations before submitting any tasks.
for scale in "${scales[@]}"; do
  if [[ ! "$scale" =~ ^[1-9][0-9]*[[:space:]]+[1-9][0-9]*[[:space:]]+[1-9][0-9]*[[:space:]]+[1-9][0-9]*$ ]]; then
    printf 'Invalid scale tuple (expected four positive integers): %s\n' "$scale" >&2
    exit 2
  fi
done

count=0
for scale in "${scales[@]}"; do
  read -r num_blocks num_heads head_dim max_seq_len <<< "$scale"
  embed_dim=$((num_heads * head_dim))
  scale_tag="b${num_blocks}_h${num_heads}_hd${head_dim}_d${embed_dim}_len${max_seq_len}"
  for dataset in "${datasets[@]}"; do
    for align_target in "${align_targets[@]}"; do
      for align_item in "${align_items[@]}"; do
        for mse_loss_weight in "${mse_loss_weights[@]}"; do
          for mse_loss_mode in "${mse_loss_modes[@]}"; do
            for learning_rate in "${learning_rates[@]}"; do
              for align_loss_type in "${align_loss_types[@]}"; do
                ratios=(0.5)
                if [[ "$align_loss_type" == cosgram && "$mse_loss_weight" != 0 ]]; then
                  ratios=("${cos_loss_weights[@]}")
                fi
                for cos_loss_weight in "${ratios[@]}"; do
                  suffix="${align_target}_${align_item}_${align_loss_type}${mse_loss_weight}_${mse_loss_mode}_lr${learning_rate}"
                  # Keep dimensions and output paths tied to the tuple, even with extra args.
                  cmd=(
                    mytask ./scripts/train.sh "$model" "$dataset"
                    --gin "train.align_target='${align_target}'"
                    --gin "train.align_item='${align_item}'"
                    --gin "train.mse_loss_weight=${mse_loss_weight}"
                    --gin "train.mse_loss_mode='${mse_loss_mode}'"
                    --gin "train.align_loss_type='${align_loss_type}'"
                    --gin "train.cos_loss_weight=${cos_loss_weight}"
                    --gin "train.learning_rate=${learning_rate}"
                    --gin "train.infer_batch_size=${infer_batch_size}"
                    "${extra_args[@]}"
                    --gin "train.num_blocks=${num_blocks}"
                    --gin "train.num_heads=${num_heads}"
                    --gin "train.embed_dim=${embed_dim}"
                    --gin "train.max_seq_len=${max_seq_len}"
                    --gin "train.save_dir_root='out_hstu_scale_ffn/${model}/amazon/${dataset}/scale_${scale_tag}/align_${suffix}'"
                    -m "irea_${model}_scale_ffn_${dataset}_${scale_tag}_${suffix}_cw${cos_loss_weight}"
                  )
                  if "$dry_run"; then
                    printf '%q ' "${cmd[@]}"
                    printf '\n'
                  else
                    "${cmd[@]}"
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
printf 'Tasks: %d\n' "$count"

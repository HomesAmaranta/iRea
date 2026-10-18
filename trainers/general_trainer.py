"""Shared target-only SID NTP training for TIGER, SASRec and HSTU."""

import copy
import json
import math
import os
from contextlib import nullcontext
from functools import partial
from uuid import uuid4

import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.sid import sid_collate
from models.factory import build_sid_model
from modules.utils import get_run_split, parse_config, setup_logger
from trainers.affine_probe import PROTOCOL, RECALL_KS, validate_fit_config
from trainers.item_probe import ItemLinearProbe
from trainers.representation_geometry import (
    GEOMETRY_DESCRIPTION, SequenceGeometryEpoch, geometry_log_metrics,
)
from trainers.trainer_utils import set_seed, setup_wandb


def move_batch(batch, device, include_labels=True):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if include_labels or key not in ("labels", "align_target_emb")
    }


def configure_alignment(model, target_dim, align_loss_type="cos"):
    """Register only the required heads before creating the optimizer/checkpoint."""
    if align_loss_type not in ("cos", "gram", "cosgram"):
        raise ValueError("align_loss_type must be cos, gram or cosgram")
    if target_dim <= 0:
        raise ValueError("Alignment target dimension must be positive")
    embedding = (model.model.get_input_embeddings() if hasattr(model, "model")
                 else model.item_embedding)
    hidden_dim = embedding.weight.shape[1]
    if align_loss_type in ("cos", "cosgram"):
        model.hidden_to_item_emb = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim, bias=False),
        ).to(device=embedding.weight.device, dtype=embedding.weight.dtype)
    if align_loss_type in ("gram", "cosgram"):
        model.hidden_to_gram = nn.Linear(hidden_dim, hidden_dim, bias=False).to(
            device=embedding.weight.device, dtype=embedding.weight.dtype
        )


def compute_alignment_loss(model, output, inp, align_loss_type="cos", cos_loss_weight=0.5):
    """Align the first target-prediction slot to a frozen current/next item vector."""
    if align_loss_type not in ("cos", "gram", "cosgram"):
        raise ValueError("align_loss_type must be cos, gram or cosgram")
    if not 0 <= cos_loss_weight <= 1:
        raise ValueError("cos_loss_weight must be finite and in [0, 1]")
    # SASRec/HSTU: b_last. TIGER: decoder start PAD, before reading any target code.
    hidden = output.last_hidden_state[:, 0, :]
    target = inp.get("align_target_emb")
    if target is None:
        raise ValueError("Alignment requires align_target_emb")
    if target.ndim != 2 or target.shape[0] != hidden.shape[0]:
        raise ValueError("Alignment requires one target vector per sample")
    if align_loss_type in ("gram", "cosgram"):
        # Gram compares sample similarities, so target feature dimensions may differ.
        prediction = model.hidden_to_gram(hidden)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            normalized_prediction = F.normalize(prediction.float(), dim=-1, eps=1e-8)
            reference = F.normalize(target.detach().float(), dim=-1, eps=1e-8)
            gram_loss = F.mse_loss(
                normalized_prediction @ normalized_prediction.T,
                reference @ reference.T, reduction="mean"
            )
        if align_loss_type == "gram":
            return gram_loss
    prediction = model.hidden_to_item_emb(hidden)
    if prediction.shape != target.shape:
        raise ValueError("Projected hidden and alignment target shapes must match")
    # FP32 cosine avoids small-norm instability under mixed precision.
    cos_loss = 1.0 - F.cosine_similarity(
        prediction.float(), target.detach().float(), dim=-1
    ).mean()
    return (cos_loss_weight * cos_loss + (1 - cos_loss_weight) * gram_loss
            if align_loss_type == "cosgram" else cos_loss)


def autocast_context(device, mixed_precision):
    if mixed_precision == "no":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def evaluate_ce(model, loader, device, mixed_precision="no", desc="Valid CE"):
    """Target-only teacher-forced CE, without generation or alignment loss."""
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    token_count = 0
    try:
        with torch.no_grad(), autocast_context(device, mixed_precision):
            for batch in tqdm(loader, desc=desc):
                inp = move_batch(
                    {key: value for key, value in batch.items() if key != "align_target_emb"},
                    device,
                )
                output = model(inp)
                if output.loss is None or not torch.isfinite(output.loss):
                    raise RuntimeError("Model returned a missing or non-finite validation CE")
                # SID batches contain complete fixed-length targets, without padding.
                count = inp["labels"].numel()
                loss_sum += output.loss.item() * count
                token_count += count
    finally:
        model.train(was_training)
    if token_count == 0:
        raise ValueError("Evaluation dataset is empty")
    return {"ce": loss_sum / token_count}


def evaluate(model, loader, device, beam_size=30, top_ks=(5, 10),
             constrain_sid=False, mixed_precision="no", desc="Eval"):
    if not top_ks or min(top_ks) <= 0 or max(top_ks) > beam_size:
        raise ValueError("Evaluation k values must be positive and <= beam_size")
    was_training = model.training
    model.eval()
    total = 0
    sums = {f"{metric}@{k}": 0.0 for k in top_ks for metric in ("Recall", "NDCG")}
    try:
        with torch.no_grad(), autocast_context(device, mixed_precision):
            for batch in tqdm(loader, desc=desc):
                # Keep ground truth outside the inference input to prevent target leakage.
                inp = move_batch(batch, device, include_labels=False)
                generated = model.generate(
                    inp, num_beams=beam_size, constrain_sid=constrain_sid
                )
                targets = batch["labels"].to(device)
                match = generated.sequences.eq(targets.unsqueeze(1)).all(dim=-1)
                first_match = match & (match.long().cumsum(dim=1) == 1)
                ranks = torch.arange(1, beam_size + 1, device=device).float()
                discounts = torch.log2(ranks + 1).reciprocal()
                for k in top_ks:
                    hits = first_match[:, :k]
                    sums[f"Recall@{k}"] += hits.sum().item()
                    sums[f"NDCG@{k}"] += (hits * discounts[:k]).sum().item()
                total += targets.shape[0]
    finally:
        model.train(was_training)
    if total == 0:
        raise ValueError("Evaluation dataset is empty")
    return {key: value / total for key, value in sums.items()}


@gin.configurable
def train(
    model_name="tiger",
    model_config=None,
    num_layers=None,
    num_decoder_layers=None,
    d_model=None,
    d_ff=None,
    num_heads=None,
    d_kv=None,
    dropout_rate=None,
    feed_forward_proj=None,
    embed_dim=None,
    num_blocks=None,
    ffn_dim=None,
    dropout=None,
    num_position_buckets=None,
    num_time_buckets=None,
    use_temporal_bias=None,
    dataset=None,
    dataset_folder="dataset/amazon",
    pretrained_rqvae_path="out/tiger/amazon/{split}/rqvae/checkpoint_epoch_4999.pt",
    save_dir_root="out/{model}/amazon/{split}",
    codebook_size=256,
    sem_id_dim=3,
    max_seq_len=20,
    epochs=200,
    batch_size=256,
    infer_batch_size=96,
    learning_rate=1e-4,
    weight_decay=0.0,
    lr_schedule="none",
    num_warmup_steps=0,
    beam_size=30,
    top_ks=(5, 10),
    constrain_sid=False,
    early_stop_patience=10,
    early_stop_metric="ce",
    eval_valid_every_n_epochs=1,
    eval_test_every_epoch=2,
    num_workers=4,
    mixed_precision="no",
    seed=42,
    device=None,
    wandb_logging=False,
    wandb_project="{model}_{split}_training",
    wandb_log_interval=10,
    align_target="quantized",
    align_item="current",
    mse_loss_weight=0.0,
    mse_loss_mode="last",
    align_loss_type="cos",
    cos_loss_weight=0.5,
    linear_probe=True,
    linear_probe_batch_size=256,
    linear_probe_seed=42,
    linear_probe_learning_rate=1e-3,
    linear_probe_max_epochs=3000,
    linear_probe_patience=50,
    linear_probe_reference_path=None,
    category_probe=True,
    sequence_geometry=True,
    sequence_geometry_cosine_pairs=4096,
    sequence_geometry_seed=42,
    tiger_encoder_probe=True,
):
    if min(epochs, batch_size, infer_batch_size, early_stop_patience,
           eval_valid_every_n_epochs, eval_test_every_epoch, wandb_log_interval) <= 0:
        raise ValueError("Epochs, batch sizes and scheduling intervals must be positive")
    if mixed_precision not in ("no", "bf16"):
        raise ValueError("mixed_precision must be no or bf16")
    if early_stop_metric not in ("ce", "hr"):
        raise ValueError("early_stop_metric must be ce or hr")
    if num_workers < 0 or lr_schedule not in ("none", "cosine"):
        raise ValueError("Invalid worker count or learning-rate schedule")
    if not top_ks or min(top_ks) <= 0 or max(top_ks) > beam_size:
        raise ValueError("Evaluation k values must be positive and <= beam_size")
    if align_target not in ("item", "latent", "quantized"):
        raise ValueError("align_target must be item, latent or quantized")
    if align_item not in ("current", "next"):
        raise ValueError("align_item must be current or next")
    if mse_loss_mode != "last" or align_loss_type not in ("cos", "gram", "cosgram"):
        raise ValueError("Only mse_loss_mode=last and align_loss_type=cos/gram/cosgram are supported")
    if not math.isfinite(mse_loss_weight) or mse_loss_weight < 0:
        raise ValueError("mse_loss_weight must be finite and nonnegative")
    if not 0 <= cos_loss_weight <= 1:
        raise ValueError("cos_loss_weight must be finite and in [0, 1]")
    if not isinstance(linear_probe, bool):
        raise ValueError("linear_probe must be a boolean")
    if not isinstance(tiger_encoder_probe, bool):
        raise ValueError("tiger_encoder_probe must be a boolean")
    if not isinstance(category_probe, bool):
        raise ValueError("category_probe must be a boolean")
    if not isinstance(sequence_geometry, bool):
        raise ValueError("sequence_geometry must be a boolean")
    if (isinstance(sequence_geometry_cosine_pairs, bool)
            or not isinstance(sequence_geometry_cosine_pairs, int)
            or sequence_geometry_cosine_pairs <= 0):
        raise ValueError("sequence_geometry_cosine_pairs must be a positive integer")
    if (isinstance(linear_probe_batch_size, bool)
            or not isinstance(linear_probe_batch_size, int) or linear_probe_batch_size <= 0):
        raise ValueError("linear_probe_batch_size must be a positive integer")
    validate_fit_config(
        linear_probe_learning_rate, linear_probe_max_epochs, linear_probe_patience
    )
    alignment_config = dict(
        align_target=align_target, align_item=align_item, mse_loss_weight=mse_loss_weight,
        mse_loss_mode=mse_loss_mode, align_loss_type=align_loss_type,
        cos_loss_weight=cos_loss_weight,
    )
    model_config = dict(model_config or {})
    architecture = {
        "tiger": dict(num_layers=num_layers, num_decoder_layers=num_decoder_layers,
                      d_model=d_model, d_ff=d_ff, num_heads=num_heads, d_kv=d_kv,
                      dropout_rate=dropout_rate, feed_forward_proj=feed_forward_proj),
        "sasrec": dict(embed_dim=embed_dim, num_heads=num_heads, num_blocks=num_blocks,
                       ffn_dim=ffn_dim, dropout=dropout, d_kv=d_kv,
                       feed_forward_proj=feed_forward_proj),
        "hstu": dict(embed_dim=embed_dim, num_heads=num_heads, num_blocks=num_blocks,
                     dropout=dropout, num_position_buckets=num_position_buckets,
                     num_time_buckets=num_time_buckets, use_temporal_bias=use_temporal_bias),
    }
    if model_name not in architecture:
        raise ValueError("model_name must be tiger, hstu or sasrec")
    model_config.update({
        key: value for key, value in architecture[model_name].items() if value is not None
    })
    set_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    paths = {"model": model_name, "split": get_run_split("beauty")}
    save_dir_root = save_dir_root.format(**paths)
    pretrained_rqvae_path = pretrained_rqvae_path.format(**paths)
    reference_path = (linear_probe_reference_path or os.path.join(
        dataset_folder, "processed", "{split}", "item_emb_sentence-t5-xl.parquet"
    )).format(**paths)
    probe_config = dict(
        enabled=linear_probe, reference_path=reference_path, batch_size=linear_probe_batch_size,
        seed=linear_probe_seed, protocol=PROTOCOL,
        learning_rate=linear_probe_learning_rate, max_epochs=linear_probe_max_epochs,
        patience=linear_probe_patience, recall_top_ks=list(RECALL_KS),
        category_enabled=linear_probe and category_probe,
        category_level=2,
        tiger_encoder_mean_enabled=linear_probe and tiger_encoder_probe and model_name == "tiger",
    )
    artifact_suffix = ""
    if mse_loss_weight > 0:
        weight_tag = str(mse_loss_weight).removesuffix(".0")
        artifact_suffix = (
            f"_{align_target}_{align_item}_{align_loss_type}{weight_tag}_{mse_loss_mode}"
        )
        if align_loss_type == "cosgram" and cos_loss_weight != 0.5:
            ratio_tag = f"_cw{cos_loss_weight}"
            artifact_suffix += ratio_tag
            save_dir_root += ratio_tag
    logger = setup_logger(save_dir_root, name=f"{model_name}{artifact_suffix}")
    sequence_geometry_config = dict(
        enabled=sequence_geometry, cosine_pairs_per_batch=sequence_geometry_cosine_pairs,
        seed=sequence_geometry_seed, method=GEOMETRY_DESCRIPTION,
    )
    sequence_geometry_path = (os.path.join(
        save_dir_root, f"sequence_geometry{artifact_suffix}_{uuid4().hex}.jsonl"
    ) if sequence_geometry else None)
    sequence_geometry_latest = None
    if sequence_geometry:
        logger.info(
            "[SequenceGeometry] each training batch: output.last_hidden_state "
            "(all target-prediction slots, not full history); epoch-end batch mean; "
            f"cosine pairs/batch={sequence_geometry_cosine_pairs}; file={sequence_geometry_path}"
        )

    if dataset is None:
        from data.amazon import AmazonSeqDataset
        dataset = AmazonSeqDataset
    datasets = {
        split: dataset(
            root=dataset_folder, train_test_split=split, max_seq_len=max_seq_len,
            pretrained_rqvae_path=pretrained_rqvae_path,
            **(dict(align_target=align_target, align_item=align_item)
               if split == "train" and mse_loss_weight > 0 else {}),
        )
        for split in ("train", "valid", "test")
    }
    if any(len(value) == 0 for value in datasets.values()):
        raise ValueError("Train, valid and test datasets must all be nonempty")
    collate_fn = partial(
        sid_collate, codebook_size=codebook_size, sem_id_dim=sem_id_dim,
        max_seq_len=max_seq_len,
        require_timestamps=model_name == "hstu" and model_config.get("use_temporal_bias", False),
    )
    loaders = {
        split: DataLoader(
            value, batch_size=batch_size if split == "train" else infer_batch_size,
            shuffle=split == "train", num_workers=num_workers,
            pin_memory=device.type == "cuda", collate_fn=collate_fn,
        )
        for split, value in datasets.items()
    }
    model = build_sid_model(
        model_name, codebook_size, sem_id_dim, max_seq_len, model_config
    ).to(device)
    if mse_loss_weight > 0:
        target = getattr(datasets["train"][0], "align_target_emb", None)
        if target is None or torch.as_tensor(target).ndim != 1:
            raise ValueError("Training dataset must provide an align_target_emb vector")
        configure_alignment(model, len(target), align_loss_type)
    item_probe = None
    encoder_probe = None
    if linear_probe:
        item_probe = ItemLinearProbe(
            datasets["train"], reference_path, save_dir_root, logger,
            batch_size=linear_probe_batch_size, seed=linear_probe_seed,
            learning_rate=linear_probe_learning_rate,
            max_epochs=linear_probe_max_epochs, patience=linear_probe_patience,
            category_probe=category_probe,
        )
        logger.info(
            "[ItemProbe] last position from normal single-item forward: "
            + ("TIGER decoder start" if model_name == "tiger" else "last input SID token")
        )
        if model_name == "tiger" and tiger_encoder_probe:
            encoder_probe = ItemLinearProbe(
                datasets["train"], reference_path, save_dir_root, logger,
                batch_size=linear_probe_batch_size, seed=linear_probe_seed,
                learning_rate=linear_probe_learning_rate,
                max_epochs=linear_probe_max_epochs, patience=linear_probe_patience,
                category_probe=category_probe, hidden_source="encoder_mean",
            )
            logger.info("[ItemProbe:encoder_mean] mean of TIGER encoder's three SID tokens")
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = None
    if lr_schedule == "cosine":
        total_steps = len(loaders["train"]) * epochs
        if not 0 <= num_warmup_steps < total_steps:
            raise ValueError("Warmup steps must be in [0, total training steps)")
        cosine = CosineAnnealingLR(optimizer, T_max=total_steps - num_warmup_steps)
        if num_warmup_steps:
            warmup = LinearLR(optimizer, start_factor=0.01, total_iters=num_warmup_steps)
            scheduler = SequentialLR(
                optimizer, [warmup, cosine], milestones=[num_warmup_steps]
            )
        else:
            scheduler = cosine

    logger.info(
        f"Model: {model_name}, Device: {device}, "
        f"Params: {sum(p.numel() for p in model.parameters()):,}, "
        f"Samples: { {key: len(value) for key, value in datasets.items()} }"
    )
    logger.info(f"Alignment: {alignment_config}")
    monitor = "ce" if early_stop_metric == "ce" else f"Recall@{max(top_ks)}"
    logger.info(
        f"Early stopping: {early_stop_metric}, monitor: {monitor}, "
        f"direction: {'min' if early_stop_metric == 'ce' else 'max'}, "
        f"patience: {early_stop_patience} validations"
    )
    wandb = None
    if wandb_logging:
        import wandb
        setup_wandb(
            project=wandb_project.format(**paths),
            config={
                "model_name": model_name, "model_config": model_config or {},
                "codebook_size": codebook_size, "sem_id_dim": sem_id_dim,
                "max_seq_len": max_seq_len, "learning_rate": learning_rate,
                "batch_size": batch_size, "beam_size": beam_size, "seed": seed,
                "early_stop_metric": early_stop_metric,
                "early_stop_patience": early_stop_patience,
                "linear_probe": probe_config,
                  "sequence_geometry": sequence_geometry_config,
                **alignment_config,
            },
        )

    best_value = float("inf") if early_stop_metric == "ce" else -float("inf")
    best_epoch = -1
    best_model = None
    wait = 0
    global_step = 0
    best_path = os.path.join(save_dir_root, f"best_model{artifact_suffix}.pt")
    eval_kwargs = dict(
        device=device, beam_size=beam_size, top_ks=top_ks,
        constrain_sid=constrain_sid, mixed_precision=mixed_precision,
    )
    for epoch in range(epochs):
        model.train()
        sequence_stats = (SequenceGeometryEpoch(
            device, sequence_geometry_cosine_pairs, sequence_geometry_seed + epoch
        ) if sequence_geometry else None)
        loss_sum = 0.0
        ntp_loss_sum = 0.0
        align_loss_sum = 0.0
        sample_count = 0
        for batch in tqdm(loaders["train"], desc=f"Epoch {epoch}"):
            inp = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, mixed_precision):
                output = model(inp)
                align_loss = (
                    compute_alignment_loss(model, output, inp, align_loss_type, cos_loss_weight)
                    if mse_loss_weight > 0 else output.logits.new_zeros(())
                )
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError("Model returned a missing or non-finite NTP loss")
            loss = output.loss + mse_loss_weight * align_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite combined alignment loss")
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if sequence_stats is not None:
                # SIDOutput is aligned with target labels, not the history attention mask.
                # TIGER's decoder-start slot is a valid prediction position.
                sequence_stats.update(output.last_hidden_state, inp["labels"] > 0)
            size = inp["labels"].shape[0]
            loss_sum += loss.item() * size
            ntp_loss_sum += output.loss.item() * size
            align_loss_sum += align_loss.item() * size
            sample_count += size
            global_step += 1
            if wandb is not None and global_step % wandb_log_interval == 0:
                wandb.log({
                    "global_step": global_step, "train/loss": loss.item(),
                    "train/ntp_loss": output.loss.item(), "train/align_loss": align_loss.item(),
                })

        log = {
            "epoch": epoch, "train/loss": loss_sum / sample_count,
            "train/ntp_loss": ntp_loss_sum / sample_count,
            "train/align_loss": align_loss_sum / sample_count,
        }
        logger.info(
            f"Epoch {epoch} - loss: {log['train/loss']:.6f}, "
            f"ntp: {log['train/ntp_loss']:.6f}, align: {log['train/align_loss']:.6f}"
        )
        if sequence_stats is not None:
            sequence_geometry_latest = {
                "epoch": epoch, "global_step": global_step,
                **sequence_stats.result(), "config": sequence_geometry_config,
            }
            log.update(geometry_log_metrics(sequence_geometry_latest, "geometry/sequence"))
            logger.info(
                "[SequenceGeometry] epoch=%d mean_effective_rank=%s mean_avg_cosine=%s "
                "batches=%d valid_batches=%s",
                epoch, sequence_geometry_latest["effective_rank"],
                sequence_geometry_latest["avg_cosine"], sequence_stats.batches,
                sequence_geometry_latest["valid_batches"],
            )
            with open(sequence_geometry_path, "a") as handle:
                handle.write(json.dumps(sequence_geometry_latest, allow_nan=False) + "\n")
        if item_probe is not None:
            log.update(item_probe.evaluate(model, epoch, mixed_precision))
        if encoder_probe is not None:
            log.update(encoder_probe.evaluate(model, epoch, mixed_precision))
        if epoch == 0 or (epoch + 1) % eval_valid_every_n_epochs == 0:
            if early_stop_metric == "ce":
                valid = evaluate_ce(
                    model, loaders["valid"], device, mixed_precision=mixed_precision
                )
            else:
                valid = evaluate(model, loaders["valid"], desc="Valid", **eval_kwargs)
            logger.info(f"Epoch {epoch} - Valid: {valid}")
            log.update({f"valid/{key}": value for key, value in valid.items()})
            value = valid[monitor]
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite validation metric: {monitor}")
            improved = value < best_value if early_stop_metric == "ce" else value > best_value
            if improved:
                best_value, best_epoch, wait = value, epoch, 0
                best_model = copy.deepcopy(model).eval()
                torch.save(model.state_dict(), best_path)
            else:
                wait += 1
            # Test only once using the independent best-model copy below.
            # if improved or (epoch + 1) % eval_test_every_epoch == 0:
            #     test = evaluate(model, loaders["test"], desc="Test", **eval_kwargs)
            #     logger.info(f"Epoch {epoch} - Test: {test}")
            #     log.update({f"test/{key}": value for key, value in test.items()})
        if wandb is not None:
            wandb.log(log)
        if wait >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch}; best epoch: {best_epoch}")
            break

    if best_model is None:
        raise RuntimeError("No best model was selected during validation")
    final_test = evaluate(best_model, loaders["test"], desc="Best model test", **eval_kwargs)
    results = {
        "model_name": model_name, "best_epoch": best_epoch, "seed": seed,
        "early_stop_metric": early_stop_metric,
        "best_valid_metric": monitor, "best_valid_value": best_value,
        "best_valid_ce": best_value if early_stop_metric == "ce" else None,
        "best_valid_recall": best_value if early_stop_metric == "hr" else None,
        "test": final_test,
        "model_config": model_config or {}, "codebook_size": codebook_size,
        "sem_id_dim": sem_id_dim, "max_seq_len": max_seq_len,
        "linear_probe": probe_config,
        "linear_probe_latest": item_probe.latest if item_probe is not None else None,
        "linear_probe_dir": str(item_probe.directory) if item_probe is not None else None,
        "encoder_probe_latest": encoder_probe.latest if encoder_probe is not None else None,
        "encoder_probe_dir": str(encoder_probe.directory) if encoder_probe is not None else None,
        "sequence_geometry": sequence_geometry_config,
        "sequence_geometry_latest": sequence_geometry_latest,
        "sequence_geometry_path": sequence_geometry_path,
        **alignment_config,
    }
    with open(os.path.join(save_dir_root, "results.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    logger.info(f"Best epoch: {best_epoch}, final test: {final_test}")
    if wandb is not None:
        wandb.log({f"final_test/{key}": value for key, value in final_test.items()})
        wandb.finish()
    return results


if __name__ == "__main__":
    parse_config()
    train()

"""Shared target-only SID NTP training for TIGER, SASRec and HSTU."""

import json
import os
from contextlib import nullcontext
from functools import partial

import gin
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.sid import sid_collate
from models.factory import build_sid_model
from modules.utils import get_run_split, parse_config, setup_logger
from trainers.trainer_utils import set_seed, setup_wandb


def move_batch(batch, device, include_labels=True):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if include_labels or key != "labels"
    }


def autocast_context(device, mixed_precision):
    if mixed_precision == "no":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


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
    eval_valid_every_n_epochs=1,
    eval_test_every_epoch=2,
    num_workers=4,
    mixed_precision="no",
    seed=42,
    device=None,
    wandb_logging=False,
    wandb_project="{model}_{split}_training",
    wandb_log_interval=10,
):
    if min(epochs, batch_size, infer_batch_size, early_stop_patience,
           eval_valid_every_n_epochs, eval_test_every_epoch, wandb_log_interval) <= 0:
        raise ValueError("Epochs, batch sizes and scheduling intervals must be positive")
    if mixed_precision not in ("no", "bf16"):
        raise ValueError("mixed_precision must be no or bf16")
    if num_workers < 0 or lr_schedule not in ("none", "cosine"):
        raise ValueError("Invalid worker count or learning-rate schedule")
    if not top_ks or min(top_ks) <= 0 or max(top_ks) > beam_size:
        raise ValueError("Evaluation k values must be positive and <= beam_size")
    model_config = dict(model_config or {})
    architecture = {
        "tiger": dict(num_layers=num_layers, num_decoder_layers=num_decoder_layers,
                      d_model=d_model, d_ff=d_ff, num_heads=num_heads, d_kv=d_kv,
                      dropout_rate=dropout_rate, feed_forward_proj=feed_forward_proj),
        "sasrec": dict(embed_dim=embed_dim, num_heads=num_heads, num_blocks=num_blocks,
                       ffn_dim=ffn_dim, dropout=dropout),
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
    logger = setup_logger(save_dir_root, name=model_name)

    if dataset is None:
        from data.amazon import AmazonSeqDataset
        dataset = AmazonSeqDataset
    datasets = {
        split: dataset(
            root=dataset_folder, train_test_split=split, max_seq_len=max_seq_len,
            pretrained_rqvae_path=pretrained_rqvae_path,
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
            },
        )

    best_recall = -float("inf")
    best_epoch = -1
    wait = 0
    global_step = 0
    monitor = f"Recall@{max(top_ks)}"
    best_path = os.path.join(save_dir_root, "best_model.pt")
    eval_kwargs = dict(
        device=device, beam_size=beam_size, top_ks=top_ks,
        constrain_sid=constrain_sid, mixed_precision=mixed_precision,
    )
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in tqdm(loaders["train"], desc=f"Epoch {epoch}"):
            inp = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, mixed_precision):
                output = model(inp)
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError("Model returned a missing or non-finite NTP loss")
            output.loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            size = inp["labels"].shape[0]
            loss_sum += output.loss.item() * size
            sample_count += size
            global_step += 1
            if wandb is not None and global_step % wandb_log_interval == 0:
                wandb.log({"global_step": global_step, "train/loss": output.loss.item()})

        log = {"epoch": epoch, "train/loss": loss_sum / sample_count}
        logger.info(f"Epoch {epoch} - loss: {log['train/loss']:.6f}")
        if epoch == 0 or (epoch + 1) % eval_valid_every_n_epochs == 0:
            valid = evaluate(model, loaders["valid"], desc="Valid", **eval_kwargs)
            logger.info(f"Epoch {epoch} - Valid: {valid}")
            log.update({f"valid/{key}": value for key, value in valid.items()})
            improved = valid[monitor] > best_recall
            if improved:
                best_recall, best_epoch, wait = valid[monitor], epoch, 0
                torch.save(model.state_dict(), best_path)
            else:
                wait += 1
            if improved or (epoch + 1) % eval_test_every_epoch == 0:
                test = evaluate(model, loaders["test"], desc="Test", **eval_kwargs)
                logger.info(f"Epoch {epoch} - Test: {test}")
                log.update({f"test/{key}": value for key, value in test.items()})
        if wandb is not None:
            wandb.log(log)
        if wait >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch}; best epoch: {best_epoch}")
            break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    final_test = evaluate(model, loaders["test"], desc="Best model test", **eval_kwargs)
    results = {
        "model_name": model_name, "best_epoch": best_epoch, "seed": seed,
        "best_valid_recall": best_recall, "test": final_test,
        "model_config": model_config or {}, "codebook_size": codebook_size,
        "sem_id_dim": sem_id_dim, "max_seq_len": max_seq_len,
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

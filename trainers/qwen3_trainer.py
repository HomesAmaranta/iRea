"""Prompted Qwen3 LoRA training for LCRec and OneRec-style RQ-KMeans SIDs."""

import copy
import json
import logging
import math
import os
from uuid import uuid4
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import gin
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler

from data.lcrec import DEFAULT_PROMPT, LCSIDCollator, load_lm_datasets, register_sid_tokens
from models.lcrec import build_lcrec
from modules.utils import get_run_split, parse_config, setup_logger
from trainers.qwen3_item_probe import Qwen3ItemProbe, on_main_process
from trainers.affine_probe import validate_fit_config
from trainers.representation_geometry import (
    GEOMETRY_DESCRIPTION, SequenceGeometryEpoch, geometry_log_metrics,
)
from trainers.trainer_utils import save_run_results, set_seed, setup_accelerator


def setup_training_logger(accelerator, save_dir, config, paths):
    name = config["model_name"]
    if config["mse_loss_weight"] > 0:
        name += (
            f"_{config['align_target']}_{config['align_item']}"
            f"_{config['align_loss_type']}{config['mse_loss_weight']:g}_{config['align_source']}"
        )
    if not accelerator.is_main_process:
        # No file/console handlers on workers, including when root logging is enabled.
        logger = logging.Logger(name)
        logger.disabled = True
        logger.propagate = False
        return logger
    logger = setup_logger(save_dir, name=name)
    logger.propagate = False
    effective = dict(config)
    for key in ("dataset_folder", "pretrained_rqvae_path", "pretrained_rqkmeans_path"):
        effective[key] = config[key].format(**paths)
    effective.update(split=paths["split"], save_dir_root=save_dir)
    logger.info("Effective training arguments:\n%s", json.dumps(effective, indent=2))
    logger.info("Alignment: %s", {key: config[key] for key in (
        "align_target", "align_item", "mse_loss_weight", "align_source", "align_loss_type",
    )})
    logger.info(
        "Batch: per-device=%d, accumulation=%d, world_size=%d, effective=%d",
        config["batch_size"], config["gradient_accumulate_every"], accelerator.num_processes,
        config["batch_size"] * config["gradient_accumulate_every"] * accelerator.num_processes,
    )
    return logger


@contextmanager
def main_process_loading_output(accelerator):
    """Keep third-party loading prints/progress on rank zero; exceptions still propagate."""
    if accelerator.is_main_process:
        yield
    else:
        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            yield


def aggregate_training_log(accelerator, epoch, ce_total, align_total, count, weight):
    # Every rank must participate, even though only rank zero emits the result.
    totals = accelerator.reduce(
        torch.tensor([ce_total, align_total, count], device=accelerator.device, dtype=torch.float64),
        reduction="sum",
    ).tolist()
    if totals[2] <= 0:
        raise ValueError("No training samples")
    ce, align = totals[0] / totals[2], totals[1] / totals[2]
    return {"epoch": epoch, "train/ce": ce, "train/align": align, "train/loss": ce + weight * align}


def aggregate_sequence_geometry(accelerator, stats):
    """Reduce sums and valid counts, not per-rank averages or pooled hidden."""
    keys = ("effective_rank", "avg_cosine")
    totals = accelerator.reduce(torch.tensor(
        [stats.sums[key] for key in keys] + [stats.counts[key] for key in keys]
        + [stats.batches, stats.tokens, stats.nonzero_tokens, stats.sampled_pairs],
        device=accelerator.device, dtype=torch.float64,
    ), reduction="sum").tolist()
    counts = dict(zip(keys, map(int, totals[2:4])))
    batches, tokens, nonzero_tokens, pairs = map(int, totals[4:])
    return {
        **{key: totals[i] / counts[key] if counts[key] else None
           for i, key in enumerate(keys)},
        "batches": batches, "tokens": tokens, "nonzero_tokens": nonzero_tokens,
        "sampled_pairs": pairs, "valid_batches": counts,
        "skipped_batches": {key: batches - counts[key] for key in keys},
    }


@torch.no_grad()
def evaluate_lm(model, loader, accelerator, metric="ce", beam_size=20,
                top_ks=(5, 10), constrain_sid=True):
    """Gather per-example values, trimming distributed sampler padding."""
    was_training = model.training
    model.eval()
    values = []
    try:
        for batch in tqdm(loader, desc=f"Eval {metric}", disable=not accelerator.is_main_process):
            with accelerator.autocast():
                if metric == "ce":
                    inputs = {key: value for key, value in batch.items()
                              if key != "align_target_emb"}
                    result = model(inputs).ce_per_example[:, None]
                elif metric == "hr":
                    inputs = {key: batch[key] for key in ("input_ids", "attention_mask")}
                    sequences = model.generate(
                        inputs, num_beams=beam_size, constrain_sid=constrain_sid
                    ).sequences
                    match = sequences.eq(batch["labels"][:, None]).all(-1)
                    match &= match.long().cumsum(1).eq(1)
                    ranks = torch.arange(1, beam_size + 1, device=match.device).float()
                    discounts = torch.log2(ranks + 1).reciprocal()
                    result = torch.stack([
                        value
                        for k in top_ks
                        for value in (
                            match[:, :k].sum(1).float(),
                            (match[:, :k] * discounts[:k]).sum(1),
                        )
                    ], dim=1)
                else:
                    raise ValueError("metric must be ce or hr")
            values.append(accelerator.gather_for_metrics(result).float().cpu())
    finally:
        model.train(was_training)
    if not values:
        raise ValueError("Evaluation dataset is empty")
    means = torch.cat(values).mean(0)
    if not torch.isfinite(means).all():
        raise RuntimeError("Non-finite validation/test metric")
    keys = ["ce"] if metric == "ce" else [
        f"{name}@{k}" for k in top_ks for name in ("Recall", "NDCG")
    ]
    return dict(zip(keys, means.tolist()))


@gin.configurable
def train_lcrec(
    model_name="lcrec",
    quantizer="rqvae",
    base_model="/opt/tiger/project/hf_models/Qwen3-1.7B",
    encoder_model_name="/opt/tiger/project/hf_models/sentence-t5-xl",
    dataset_folder="dataset/amazon",
    pretrained_rqvae_path="out/tiger/amazon/{split}/rqvae/checkpoint_epoch_4999.pt",
    pretrained_rqkmeans_path="out/onerec/amazon/{split}/rqkmeans/rqkmeans.pt",
    rqvae_config=None,
    save_dir_root="out/{model}/amazon/{split}/sft",
    codebook_size=256,
    sem_id_dim=3,
    max_seq_len=20,
    model_max_length=1024,
    prompt_template=DEFAULT_PROMPT,
    epochs=10,
    batch_size=128,
    infer_batch_size=8,
    gradient_accumulate_every=1,
    learning_rate=1e-3,
    weight_decay=0.0,
    lr_schedule="cosine",
    num_warmup_steps=200,
    max_grad_norm=10.0,
    lora_r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules=("q_proj", "v_proj", "o_proj", "up_proj", "down_proj"),
    gradient_checkpointing=True,
    early_stop_metric="ce",
    early_stop_patience=2,
    eval_valid_every_n_epochs=1,
    beam_size=20,
    top_ks=(5, 10),
    constrain_sid=True,
    mixed_precision="bf16",
    num_workers=0,
    seed=42,
    mse_loss_weight=0.0,
    align_target="quantized",
    align_item="current",
    align_source="last_inp",
    align_loss_type="cos",
    wandb_logging=False,
    linear_probe=None,
    linear_probe_every_steps=50,
    linear_probe_batch_size=32,
    linear_probe_seed=42,
    linear_probe_reference_path=None,
    linear_probe_learning_rate=1e-3,
    linear_probe_max_epochs=3000,
    linear_probe_patience=50,
    sequence_geometry=True,
    sequence_geometry_cosine_pairs=4096,
    sequence_geometry_seed=42,
):
    if linear_probe is None:
        linear_probe = model_name == "lcrec"
    if not isinstance(linear_probe_every_steps, int) or linear_probe_every_steps <= 0:
        raise ValueError("linear_probe_every_steps must be a positive integer")
    if not isinstance(linear_probe, bool):
        raise ValueError("linear_probe must be a boolean or None")
    if linear_probe and (model_name != "lcrec" or quantizer != "rqvae"):
        raise ValueError("The Qwen3 item probe currently supports LCRec with RQVAE only")
    if linear_probe and linear_probe_batch_size <= 0:
        raise ValueError("Invalid item probe batch size")
    validate_fit_config(linear_probe_learning_rate, linear_probe_max_epochs, linear_probe_patience)
    if not isinstance(sequence_geometry, bool):
        raise ValueError("sequence_geometry must be a boolean")
    if (isinstance(sequence_geometry_cosine_pairs, bool)
            or not isinstance(sequence_geometry_cosine_pairs, int)
            or sequence_geometry_cosine_pairs <= 0):
        raise ValueError("sequence_geometry_cosine_pairs must be a positive integer")
    run_config = dict(locals())
    if model_name not in ("lcrec", "onerec") or quantizer not in ("rqvae", "rqkmeans"):
        raise ValueError("Use lcrec/onerec and rqvae/rqkmeans")
    if min(epochs, batch_size, infer_batch_size, gradient_accumulate_every,
           early_stop_patience, eval_valid_every_n_epochs, lora_r,
           codebook_size, sem_id_dim, max_seq_len) <= 0:
        raise ValueError("Training sizes, SID dimensions and intervals must be positive")
    if early_stop_metric not in ("ce", "hr") or mixed_precision not in ("no", "bf16"):
        raise ValueError("Use early_stop_metric=ce/hr and mixed_precision=no/bf16")
    if not top_ks or min(top_ks) <= 0 or max(top_ks) > beam_size:
        raise ValueError("Evaluation k values must be positive and <= beam_size")
    if num_workers < 0 or num_warmup_steps < 0 or lr_schedule not in ("cosine", "linear", "constant"):
        raise ValueError("Invalid worker count, warmup or learning-rate schedule")
    if not math.isfinite(mse_loss_weight) or mse_loss_weight < 0:
        raise ValueError("mse_loss_weight must be finite and nonnegative")
    if align_target not in ("item", "latent", "quantized") or align_item not in ("current", "next"):
        raise ValueError("Invalid alignment target or item")
    if quantizer == "rqkmeans" and align_target == "latent" and mse_loss_weight > 0:
        raise ValueError("RQ-KMeans has no latent target; use item or quantized")
    if align_source not in ("last_inp", "last_out", "mean_out") or align_loss_type not in (
        "cos", "gram", "cosgram"
    ):
        raise ValueError("Invalid alignment source or loss")
    set_seed(seed)
    accelerator = setup_accelerator(
        split_batches=False, gradient_accumulate_every=gradient_accumulate_every,
        amp=mixed_precision != "no", mixed_precision_type=mixed_precision,
        timeout_minutes=60 if linear_probe else 3,
    )
    paths = {"model": model_name, "split": get_run_split("beauty")}
    save_dir = save_dir_root.format(**paths)
    if mse_loss_weight > 0:
        save_dir += f"_{align_target}_{align_item}_{align_source}_{align_loss_type}{mse_loss_weight:g}"
    logger = setup_training_logger(accelerator, save_dir, run_config, paths)
    sequence_config = dict(
        enabled=sequence_geometry, cosine_pairs_per_batch=sequence_geometry_cosine_pairs,
        seed=sequence_geometry_seed,
        method={**GEOMETRY_DESCRIPTION,
                "sequence_hidden": "Qwen3 final decoder layer: answer separator, SID0, SID1; "
                                   "three supervised SID prediction positions, no prompt/padding",
                "epoch_reduction": "Arithmetic mean of valid local microbatch metrics across "
                                   "all ranks; includes actual distributed sampler padding"},
    )
    sequence_path = (os.path.join(save_dir, f"sequence_geometry_{uuid4().hex}.jsonl")
                     if sequence_geometry and accelerator.is_main_process else None)
    sequence_latest = None
    # Let rank zero populate missing embedding caches before other ranks read them.
    with accelerator.main_process_first(), main_process_loading_output(accelerator):
        datasets = load_lm_datasets(
            root=dataset_folder.format(**paths), split=paths["split"], max_seq_len=max_seq_len,
            quantizer=quantizer, encoder_model_name=encoder_model_name,
            pretrained_rqvae_path=pretrained_rqvae_path.format(**paths),
            pretrained_rqkmeans_path=pretrained_rqkmeans_path.format(**paths),
            codebook_size=codebook_size, sem_id_dim=sem_id_dim, rqvae_config=rqvae_config,
            align_target=align_target if mse_loss_weight > 0 else None, align_item=align_item,
        )
    if any(len(value) == 0 for value in datasets.values()):
        raise ValueError("Train, valid and test datasets must all be nonempty")
    item_probe = None
    if linear_probe:
        reference_path = (
            linear_probe_reference_path.format(**paths) if linear_probe_reference_path else
            os.path.join(dataset_folder.format(**paths), "processed", paths["split"],
                         "item_emb_sentence-t5-xl.parquet")
        )
    with main_process_loading_output(accelerator):
        tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    if tokenizer.eos_token_id is None:
        raise ValueError("Base tokenizer must define EOS")
    # Match the reference, but use attention masks rather than token IDs to mask padding.
    tokenizer.pad_token = tokenizer.eos_token
    sid_token_ids = register_sid_tokens(tokenizer, codebook_size, sem_id_dim)
    collator = LCSIDCollator(
        tokenizer, sid_token_ids, max_seq_len, model_max_length, prompt_template
    )
    if linear_probe:
        item_probe = on_main_process(accelerator, lambda: Qwen3ItemProbe(
            datasets["train"], reference_path, save_dir, logger, collator=collator,
            batch_size=linear_probe_batch_size, seed=linear_probe_seed,
            learning_rate=linear_probe_learning_rate, max_epochs=linear_probe_max_epochs,
            patience=linear_probe_patience,
        ))
    loaders = {
        split: DataLoader(
            dataset, batch_size=batch_size if split == "train" else infer_batch_size,
            shuffle=split == "train", collate_fn=collator, num_workers=num_workers,
            pin_memory=accelerator.device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }
    target = datasets["train"][0].align_target_emb
    align_dim = len(target) if mse_loss_weight > 0 and target is not None else None
    if mse_loss_weight > 0 and align_dim is None:
        raise ValueError("The training dataset must supply alignment vectors")
    with main_process_loading_output(accelerator):
        model = build_lcrec(
            base_model, tokenizer, sid_token_ids, datasets["train"].sem_ids_list,
            mixed_precision=mixed_precision, lora_r=lora_r, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout, lora_target_modules=lora_target_modules,
            gradient_checkpointing=gradient_checkpointing, align_dim=align_dim,
            align_source=align_source, align_loss_type=align_loss_type,
        )
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad),
                      lr=learning_rate, weight_decay=weight_decay)
    model, optimizer, *prepared = accelerator.prepare(
        model, optimizer, loaders["train"], loaders["valid"], loaders["test"]
    )
    loaders = dict(zip(("train", "valid", "test"), prepared))
    updates_per_epoch = math.ceil(len(loaders["train"]) / gradient_accumulate_every)
    scheduler = get_scheduler(
        lr_schedule, optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=updates_per_epoch * epochs,
    )
    monitor = "ce" if early_stop_metric == "ce" else f"Recall@{max(top_ks)}"
    logger.info(f"Backbone: {base_model}; SID: {quantizer}, encoder: {encoder_model_name}")
    logger.info("Checkpoint writing disabled; retaining results, logs and metadata only.")
    logger.info(f"Early stopping: {monitor}, patience: {early_stop_patience} validations")
    logger.info(f"Samples: { {key: len(value) for key, value in datasets.items()} }")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    wandb = None
    if wandb_logging and accelerator.is_main_process:
        import wandb
        wandb.init(project=f"{model_name}_{paths['split']}_training", config=run_config)
    best_value = float("inf") if early_stop_metric == "ce" else -float("inf")
    best_model, best_epoch, wait = None, -1, 0
    best_test = None
    best_probe = None
    global_step, best_step = 0, -1
    logger.info("Validation: every %d epochs; item probes: every %d optimizer updates (enabled=%s)",
                eval_valid_every_n_epochs, linear_probe_every_steps, linear_probe)

    def run_evaluation(epoch):
        nonlocal best_value, best_epoch, best_step, wait, best_probe, best_model, best_test
        metrics = {"epoch": epoch, "global_step": global_step}
        unwrapped = accelerator.unwrap_model(model, keep_fp32_wrapper=False)
        valid = evaluate_lm(
            unwrapped, loaders["valid"], accelerator, metric=early_stop_metric,
            beam_size=beam_size, top_ks=top_ks, constrain_sid=constrain_sid,
        )
        value = valid[monitor]
        improved = value < best_value if early_stop_metric == "ce" else value > best_value
        if improved:
            best_value, best_epoch, best_step, wait = value, epoch, global_step, 0
            if item_probe is not None:
                # A preceding probe step is not the epoch-end selected model.
                latest = item_probe.latest
                best_probe = latest if latest and latest["global_step"] == global_step else None
            best_model = None
            best_model = copy.deepcopy(unwrapped).eval().requires_grad_(False)
            if accelerator.is_main_process:
                with open(os.path.join(save_dir, "sid_catalogue.json"), "w") as handle:
                    json.dump(datasets["train"].sem_ids_list, handle)
                with open(os.path.join(save_dir, "run_config.json"), "w") as handle:
                    json.dump(run_config, handle, indent=2)
            best_test = evaluate_lm(
                best_model, loaders["test"], accelerator, metric="hr",
                beam_size=beam_size, top_ks=top_ks, constrain_sid=constrain_sid,
            )
            metrics.update({f"test/{key}": value for key, value in best_test.items()})
            logger.info(f"New best epoch: {best_epoch}, step: {best_step}; test: {best_test}")
        else:
            wait += 1
        metrics.update({f"valid/{key}": value for key, value in valid.items()})
        logger.info(f"Epoch {epoch} step={global_step} - Valid: {valid}; "
                    f"best epoch: {best_epoch}; best step: {best_step}; wait: {wait}")
        if wandb is not None:
            wandb.log(metrics)

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        model.train()
        sequence_stats = (SequenceGeometryEpoch(
            accelerator.device, sequence_geometry_cosine_pairs,
            sequence_geometry_seed + epoch * accelerator.num_processes + accelerator.process_index,
        ) if sequence_geometry else None)
        ce_total, align_total, count = 0.0, 0.0, 0
        for step, batch in enumerate(tqdm(
            loaders["train"], desc=f"Epoch {epoch}", disable=not accelerator.is_main_process
        )):
            updated = False
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    output = (model(batch, return_last_hidden_state=True)
                              if sequence_geometry else model(batch))
                    loss = output.loss + mse_loss_weight * output.align_loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                # Accelerator divides by the configured accumulation count, including
                # the last window; compensate when that window is shorter.
                window_size = min(gradient_accumulate_every, len(loaders["train"])
                                  - (step // gradient_accumulate_every) * gradient_accumulate_every)
                accelerator.backward(loss * gradient_accumulate_every / window_size)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                    global_step += 1
                    updated = True
                optimizer.zero_grad(set_to_none=True)
            size = len(batch["labels"])
            ce_total += output.loss.item() * size
            align_total += output.align_loss.item() * size
            count += size
            if sequence_stats is not None:
                sequence_stats.update(
                    output.last_hidden_state,
                    torch.ones_like(batch["labels"], dtype=torch.bool),
                )
            if linear_probe and updated and global_step % linear_probe_every_steps == 0:
                probe_metrics = on_main_process(accelerator, lambda: item_probe.evaluate(
                    accelerator.unwrap_model(model, keep_fp32_wrapper=False),
                    epoch, mixed_precision, global_step=global_step,
                ))
                if wandb is not None and probe_metrics is not None:
                    wandb.log({"epoch": epoch, "global_step": global_step, **probe_metrics})
        log = aggregate_training_log(
            accelerator, epoch, ce_total, align_total, count, mse_loss_weight
        )
        if sequence_stats is not None:
            sequence_latest = {
                "epoch": epoch, "global_step": global_step,
                **aggregate_sequence_geometry(accelerator, sequence_stats),
                "config": sequence_config,
            }
            log.update(geometry_log_metrics(sequence_latest, "geometry/sequence"))
            if accelerator.is_main_process:
                with open(sequence_path, "a") as handle:
                    handle.write(json.dumps(sequence_latest, allow_nan=False) + "\n")
                logger.info(
                    "[SequenceGeometry] epoch=%d mean_effective_rank=%s mean_avg_cosine=%s "
                    "batches=%d valid_batches=%s",
                    epoch, sequence_latest["effective_rank"], sequence_latest["avg_cosine"],
                    sequence_latest["batches"], sequence_latest["valid_batches"],
                )
        logger.info(
            f"Epoch {epoch} - loss: {log['train/loss']:.6f}, "
            f"ntp: {log['train/ce']:.6f}, align: {log['train/align']:.6f}"
        )
        if epoch == 0 or (epoch + 1) % eval_valid_every_n_epochs == 0:
            run_evaluation(epoch)
        if wandb is not None:
            wandb.log(log)
        if wait >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch}, step {global_step}; "
                        f"best epoch: {best_epoch}, best step: {best_step}")
            break
    if best_test is None:
        raise RuntimeError("No best model selected")
    # Reuse the test metrics from the latest validation improvement, without
    # evaluating again or selecting a checkpoint using test performance.
    results = dict(best_epoch=best_epoch, best_step=best_step, global_step=global_step,
                   best_valid_metric=monitor,
                   best_valid_value=best_value, test=best_test)
    results.update(sequence_geometry=sequence_config, sequence_geometry_latest=sequence_latest,
                   sequence_geometry_path=sequence_path)
    if accelerator.is_main_process:
        if item_probe is not None:
            results.update(linear_probe_latest=item_probe.latest, linear_probe_best=best_probe,
                           linear_probe_directory=str(item_probe.directory))
        save_run_results(save_dir, model_name, paths["split"], seed, best_test,
                         config=run_config, extra=results)
        logger.info(f"Best epoch: {best_epoch}; cached best-model test: {best_test}")
        if wandb is not None:
            wandb.log({f"best_test/{key}": value for key, value in best_test.items()})
            wandb.finish()
    accelerator.wait_for_everyone()
    return results


if __name__ == "__main__":
    parse_config()
    train_lcrec()

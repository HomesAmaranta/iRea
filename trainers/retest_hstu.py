"""Evaluate an existing HSTU checkpoint without rebuilding training state.

Use the original general_trainer gin config/overrides, plus
--gin "retest.checkpoint_path='...'" and --gin "retest.training_log='...'".
retest.smoke_batches > 0 runs a bounded check without writing results.json.
"""

import ast
import inspect
import itertools
import json
import math
from pathlib import Path
import re

import gin
import torch
from torch.utils.data import DataLoader

from data.sid import sid_collate
from functools import partial
from models.factory import build_sid_model
from modules.utils import get_run_split, parse_config, setup_logger
from trainers.general_trainer import configure_alignment, evaluate, train
from trainers.trainer_utils import set_seed


def read_best_validation(training_log, monitor, mode):
    """Recover the original validation selection, never select on test metrics."""
    candidates = []
    stopped_best = None
    with open(training_log) as handle:
        for line in handle:
            match = re.search(r"Epoch (\d+) - Valid: (\{.*\})", line)
            if match:
                metrics = ast.literal_eval(match[2])
                value = float(metrics[monitor])
                if not math.isfinite(value):
                    raise ValueError("Non-finite validation metric in training log")
                candidates.append((int(match[1]), value))
            match = re.search(r"Early stopping at epoch \d+; best epoch: (\d+)", line)
            if match:
                stopped_best = int(match[1])
    if not candidates:
        raise ValueError("Training log has no validation metrics")
    select = min if mode == "ce" else max
    best_epoch, best_value = select(candidates, key=lambda item: item[1])
    if stopped_best is not None and best_epoch != stopped_best:
        raise ValueError("Recovered best epoch disagrees with early stopping log")
    return best_epoch, best_value


@gin.configurable
def retest(checkpoint_path, training_log, infer_batch_size=64, smoke_batches=0):
    if infer_batch_size <= 0 or smoke_batches < 0:
        raise ValueError("Batch size must be positive and smoke_batches nonnegative")
    options = {
        key: parameter.default
        for key, parameter in inspect.signature(train).parameters.items()
    }
    options.update(gin.get_bindings(train))
    if options["model_name"] != "hstu":
        raise ValueError("This checkpoint-only evaluator supports HSTU")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file() or not Path(training_log).is_file():
        raise FileNotFoundError("Both checkpoint and original training log are required")
    split = get_run_split("beauty")
    root = Path(options["save_dir_root"].format(model="hstu", split=split))
    if options["mse_loss_weight"] > 0 and options["align_loss_type"] == "cosgram":
        if options["cos_loss_weight"] != 0.5:
            root = Path(str(root) + f"_cw{options['cos_loss_weight']}")
    if checkpoint_path.parent.resolve() != root.resolve():
        raise ValueError("Checkpoint must belong to the configured experiment directory")
    if Path(training_log).parent.resolve() != root.resolve():
        raise ValueError("Training log must belong to the checkpoint experiment directory")
    result_path = root / "results.json"
    if not smoke_batches and result_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {result_path}")
    mode = options["early_stop_metric"]
    monitor = "ce" if mode == "ce" else f"Recall@{max(options['top_ks'])}"
    best_epoch, best_value = read_best_validation(training_log, monitor, mode)
    set_seed(options["seed"])
    device = torch.device(options["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_config = dict(options["model_config"] or {})
    for key in ("embed_dim", "num_heads", "num_blocks", "dropout",
                "num_position_buckets", "num_time_buckets", "use_temporal_bias"):
        if options[key] is not None:
            model_config[key] = options[key]
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    # No optimizer or second model copy; avoid allocating/initializing a large CPU model.
    with torch.device("meta"):
        model = build_sid_model(
            "hstu", options["codebook_size"], options["sem_id_dim"],
            options["max_seq_len"], model_config,
        )
        if options["mse_loss_weight"] > 0:
            target_dim = (
                state["hidden_to_item_emb.2.weight"].shape[0]
                if options["align_loss_type"] != "gram" else 1
            )
            configure_alignment(model, target_dim, options["align_loss_type"])
    model.load_state_dict(state, strict=True, assign=True)
    model = model.to(device).eval()
    del state
    dataset_cls = options["dataset"]
    if dataset_cls is None:
        from data.amazon import AmazonSeqDataset
        dataset_cls = AmazonSeqDataset
    dataset = dataset_cls(
        root=options["dataset_folder"], train_test_split="test",
        max_seq_len=options["max_seq_len"],
        pretrained_rqvae_path=options["pretrained_rqvae_path"].format(model="hstu", split=split),
    )
    loader = DataLoader(
        dataset, batch_size=infer_batch_size, shuffle=False,
        num_workers=options["num_workers"], pin_memory=device.type == "cuda",
        collate_fn=partial(
            sid_collate, codebook_size=options["codebook_size"],
            sem_id_dim=options["sem_id_dim"], max_seq_len=options["max_seq_len"],
            require_timestamps=model_config.get("use_temporal_bias", False),
        ),
    )
    logger = setup_logger(str(root), name="hstu_retest")
    logger.info(
        f"Checkpoint: {checkpoint_path}; best epoch: {best_epoch}; "
        f"test samples: {len(dataset)}; infer batch: {infer_batch_size}; "
        f"beam: {options['beam_size']}; precision: {options['mixed_precision']}"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metrics = evaluate(
        model, itertools.islice(loader, smoke_batches) if smoke_batches else loader,
        device, beam_size=options["beam_size"], top_ks=options["top_ks"],
        constrain_sid=options["constrain_sid"], mixed_precision=options["mixed_precision"],
        desc="Checkpoint smoke test" if smoke_batches else "Best model test",
    )
    if device.type == "cuda":
        logger.info(f"Peak allocated GPU memory: {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB")
    if smoke_batches:
        logger.info(f"Smoke metrics (not a full test result): {metrics}")
        return metrics
    results = {
        "model_name": "hstu", "best_epoch": best_epoch, "seed": options["seed"],
        "early_stop_metric": mode, "best_valid_metric": monitor,
        "best_valid_value": best_value,
        "best_valid_ce": best_value if mode == "ce" else None,
        "best_valid_recall": best_value if mode == "hr" else None,
        "test": metrics, "model_config": model_config,
        **{key: options[key] for key in (
            "codebook_size", "sem_id_dim", "max_seq_len", "align_target",
            "align_item", "mse_loss_weight", "mse_loss_mode",
            "align_loss_type", "cos_loss_weight",
        )},
        "evaluation": {
            "checkpoint": str(checkpoint_path), "training_log": str(training_log),
            "infer_batch_size": infer_batch_size, "beam_size": options["beam_size"],
            "mixed_precision": options["mixed_precision"],
            "constrain_sid": options["constrain_sid"], "test_samples": len(dataset),
        },
    }
    # Only a completed, full test can create the final artifact.
    with result_path.open("x") as handle:
        json.dump(results, handle, indent=2)
    logger.info(f"Best epoch: {best_epoch}, final test: {metrics}")
    return results


if __name__ == "__main__":
    parse_config()
    retest()

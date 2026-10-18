"""Rank-zero, frozen single-item diagnostics for the LCRec Qwen3 backbone."""

import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import torch
from torch.nn import functional as F
from accelerate.utils import broadcast_object_list

from trainers.affine_probe import (
    DEFAULT_LR, DEFAULT_MAX_EPOCHS, DEFAULT_PATIENCE, PROTOCOL, RECALL_KS,
    fit_probe, retrieval_metrics, validate_fit_config,
)
from trainers.category_probe import (
    CategoryProbeNumericalError, ItemCategoryProbe, load_dataset_labels,
)
from trainers.item_probe import (
    CKA_DESCRIPTION, _center_cosine_features, _cka_centered, _debiased_self_norm,
    GENREC_CKA_DESCRIPTION, genrec_linear_cka, load_reference, split_items,
)
from trainers.representation_geometry import (
    GEOMETRY_DESCRIPTION, geometry_log_metrics, item_geometry,
)

RAW_SID_FORMAT = "single_item_raw_sid_last_v1"
RAW_SID_DESCRIPTION = (
    "Single item, exactly three registered SID tokens, no prompt/BOS/EOS/separator; "
    "last SID token of final decoder layer; no projection"
)


def on_main_process(accelerator, action):
    """Workers wait without running duplicate probes; propagate rank-zero failures."""
    accelerator.wait_for_everyone()
    result, failure = None, [None]
    if accelerator.is_main_process:
        try:
            result = action()
        except Exception as error:
            failure[0] = f"{type(error).__name__}: {error}"
    if accelerator.num_processes > 1:
        broadcast_object_list(failure, from_process=0)
    if failure[0] is not None:
        raise RuntimeError(f"LCRec item probe failed on rank zero: {failure[0]}")
    return result


@torch.no_grad()
def extract_qwen3_item_hidden(model, sem_ids, batch_size=32, mixed_precision="bf16", *,
                             collator=None, raw_sid=False):
    """Run only the adapted decoder, avoiding vocabulary logits and layer snapshots."""
    if batch_size <= 0 or mixed_precision not in ("no", "bf16"):
        raise ValueError("Invalid LCRec probe batch size or precision")
    codes = torch.as_tensor(sem_ids)
    table = model.sid_token_ids
    if (codes.ndim != 2 or len(codes) == 0 or codes.shape[1] != table.shape[0]
            or codes.dtype not in (torch.int32, torch.int64)
            or (codes < 0).any() or (codes >= table.shape[1]).any()):
        raise ValueError("Expected one complete, unoffset SID per item")
    if raw_sid and table.shape[0] != 3:
        raise ValueError("Raw SID diagnostics require exactly three SID tokens")
    if not raw_sid and collator is None:
        raise ValueError("Prompt diagnostics require a collator")
    device = next(model.parameters()).device
    codes = codes.cpu().long()
    backbone = model.backbone
    # get_base_model keeps injected LoRA layers and modules_to_save wrappers active.
    causal_lm = backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone
    decoder = causal_lm.model
    states = [(module, module.training) for module in model.modules()]
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    try:
        model.eval()
        hidden = []
        with torch.random.fork_rng(devices=cuda_devices):
            for start in range(0, len(codes), batch_size):
                if raw_sid:
                    batch_codes = codes[start:start + batch_size].to(table.device)
                    ids = table[torch.arange(3, device=table.device), batch_codes].to(device)
                    mask = torch.ones_like(ids)
                else:
                    # Targets only satisfy the collator interface; never pass them to the model.
                    batch = collator([SimpleNamespace(item_ids=code, target_ids=code)
                                      for code in codes[start:start + batch_size]])
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                positions = mask.cumsum(-1) - 1
                positions.masked_fill_(mask == 0, 0)
                if not raw_sid and (ids.shape[1] < 2 or not mask[:, -2:].all()):
                    raise ValueError("Probe prompt must end with text then the answer separator")
                context = (nullcontext() if mixed_precision == "no" else
                           torch.autocast(device_type=device.type, dtype=torch.bfloat16))
                with context:
                    output = decoder(
                        input_ids=ids, attention_mask=mask, position_ids=positions,
                        use_cache=False, output_hidden_states=False, return_dict=True,
                    )
                hidden.append(output.last_hidden_state[:, -1 if raw_sid else -2, :].detach().float())
        values = torch.cat(hidden)
        if not torch.isfinite(values).all():
            raise RuntimeError("Non-finite LCRec probe hidden")
        return values
    finally:
        for module, training in states:
            module.training = training


class Qwen3ItemProbe:
    def __init__(self, dataset, reference_path, save_dir, logger, batch_size=32,
                 seed=42, *, collator, learning_rate=DEFAULT_LR,
                 max_epochs=DEFAULT_MAX_EPOCHS, patience=DEFAULT_PATIENCE):
        validate_fit_config(learning_rate, max_epochs, patience)
        self.fit_config = dict(
            learning_rate=learning_rate, max_epochs=max_epochs, patience=patience, seed=seed,
        )
        self.collator = collator
        self.item_ids = np.asarray(dataset.item_ids)
        self.splits = split_items(self.item_ids, seed)
        self.codes = torch.as_tensor(dataset.sem_ids_list)
        if self.codes.ndim != 2 or len(self.codes) != len(self.item_ids):
            raise ValueError("LCRec SID catalogue must match ItemIDs")
        self.target = load_reference(reference_path, self.item_ids)
        self.centered_target = self.target_cov_norm = None
        labels, metadata = load_dataset_labels(dataset, self.item_ids)
        if labels is None:
            raise ValueError(f"LCRec category probe unavailable: {metadata}")
        self.run_id = uuid4().hex
        self.directory = Path(save_dir) / "item_probe" / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.category = ItemCategoryProbe(
            labels, self.item_ids, self.splits, self.directory, save_weights=False,
        )
        if self.category.metadata["status"] != "ready":
            raise ValueError(f"LCRec category probe unavailable: {self.category.metadata}")
        self.batch_size, self.logger, self.latest = batch_size, logger, None
        metadata = {
            "run_id": self.run_id, "seed": seed, "batch_size": batch_size,
            "reference_path": str(Path(reference_path).resolve()),
            "reference_sha256": hashlib.sha256(self.target.numpy().tobytes()).hexdigest(),
            "sid_sha256": hashlib.sha256(self.codes.numpy().tobytes()).hexdigest(),
            "hidden": "Training prompt with one history item; last_inp before answer separator",
            "probe_input_format": "single_item_prompt_last_inp_v1",
            "prompt_template": collator.prompt_template,
            "cka": CKA_DESCRIPTION,
            "cka_genrec": GENREC_CKA_DESCRIPTION,
            "raw_sid": {
                "enabled": True, "probe_input_format": RAW_SID_FORMAT,
                "hidden": RAW_SID_DESCRIPTION,
                "cka": CKA_DESCRIPTION, "cka_genrec": GENREC_CKA_DESCRIPTION,
                "scope": "all catalogue items, ItemID-aligned Sentence-T5; CKA only",
            },
            "geometry": GEOMETRY_DESCRIPTION,
            "category_probe": self.category.metadata,
            "split_item_ids": {s: self.item_ids[v.numpy()].tolist() for s, v in self.splits.items()},
            "compute_device": "rank-zero recommender device; workers wait; no CPU numerical fallback",
            "save_weights": False,
            "protocol": PROTOCOL,
            "fit_config": self.fit_config,
            "fit": "Independent raw-hidden Linear -> Sentence-T5; MSE + (1 - mean cosine)",
            "selection": "minimum validation MSE+cosine; fresh fit each interval; restore best",
            "retrieval": {"top_ks": list(RECALL_KS), "queries": "test items",
                          "candidates": "all Sentence-T5 items", "ties": "ascending ItemID"},
        }
        (self.directory / "config.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
        logger.info(
            "[ItemProbe] LCRec enabled at evaluation intervals: items=%d category_level=2 details=%s",
            len(self.item_ids), self.directory,
        )

    def evaluate(self, model, epoch, mixed_precision="bf16", global_step=None):
        start = time.perf_counter()
        hidden = extract_qwen3_item_hidden(
            model, self.codes, self.batch_size, mixed_precision, collator=self.collator)
        if self.centered_target is None or self.centered_target.device != hidden.device:
            self.target = self.target.to(hidden.device)
            self.centered_target = _center_cosine_features(self.target)
            self.target_cov_norm = _debiased_self_norm(self.centered_target)
        with torch.no_grad(), torch.autocast(device_type=hidden.device.type, enabled=False):
            cka = _cka_centered(_center_cosine_features(hidden), self.centered_target, self.target_cov_norm)
            cka_genrec = genrec_linear_cka(hidden, self.target)
            geometry = item_geometry(hidden)
        raw_hidden = extract_qwen3_item_hidden(
            model, self.codes, self.batch_size, mixed_precision, raw_sid=True)
        with torch.no_grad(), torch.autocast(device_type=hidden.device.type, enabled=False):
            raw_cka = _cka_centered(
                _center_cosine_features(raw_hidden), self.centered_target, self.target_cov_norm)
            raw_genrec = genrec_linear_cka(raw_hidden, self.target)
        raw_record = {
            "probe_input_format": RAW_SID_FORMAT, "hidden": RAW_SID_DESCRIPTION,
            "items": len(raw_hidden), "hidden_dim": raw_hidden.shape[1],
            "cka": raw_cka, "cka_status": "valid" if raw_cka is not None else "degenerate",
            "cka_method": CKA_DESCRIPTION,
            "cka_genrec": raw_genrec,
            "cka_genrec_status": "valid" if raw_genrec is not None else "undefined",
            "cka_genrec_method": GENREC_CKA_DESCRIPTION,
        }
        del raw_hidden
        state, fit = fit_probe(hidden, self.target, self.splits, **self.fit_config)
        test_rows = self.splits["test"].to(hidden.device)
        with torch.no_grad(), torch.autocast(device_type=hidden.device.type, enabled=False):
            prediction = F.linear(hidden[test_rows].float(), **state)
            recall, _ = retrieval_metrics(
                prediction, self.target, test_rows, batch_size=self.batch_size,
                top_ks=RECALL_KS, item_ids=self.item_ids,
            )
        fit["splits"]["test"].update({key: value["value"] for key, value in recall.items()})
        # fit_category_probe enables gradients only for its detached affine classifier.
        try:
            category = self.category.evaluate(hidden, epoch)
        except CategoryProbeNumericalError as error:
            category = {
                "status": "nonfinite", "reason": str(error),
                "category_level": 2, "device": str(hidden.device),
                "failed_epoch": error.epoch,
            }
            self.logger.warning(
                "[ItemProbe] epoch=%d step=%s category_status=nonfinite; "
                "acc unavailable, keeping CKA and continuing training. reason=%s",
                epoch, global_step, error,
            )
        record = {
            "run_id": self.run_id, "epoch": epoch, "global_step": global_step, "items": len(hidden),
            "probe_input_format": "single_item_prompt_last_inp_v1",
            "device": str(hidden.device), "hidden_dim": hidden.shape[1],
            "target_dim": self.target.shape[1], "cka": cka,
            "cka_status": "valid" if cka is not None else "degenerate",
            "cka_method": CKA_DESCRIPTION,
            "cka_genrec": cka_genrec,
            "cka_genrec_status": "valid" if cka_genrec is not None else "undefined",
            "cka_genrec_method": GENREC_CKA_DESCRIPTION,
            "geometry": geometry,
            "raw_sid": raw_record,
            "category": category,
            "test": {
                **fit["splits"]["test"],
                "acc": (category["splits"]["test"]["accuracy"]
                        if category["status"] == "valid" else None),
            },
            "retrieval": {"recall": recall, "catalogue_items": len(self.item_ids),
                          "test_queries": len(test_rows), "ties": "ascending ItemID"},
            "elapsed_seconds": time.perf_counter() - start, **fit,
        }
        with (self.directory / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        metrics = geometry_log_metrics(geometry, "geometry/item")
        metrics.update({f"linear_probe/{split}_{key}": value
                        for split, values in fit["splits"].items() for key, value in values.items()})
        metrics.update({"linear_probe/best_epoch": fit["best_epoch"],
                        "linear_probe/best_valid_loss": fit["best_valid_loss"]})
        if cka is not None:
            metrics["linear_probe/cka"] = cka
        if cka_genrec is not None:
            metrics["linear_probe/cka_genrec"] = cka_genrec
        if raw_cka is not None:
            metrics["raw_sid/cka"] = raw_cka
        if raw_genrec is not None:
            metrics["raw_sid/cka_genrec"] = raw_genrec
        for split, values in category.get("splits", {}).items():
            metrics[f"category_probe/{split}_acc"] = values["accuracy"]
            metrics[f"category_probe/{split}_macro_f1"] = values["macro_f1"]
        if category["status"] == "valid":
            metrics["category_probe/best_valid_ce"] = category["best_valid_ce"]
            metrics["category_probe/best_epoch"] = category["best_epoch"]
        self.logger.info(
            "[ItemProbe] epoch=%d step=%s device=%s clipped_debiased_cka=%s cka_genrec=%s "
            "category_level=2 category_test_acc=%s category_test_macro_f1=%s "
            "category_valid_ce=%s category_best_epoch=%s seconds=%.2f",
            epoch, global_step, hidden.device, "NA" if cka is None else f"{cka:.6f}",
            "NA" if cka_genrec is None else f"{cka_genrec:.6f}",
            (f"{category['splits']['test']['accuracy']:.6f}" if category["status"] == "valid" else "NA"),
            (f"{category['splits']['test']['macro_f1']:.6f}" if category["status"] == "valid" else "NA"),
            category.get("best_valid_ce", "NA"), category.get("best_epoch", "NA"),
            record["elapsed_seconds"],
        )
        self.logger.info(
            "[ItemGeometry] epoch=%d effective_rank=%s avg_cosine=%s",
            epoch, geometry["effective_rank"], geometry["avg_cosine"],
        )
        self.logger.info(
            "[RawSIDCKA] epoch=%d step=%s device=%s cka=%s cka_genrec=%s",
            epoch, global_step, hidden.device,
            "NA" if raw_cka is None else f"{raw_cka:.6f}",
            "NA" if raw_genrec is None else f"{raw_genrec:.6f}",
        )
        self.logger.info(
            "[ItemRegression] epoch=%d step=%s test_mse=%.8g test_cos=%.6f "
            "recall=%s best_epoch=%d best_valid_loss=%.8g",
            epoch, global_step, fit["splits"]["test"]["mse"], fit["splits"]["test"]["cosine"],
            {key: value["value"] for key, value in recall.items()},
            fit["best_epoch"], fit["best_valid_loss"],
        )
        self.latest = record
        return metrics

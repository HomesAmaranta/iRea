"""Frozen single-item SID probes and full-catalogue clipped debiased CKA."""

import hashlib
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from trainers.affine_probe import (
    DEFAULT_LR, DEFAULT_MAX_EPOCHS, DEFAULT_PATIENCE, PROTOCOL, RECALL_KS,
    fit_probe, retrieval_metrics, validate_fit_config,
)
from trainers.category_probe import ItemCategoryProbe, load_dataset_labels
from trainers.representation_geometry import (
    GEOMETRY_DESCRIPTION, geometry_log_metrics, item_geometry,
)


CKA_DESCRIPTION = (
    "debiased cosine-Gram CKA (arXiv:2605.09969); all pre-projection items; "
    "clip at mean per-item absolute-feature q95, then L2 normalize; "
    "unbiased HSIC / (sqrt(HSIC_XX * HSIC_YY) + 1e-6); "
    "FP32 preprocessing, FP64 equivalent feature-space computation"
)

GENREC_CKA_DESCRIPTION = (
    "genrec/modules/cka.py::linear_cka; centered raw-feature linear CKA; "
    "all pre-projection item hidden vs ItemID-aligned Sentence-T5; "
    "FP32 with autocast disabled; no clipping, L2 normalization, debiasing, epsilon or clamping; "
    "undefined/nonfinite scores serialized as null"
)


@torch.no_grad()
def genrec_linear_cka(x, y):
    """Match genrec's linear_cka numerics; convert undefined NaN to JSON-safe None."""
    if x.ndim != 2 or y.ndim != 2 or x.size(0) != y.size(0):
        raise ValueError("CKA requires two matrices with matching item rows")
    if x.size(0) < 2 or min(x.size(1), y.size(1)) < 1:
        raise ValueError("CKA requires at least two items and nonempty features")
    if x.device != y.device:
        raise ValueError("CKA matrices must be on the same device")
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, y = x.float(), y.float()
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        cross = (x.T @ y).square().sum()
        denominator = torch.linalg.vector_norm(x.T @ x) * torch.linalg.vector_norm(y.T @ y)
        score = torch.where(
            denominator > 0, cross / denominator, denominator.new_full((), float("nan"))
        )
    return float(score) if torch.isfinite(score) else None


def split_items(item_ids, seed):
    """Stable 80/10/10 item splits, independent of catalogue row ordering."""
    ids = torch.as_tensor(np.asarray(item_ids), dtype=torch.long)
    if ids.ndim != 1 or ids.unique().numel() != ids.numel() or ids.numel() < 10:
        raise ValueError("Item probe requires at least 10 unique integer ItemIDs")
    order = ids.argsort()
    permutation = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed))
    order = order[permutation]
    train_end, valid_end = int(0.8 * len(ids)), int(0.9 * len(ids))
    return dict(train=order[:train_end], valid=order[train_end:valid_end], test=order[valid_end:])


def load_reference(path, item_ids):
    """Join the frozen Sentence-T5 cache to SID rows by ItemID, never row position."""
    frame = pd.read_parquet(path, columns=["ItemID", "embedding"])
    ids = np.asarray(item_ids)
    if (not np.issubdtype(ids.dtype, np.integer)
            or not pd.api.types.is_integer_dtype(frame["ItemID"])):
        raise ValueError("Probe ItemIDs and reference ItemIDs must be integers")
    if (frame["ItemID"].duplicated().any() or len(np.unique(ids)) != len(ids)
            or set(frame["ItemID"]) != set(ids.tolist())):
        raise ValueError("Probe catalogue and Sentence-T5 cache must have identical unique ItemIDs")
    values = np.stack(frame.set_index("ItemID").loc[ids, "embedding"].to_numpy())
    target = torch.as_tensor(values, dtype=torch.float64)
    if target.ndim != 2 or target.shape[1] == 0 or not torch.isfinite(target).all():
        raise ValueError("Sentence-T5 probe targets must be finite nonempty vectors")
    return target


HIDDEN_DESCRIPTIONS = {
    "last": "normal forward without labels; last_hidden_state[:, -1, :]",
    "encoder_mean": "TIGER encoder last layer; FP32 mean over all three SID tokens; "
                    "no labels/history/prefix, normalization or projection before pooling",
}


@torch.no_grad()
def extract_item_hidden(model, sem_ids, batch_size=256, mixed_precision="no",
                        hidden_source="last"):
    """Extract isolated item representations without changing model or RNG state."""
    if hidden_source not in HIDDEN_DESCRIPTIONS:
        raise ValueError("Unknown item hidden source")
    if hidden_source == "encoder_mean" and (
        not callable(getattr(model, "encode_item_hidden", None)) or model.sem_id_dim != 3
    ):
        raise ValueError("encoder_mean requires TIGER with three SID tokens")
    if batch_size <= 0 or mixed_precision not in ("no", "bf16"):
        raise ValueError("Invalid probe extraction batch size or precision")
    codes = torch.as_tensor(sem_ids)
    if (codes.ndim != 2 or codes.shape[0] == 0 or codes.shape[1] != model.sem_id_dim
            or codes.dtype not in (torch.int32, torch.int64)
            or (codes < 0).any() or (codes >= model.codebook_size).any()):
        raise ValueError("Probe needs exactly one complete, unoffset SID per catalogue item")
    device = next(model.parameters()).device
    offsets = torch.arange(model.sem_id_dim, device=device) * model.codebook_size + 1
    tokens = codes.to(device=device, dtype=torch.long) + offsets
    states = [(module, module.training) for module in model.modules()]
    # Protect the training RNG even if a future model samples during evaluation.
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    try:
        model.eval()
        hidden = []
        with torch.random.fork_rng(devices=cuda_devices):
            for start in range(0, len(tokens), batch_size):
                ids = tokens[start:start + batch_size].to(device)
                inp = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
                if getattr(model, "use_temporal_bias", False):
                    # All SID tokens belong to the same item, so time differences are zero.
                    inp["timestamps"] = torch.zeros_like(ids)
                context = (nullcontext() if mixed_precision == "no"
                           else torch.autocast(device_type=device.type, dtype=torch.bfloat16))
                with context:
                    value = (model.encode_item_hidden(inp) if hidden_source == "encoder_mean"
                             else model(inp).last_hidden_state[:, -1, :])
                hidden.append(value.detach().float())
        result = torch.cat(hidden)
        if not torch.isfinite(result).all():
            raise RuntimeError("Non-finite single-item hidden state in linear probe")
        return result
    finally:
        for module, training in states:
            module.training = training


def _center_cosine_features(features):
    # Match normalized_pair/remove_outliers in sophicle/tokens, metrics.py
    # (856f30932826c715b3bb88fbdfe146d583f71384): one shared threshold per space.
    values = features.detach().to(dtype=torch.float32)
    threshold = torch.quantile(values.abs(), 0.95, dim=1).mean()
    values = values.clamp(-threshold, threshold)
    values = F.normalize(values, dim=1, eps=1e-12)
    values = values.to(dtype=torch.float64)
    return values - values.mean(dim=0, keepdim=True)


def _debiased_hsic(x, y):
    """Unbiased HSIC for column-centered features, without N-by-N Gram matrices."""
    n = len(x)
    if n < 4:
        raise ValueError("Debiased HSIC requires at least four samples")
    x_sq, y_sq = x.square().sum(dim=1), y.square().sum(dim=1)
    # U-centering cancels feature translations. This is the hollow-Gram HSIC
    # identity applied after ordinary centering to reduce cancellation error.
    numerator = (
        (x.T @ y).square().sum()
        - n / (n - 2) * torch.dot(x_sq, y_sq)
        + x_sq.sum() * y_sq.sum() / ((n - 1) * (n - 2))
    )
    return numerator / (n * (n - 3))


def _debiased_self_norm(features):
    # Self-HSIC is nonnegative; roundoff in the equivalent formula may go below 0.
    return _debiased_hsic(features, features).clamp_min(0).sqrt()


def _cka_centered(x, y, target_cov_norm=None):
    x_norm = _debiased_self_norm(x)
    y_norm = (_debiased_self_norm(y)
              if target_cov_norm is None else target_cov_norm)
    denominator = x_norm * y_norm + 1e-6
    if not torch.isfinite(denominator):
        return None
    score = _debiased_hsic(x, y) / denominator
    return float(score) if torch.isfinite(score) else None


@torch.no_grad()
def cosine_gram_cka(hidden, target):
    """Paper's clipped debiased CKA over all items; no batch averaging or sampling."""
    if (hidden.ndim != 2 or target.ndim != 2 or len(hidden) != len(target)
            or len(hidden) < 4 or hidden.shape[1] == 0 or target.shape[1] == 0):
        raise ValueError("CKA requires nonempty feature matrices for the same four or more items")
    if not torch.isfinite(hidden).all() or not torch.isfinite(target).all():
        raise ValueError("CKA requires finite features")
    return _cka_centered(
        _center_cosine_features(hidden), _center_cosine_features(target.to(hidden.device)))


@torch.no_grad()
def regression_metrics(prediction, target):
    mse = F.mse_loss(prediction, target).item()
    cosine = F.cosine_similarity(prediction, target, dim=-1, eps=1e-12).mean().item()
    if not math.isfinite(mse) or not math.isfinite(cosine):
        raise RuntimeError("Non-finite linear probe regression metrics")
    return {"mse": mse, "cosine": cosine}


@torch.no_grad()
def fit_linear_probe(hidden, target, splits, alphas=(0., 1e-6, 1e-4, 1e-2, 1.)):
    """Legacy offline ridge baseline; epoch probes use fit_probe (MSE + cosine)."""
    if not alphas or any(not math.isfinite(a) or a < 0 for a in alphas):
        raise ValueError("Probe ridge strengths must be finite and nonnegative")
    x = hidden.detach().to(dtype=torch.float64)
    y = target.detach().to(device=x.device, dtype=torch.float64)
    splits = {name: torch.as_tensor(rows, device=x.device, dtype=torch.long)
              for name, rows in splits.items()}
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y)
            or not torch.isfinite(x).all() or not torch.isfinite(y).all()):
        raise ValueError("Probe features and targets must be finite, aligned matrices")
    train, valid = splits["train"], splits["valid"]
    x_train, y_train = x[train], y[train]
    mean = x_train.mean(dim=0)
    scale = x_train.std(dim=0, correction=0)
    scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
    y_mean = y_train.mean(dim=0)
    z = (x_train - mean) / scale
    u, singular, vh = torch.linalg.svd(z, full_matrices=False)
    rhs = u.T @ (y_train - y_mean)
    cutoff = torch.finfo(z.dtype).eps * max(z.shape) * singular.max()
    best = None
    candidates = []
    for alpha in alphas:
        if alpha == 0:
            inverse = torch.zeros_like(singular)
            active = singular > cutoff
            inverse[active] = singular[active].reciprocal()
        else:
            inverse = singular / (singular.square() + len(train) * alpha)
        standardized_weight = vh.T @ (inverse[:, None] * rhs)
        # Fold train-only standardization into an ordinary Linear(d, target_dim, bias=True).
        weight = (standardized_weight / scale[:, None]).T.contiguous()
        bias = y_mean - mean @ weight.T
        metrics = regression_metrics(F.linear(x[valid], weight, bias), y[valid])
        candidates.append({"alpha": float(alpha), "valid_mse": metrics["mse"]})
        if best is None or metrics["mse"] < best["valid_mse"]:
            best = dict(weight=weight, bias=bias, alpha=float(alpha), valid_mse=metrics["mse"])
    prediction = F.linear(x, best["weight"], best["bias"])
    metrics = {
        split: regression_metrics(prediction[indices], y[indices])
        for split, indices in splits.items()
    }
    return {"weight": best["weight"], "bias": best["bias"]}, {
        "alpha": best["alpha"], "candidates": candidates, "splits": metrics,
    }


class ItemLinearProbe:
    def __init__(self, dataset, reference_path, save_dir, logger, batch_size=256,
                 seed=42, category_probe=True, learning_rate=DEFAULT_LR,
                 max_epochs=DEFAULT_MAX_EPOCHS, patience=DEFAULT_PATIENCE,
                 hidden_source="last"):
        if hidden_source not in HIDDEN_DESCRIPTIONS:
            raise ValueError("Unknown item hidden source")
        self.hidden_source = hidden_source
        self.metric_prefix = "" if hidden_source == "last" else f"{hidden_source}/"
        if not hasattr(dataset, "item_ids") or not hasattr(dataset, "sem_ids_list"):
            raise ValueError("Linear probe requires dataset.item_ids and dataset.sem_ids_list")
        if batch_size <= 0:
            raise ValueError("Invalid linear probe batch size")
        validate_fit_config(learning_rate, max_epochs, patience)
        if not isinstance(category_probe, bool):
            raise ValueError("Invalid category probe switch")
        self.item_ids = np.asarray(dataset.item_ids)
        self.splits = split_items(self.item_ids, seed)
        self.codes = torch.as_tensor(dataset.sem_ids_list)
        if self.codes.ndim != 2 or len(self.codes) != len(self.item_ids):
            raise ValueError("Probe SID rows must correspond one-to-one to ItemIDs")
        self.target = load_reference(reference_path, self.item_ids)
        # Prepare and cache numerical reference statistics on the model device at first use.
        self.centered_target = None
        self.target_cov_norm = None
        self.batch_size = batch_size
        self.fit_config = dict(
            learning_rate=learning_rate, max_epochs=max_epochs, patience=patience, seed=seed,
        )
        self.logger = logger
        self.run_id = uuid4().hex
        folder = "item_probe" if hidden_source == "last" else f"item_probe_{hidden_source}"
        self.directory = Path(save_dir) / folder / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "metrics.jsonl"
        self.category_probe = None
        category_metadata = {"status": "disabled"}
        if category_probe:
            logger.info("[CategoryProbe] loading second-level category labels (missing = separate class)...")
            labels, category_metadata = load_dataset_labels(dataset, self.item_ids)
            if labels is not None:
                self.category_probe = ItemCategoryProbe(
                    labels, self.item_ids, self.splits, self.directory,
                )
                category_metadata.update(self.category_probe.metadata)
            logger.info(
                f"[CategoryProbe] status={category_metadata['status']} "
                f"counts={category_metadata.get('class_counts', {})} "
                f"reason={category_metadata.get('reason', '')}"
            )
        self.category_metadata = category_metadata
        metadata = {
            "run_id": self.run_id, "seed": seed, "reference_path": str(Path(reference_path).resolve()),
            "reference_sha256": hashlib.sha256(self.target.numpy().tobytes()).hexdigest(),
            "sid_sha256": hashlib.sha256(self.codes.numpy().tobytes()).hexdigest(),
            "batch_size": batch_size, "bias": True, "protocol": PROTOCOL,
            "fit": "Independent raw-hidden Linear -> Sentence-T5; MSE + (1 - mean cosine)",
            "fit_config": self.fit_config,
            "selection": "minimum validation MSE+cosine loss; fresh fit each epoch; restore best",
            "retrieval": {
                "queries": "test items only", "candidates": "all Sentence-T5 catalogue items",
                "positive": "exact ItemID", "ties": "ascending ItemID",
                "top_ks": list(RECALL_KS),
            },
            "hidden_source": hidden_source,
            "hidden": HIDDEN_DESCRIPTIONS[hidden_source],
            "cka": CKA_DESCRIPTION,
            "cka_genrec": GENREC_CKA_DESCRIPTION,
            "geometry": GEOMETRY_DESCRIPTION,
            "category_probe": category_metadata,
            "compute_device": "same as recommender; no CPU numerical fallback",
            "split_item_ids": {
                name: self.item_ids[indices.numpy()].tolist() for name, indices in self.splits.items()
            },
        }
        with open(self.directory / "config.json", "w") as handle:
            json.dump(metadata, handle, indent=2, allow_nan=False)
        logger.info(
            f"[ItemProbe] enabled: every epoch, items={len(self.item_ids)}, "
            f"split={ {k: len(v) for k, v in self.splits.items()} }, "
            f"target_dim={self.target.shape[1]}, Linear(bias=True), details={self.directory}"
        )
        self.latest = None

    def evaluate(self, model, epoch, mixed_precision="no"):
        start = time.perf_counter()
        hidden = extract_item_hidden(
            model, self.codes, self.batch_size, mixed_precision, self.hidden_source
        )
        if self.centered_target is None or self.target.device != hidden.device:
            self.target = self.target.to(hidden.device)
            self.centered_target = _center_cosine_features(self.target)
            self.target_cov_norm = _debiased_self_norm(self.centered_target)
        cka = _cka_centered(
            _center_cosine_features(hidden), self.centered_target, self.target_cov_norm
        )
        cka_genrec = genrec_linear_cka(hidden, self.target)
        geometry = item_geometry(hidden)
        state, fit = fit_probe(hidden, self.target, self.splits, **self.fit_config)
        test_rows = self.splits["test"].to(hidden.device)
        with torch.no_grad(), torch.autocast(device_type=hidden.device.type, enabled=False):
            prediction = F.linear(hidden[test_rows].float(), **state)
            recall, ranks = retrieval_metrics(
                prediction, self.target, test_rows, batch_size=self.batch_size,
                top_ks=RECALL_KS, item_ids=self.item_ids,
            )
        fit["splits"]["test"].update({key: value["value"] for key, value in recall.items()})
        # This classifier sees raw hidden, never regression predictions or weights.
        category = (self.category_probe.evaluate(hidden, epoch) if self.category_probe is not None
                    else {"status": self.category_metadata["status"],
                          "reason": self.category_metadata.get("reason")})
        test_summary = {
            **fit["splits"]["test"],
            "acc": (category["splits"]["test"]["accuracy"]
                    if category["status"] == "valid" else None),
        }
        record = {
            "run_id": self.run_id, "epoch": epoch, "items": len(hidden),
            "hidden_source": self.hidden_source,
            "device": str(hidden.device),
            "hidden_dim": hidden.shape[1], "target_dim": self.target.shape[1],
            "cka": cka, "cka_status": "valid" if cka is not None else "degenerate",
            "cka_method": CKA_DESCRIPTION,
            "cka_genrec": cka_genrec,
            "cka_genrec_status": "valid" if cka_genrec is not None else "undefined",
            "cka_genrec_method": GENREC_CKA_DESCRIPTION,
            "geometry": geometry,
            "category": category,
            "test": test_summary,
            "retrieval": {
                "recall": recall, "catalogue_items": len(self.item_ids),
                "test_queries": len(test_rows), "ties": "ascending ItemID",
            },
            "elapsed_seconds": time.perf_counter() - start, **fit,
        }
        # Keep the latest probe separately: never register it on the recommender or optimizer.
        torch.save({"epoch": epoch, "protocol": fit["protocol"],
                    "hidden_source": self.hidden_source,
                    "best_epoch": fit["best_epoch"], "best_valid_loss": fit["best_valid_loss"],
                    "state_dict": state},
                   self.directory / "latest_linear.pt")
        pd.DataFrame({
            "ItemID": self.item_ids[self.splits["test"].numpy()],
            "rank": ranks.cpu().numpy(),
        }).to_csv(self.directory / "latest_test_ranks.csv", index=False)
        with open(self.path, "a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        metrics = {f"linear_probe/{split}_{metric}": value
                   for split, values in fit["splits"].items() for metric, value in values.items()}
        metrics["linear_probe/best_epoch"] = fit["best_epoch"]
        metrics["linear_probe/best_valid_loss"] = fit["best_valid_loss"]
        if cka is not None:
            metrics["linear_probe/cka"] = cka
        if cka_genrec is not None:
            metrics["linear_probe/cka_genrec"] = cka_genrec
        metrics.update(geometry_log_metrics(geometry, "geometry/item"))
        category_text = ""
        if category["status"] == "valid":
            for split, values in category["splits"].items():
                metrics[f"category_probe/{split}_acc"] = values["accuracy"]
                metrics[f"category_probe/{split}_macro_f1"] = values["macro_f1"]
            metrics["category_probe/best_valid_ce"] = category["best_valid_ce"]
            metrics["category_probe/best_epoch"] = category["best_epoch"]
            metrics["category_probe/majority_test_acc"] = category["majority_test_acc"]
            category_text = (
                f" category_level=2 category_test_acc={category['splits']['test']['accuracy']:.6f}"
                f" category_test_macro_f1={category['splits']['test']['macro_f1']:.6f}"
                f" category_valid_ce={category['best_valid_ce']:.6f}"
                f" category_best_epoch={category['best_epoch']}"
            )
        cka_text = "NA (degenerate)" if cka is None else f"{cka:.6f}"
        genrec_text = "NA (undefined)" if cka_genrec is None else f"{cka_genrec:.6f}"
        tag = "[ItemProbe]" if self.hidden_source == "last" else "[ItemProbe:encoder_mean]"
        self.logger.info(
            f"{tag} epoch={epoch} device={hidden.device} "
            + " ".join(f"{split}_mse={v['mse']:.8g} {split}_cos={v['cosine']:.6f}"
                       for split, v in fit["splits"].items())
            + " " + " ".join(f"test_{key}={value['value']:.6f}" for key, value in recall.items())
            + f" clipped_debiased_cka={cka_text} cka_genrec={genrec_text} "
            f"regression_best_epoch={fit['best_epoch']}{category_text} "
            f"item_effective_rank={geometry['effective_rank']} item_avg_cosine={geometry['avg_cosine']} "
            f"seconds={record['elapsed_seconds']:.2f}"
        )
        self.latest = record
        return {self.metric_prefix + key: value for key, value in metrics.items()}

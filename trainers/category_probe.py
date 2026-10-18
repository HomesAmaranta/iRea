"""Train-only affine category probes on frozen single-item representations."""

import ast
import gzip
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.nn import functional as F


PROBE_PROTOCOL = "affine_ce_valid_early_stop_v1"
DEFAULT_LR = 0.01
DEFAULT_MAX_EPOCHS = 1000
DEFAULT_PATIENCE = 20


class CategoryProbeNumericalError(RuntimeError):
    """Numerical fit failure with diagnostics, distinct from invalid data or OOM."""

    def __init__(self, device, epoch):
        super().__init__(f"Non-finite category probe CE on {device} at epoch {epoch}")
        self.epoch = epoch


MISSING_SECOND_LEVEL = "__missing_second_level__"
TAXONOMIES = {
    "beauty": ("Beauty", "Beauty"),
    "sports": ("Sports_and_Outdoors", "Sports & Outdoors"),
    "toys": ("Toys_and_Games", "Toys & Games"),
}


def records(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield ast.literal_eval(line)


def category_children(metadata, root="Beauty"):
    """Second-level labels retain their parent to distinguish e.g. Face categories."""
    return sorted({
        " > ".join(path[1:3]) for path in metadata.get("categories", []) or []
        if isinstance(path, (list, tuple)) and len(path) > 2 and path[0] == root
    })


def read_labels(raw_dir, ids, stem="Beauty", category_root="Beauty"):
    """Mirror AmazonItemDataset's first-seen ASIN -> 1-based ItemID mapping."""
    mapping = {}
    for review in records(Path(raw_dir) / f"reviews_{stem}_5.json.gz"):
        asin = review.get("asin")
        if asin and asin not in mapping:
            mapping[asin] = len(mapping) + 1
    metadata = {}
    for item in records(Path(raw_dir) / f"meta_{stem}.json.gz"):
        if item.get("asin") in mapping:
            metadata[mapping[item["asin"]]] = item
    if not set(ids) <= set(mapping.values()):
        raise ValueError("Cache ItemIDs not present in review mapping")
    labels = []
    for item_id in ids:
        item = metadata.get(item_id)
        categories = category_children(item, category_root) if item else []
        missing_second_level = item is not None and not categories
        if missing_second_level:
            categories = [MISSING_SECOND_LEVEL]
        status = ("missing_metadata" if item is None else
                  "multiple_labels" if len(categories) > 1 else "included")
        labels.append({
            "ItemID": int(item_id), "asin": item["asin"] if item else "",
            "category": categories[0] if len(categories) == 1 else "",
            "status": status, "all_children": json.dumps(categories),
            "missing_second_level": missing_second_level,
        })
    return pd.DataFrame(labels)


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset_labels(dataset, item_ids):
    """Cache labels by raw-file contents and catalogue; publish atomically."""
    split = getattr(dataset, "split", None)
    if split not in TAXONOMIES or not hasattr(dataset, "root"):
        return None, {"status": "skipped", "reason": "unsupported dataset; requires Amazon 2014 beauty/sports/toys"}
    stem, root = TAXONOMIES[split]
    raw = Path(dataset.root) / "raw" / split
    fingerprint = {
        "version": 2, "root": root, "category_level": 2,
        "missing_label": MISSING_SECOND_LEVEL,
        "reviews": file_digest(raw / f"reviews_{stem}_5.json.gz"),
        "metadata": file_digest(raw / f"meta_{stem}.json.gz"),
        "item_ids": sorted(int(i) for i in item_ids),
    }
    cache_dir = Path(dataset.root) / "processed" / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "item_category_probe_v2.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        if data["fingerprint"] == fingerprint:
            frame = pd.DataFrame(data["labels"])
            return frame, {"status": "ready", "cache": str(cache.resolve()), "fingerprint": fingerprint}
    frame = read_labels(raw, fingerprint["item_ids"], stem, root)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=cache_dir, delete=False) as handle:
            temporary = handle.name
            json.dump({"fingerprint": fingerprint, "labels": frame.to_dict("records")}, handle)
        os.replace(temporary, cache)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return frame, {"status": "ready", "cache": str(cache.resolve()), "fingerprint": fingerprint}


def classification_metrics(truth, prediction, classes):
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, labels=classes, average="weighted", zero_division=0)),
        "per_class": classification_report(
            truth, prediction, labels=classes, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=classes).tolist(),
    }


@torch.no_grad()
def _tensor_classification_metrics(truth, prediction, classes):
    """Compute every metric on-device; copy only final small summaries for JSON."""
    size = len(classes)
    cm = torch.bincount(truth * size + prediction, minlength=size * size).reshape(size, size)
    support = cm.sum(1).double()
    predicted = cm.sum(0).double()
    true_positive = cm.diag().double()
    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * true_positive / (support + predicted).clamp_min(1)
    accuracy = true_positive.sum() / support.sum()
    macro = torch.stack((precision.mean(), recall.mean(), f1.mean()))
    weighted = torch.stack(tuple((v * support).sum() / support.sum()
                                 for v in (precision, recall, f1)))
    values = torch.stack((precision, recall, f1, support), dim=1).tolist()
    per_class = {
        name: dict(zip(("precision", "recall", "f1-score", "support"), row))
        for name, row in zip(classes, values)
    }
    per_class["accuracy"] = accuracy.item()
    for name, summary in (("macro avg", macro), ("weighted avg", weighted)):
        per_class[name] = dict(zip(("precision", "recall", "f1-score"), summary.tolist()))
        per_class[name]["support"] = int(truth.numel())
    return {
        "accuracy": accuracy.item(), "macro_f1": f1.mean().item(),
        "weighted_f1": weighted[2].item(), "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


@torch.enable_grad()
def fit_category_probe(hidden, labels, splits, *, learning_rate=DEFAULT_LR,
                       max_epochs=DEFAULT_MAX_EPOCHS, patience=DEFAULT_PATIENCE):
    """Fit one affine classifier; restore minimum valid CE before evaluating test."""
    if (not math.isfinite(learning_rate) or learning_rate <= 0
            or not isinstance(max_epochs, int) or max_epochs <= 0
            or not isinstance(patience, int) or patience <= 0):
        raise ValueError("Category probe learning rate, max_epochs and patience must be positive")
    x = hidden.detach().double()
    labels = np.asarray(labels, dtype=str)
    if x.ndim != 2 or len(x) != len(labels) or not torch.isfinite(x).all():
        raise ValueError("Category features must be finite and aligned with labels")
    rows = {s: np.asarray(v, dtype=np.int64) for s, v in splits.items()}
    classes = sorted(set(labels[np.concatenate(list(rows.values()))].tolist()))
    if any(len(v) == 0 for v in rows.values()) or len(set(labels[rows["train"]])) < 2:
        raise ValueError("Category probe requires nonempty splits and at least two train classes")
    train, valid = rows["train"], rows["valid"]
    # Include validation-only categories so their CE is defined. No test labels
    # determine the output layer, and only train examples contribute gradients.
    fit_classes = sorted(set(labels[np.concatenate([train, valid])].tolist()))
    train_classes = sorted(set(labels[train].tolist()))
    rows_device = {s: torch.as_tensor(v, device=x.device) for s, v in rows.items()}
    train_labels = torch.tensor(
        [fit_classes.index(label) for label in labels[train]], device=x.device)
    mean = x[rows_device["train"]].mean(0)
    scale = x[rows_device["train"]].std(0, correction=0)
    scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
    z = (x - mean) / scale
    z_train = z[rows_device["train"]]
    valid_labels = torch.tensor(
        [fit_classes.index(label) for label in labels[valid]], device=x.device)
    # Zero initialization does not consume the recommender RNG.
    weight = torch.zeros((len(fit_classes), x.shape[1]), device=x.device,
                         dtype=x.dtype, requires_grad=True)
    bias = torch.zeros(len(fit_classes), device=x.device, dtype=x.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=learning_rate, weight_decay=0)
    best_valid_ce, best_epoch, wait = math.inf, 0, 0
    history = []
    stop_reason = "max_epochs"
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(F.linear(z_train, weight, bias), train_labels)
        if not torch.isfinite(loss):
            raise CategoryProbeNumericalError(x.device, epoch)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            valid_ce = F.cross_entropy(
                F.linear(z[rows_device["valid"]], weight, bias), valid_labels).item()
        if not math.isfinite(valid_ce):
            raise CategoryProbeNumericalError(x.device, epoch)
        history.append({"epoch": epoch, "train_ce": loss.item(), "valid_ce": valid_ce})
        if valid_ce < best_valid_ce:
            best_valid_ce, best_epoch, wait = valid_ce, epoch, 0
            best_weight, best_bias = weight.detach().clone(), bias.detach().clone()
        else:
            wait += 1
            if wait >= patience:
                stop_reason = "patience"
                break
    weight, bias = best_weight, best_bias
    weight = weight / scale[None, :]
    bias = bias - weight @ mean
    state = {"weight": weight, "bias": bias}
    mapping = torch.tensor([classes.index(label) for label in fit_classes], device=x.device)
    metrics = {}
    for s, v in rows.items():
        truth = torch.tensor([classes.index(label) for label in labels[v]], device=x.device)
        predicted = F.linear(x[rows_device[s]], weight, bias).argmax(1)
        metrics[s] = _tensor_classification_metrics(truth, mapping[predicted], classes)
    return state, {"status": "valid", "protocol": PROBE_PROTOCOL, "class_weight": None,
                   "backend": "torch_adam", "device": str(x.device),
                   "loss": "unweighted cross-entropy", "weight_decay": 0,
                   "learning_rate": learning_rate, "max_epochs": max_epochs, "patience": patience,
                   "best_epoch": best_epoch, "best_valid_ce": best_valid_ce,
                   "epochs_trained": epoch, "stop_reason": stop_reason,
                   "selection": "minimum validation CE; strict improvement; epoch is 1-based",
                   "class_order": fit_classes,
                   "class_vocabulary": "train and valid labels only; test-only classes evaluated as errors",
                   "metric_class_order": classes,
                   "unseen_train_classes": sorted(set(classes) - set(train_classes)),
                   "history": history, "splits": metrics}


class ItemCategoryProbe:
    def __init__(self, frame, item_ids, splits, directory, save_weights=True):
        self.save_weights = save_weights
        if not frame.ItemID.is_unique or set(frame.ItemID) != set(item_ids):
            raise ValueError("Category labels must cover the exact unique probe ItemIDs")
        self.directory = Path(directory)
        ordered = frame.set_index("ItemID").loc[item_ids].reset_index()
        self.labels = ordered.category.fillna("").to_numpy(dtype=str)
        valid = (ordered.status == "included").to_numpy()
        self.splits = {s: v.numpy()[valid[v.numpy()]] for s, v in splits.items()}
        self.metadata = {
            "status": "ready", "protocol": PROBE_PROTOCOL, "class_weight": None, "bias": True,
            "category_level": 2, "missing_label": MISSING_SECOND_LEVEL,
            "rare_class_policy": "Keep all labelled items; no minimum frequency or resampling",
            "loss": "unweighted cross-entropy", "weight_decay": 0,
            "fit": "Fresh affine classifier; train-only standardization; full-batch updates",
            "backend": "torch_adam", "compute_device": "same as hidden",
            "learning_rate": DEFAULT_LR, "max_epochs": DEFAULT_MAX_EPOCHS,
            "patience": DEFAULT_PATIENCE,
            "selection": "Minimum validation CE; stop after patience checks without strict improvement",
            "class_vocabulary": "train and valid labels only; test-only classes evaluated as errors",
            "excluded_counts": ordered.loc[~valid, "status"].value_counts().to_dict(),
            "split_item_ids": {s: np.asarray(item_ids)[v].tolist() for s, v in self.splits.items()},
            "class_counts": {s: pd.Series(self.labels[v]).value_counts().to_dict()
                             for s, v in self.splits.items()},
        }
        if any(len(v) == 0 for v in self.splits.values()) or len(set(self.labels[self.splits["train"]])) < 2:
            self.metadata.update(status="skipped", reason="insufficient labelled splits or train classes")
        else:
            majority = pd.Series(self.labels[self.splits["train"]]).value_counts().index[0]
            self.metadata["majority_test_acc"] = float(
                (self.labels[self.splits["test"]] == majority).mean())
        ordered.to_csv(self.directory / "category_labels.csv", index=False)

    def evaluate(self, hidden, epoch):
        if self.metadata["status"] != "ready":
            return dict(self.metadata)
        start = time.perf_counter()
        state, result = fit_category_probe(hidden, self.labels, self.splits)
        result["elapsed_seconds"] = time.perf_counter() - start
        result["category_level"] = 2
        result["missing_label"] = MISSING_SECOND_LEVEL
        result["majority_test_acc"] = self.metadata["majority_test_acc"]
        if self.save_weights:
            torch.save({"epoch": epoch, "protocol": result["protocol"],
                        "best_epoch": result["best_epoch"], "best_valid_ce": result["best_valid_ce"],
                        "classes": result["class_order"],
                        "category_level": 2, "missing_label": MISSING_SECOND_LEVEL,
                        "state_dict": state}, self.directory / "latest_category_linear.pt")
        return result

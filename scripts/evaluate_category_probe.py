"""Frozen linear classification of Beauty metadata categories from cached embeddings."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trainers.category_probe import (
    MISSING_SECOND_LEVEL,
    PROBE_PROTOCOL,
    category_children,
    classification_metrics as metrics,
    fit_category_probe,
    read_labels,
)

def beauty_categories(metadata):
    """Use the same second-level paths as the per-epoch category probe."""
    return category_children(metadata, "Beauty")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset/amazon/raw/beauty"))
    parser.add_argument("--cache-dir", type=Path, default=Path("dataset/amazon/processed/beauty"))
    parser.add_argument("--split-file", type=Path,
                        default=Path("out/beauty_embedding_retrieval/results.json"))
    parser.add_argument("--output", type=Path, default=Path("out/beauty_category_probe_level2"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    saved_split = json.loads(args.split_file.read_text())["split_item_ids"]
    reference = args.cache_dir / "item_emb_sentence-t5-xl.parquet"
    ids = np.sort(pd.read_parquet(reference, columns=["ItemID"]).ItemID.to_numpy())
    assert len(np.unique(ids)) == len(ids)
    split_for = {}
    for split, values in saved_split.items():
        for item_id in values:
            assert item_id not in split_for, "Splits must be disjoint"
            split_for[item_id] = split
    assert set(split_for) == set(ids)
    args.output.mkdir(parents=True, exist_ok=False)
    print("Reading raw categories and reconstructing ASIN -> ItemID...", flush=True)
    labels = read_labels(args.raw_dir, ids)
    labels["split"] = labels.ItemID.map(split_for)
    labels.to_csv(args.output / "item_labels.csv", index=False)
    usable = labels[labels.status == "included"].copy()
    classes = sorted(usable.category.unique())
    truth = usable.category.to_numpy()
    masks = {s: (usable.split == s).to_numpy() for s in saved_split}
    assert all(m.any() for m in masks.values())
    splits = {s: np.flatnonzero(mask) for s, mask in masks.items()}
    distributions = {
        s: dict(sorted(Counter(truth[m]).items())) for s, m in masks.items()
    }
    report = {
        "dataset": "beauty", "label": "Second-level Beauty path including parent",
        "category_level": 2, "missing_label": MISSING_SECOND_LEVEL,
        "unseen_train_classes": sorted(set(classes) - set(truth[masks["train"]])),
        "caveat": "Categories were provided in encoder input; not unknown-attribute inference",
        "catalogue_items": len(ids), "usable_items": len(usable),
        "filter_counts": labels.status.value_counts().to_dict(),
        "split_source": str(args.split_file.resolve()),
        "split_counts": {s: int(m.sum()) for s, m in masks.items()},
        "class_order": classes, "class_counts": distributions,
        "protocol": PROBE_PROTOCOL,
        "selection": "Minimum validation CE with early stopping",
        "classifier": "Train-only standardization + affine classifier with bias; torch Adam",
        "class_weight": None,
        "loss": "unweighted cross-entropy; no regularization",
        "models": {},
    }
    majority = Counter(truth[masks["train"]]).most_common(1)[0][0]
    report["majority_baseline"] = {
        "category": majority,
        **metrics(truth[masks["test"]], np.repeat(majority, masks["test"].sum()), classes),
    }
    print(json.dumps({k: report[k] for k in (
        "filter_counts", "split_counts", "class_counts")}), flush=True)
    for name in ("Qwen3-Embedding-8B", "bge-m3", "all-mpnet-base-v2", "sentence-t5-xl"):
        start = time.perf_counter()
        print(f"Fitting {name}...", flush=True)
        frame = pd.read_parquet(args.cache_dir / f"item_emb_{name}.parquet")
        assert frame.ItemID.is_unique and set(frame.ItemID) == set(ids)
        x = np.stack(frame.set_index("ItemID").loc[usable.ItemID, "embedding"].to_numpy())
        assert x.ndim == 2 and np.isfinite(x).all()
        features = torch.as_tensor(x, device=args.device, dtype=torch.float64)
        state, category = fit_category_probe(features, truth, splits)
        test_rows = torch.as_tensor(splits["test"], device=features.device)
        predicted_ids = torch.nn.functional.linear(
            features[test_rows], **state).argmax(1).cpu().numpy()
        prediction = np.asarray(category["class_order"])[predicted_ids]
        test = category["splits"]["test"]
        record = {
            "dimension": x.shape[1], "category": category, "test": test,
            "feature_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            "elapsed_seconds": time.perf_counter() - start,
        }
        report["models"][name] = record
        np.savez(args.output / f"{name}_linear.npz",
                 weight=state["weight"].cpu().numpy(), bias=state["bias"].cpu().numpy(),
                 classes=np.asarray(category["class_order"]))
        pd.DataFrame({
            "ItemID": usable.ItemID.to_numpy()[masks["test"]],
            "truth": truth[masks["test"]], "prediction": prediction,
        }).to_csv(args.output / f"{name}_test_predictions.csv", index=False)
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps({"model": name, "test_accuracy": test["accuracy"],
                          "test_macro_f1": test["macro_f1"],
                          "best_valid_ce": category["best_valid_ce"],
                          "best_epoch": category["best_epoch"],
                          "seconds": record["elapsed_seconds"]}), flush=True)
    print(f"Finished; results: {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()

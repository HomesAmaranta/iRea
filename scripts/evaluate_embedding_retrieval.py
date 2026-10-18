"""Evaluate held-out cross-encoder item retrieval through train-only ridge probes."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from trainers.affine_probe import retrieval_metrics
from trainers.item_probe import fit_linear_probe, load_reference, split_items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("dataset/amazon/processed/beauty"))
    parser.add_argument("--output", type=Path, default=Path("out/beauty_embedding_retrieval"))
    args = parser.parse_args()
    config = json.loads(args.probe_config.read_text())
    # This standalone script remains an explicitly labelled legacy ridge baseline.
    alphas = config.get("alphas", (0., 1e-6, 1e-4, 1e-2, 1.))
    reference_path = args.cache_dir / "item_emb_sentence-t5-xl.parquet"
    ids = np.sort(pd.read_parquet(reference_path, columns=["ItemID"]).ItemID.to_numpy())
    target = load_reference(reference_path, ids)
    splits = split_items(ids, config["seed"])
    for name, rows in splits.items():
        assert ids[rows.numpy()].tolist() == config["split_item_ids"][name]
    # The original probe hash uses its own catalogue order, not sorted ItemIDs.
    original_ids = pd.read_parquet(reference_path, columns=["ItemID"]).ItemID.to_numpy()
    original_target = load_reference(reference_path, original_ids)
    reference_hash = hashlib.sha256(original_target.numpy().tobytes()).hexdigest()
    assert reference_hash == config["reference_sha256"], "Reference cache changed"
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "dataset": "beauty", "catalogue_items": len(ids), "test_queries": len(splits["test"]),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "split_item_ids": config["split_item_ids"],
        "probe_config": str(args.probe_config.resolve()), "seed": config["seed"],
        "reference_sha256": reference_hash, "alphas": alphas,
        "fit": "Frozen embeddings; train-only standardization; ridge Linear(bias=True)",
        "selection": "Minimum validation MSE; no test-based selection or refitting",
        "retrieval": "Normalized mapped test queries @ normalized full Sentence-T5 catalogue.T",
        "ties": "Ascending ItemID; ties are not all counted as correct",
        "models": {},
    }
    test = splits["test"]
    for name in ("Qwen3-Embedding-8B", "bge-m3", "all-mpnet-base-v2"):
        start = time.perf_counter()
        print(f"Fitting {name} on CPU, float64...", flush=True)
        source_path = args.cache_dir / f"item_emb_{name}.parquet"
        source = load_reference(source_path, ids)
        state, fit = fit_linear_probe(source, target, splits, alphas)
        prediction = F.linear(source[test], state["weight"], state["bias"])
        recall, ranks = retrieval_metrics(prediction, target, test)
        record = {
            "source_path": str(source_path.resolve()), "source_dim": source.shape[1],
            "source_sha256_sorted": hashlib.sha256(source.numpy().tobytes()).hexdigest(),
            "recall": recall, "elapsed_seconds": time.perf_counter() - start, **fit,
        }
        report["models"][name] = record
        torch.save({"state_dict": state, "alpha": fit["alpha"]}, args.output / f"{name}_linear.pt")
        pd.DataFrame({"ItemID": ids[test.numpy()], "rank": ranks.numpy()}).to_csv(
            args.output / f"{name}_test_ranks.csv", index=False)
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps({"model": name, "alpha": fit["alpha"], "recall": recall,
                          "test": fit["splits"]["test"],
                          "seconds": record["elapsed_seconds"]}), flush=True)
    self_recall, _ = retrieval_metrics(target[test], target, test)
    mean_prediction = target[splits["train"]].mean(dim=0).expand(len(test), -1)
    mean_recall, _ = retrieval_metrics(mean_prediction, target, test)
    report["controls"] = {
        "sentence_t5_self": self_recall, "train_mean_only": mean_recall,
        "uniform_random_expected": {f"Recall@{k}": k / len(ids) for k in (1, 5, 10)},
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print("CONTROLS", json.dumps(report["controls"]), flush=True)


if __name__ == "__main__":
    main()

"""Fit an affine probe between checkpoint hidden and Sentence-T5, then retrieve items."""

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from models.factory import build_sid_model
from models.rqvae import RqVae
from trainers.affine_probe import RECALL_KS, cosine_mse, fit_probe, retrieval_metrics
from trainers.category_probe import file_digest
from trainers.general_trainer import configure_alignment
from trainers.item_probe import extract_item_hidden, load_reference, split_items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("hstu", "tiger"), default="hstu")
    parser.add_argument("--run-dir", type=Path, default=Path(
        "out_linear2gram/hstu/amazon/beauty/align_item_current_cos0_last_lr5e-4"))
    parser.add_argument("--rqvae", type=Path,
                        default=Path("out/tiger/amazon/beauty/rqvae/checkpoint_epoch_4999.pt"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--loss", choices=("cosine", "cosine_mse"), default="cosine_mse")
    parser.add_argument("--reverse", action="store_true",
                        help="Map Sentence-T5 to checkpoint hidden and retrieve in hidden space")
    parser.add_argument("--checkpoint-spec", type=Path,
                        help="Explicit checkpoint and provenance; run-dir supplies architecture and probe config")
    args = parser.parse_args()
    torch.set_num_threads(2)
    start = time.perf_counter()
    device = torch.device(args.device)
    result = json.loads((args.run_dir / "results.json").read_text())
    checkpoint_spec = None
    if args.checkpoint_spec:
        checkpoint_spec = json.loads(args.checkpoint_spec.read_text())
        for key in ("best_epoch", "align_target", "align_item", "align_loss_type", "mse_loss_weight"):
            result[key] = checkpoint_spec[key]
    aligned = result["mse_loss_weight"] > 0
    if aligned and result["align_target"] != "item":
        raise ValueError("Aligned checkpoint loading currently supports item targets only")
    variant = result["align_item"] if aligned else "baseline"
    config = json.loads((Path(result["linear_probe_dir"]) / "config.json").read_text())
    reference = Path(config["reference_path"])
    ids = pd.read_parquet(reference, columns=["ItemID"]).ItemID.to_numpy()
    target = load_reference(reference, ids)
    if hashlib.sha256(target.numpy().tobytes()).hexdigest() != config["reference_sha256"]:
        raise ValueError("Sentence-T5 reference no longer matches original probe")
    splits = split_items(ids, config["seed"])
    if {s: ids[v.numpy()].tolist() for s, v in splits.items()} != config["split_item_ids"]:
        raise ValueError("Item split differs from original experiment")
    rq = RqVae(
        input_dim=768, embed_dim=32, hidden_dims=[512, 256, 128, 64],
        codebook_size=256, codebook_kmeans_init=False, codebook_normalize=False,
        codebook_sim_vq=False, n_layers=3, n_cat_features=0, commitment_weight=.25,
    )
    rq.load_pretrained(str(args.rqvae))
    rq.to(device).eval().requires_grad_(False)
    with torch.no_grad():
        codes = rq.get_semantic_ids(target.to(device).float()).sem_ids.cpu()
    if hashlib.sha256(codes.numpy().tobytes()).hexdigest() != config["sid_sha256"]:
        raise ValueError("Reconstructed SIDs differ from checkpoint experiment")
    del rq
    checkpoint = args.run_dir / "best_model.pt"
    model = build_sid_model(args.model, result["codebook_size"], result["sem_id_dim"],
                            result["max_seq_len"], result["model_config"])
    if aligned:
        weight_tag = str(result["mse_loss_weight"]).removesuffix(".0")
        checkpoint = args.run_dir / (
            f"best_model_item_{variant}_{result['align_loss_type']}{weight_tag}"
            f"_{result.get('mse_loss_mode', 'last')}.pt")
        configure_alignment(model, target.shape[1], result["align_loss_type"])
    if checkpoint_spec:
        checkpoint = Path(checkpoint_spec["checkpoint"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval().requires_grad_(False)
    hidden = extract_item_hidden(model, codes, 256, "no")
    del model
    # Ascending ItemID makes exact retrieval ties deterministic and interpretable.
    order = np.argsort(ids)
    ids, hidden, target = ids[order], hidden[order], target[order].to(device)
    splits = split_items(ids, config["seed"])
    source, catalogue = (target, hidden) if args.reverse else (hidden, target)
    direction = "sentence_to_hidden" if args.reverse else "hidden_to_sentence"
    direction_tag = "sentence_to_hidden_" if args.reverse else ""
    output = args.output or Path("out") / (
        f"beauty_{args.model}_{variant}_retrieval_{direction_tag}{args.loss}_"
        + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    print(f"Frozen {args.model} checkpoint epoch={result['best_epoch']} hidden={tuple(hidden.shape)} "
          f"splits={ {s: len(v) for s, v in splits.items()} } device={device}", flush=True)
    state, fit = fit_probe(
        source, catalogue, splits, learning_rate=args.learning_rate,
        max_epochs=args.max_epochs, patience=args.patience, seed=config["seed"],
        loss_type=args.loss, log_every=100)
    test = splits["test"].to(device)
    with torch.no_grad():
        prediction = F.linear(source[test].float(), **state)
    recall, ranks = retrieval_metrics(prediction, catalogue, test, top_ks=RECALL_KS, item_ids=ids)
    self_recall, _ = retrieval_metrics(catalogue[test], catalogue, test, top_ks=RECALL_KS, item_ids=ids)
    mean_recall, _ = retrieval_metrics(
        catalogue[splits["train"].to(device)].mean(0).expand(len(test), -1), catalogue, test,
        top_ks=RECALL_KS, item_ids=ids)
    report = {
        "dataset": "beauty", "model": args.model, "variant": variant, "device": str(device),
        "backbone_alignment": {k: result.get(k) for k in
                               ("align_target", "align_item", "align_loss_type", "mse_loss_weight")},
        "uses_backbone_alignment_head": False,
        "direction": direction, "source_dim": source.shape[1], "target_dim": catalogue.shape[1],
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_digest(checkpoint),
        "checkpoint_spec": checkpoint_spec,
        "architecture_and_probe_config_source": str(args.run_dir.resolve()),
        "checkpoint_epoch": result["best_epoch"], "checkpoint_selection": "recommender valid CE",
        "reference_path": str(reference.resolve()), "reference_sha256": config["reference_sha256"],
        "rqvae_path": str(args.rqvae.resolve()), "sid_sha256": config["sid_sha256"],
        "hidden": ("single item, last SID token, final normalized hidden" if args.model == "hstu"
                   else "single item, decoder start token, final hidden"),
        "hidden_precision": "float32", "hidden_dimension": hidden.shape[1],
        "catalogue_items": len(ids), "test_queries": len(test),
        "split_counts": {s: len(v) for s, v in splits.items()},
        "split_item_ids": {s: ids[v.numpy()].tolist() for s, v in splits.items()},
        "retrieval": ("cosine against all checkpoint item hidden; exact ItemID match" if args.reverse
                      else "cosine against all Sentence-T5 items; exact ItemID match"),
        "ties": "ascending ItemID; equal-score candidates not all counted correct",
        "caveat": "Probe-test items held out of linear fitting, not necessarily from backbone training",
        "fit": fit, "recall": recall,
        "controls": {("hidden_self" if args.reverse else "sentence_t5_self"): self_recall,
                     "train_mean_only": mean_recall},
        "elapsed_seconds": time.perf_counter() - start,
    }
    torch.save({"state_dict": {k: v.cpu() for k, v in state.items()},
                "best_epoch": fit["best_epoch"], "protocol": fit["protocol"],
                "direction": direction},
               output / "linear.pt")
    pd.DataFrame({"ItemID": ids[splits["test"].numpy()], "rank": ranks.cpu().numpy()}).to_csv(
        output / "test_ranks.csv", index=False)
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"recall": recall, "best_epoch": fit["best_epoch"],
                      "epochs_trained": fit["epochs_trained"], "stop_reason": fit["stop_reason"],
                      "test": fit["splits"]["test"], "seconds": report["elapsed_seconds"],
                      "results": str(output / "results.json")}), flush=True)


if __name__ == "__main__":
    main()

"""Fit OneRec RQ-KMeans: python -m trainers.rqkmeans_trainer config/rqkmeans.gin --split beauty."""

import os

import gin
import numpy as np

from data.amazon import AmazonItemDataset
from models.rqkmeans import RqKmeans
from modules.utils import get_run_split, parse_config
from trainers.trainer_utils import save_run_results, set_seed


@gin.configurable
def fit_rqkmeans(
    dataset=AmazonItemDataset,
    dataset_folder: str = "dataset/amazon",
    encoder_model_name: str = "/opt/tiger/project/hf_models/Qwen3-Embedding-8B",
    n_layers: int = 3,
    codebook_size: int = 256,
    save_dir_root: str = "out/onerec/amazon/{split}/rqkmeans",
    random_state: int = 42,
    max_iter: int = 100,
    batch_size: int = 4096,
    seed: int = 42,
    backend: str = "auto",
    n_init: int = 10,
    dead_reinit: bool = True,
):
    """Fit all item embeddings and save compatible centroids plus collision metrics."""
    run_config = dict(locals())
    set_seed(seed)
    split = get_run_split("beauty")
    save_dir_root = save_dir_root.format(split=split)
    dataset_folder = dataset_folder.format(split=split)
    rqkmeans = RqKmeans(
        n_layers=n_layers, codebook_size=codebook_size, random_state=random_state,
        max_iter=max_iter, batch_size=batch_size, backend=backend,
        n_init=n_init, dead_reinit=dead_reinit,
    )
    print("=== OneRec RQ-KMeans fitting ===")
    print(f"  encoder: {encoder_model_name}")
    print(f"  codebook_size: {codebook_size}, n_layers: {n_layers}")
    print(f"  save_dir: {save_dir_root}")
    item_dataset = dataset(
        root=dataset_folder,
        train_test_split="all",
        encoder_model_name=encoder_model_name,
    )
    embeddings = np.asarray(item_dataset.embeddings, dtype=np.float32)
    rqkmeans.fit(embeddings)
    stats = rqkmeans.collision_rate(embeddings)
    print(f"Collision statistics: {stats}")

    os.makedirs(save_dir_root, exist_ok=True)
    save_path = os.path.join(save_dir_root, "rqkmeans.pt")
    rqkmeans.save(save_path)
    save_run_results(
        save_dir=save_dir_root, model="rqkmeans", split=split, seed=seed,
        metrics={
            "collision_rate": float(stats["collision_rate"]),
            "unique_sids": int(stats["unique"]),
            "total_items": int(stats["total"]),
            "max_collision": int(stats["max_collision"]),
        },
        config=run_config,
        extra={
            "encoder_model_name": encoder_model_name,
            "embedding_dim": int(embeddings.shape[1]),
            "checkpoint": save_path,
        },
    )
    print(f"=== Done. Saved to {save_path} ===")
    return stats


if __name__ == "__main__":
    parse_config()
    fit_rqkmeans()

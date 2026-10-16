"""OneRec residual K-means quantization, ported from genrec.models.rqkmeans.

Fits codebooks directly on item embeddings, without an encoder or decoder.
The centroid checkpoint schema is compatible with the genrec implementation.
"""

from collections import Counter
from typing import List, NamedTuple

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans


class RqKmeansOutput(NamedTuple):
    sem_ids: np.ndarray  # [N, n_layers], zero-based codebook indices


class RqKmeans:
    """Fit successive K-means codebooks on the previous layer's residual."""

    def __init__(
        self,
        n_layers: int = 3,
        codebook_size: int = 8192,
        random_state: int = 42,
        max_iter: int = 100,
        batch_size: int = 4096,
        backend: str = "auto",
        n_init: int = 10,
        dead_reinit: bool = True,
    ):
        if min(n_layers, codebook_size, max_iter, batch_size, n_init) <= 0:
            raise ValueError("Layer count, codebook size, iteration counts and batch size must be positive")
        if backend not in ("auto", "full", "minibatch"):
            raise ValueError("backend must be auto, full or minibatch")
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.backend = backend
        self.n_init = n_init
        self.dead_reinit = dead_reinit
        self.centroids: List[np.ndarray] = []

    @staticmethod
    def _validate_embeddings(embeddings):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or min(embeddings.shape) == 0:
            raise ValueError("embeddings must have nonempty shape [N, D]")
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must be finite")
        return embeddings

    def fit(self, embeddings: np.ndarray) -> "RqKmeans":
        embeddings = self._validate_embeddings(embeddings)
        n_items = len(embeddings)
        if n_items < self.codebook_size:
            raise ValueError("Number of items must be at least codebook_size")
        self.centroids = []
        residual = embeddings.copy()
        backend = self.backend
        if backend == "auto":
            backend = "full" if n_items * self.codebook_size < 5e7 else "minibatch"

        for layer in range(self.n_layers):
            print(
                f"Fitting layer {layer}/{self.n_layers} [{backend}], "
                f"N={n_items}, K={self.codebook_size}, "
                f"residual norm: {np.linalg.norm(residual, axis=1).mean():.4f}"
            )
            if backend == "full":
                kmeans = KMeans(
                    n_clusters=self.codebook_size, init="k-means++",
                    n_init=self.n_init, max_iter=self.max_iter,
                    random_state=self.random_state, verbose=0,
                )
            else:
                kmeans = MiniBatchKMeans(
                    n_clusters=self.codebook_size, random_state=self.random_state,
                    max_iter=self.max_iter, batch_size=min(self.batch_size, n_items),
                    n_init=self.n_init, verbose=0,
                )
            kmeans.fit(residual)
            centroids = kmeans.cluster_centers_.copy()
            codes = kmeans.predict(residual)

            # Match the source: reseed unused centroids, then reassign residuals.
            if self.dead_reinit:
                used = set(codes.tolist())
                dead = [code for code in range(self.codebook_size) if code not in used]
                if dead:
                    print(f"  Layer {layer}: reinit {len(dead)} dead clusters")
                    error = ((residual - centroids[codes]) ** 2).sum(axis=1)
                    farthest = np.argsort(-error)
                    for index, code in enumerate(dead):
                        centroids[code] = residual[farthest[index % n_items]]
                    codes = self._nearest(residual, centroids)

            self.centroids.append(centroids)
            residual = residual - centroids[codes]
            counts = Counter(codes.tolist())
            print(
                f"  Layer {layer} usage: {len(counts)}/{self.codebook_size} "
                f"({100 * len(counts) / self.codebook_size:.1f}%), "
                f"max_freq={max(counts.values())} "
                f"({100 * max(counts.values()) / n_items:.1f}% of items)"
            )

        print(f"Final residual norm: {np.linalg.norm(residual, axis=1).mean():.4f}")
        return self

    @staticmethod
    def _nearest(residual, centroids):
        x_sq = (residual ** 2).sum(axis=1, keepdims=True)
        c_sq = (centroids ** 2).sum(axis=1, keepdims=True).T
        return (x_sq + c_sq - 2 * residual @ centroids.T).argmin(axis=1)

    def assign(self, embeddings: np.ndarray) -> RqKmeansOutput:
        if len(self.centroids) != self.n_layers:
            raise RuntimeError("Must call fit() or load() before assigning SIDs")
        embeddings = self._validate_embeddings(embeddings)
        if embeddings.shape[1] != self.centroids[0].shape[1]:
            raise ValueError("Embedding dimension does not match the codebooks")
        sem_ids = np.zeros((len(embeddings), self.n_layers), dtype=np.int64)
        residual = embeddings.copy()
        for layer, centroids in enumerate(self.centroids):
            codes = self._nearest(residual, centroids)
            sem_ids[:, layer] = codes
            residual = residual - centroids[codes]
        return RqKmeansOutput(sem_ids=sem_ids)

    def save(self, path: str) -> None:
        if len(self.centroids) != self.n_layers:
            raise RuntimeError("Must call fit() before saving codebooks")
        torch.save({
            "n_layers": self.n_layers,
            "codebook_size": self.codebook_size,
            "centroids": [centroids.copy() for centroids in self.centroids],
        }, path)
        print(f"Saved RQ-KMeans to {path}")

    @classmethod
    def load(cls, path: str) -> "RqKmeans":
        """Load a trusted local checkpoint (legacy format contains NumPy pickle data)."""
        if not hasattr(np, "_core"):
            import sys
            sys.modules.setdefault("numpy._core", np.core)
            sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
            sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
        state = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(n_layers=state["n_layers"], codebook_size=state["codebook_size"])
        obj.centroids = [np.asarray(c, dtype=np.float32) for c in state["centroids"]]
        if len(obj.centroids) != obj.n_layers:
            raise ValueError("Checkpoint has an incorrect number of codebooks")
        dimension = None
        for centroids in obj.centroids:
            cls._validate_embeddings(centroids)
            if centroids.shape[0] != obj.codebook_size:
                raise ValueError("Checkpoint codebook size does not match its metadata")
            if dimension is not None and centroids.shape[1] != dimension:
                raise ValueError("Checkpoint codebooks have inconsistent dimensions")
            dimension = centroids.shape[1]
        print(f"Loaded RQ-KMeans from {path} "
              f"(layers={obj.n_layers}, codebook={obj.codebook_size}, dim={dimension})")
        return obj

    def collision_rate(self, embeddings: np.ndarray) -> dict:
        sem_ids = self.assign(embeddings).sem_ids
        counts = Counter(tuple(row) for row in sem_ids)
        total = len(sem_ids)
        return {
            "total": total,
            "unique": len(counts),
            "collision_rate": 1 - len(counts) / total,
            "max_collision": max(counts.values()),
        }

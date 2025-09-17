import time
import os
import psutil
import numpy as np
from typing import Optional
from sklearn.neighbors import NearestNeighbors


class Algo:
    def __init__(self, metric: str = "l2", **params):
        self.metric = metric
        self.algorithm = params.get("algorithm", "brute")  # Use fastest algorithm as default
        self.leaf_size = params.get("leaf_size", 30)  # Leaf size for tree algorithms
        self.params = params
        self.index = None
        self.is_built = False
        self._stats = {
            "build_time_s": None,
            "ram_rss_mb_after_build": None,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
        }

    def build(self, xb: np.ndarray, metric: Optional[str] = None) -> None:
        if metric is not None:
            self.metric = metric

        start_time = time.perf_counter()

        # Convert metric names to sklearn format
        if self.metric == "l2":
            sklearn_metric = "euclidean"
        elif self.metric == "cos":
            sklearn_metric = "cosine"
        else:
            raise ValueError(f"Unsupported metric for sklearn KNN: {self.metric}. Supported: l2, cos")

        # Create and fit the model
        self.index = NearestNeighbors(
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=sklearn_metric
        )

        # Sklearn expects float64 by default but can handle float32
        xb_clean = xb.astype(np.float32, copy=False)
        self.index.fit(xb_clean)
        self.is_built = True

        build_time = time.perf_counter() - start_time
        self._stats["build_time_s"] = float(build_time)
        self._stats["ram_rss_mb_after_build"] = psutil.Process(os.getpid()).memory_info().rss / 1e6

    def query(self, xq: np.ndarray, k: int):
        if not self.is_built:
            raise RuntimeError("Index not built. Call build() first.")

        # Sklearn expects consistent data types
        xq_clean = xq.astype(np.float32, copy=False)

        # Find k nearest neighbors
        distances, indices = self.index.kneighbors(xq_clean, n_neighbors=k)

        # Convert to int64 for indices
        indices = indices.astype(np.int64)
        distances = distances.astype(np.float32)

        return indices, distances

    def stats(self) -> dict:
        return dict(self._stats)
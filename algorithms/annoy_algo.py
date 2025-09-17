import time
import os
import psutil
import numpy as np
from typing import Optional
from annoy import AnnoyIndex


class Algo:
    def __init__(self, metric: str = "l2", **params):
        self.metric = metric
        self.n_trees = params.get("n_trees", 50)
        self.search_k = params.get("search_k", -1)  # -1 means n_trees * k
        self.params = params
        self.index = None
        self.dimension = None
        self.is_built = False
        self._stats = {
            "build_time_s": None,
            "ram_rss_mb_after_build": None,
            "n_trees": self.n_trees,
            "search_k": self.search_k,
        }

    def build(self, xb: np.ndarray, metric: Optional[str] = None) -> None:
        if metric is not None:
            self.metric = metric

        start_time = time.perf_counter()

        # Convert metric names to Annoy format
        if self.metric == "l2":
            annoy_metric = "euclidean"
        elif self.metric == "cos":
            annoy_metric = "angular"
        else:
            raise ValueError(f"Unsupported metric for Annoy: {self.metric}. Supported: l2, cos")

        self.dimension = xb.shape[1]
        self.index = AnnoyIndex(self.dimension, annoy_metric)

        # Add all vectors to the index
        xb_list = xb.tolist()
        for i, vector in enumerate(xb_list):
            self.index.add_item(i, vector)

        # Build the index with specified number of trees
        self.index.build(self.n_trees)
        self.is_built = True

        build_time = time.perf_counter() - start_time
        self._stats["build_time_s"] = float(build_time)
        self._stats["ram_rss_mb_after_build"] = psutil.Process(os.getpid()).memory_info().rss / 1e6

    def query(self, xq: np.ndarray, k: int):
        if not self.is_built:
            raise RuntimeError("Index not built. Call build() first.")

        n_queries = xq.shape[0]
        indices = np.empty((n_queries, k), dtype=np.int64)
        distances = np.empty((n_queries, k), dtype=np.float32)

        # Calculate search_k_param once before the loop
        search_k_param = self.search_k if self.search_k > 0 else self.n_trees * k

        for i, query in enumerate(xq):
            # Get k nearest neighbors with search_k parameter for better recall
            neighbors, dists = self.index.get_nns_by_vector(
                query.tolist(), k, search_k=search_k_param, include_distances=True
            )

            # Pad results if we get fewer than k neighbors
            while len(neighbors) < k:
                neighbors.append(-1)  # Invalid index
                dists.append(float('inf'))

            indices[i] = neighbors[:k]
            distances[i] = dists[:k]

        return indices, distances

    def stats(self) -> dict:
        return dict(self._stats)
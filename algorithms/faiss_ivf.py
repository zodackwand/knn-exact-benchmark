import time
import os
import psutil
import numpy as np
from typing import Optional
import faiss


class Algo:
    def __init__(self, metric: str = "l2", **params):
        self.metric = metric
        self.nlist = params.get("nlist", 100)  # Number of cluster centroids
        self.nprobe = params.get("nprobe", 10)  # Number of clusters to search
        self.params = params
        self.index = None
        self.dimension = None
        self.is_built = False
        self._stats = {
            "build_time_s": None,
            "ram_rss_mb_after_build": None,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
        }

    def build(self, xb: np.ndarray, metric: Optional[str] = None) -> None:
        if metric is not None:
            self.metric = metric

        start_time = time.perf_counter()

        self.dimension = xb.shape[1]

        # Create quantizer for IVF
        if self.metric == "l2":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
        elif self.metric == "ip":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
        else:
            raise ValueError(f"Unsupported metric for FAISS IVF: {self.metric}. Supported: l2, ip")

        # FAISS expects float32
        xb_float32 = xb.astype(np.float32, copy=False)

        # Train the index (required for IVF)
        self.index.train(xb_float32)

        # Add vectors to the index
        self.index.add(xb_float32)
        self.is_built = True

        build_time = time.perf_counter() - start_time
        self._stats["build_time_s"] = float(build_time)
        self._stats["ram_rss_mb_after_build"] = psutil.Process(os.getpid()).memory_info().rss / 1e6

    def query(self, xq: np.ndarray, k: int):
        if not self.is_built:
            raise RuntimeError("Index not built. Call build() first.")

        # Set search parameters
        self.index.nprobe = self.nprobe

        # FAISS expects float32
        xq_float32 = xq.astype(np.float32, copy=False)

        # Search for k nearest neighbors
        distances, indices = self.index.search(xq_float32, k)

        # Convert to int64 for indices
        indices = indices.astype(np.int64)

        return indices, distances

    def stats(self) -> dict:
        return dict(self._stats)
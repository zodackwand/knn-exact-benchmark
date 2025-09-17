import time
import os
import psutil
import numpy as np
from typing import Optional
import faiss


class Algo:
    def __init__(self, metric: str = "l2", **params):
        self.metric = metric
        self.params = params
        self.index = None
        self.dimension = None
        self.is_built = False
        self._stats = {
            "build_time_s": None,
            "ram_rss_mb_after_build": None,
            "index_size_bytes": None,
        }

    def build(self, xb: np.ndarray, metric: Optional[str] = None) -> None:
        if metric is not None:
            self.metric = metric

        start_time = time.perf_counter()

        self.dimension = xb.shape[1]

        # Create the appropriate FAISS index based on metric
        if self.metric == "l2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.metric == "ip":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Unsupported metric for FAISS Flat: {self.metric}. Supported: l2, ip")

        # Add vectors to the index
        # FAISS expects float32
        xb_float32 = xb.astype(np.float32, copy=False)
        self.index.add(xb_float32)
        self.is_built = True

        build_time = time.perf_counter() - start_time
        self._stats["build_time_s"] = float(build_time)
        self._stats["ram_rss_mb_after_build"] = psutil.Process(os.getpid()).memory_info().rss / 1e6

        # Calculate index size (FAISS stores the vectors directly for flat index)
        self._stats["index_size_bytes"] = xb_float32.nbytes

    def query(self, xq: np.ndarray, k: int):
        if not self.is_built:
            raise RuntimeError("Index not built. Call build() first.")

        # FAISS expects float32
        xq_float32 = xq.astype(np.float32, copy=False)

        # Search for k nearest neighbors
        distances, indices = self.index.search(xq_float32, k)

        # Convert to int64 for indices (FAISS returns int64 on some systems, int32 on others)
        indices = indices.astype(np.int64)

        return indices, distances

    def stats(self) -> dict:
        return dict(self._stats)
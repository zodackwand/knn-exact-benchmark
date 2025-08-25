import time, os
from typing import Optional, Tuple
import numpy as np
import psutil


class Algo:
    def __init__(self, metric: str = "l2", **params):
        self.metric = metric
        self.params = params
        self._stats = {"build_time_s": None, "ram_rss_mb_after_build": None}
        self._built = False

    def build(self, xb: np.ndarray, metric: Optional[str] = None) -> None:
        if metric is not None:
            self.metric = metric
        t0 = time.perf_counter()
        # TODO: build your index here using xb (np.float32)
        # self.index = ...
        self._built = True
        self._stats["build_time_s"] = time.perf_counter() - t0
        self._stats["ram_rss_mb_after_build"] = psutil.Process(os.getpid()).memory_info().rss / 1e6

    def query(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self._built:
            raise RuntimeError("Index not built. Call build() first.")
        # TODO: compute neighbors
        nq = xq.shape[0]
        I = np.zeros((nq, k), dtype=np.int64)
        D = np.zeros((nq, k), dtype=np.float32)
        return I, D

    def stats(self) -> dict:
        return dict(self._stats)


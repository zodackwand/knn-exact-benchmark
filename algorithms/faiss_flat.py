# algorithms/faiss_flat.py - FAISS Flat (exact brute force) adapter
import time
import os
import psutil
import numpy as np
from typing import Optional, Tuple

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class Algo:
    """FAISS Flat (exact brute force) algorithm adapter"""
    
    def __init__(self, metric: str = "l2", **params):
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available. Install with: pip install faiss-cpu")
            
        self.metric = metric
        self.params = params
        self.index = None
        self.xb = None
        self._stats = {
            "build_time_s": None,
            "ram_rss_mb_after_build": None,
            "index_size_bytes": None,
            "used_metric": None,
        }

    def build(self, xb: np.ndarray, metric: Optional[str] = None) -> None:
        if metric is not None:
            self.metric = metric
            
        t0 = time.perf_counter()
        
        # Store reference for stats
        self.xb = xb.astype(np.float32, copy=False)
        D = self.xb.shape[1]
        
        # Create appropriate FAISS index based on metric
        if self.metric == "l2":
            self.index = faiss.IndexFlatL2(D)
            used_metric = "l2"
        elif self.metric == "ip":
            self.index = faiss.IndexFlatIP(D)
            used_metric = "ip"
        elif self.metric == "cos":
            # For cosine, we normalize vectors and use inner product
            self.index = faiss.IndexFlatIP(D)
            # Normalize base vectors
            faiss.normalize_L2(self.xb)
            used_metric = "cos"
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        # Add vectors to index
        self.index.add(self.xb)
        
        build_time = time.perf_counter() - t0
        
        # Record stats
        self._stats["build_time_s"] = float(build_time)
        self._stats["ram_rss_mb_after_build"] = psutil.Process(os.getpid()).memory_info().rss / 1e6
        self._stats["index_size_bytes"] = self.xb.nbytes
        self._stats["used_metric"] = used_metric

    def query(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")
        
        xq = xq.astype(np.float32, copy=False)
        
        # Handle cosine metric (normalize queries)
        if self.metric == "cos":
            xq_normalized = xq.copy()
            faiss.normalize_L2(xq_normalized)
            distances, indices = self.index.search(xq_normalized, k)
        else:
            distances, indices = self.index.search(xq, k)
        
        return indices.astype(np.int64), distances.astype(np.float32)

    def stats(self) -> dict:
        return dict(self._stats)
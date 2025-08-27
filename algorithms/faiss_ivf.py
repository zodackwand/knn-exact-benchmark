# algorithms/faiss_ivf.py - FAISS IVF (Inverted File) adapter
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
    """FAISS IVF (Inverted File Index) algorithm adapter"""
    
    def __init__(self, metric: str = "l2", nlist: int = 100, nprobe: int = 10, **params):
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available. Install with: pip install faiss-cpu")
            
        self.metric = metric
        self.nlist = nlist  # Number of cluster centroids 
        self.nprobe = nprobe  # Number of clusters to search
        self.params = params
        self.index = None
        self.xb = None
        self._stats = {
            "build_time_s": None,
            "ram_rss_mb_after_build": None,
            "index_size_bytes": None,
            "used_metric": None,
            "nlist": nlist,
            "nprobe": nprobe,
            "trained_vectors": None,
        }

    def build(self, xb: np.ndarray, metric: Optional[str] = None) -> None:
        if metric is not None:
            self.metric = metric
            
        t0 = time.perf_counter()
        
        # Store reference for stats
        self.xb = xb.astype(np.float32, copy=False)
        D = self.xb.shape[1]
        N = self.xb.shape[0]
        
        # Adjust nlist based on dataset size (rule of thumb: nlist = sqrt(N))
        if N < 1000:
            actual_nlist = max(1, min(self.nlist, N // 10))  # At least 10 points per cluster
        else:
            actual_nlist = min(self.nlist, int(np.sqrt(N)))
        
        # Create quantizer (flat index for cluster centroids)
        if self.metric == "l2":
            quantizer = faiss.IndexFlatL2(D)
            used_metric = "l2"
        elif self.metric == "ip":
            quantizer = faiss.IndexFlatIP(D)
            used_metric = "ip"
        elif self.metric == "cos":
            quantizer = faiss.IndexFlatIP(D)
            # Normalize base vectors for cosine
            faiss.normalize_L2(self.xb)
            used_metric = "cos"
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        # Create IVF index
        self.index = faiss.IndexIVFFlat(quantizer, D, actual_nlist)
        self.index.nprobe = min(self.nprobe, actual_nlist)
        
        # Train the index (learns cluster centroids)
        if not self.index.is_trained:
            # Use subset for training if dataset is large
            train_size = min(N, max(actual_nlist * 50, 10000))  # At least 50 points per centroid
            if train_size < N:
                indices = np.random.choice(N, train_size, replace=False)
                train_data = self.xb[indices]
            else:
                train_data = self.xb
            self.index.train(train_data)
        
        # Add all vectors to index
        self.index.add(self.xb)
        
        build_time = time.perf_counter() - t0
        
        # Record stats
        self._stats["build_time_s"] = float(build_time)
        self._stats["ram_rss_mb_after_build"] = psutil.Process(os.getpid()).memory_info().rss / 1e6
        self._stats["index_size_bytes"] = self.xb.nbytes  # Approximate
        self._stats["used_metric"] = used_metric
        self._stats["nlist"] = actual_nlist
        self._stats["nprobe"] = self.index.nprobe
        self._stats["trained_vectors"] = train_size if 'train_size' in locals() else N

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
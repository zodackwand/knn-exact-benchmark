# algorithms/faiss_hnsw.py - FAISS HNSW (Hierarchical Navigable Small World) adapter
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
    """FAISS HNSW (Hierarchical Navigable Small World) algorithm adapter"""
    
    def __init__(self, metric: str = "l2", M: int = 16, efConstruction: int = 200, efSearch: int = 16, **params):
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available. Install with: pip install faiss-cpu")
            
        self.metric = metric
        self.M = M  # Number of connections for each vertex
        self.efConstruction = efConstruction  # Size of dynamic candidate list during construction
        self.efSearch = efSearch  # Size of dynamic candidate list during search
        self.params = params
        self.index = None
        self.xb = None
        self._stats = {
            "build_time_s": None,
            "ram_rss_mb_after_build": None,
            "index_size_bytes": None,
            "used_metric": None,
            "M": M,
            "efConstruction": efConstruction,
            "efSearch": efSearch,
            "edges": None,
            "avg_out_degree": None,
        }

    def build(self, xb: np.ndarray, metric: Optional[str] = None) -> None:
        if metric is not None:
            self.metric = metric
            
        t0 = time.perf_counter()
        
        # Store reference for stats
        self.xb = xb.astype(np.float32, copy=False)
        D = self.xb.shape[1]
        N = self.xb.shape[0]
        
        # Create appropriate HNSW index based on metric
        if self.metric == "l2":
            self.index = faiss.IndexHNSWFlat(D, self.M)
            used_metric = "l2"
        elif self.metric == "ip":
            # HNSW doesn't directly support IP, use L2 with preprocessing
            self.index = faiss.IndexHNSWFlat(D, self.M)
            used_metric = "ip_via_l2"
            # Note: This is a limitation - true IP requires different preprocessing
        elif self.metric == "cos":
            self.index = faiss.IndexHNSWFlat(D, self.M)
            # Normalize base vectors for cosine
            faiss.normalize_L2(self.xb)
            used_metric = "cos"
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        # Set construction parameters
        self.index.hnsw.efConstruction = self.efConstruction
        self.index.hnsw.efSearch = self.efSearch
        
        # Add vectors to index (HNSW builds incrementally)
        self.index.add(self.xb)
        
        build_time = time.perf_counter() - t0
        
        # Calculate graph statistics
        total_edges = 0
        if hasattr(self.index, 'hnsw') and hasattr(self.index.hnsw, 'graph'):
            # Approximate edge count (FAISS doesn't expose this easily)
            total_edges = N * self.M  # Rough estimate
        
        # Record stats
        self._stats["build_time_s"] = float(build_time)
        self._stats["ram_rss_mb_after_build"] = psutil.Process(os.getpid()).memory_info().rss / 1e6
        self._stats["index_size_bytes"] = self.xb.nbytes + total_edges * 4  # Approximate
        self._stats["used_metric"] = used_metric
        self._stats["edges"] = total_edges
        self._stats["avg_out_degree"] = float(total_edges / N) if N > 0 else 0.0

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
# algorithms/annoy_adapter.py - Annoy (Spotify) adapter
import time
import os
import psutil
import numpy as np
from typing import Optional, Tuple

try:
    from annoy import AnnoyIndex
    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False


class Algo:
    """Annoy (Approximate Nearest Neighbors Oh Yeah) algorithm adapter"""
    
    def __init__(self, metric: str = "l2", n_trees: int = 10, **params):
        if not ANNOY_AVAILABLE:
            raise RuntimeError("Annoy not available. Install with: pip install annoy")
            
        self.metric = metric
        self.n_trees = n_trees  # More trees = better accuracy, slower build
        self.params = params
        self.index = None
        self.xb = None
        self.D = None
        self._stats = {
            "build_time_s": None,
            "ram_rss_mb_after_build": None,
            "index_size_bytes": None,
            "used_metric": None,
            "n_trees": n_trees,
            "edges": None,
        }

    def build(self, xb: np.ndarray, metric: Optional[str] = None) -> None:
        if metric is not None:
            self.metric = metric
            
        t0 = time.perf_counter()
        
        # Store reference for stats
        self.xb = xb.astype(np.float32, copy=False)
        self.D = self.xb.shape[1]
        N = self.xb.shape[0]
        
        # Map our metrics to Annoy metrics
        if self.metric == "l2":
            annoy_metric = "euclidean"
            used_metric = "l2"
        elif self.metric == "ip":
            # Annoy doesn't support IP directly, use dot product
            annoy_metric = "dot"
            used_metric = "ip_via_dot"
        elif self.metric == "cos":
            annoy_metric = "angular"  # Annoy's angular = cosine distance
            used_metric = "cos"
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        # Create Annoy index
        self.index = AnnoyIndex(self.D, annoy_metric)
        
        # Add all vectors
        for i in range(N):
            self.index.add_item(i, self.xb[i])
        
        # Build the index (creates the trees)
        self.index.build(self.n_trees)
        
        build_time = time.perf_counter() - t0
        
        # Estimate index size (Annoy doesn't provide direct access)
        # Rough estimate: trees * nodes_per_tree * node_size
        estimated_size = self.xb.nbytes + (self.n_trees * N * 8)  # Approximate
        
        # Record stats
        self._stats["build_time_s"] = float(build_time)
        self._stats["ram_rss_mb_after_build"] = psutil.Process(os.getpid()).memory_info().rss / 1e6
        self._stats["index_size_bytes"] = estimated_size
        self._stats["used_metric"] = used_metric
        self._stats["edges"] = N * self.n_trees  # Rough estimate

    def query(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")
        
        xq = xq.astype(np.float32, copy=False)
        nq = xq.shape[0]
        
        # Prepare output arrays
        indices = np.zeros((nq, k), dtype=np.int64)
        distances = np.zeros((nq, k), dtype=np.float32)
        
        # Query each vector individually (Annoy's API)
        for i in range(nq):
            query_vec = xq[i].tolist()  # Annoy expects Python lists
            
            # Get k nearest neighbors
            nn_indices, nn_distances = self.index.get_nns_by_vector(
                query_vec, k, include_distances=True
            )
            
            # Handle case where fewer than k neighbors are found
            actual_k = len(nn_indices)
            if actual_k < k:
                # Pad with -1 for indices and inf for distances
                nn_indices.extend([-1] * (k - actual_k))
                nn_distances.extend([float('inf')] * (k - actual_k))
            
            indices[i] = nn_indices[:k]
            distances[i] = nn_distances[:k]
        
        return indices, distances

    def stats(self) -> dict:
        return dict(self._stats)
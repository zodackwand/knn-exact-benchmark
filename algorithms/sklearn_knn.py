# algorithms/sklearn_knn.py - Scikit-learn KNN adapter
import time
import os
import psutil
import numpy as np
from typing import Optional, Tuple

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class Algo:
    """Scikit-learn NearestNeighbors algorithm adapter"""
    
    def __init__(self, metric: str = "l2", algorithm: str = "auto", leaf_size: int = 30, **params):
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("Scikit-learn not available. Install with: pip install scikit-learn")
            
        self.metric = metric
        self.algorithm = algorithm  # 'ball_tree', 'kd_tree', 'brute', 'auto'
        self.leaf_size = leaf_size
        self.params = params
        self.nn_model = None
        self.xb = None
        self._stats = {
            "build_time_s": None,
            "ram_rss_mb_after_build": None,
            "index_size_bytes": None,
            "used_metric": None,
            "algorithm_used": None,
            "leaf_size": leaf_size,
        }

    def build(self, xb: np.ndarray, metric: Optional[str] = None) -> None:
        if metric is not None:
            self.metric = metric
            
        t0 = time.perf_counter()
        
        # Store reference for stats
        self.xb = xb.astype(np.float32, copy=False)
        
        # Map our metrics to sklearn metrics
        if self.metric == "l2":
            sklearn_metric = "euclidean"
            used_metric = "l2"
        elif self.metric == "ip":
            # Inner product not directly supported, we'll use cosine with normalization
            sklearn_metric = "cosine"
            used_metric = "ip_via_cosine"
            # Note: This is a limitation - true IP requires custom implementation
        elif self.metric == "cos":
            sklearn_metric = "cosine"  
            used_metric = "cos"
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        # Create and fit model
        self.nn_model = NearestNeighbors(
            n_neighbors=1,  # Will be overridden in query
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=sklearn_metric,
            **self.params
        )
        
        data_to_fit = self.xb
        if self.metric == "ip":
            # For IP via cosine, normalize the data
            norms = np.linalg.norm(data_to_fit, axis=1, keepdims=True) + 1e-12
            data_to_fit = data_to_fit / norms
        
        self.nn_model.fit(data_to_fit)
        
        build_time = time.perf_counter() - t0
        
        # Record stats
        self._stats["build_time_s"] = float(build_time)
        self._stats["ram_rss_mb_after_build"] = psutil.Process(os.getpid()).memory_info().rss / 1e6
        self._stats["index_size_bytes"] = self.xb.nbytes
        self._stats["used_metric"] = used_metric
        self._stats["algorithm_used"] = self.nn_model._fit_method

    def query(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.nn_model is None:
            raise RuntimeError("Index not built. Call build() first.")
        
        xq = xq.astype(np.float32, copy=False)
        
        # Set k for this query
        self.nn_model.set_params(n_neighbors=k)
        
        query_data = xq
        if self.metric == "ip":
            # Normalize queries for IP via cosine
            norms = np.linalg.norm(query_data, axis=1, keepdims=True) + 1e-12
            query_data = query_data / norms
        
        distances, indices = self.nn_model.kneighbors(query_data)
        
        # Convert cosine distances back to IP if needed
        if self.metric == "ip":
            # cosine distance = 1 - cosine_similarity
            # For normalized vectors: IP = cosine_similarity
            distances = 1.0 - distances  # Convert to similarity
            distances = -distances  # Negate for IP (higher = better)
        
        return indices.astype(np.int64), distances.astype(np.float32)

    def stats(self) -> dict:
        return dict(self._stats)
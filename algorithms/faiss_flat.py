import time
import os
import psutil
import numpy as np
from typing import Optional, Tuple
import faiss


class Algo:
    def __init__(self, metric: str = "l2", **params):
        self.metric = metric
        self.params = params
        self.index = None
        self.dimension = None
        self.is_built = False
        self._id_selector = None
        self._cached_filter_range = None
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

    def _create_candidates(self, vector_metadata: np.ndarray, filter_range: Tuple[int, int]) -> np.ndarray:
        """Create candidate indices from metadata filtering.

        Args:
            vector_metadata: Array of metadata values for each vector
            filter_range: Tuple of (min_val, max_val) for filtering

        Returns:
            Array of candidate indices that pass the filter
        """
        min_val, max_val = filter_range
        mask = (vector_metadata >= min_val) & (vector_metadata <= max_val)
        candidates = np.where(mask)[0].astype(np.int64)
        return candidates

    def query(self, xq: np.ndarray, k: int, vector_metadata: Optional[np.ndarray] = None,
              filter_range: Optional[Tuple[int, int]] = None):
        if not self.is_built:
            raise RuntimeError("Index not built. Call build() first.")

        # FAISS expects float32
        xq_float32 = xq.astype(np.float32, copy=False)

        # Apply filtering if requested
        if vector_metadata is not None and filter_range is not None:
            # Cache ID selector for same filter range to avoid recreation overhead
            if self._cached_filter_range != filter_range:
                # Clean up previous selector
                if self._id_selector:
                    del self._id_selector

                # Create new candidates and ID selector
                candidates = self._create_candidates(vector_metadata, filter_range)
                self._id_selector = faiss.IDSelectorArray(candidates)
                self._cached_filter_range = filter_range

            # Create search parameters with cached ID selector
            search_params = faiss.SearchParameters()
            search_params.sel = self._id_selector

            # Search with filtering
            distances, indices = self.index.search(xq_float32, k, params=search_params)
        else:
            # Standard search without filtering
            distances, indices = self.index.search(xq_float32, k)

        # Convert to int64 for indices (FAISS returns int64 on some systems, int32 on others)
        indices = indices.astype(np.int64)

        return indices, distances

    def stats(self) -> dict:
        return dict(self._stats)
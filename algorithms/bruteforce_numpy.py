# algorithms/bruteforce_numpy.py - исправленная версия
import time
import numpy as np


class BruteForceNumpy:
    def __init__(self, metric="l2", **params):
        self.metric = metric
        self.params = params
        self.index = None
        self.is_built = False
        self._stats = {}

    def build(self, data):
        start_time = time.perf_counter()

        self.index = data.astype(np.float32, copy=False)
        if self.metric == "cos":
            # Нормализуем базу для косинусного расстояния
            norms = np.linalg.norm(self.index, axis=1, keepdims=True) + 1e-12
            self.index = self.index / norms

        self.is_built = True
        build_time = time.perf_counter() - start_time

        return build_time

    def query(self, queries, k):
        if not self.is_built:
            raise RuntimeError("Index not built. Call build() first.")

        batch_size = 256
        n_queries = queries.shape[0]
        indices = np.empty((n_queries, k), dtype=np.int64)
        distances = np.empty((n_queries, k), dtype=np.float32)

        for start in range(0, n_queries, batch_size):
            end = min(start + batch_size, n_queries)
            batch_queries = queries[start:end].astype(np.float32, copy=False)

            # Вычисляем расстояния
            if self.metric == "l2":
                dists = self._compute_l2_distances(batch_queries)
            elif self.metric == "ip":
                dists = -np.dot(batch_queries, self.index.T)
            elif self.metric == "cos":
                normalized_queries = batch_queries / (np.linalg.norm(batch_queries, axis=1, keepdims=True) + 1e-12)
                dists = -np.dot(normalized_queries, self.index.T)

            # Находим k ближайших
            idx = np.argpartition(dists, k - 1, axis=1)[:, :k]
            batch_dists = np.take_along_axis(dists, idx, axis=1)
            sort_order = np.argsort(batch_dists, axis=1)

            indices[start:end] = np.take_along_axis(idx, sort_order, axis=1)
            distances[start:end] = np.take_along_axis(batch_dists, sort_order, axis=1)

        return indices, distances

    def _compute_l2_distances(self, queries):
        # Оптимизированное вычисление L2 расстояний
        q_norm2 = np.sum(queries * queries, axis=1, keepdims=True)
        x_norm2 = np.sum(self.index * self.index, axis=1)
        dots = np.dot(queries, self.index.T)
        return q_norm2 + x_norm2[None, :] - 2.0 * dots
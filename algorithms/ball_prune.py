# algorithms/ball_prune.py
# Адаптер вокруг реализаций из allens_code (Ball-Prune / dynamic radius BFS)
import time
import os
import numpy as np
import psutil
from typing import Tuple, Optional, Dict, Any, Union
from allens_code.functions.query import bfs_within_ball_dynamic
from allens_code.functions.distance import sq_Euclidean_d
from allens_code.functions.construction import build_ball_traversable_graph, create_percentile_radii_by_node


class Algo:
    def __init__(self, metric: str = "l2", use_empirical=False, hnsw_M: int = 32, ef_search: int = 1,
                 seed_select_sample: int = 256):
        if metric != "l2":
            raise ValueError("Ball_Prune поддерживает только metric='l2' на текущем этапе")
        self.metric = metric
        self.use_empirical = use_empirical
        self.hnsw_M = hnsw_M
        self.ef_search = ef_search
        self.seed_select_sample = seed_select_sample

        self.xb: Optional[np.ndarray] = None
        self.graph = None
        self.safe_radii_by_node = None
        self.percentile_radii_by_node = None

        self._stats = {
            "build_time_s": None,
            "ram_rss_mb_after_build": None,
            "edges": None,
            "visited_nodes_avg": None,
            "visited_nodes_p95": None,
        }

    def build(self, xb: np.ndarray, metric: Optional[str] = None):
        if metric is not None and metric != self.metric:
            if metric != "l2":
                raise ValueError("Ball_Prune поддерживает только metric='l2'")
        t0 = time.perf_counter()

        self.xb = xb.astype(np.float32, copy=False)

        # Строим направленный граф и уровни безопасных радиусов
        self.graph, self.safe_radii_by_node = build_ball_traversable_graph(self.xb)

        # Необязательная эмпирическая подстройка радиусов (дорого, выключено по умолчанию)
        if self.use_empirical:
            # По умолчанию k=5, p=0.95 как в исходнике; можно настроить через конструктор при необходимости
            self.percentile_radii_by_node = create_percentile_radii_by_node(self.xb, self.safe_radii_by_node, k=5, p=0.95)
        else:
            self.percentile_radii_by_node = None

        build_time_s = time.perf_counter() - t0
        self._stats["build_time_s"] = float(build_time_s)
        self._stats["ram_rss_mb_after_build"] = psutil.Process(os.getpid()).memory_info().rss / 1e6
        # число рёбер
        if self.graph is not None:
            self._stats["edges"] = int(sum(len(v) for v in self.graph.values()))
        else:
            self._stats["edges"] = 0

    def _choose_seed(self, q: np.ndarray) -> int:
        """Простой выбор стартовой точки: перебираем небольшой семпл базы и берём ближайшую по L2."""
        n = self.xb.shape[0]
        if n <= self.seed_select_sample:
            cand = np.arange(n)
        else:
            # фиксированная подвыборка для скорости
            rng = np.random.default_rng(12345)
            cand = rng.choice(n, size=self.seed_select_sample, replace=False)
        # считаем квадрат L2 до кандидатов
        d2 = np.sum((self.xb[cand] - q) ** 2, axis=1)
        return int(cand[int(np.argmin(d2))])

    def _query_one(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = q.astype(np.float32, copy=False)
        root = self._choose_seed(q)
        expanded, knn_list, _jumps = bfs_within_ball_dynamic(
            graph=self.graph,
            dataset=self.xb,
            root=root,
            k=k,
            query=q,
            evaluated_set=set(),
            traversable_radii_by_node=self.safe_radii_by_node,
            percentile_radii_by_node=self.percentile_radii_by_node,
        )
        # knn_list: list[(dist, idx)] по возрастанию dist
        if len(knn_list) == 0:
            # fallback: если ничего не найдено (маловероятно), вернём ближайшее из семпла
            root = self._choose_seed(q)
            d = sq_Euclidean_d(q, self.xb[root])
            return np.array([root], dtype=np.int64), np.array([d], dtype=np.float32)
        idx = np.array([i for _, i in knn_list[:k]], dtype=np.int64)
        dist = np.array([d for d, _ in knn_list[:k]], dtype=np.float32)
        return idx, dist, len(expanded)

    def query(self, xq: np.ndarray, k: int):
        if self.xb is None or self.graph is None:
            raise RuntimeError("Index not built. Call build() first.")
        xq = xq.astype(np.float32, copy=False)
        nq = xq.shape[0]
        I = np.empty((nq, k), dtype=np.int64)
        D = np.empty((nq, k), dtype=np.float32)
        visited_counts = []
        for i in range(nq):
            result = self._query_one(xq[i], k)
            if len(result) == 3:
                idx, dist, visited = result
                visited_counts.append(visited)
            else:
                idx, dist = result
                visited_counts.append(0)
            if idx.shape[0] < k:
                # добьём до k простым брутфорсом по базе — редкий случай
                need = k - idx.shape[0]
                # посчитаем до небольшого семпла оставшиеся
                n = self.xb.shape[0]
                rest = np.setdiff1d(np.arange(n, dtype=np.int64), idx, assume_unique=False)
                if rest.size > 0:
                    cand = rest if rest.size <= 1024 else rest[:1024]
                    scores = np.sum((self.xb[cand] - xq[i]) ** 2, axis=1)
                    ord_ = np.argsort(scores)[:need]
                    idx = np.concatenate([idx, cand[ord_]])
                    dist = np.concatenate([dist, scores[ord_].astype(np.float32)])
            I[i, :k] = idx[:k]
            D[i, :k] = dist[:k]
        # Update stats with visited node metrics
        if visited_counts:
            self._stats["visited_nodes_avg"] = float(np.mean(visited_counts))
            self._stats["visited_nodes_p95"] = float(np.percentile(visited_counts, 95))
        return I, D

    def stats(self):
        return dict(self._stats)


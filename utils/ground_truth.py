# utils/ground_truth.py
import os
import numpy as np
from typing import Tuple


def _gt_dir(root: str = "data/ground_truth") -> str:
    os.makedirs(root, exist_ok=True)
    return root


def _gt_path(dataset_key: str, metric: str, k: int, root: str = "data/ground_truth") -> str:
    safe_metric = metric.lower()
    return os.path.join(_gt_dir(root), f"{dataset_key}_{safe_metric}_k{k}.npz")


def _compute_gt(xb: np.ndarray, xq: np.ndarray, k: int, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    # Бэтчевый брутфорс для памяти
    def scores_batch(qb: np.ndarray) -> np.ndarray:
        if metric == "l2":
            x_norm2 = np.sum(xb * xb, axis=1)  # [N]
            q_norm2 = np.sum(qb * qb, axis=1, keepdims=True)  # [b,1]
            dots = qb @ xb.T
            return (q_norm2 + x_norm2[None, :] - 2.0 * dots)
        elif metric == "ip":
            return -(qb @ xb.T)
        elif metric == "cos":
            xb_n = xb / (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-12)
            qb_n = qb / (np.linalg.norm(qb, axis=1, keepdims=True) + 1e-12)
            return -(qb_n @ xb_n.T)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    NQ = xq.shape[0]
    I_all = np.empty((NQ, k), dtype=np.int64)
    D_all = np.empty((NQ, k), dtype=np.float32)
    B = 256
    for s in range(0, NQ, B):
        e = min(s + B, NQ)
        qb = xq[s:e].astype(np.float32, copy=False)
        sc = scores_batch(qb)
        idx = np.argpartition(sc, kth=k-1, axis=1)[:, :k]
        part = np.take_along_axis(sc, idx, axis=1)
        ord_ = np.argsort(part, axis=1)
        I = np.take_along_axis(idx, ord_, axis=1)
        D = np.take_along_axis(part, ord_, axis=1)
        I_all[s:e] = I
        D_all[s:e] = D
    return I_all, D_all


def load_or_generate(dataset_key: str,
                     xb: np.ndarray,
                     xq: np.ndarray,
                     k: int,
                     metric: str,
                     root: str = "data/ground_truth") -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает кеш ground-truth, либо вычисляет и кеширует.
    Путь: data/ground_truth/{dataset_key}_{metric}_k{k}.npz с массивами I, D.
    """
    path = _gt_path(dataset_key, metric, k, root)
    if os.path.exists(path):
        with np.load(path) as npz:
            return npz["I"], npz["D"]
    I, D = _compute_gt(xb.astype(np.float32, copy=False), xq.astype(np.float32, copy=False), k, metric)
    np.savez_compressed(path, I=I, D=D)
    return I, D

#!/usr/bin/env python3
"""
Minimal smoke test for an algorithm adapter implementing the Algo interface.
- Imports algorithms.<name>.Algo
- Loads/generates a toy dataset by key
- Builds the index, runs queries, computes Recall@k vs cached ground truth
- Prints a short summary and exits with non-zero code if --min-recall is set and not met

Usage example:
  python tools/smoke_algo.py --algo bruteforce_numpy --dataset-key toy_gaussian_N10000_D64_nq200_seed42 --k 10
"""
import argparse
import importlib
import os
import sys
import time
from typing import Tuple

import numpy as np

# Local imports from project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root
from utils.make_toy import load_by_key  # noqa: E402
from utils import ground_truth as gt_mod  # noqa: E402


def recall_at_k(I_pred: np.ndarray, I_true: np.ndarray) -> float:
    k = min(I_pred.shape[1], I_true.shape[1])
    I_pred = I_pred[:, :k]
    I_true = I_true[:, :k]
    hits = 0
    for p, t in zip(I_pred, I_true):
        hits += len(set(p.tolist()) & set(t.tolist()))
    return hits / (I_true.shape[0] * k)


def load_algo(name: str, metric: str):
    mod = importlib.import_module(f"algorithms.{name}")
    AlgoClass = getattr(mod, "Algo")
    return AlgoClass(metric=metric)


def run(algo_name: str, dataset_key: str, metric: str, k: int, data_dir: str, warmup: int) -> Tuple[float, float, float]:
    xb, xq, meta = load_by_key(dataset_key, data_dir=data_dir)
    algo = load_algo(algo_name, metric)

    t0 = time.perf_counter()
    algo.build(xb, metric=metric)
    build_s = time.perf_counter() - t0

    w = min(warmup, len(xq))
    if w > 0:
        _ = algo.query(xq[:w], k)

    # run all queries once
    t0 = time.perf_counter()
    I_pred, _D_pred = algo.query(xq, k)
    query_s = time.perf_counter() - t0

    I_true, _ = gt_mod.load_or_generate(meta["key"], xb, xq, k=k, metric=metric)
    rec = recall_at_k(I_pred, I_true)
    return build_s, query_s, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, help="Algorithm module name under algorithms/ (e.g., bruteforce_numpy)")
    ap.add_argument("--dataset-key", dest="dataset_key", default="toy_gaussian_N10000_D64_nq200_seed42")
    ap.add_argument("--metric", default="l2", choices=["l2", "ip", "cos"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--data-dir", dest="data_dir", default="data")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--min-recall", dest="min_recall", type=float, default=None, help="If set, exit non-zero if Recall@k < this value")
    args = ap.parse_args()

    try:
        build_s, query_s, rec = run(args.algo, args.dataset_key, args.metric, args.k, args.data_dir, args.warmup)
    except Exception as e:
        print(f"[SMOKE] FAILED: {e}")
        sys.exit(2)

    print(f"[SMOKE] algo={args.algo} dataset={args.dataset_key} metric={args.metric} k={args.k}")
    print(f"[SMOKE] build_s={build_s:.4f}s total_query_s={query_s:.4f}s recall@{args.k}={rec:.6f}")

    if args.min_recall is not None and rec < args.min_recall:
        print(f"[SMOKE] Recall below threshold: {rec:.6f} < {args.min_recall}")
        sys.exit(1)

    print("[SMOKE] OK")


if __name__ == "__main__":
    main()

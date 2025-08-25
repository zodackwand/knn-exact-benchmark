# bench.py
import os, json, time, argparse, datetime as dt, importlib
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import psutil
import yaml
from datasets.make_toy import load_or_make, load_by_key
from datasets import ground_truth as gt_mod


def safe_load_algo(name: str, metric: str):
    try:
        mod = importlib.import_module(f"algorithms.{name}")
        AlgoClass = getattr(mod, "Algo")
        return AlgoClass(metric=metric)
    except Exception as e:
        print(f"[!] Не удалось загрузить алгоритм '{name}': {e}. Пропускаю.")
        return None


def recall_at_k(I_pred: np.ndarray, I_true: np.ndarray) -> float:
    k = I_true.shape[1]
    if I_pred.shape[1] < k:
        # подрезаем true если вдруг сравниваем с меньшим k
        k = I_pred.shape[1]
        I_true = I_true[:, :k]
    hits = 0
    for p, t in zip(I_pred, I_true):
        hits += len(set(p.tolist()) & set(t.tolist()))
    return hits / (I_true.shape[0] * k)


def run_single(algo_name: str,
               dataset_key: str,
               metric: str,
               k_values: List[int],
               warmup: int,
               data_dir: str,
               out_root: str) -> List[Dict[str, Any]]:
    """Запускает один алгоритм на одном датасете для списка k. Возвращает список сводок."""
    xb, xq, meta = load_by_key(dataset_key, data_dir=data_dir)
    run_dir = os.path.join(out_root, f"{algo_name}__{dataset_key}")
    os.makedirs(run_dir, exist_ok=True)

    algo = safe_load_algo(algo_name, metric=metric)
    if algo is None:
        return []

    rss_before = psutil.Process(os.getpid()).memory_info().rss / 1e6
    t0_build = time.perf_counter()
    algo.build(xb, metric=metric)
    build_time_s = time.perf_counter() - t0_build

    # warmup
    w = min(warmup, len(xq))
    if w > 0:
        _ = algo.query(xq[:w], max(k_values))

    # подготовим GT один раз с максимальным k
    k_max = max(k_values)
    I_true_max, _D_true_max = gt_mod.load_or_generate(meta["key"], xb, xq, k=k_max, metric=metric)

    summaries = []
    for k in k_values:
        print(f"[+] {algo_name} @ {dataset_key} metric={metric} k={k}")
        lat_ms = []
        for i in range(len(xq)):
            q = xq[i:i+1]
            t0 = time.perf_counter_ns()
            I_pred, D_pred = algo.query(q, k)
            dt_ms = (time.perf_counter_ns() - t0) / 1e6
            lat_ms.append(dt_ms)
        lat_ms = np.array(lat_ms, dtype=np.float64)
        avg_ms = float(lat_ms.mean())
        p95_ms = float(np.percentile(lat_ms, 95))

        I_pred_all, _ = algo.query(xq, k)
        rec = recall_at_k(I_pred_all, I_true_max[:, :k])

        stats = {}
        try:
            stats = algo.stats() or {}
        except Exception:
            stats = {}
        rss_after = psutil.Process(os.getpid()).memory_info().rss / 1e6

        combo_dir = os.path.join(run_dir, f"k{k}")
        os.makedirs(combo_dir, exist_ok=True)

        # сохраняем артефакты per-k
        np.savetxt(os.path.join(combo_dir, "latencies_ms.csv"), lat_ms, delimiter=",", fmt="%.6f")
        plt.figure()
        plt.bar(["avg_ms", "p95_ms"], [avg_ms, p95_ms])
        plt.title(f"Latency (ms) — {algo_name} — k={k}")
        plt.ylabel("ms")
        plt.savefig(os.path.join(combo_dir, "latency.png"), dpi=150, bbox_inches="tight")
        plt.close()

        summary = {
            "algo": algo_name,
            "dataset_key": dataset_key,
            "metric": metric,
            "k": int(k),
            "latency_ms_avg": round(avg_ms, 4),
            "latency_ms_p95": round(p95_ms, 4),
            "recall_at_k": round(rec, 6),
            "build_time_s": round(stats.get("build_time_s", build_time_s), 4),
            "ram_rss_mb_before_build": round(rss_before, 2),
            "ram_rss_mb_after_build": round(stats.get("ram_rss_mb_after_build", rss_after), 2),
            "visited_nodes": stats.get("visited_nodes"),
            "edges": stats.get("edges"),
            "combo_dir": combo_dir,
        }
        with open(os.path.join(combo_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        summaries.append(summary)

    return summaries


def run_benchmark(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_root = cfg.get("outdir", "results")
    os.makedirs(out_root, exist_ok=True)
    run_id = dt.datetime.now().strftime("run-%Y%m%d-%H%M%S")
    run_dir = os.path.join(out_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    algorithms = cfg.get("algorithms", [])
    datasets = cfg.get("utils", [])
    metrics = cfg.get("metrics", ["latency", "recall@1", "recall@10", "memory", "build_time"])  # для совместимости
    k_values = cfg.get("k_values", [10])
    data_dir = cfg.get("data_dir", "data")
    warmup = cfg.get("warmup", 50)
    metric = cfg.get("metric", "l2")

    all_rows: List[Dict[str, Any]] = []

    for ds_key in datasets:
        for algo_cfg in algorithms:
            if isinstance(algo_cfg, dict):
                name = algo_cfg.get("name")
            else:
                name = str(algo_cfg)
            if not name:
                continue
            rows = run_single(
                algo_name=name,
                dataset_key=ds_key,
                metric=metric,
                k_values=k_values,
                warmup=warmup,
                data_dir=data_dir,
                out_root=run_dir,
            )
            all_rows.extend(rows)

    # агрегаты
    agg_json_path = os.path.join(run_dir, "aggregated_results.json")
    with open(agg_json_path, "w") as f:
        json.dump(all_rows, f, indent=2)

    # CSV
    import csv
    csv_path = os.path.join(run_dir, "aggregated_results.csv")
    fieldnames = [
        "algo", "dataset_key", "metric", "k",
        "latency_ms_avg", "latency_ms_p95", "recall_at_k",
        "build_time_s", "ram_rss_mb_before_build", "ram_rss_mb_after_build",
        "visited_nodes", "edges", "combo_dir"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print("\n=== AGGREGATED SUMMARY ===")
    print(f"Wrote: {agg_json_path}\n      and {csv_path}")
    print(f"Artifacts root: {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Путь к YAML конфигу. Если задан, используется режим бенчмарка по конфигу.")
    ap.add_argument("--algo", type=str, default="faiss_exact",
                    help="имя модуля в algorithms/, напр. faiss_exact или bruteforce_numpy")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--nq", type=int, default=1_000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--metric", type=str, default="l2", choices=["l2","ip","cos"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dist", type=str, default="gaussian", choices=["gaussian","uniform"])
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--data-dir", type=str, default="data")
    args = ap.parse_args()

    if args.config:
        run_benchmark(args.config)
        return

    # Одноразовый режим (как раньше), c кеш-GT
    os.makedirs(args.outdir, exist_ok=True)
    run_id = dt.datetime.now().strftime("run-%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.outdir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"[+] Load-or-make data N={args.n}, D={args.dim}, nq={args.nq}, dist={args.dist}, seed={args.seed}")
    xb, xq, meta = load_or_make(N=args.n, D=args.dim, nq=args.nq, dist=args.dist, seed=args.seed, data_dir=args.data_dir)
    print(f"[+] Data ready (cached or generated): xb={xb.shape}, xq={xq.shape}")

    # Выбираем алгоритм
    print(f"[+] Loading algorithm: {args.algo}")
    algo = safe_load_algo(args.algo, metric=args.metric)
    if algo is None:
        return

    rss_before = psutil.Process(os.getpid()).memory_info().rss / 1e6

    print("[+] Building index...")
    t0_build = time.perf_counter()
    algo.build(xb, metric=args.metric)
    build_time_s = time.perf_counter() - t0_build

    # Warmup
    w = min(args.warmup, len(xq))
    if w > 0:
        _I, _D = algo.query(xq[:w], args.k)

    # Пер-запросные латентности
    lat_ms = []
    print("[+] Running queries...")
    for i in range(len(xq)):
        q = xq[i:i+1]
        t0 = time.perf_counter_ns()
        I_pred, D_pred = algo.query(q, args.k)
        dt_ms = (time.perf_counter_ns() - t0) / 1e6
        lat_ms.append(dt_ms)

    lat_ms = np.array(lat_ms, dtype=np.float64)
    avg_ms = float(lat_ms.mean())
    p95_ms = float(np.percentile(lat_ms, 95))

    # Точное топ-k для Recall (ground truth) — кешируем
    print("[+] Computing/loading ground-truth for Recall@k...")
    I_true, _ = gt_mod.load_or_generate(meta["key"], xb, xq, args.k, metric=args.metric)

    # Предсказания для Recall
    I_pred_all, _ = algo.query(xq, args.k)
    rec = recall_at_k(I_pred_all, I_true)

    try:
        stats = algo.stats() or {}
    except Exception:
        stats = {}
    rss_after = psutil.Process(os.getpid()).memory_info().rss / 1e6

    summary = {
        "run_id": run_id,
        "algo": args.algo,
        "dataset": {"N": args.n, "D": args.dim, "nq": args.nq, "dist": args.dist},
        "data_key": meta.get("key"),
        "metric": args.metric,
        "k": args.k,
        "latency_ms_avg": round(avg_ms, 4),
        "latency_ms_p95": round(p95_ms, 4),
        "recall_at_k": round(rec, 6),
        "visited_nodes": stats.get("visited_nodes", None),
        "edges": stats.get("edges", None),
        "build_time_s": round(stats.get("build_time_s", build_time_s), 4),
        "ram_rss_mb_before_build": round(rss_before, 2),
        "ram_rss_mb_after_build": round(stats.get("ram_rss_mb_after_build", rss_after), 2),
    }

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    np.savetxt(os.path.join(run_dir, "latencies_ms.csv"), lat_ms, delimiter=",", fmt="%.6f")

    plt.figure()
    plt.bar(["avg_ms", "p95_ms"], [avg_ms, p95_ms])
    plt.title(f"Query Latency (ms) — {args.algo}")
    plt.ylabel("ms")
    plt.savefig(os.path.join(run_dir, "latency.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("\n=== SUMMARY ===")
    import pprint; pprint.pprint(summary)
    print(f"\nArtifacts saved to: {run_dir}")

if __name__ == "__main__":
    main()

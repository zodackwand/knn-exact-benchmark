import os
import csv
import argparse
from collections import defaultdict
from typing import Dict, Tuple


Key = Tuple[str, str, str, int]  # (algo, dataset_key, metric, k)


def read_agg_csv(path: str) -> Dict[Key, dict]:
    data: Dict[Key, dict] = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key: Key = (
                row["algo"],
                row["dataset_key"],
                row["metric"],
                int(row["k"]),
            )
            data[key] = row
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Path to run A directory (containing aggregated_results.csv)")
    ap.add_argument("--b", required=True, help="Path to run B directory")
    ap.add_argument("--out", required=True, help="Output directory for diff CSV and plots")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    a_csv = os.path.join(args.a, "aggregated_results.csv")
    b_csv = os.path.join(args.b, "aggregated_results.csv")
    if not (os.path.exists(a_csv) and os.path.exists(b_csv)):
        raise FileNotFoundError("aggregated_results.csv not found in one of the runs")

    A = read_agg_csv(a_csv)
    B = read_agg_csv(b_csv)

    keys = sorted(set(A.keys()) | set(B.keys()))
    out_csv = os.path.join(args.out, "diff.csv")
    fields = [
        "algo", "dataset_key", "metric", "k",
        "recall_at_k_A", "recall_at_k_B", "delta_recall",
        "latency_ms_avg_A", "latency_ms_avg_B", "ratio_avg_latency_B_over_A",
        "latency_ms_p95_A", "latency_ms_p95_B", "ratio_p95_latency_B_over_A",
    ]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k in keys:
            a = A.get(k)
            b = B.get(k)
            row = {
                "algo": k[0], "dataset_key": k[1], "metric": k[2], "k": k[3],
                "recall_at_k_A": a and a.get("recall_at_k"),
                "recall_at_k_B": b and b.get("recall_at_k"),
                "delta_recall": (float(b["recall_at_k"]) - float(a["recall_at_k"])) if a and b else None,
                "latency_ms_avg_A": a and a.get("latency_ms_avg"),
                "latency_ms_avg_B": b and b.get("latency_ms_avg"),
                "ratio_avg_latency_B_over_A": (float(b["latency_ms_avg"]) / float(a["latency_ms_avg"])) if a and b else None,
                "latency_ms_p95_A": a and a.get("latency_ms_p95"),
                "latency_ms_p95_B": b and b.get("latency_ms_p95"),
                "ratio_p95_latency_B_over_A": (float(b["latency_ms_p95"]) / float(a["latency_ms_p95"])) if a and b else None,
            }
            w.writerow(row)

    print(f"Wrote diff: {out_csv}")


if __name__ == "__main__":
    main()


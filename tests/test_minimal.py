import os
import json
import glob
import numpy as np
import shutil
import time
import datasets.ground_truth as gt
from datasets.make_toy import load_or_make
import bench


def test_ground_truth_cache(tmp_path):
    # small dataset
    xb, xq, meta = load_or_make(N=1000, D=16, nq=50, dist="gaussian", seed=123, data_dir=str(tmp_path / "data"))
    key = meta["key"]

    # first call — creates cache
    I1, D1 = gt.load_or_generate(key, xb, xq, k=5, metric="l2", root=str(tmp_path / "data" / "ground_truth"))
    assert I1.shape == (xq.shape[0], 5)
    path = os.path.join(str(tmp_path / "data" / "ground_truth"), f"{key}_l2_k5.npz")
    assert os.path.exists(path)

    # second call — loads from cache, should match
    I2, D2 = gt.load_or_generate(key, xb, xq, k=5, metric="l2", root=str(tmp_path / "data" / "ground_truth"))
    assert np.array_equal(I1, I2)
    assert np.allclose(D1, D2)


def test_run_benchmark_minimal(tmp_path):
    # copy minimal config into tmp
    cfg_src = os.path.join("configs", "minimal.yaml")
    cfg_dst = tmp_path / "minimal.yaml"
    shutil.copy(cfg_src, cfg_dst)

    # prepare results folder in tmp
    outdir = tmp_path / "results"
    os.makedirs(outdir, exist_ok=True)

    # rewrite config so outdir and data_dir are inside tmp
    with open(cfg_dst, "r") as f:
        cfg = f.read()
    cfg = cfg.replace("outdir: results", f"outdir: {outdir}")
    cfg = cfg.replace("data_dir: data", f"data_dir: {tmp_path / 'data'}")
    with open(cfg_dst, "w") as f:
        f.write(cfg)

    # run benchmark
    bench.run_benchmark(str(cfg_dst))

    # check that a new run-* directory was created
    runs = sorted(glob.glob(os.path.join(outdir, "run-*")))
    assert runs, "run-* directory not created"
    run_dir = runs[-1]

    # aggregated files
    agg_json = os.path.join(run_dir, "aggregated_results.json")
    agg_csv = os.path.join(run_dir, "aggregated_results.csv")
    assert os.path.exists(agg_json)
    assert os.path.exists(agg_csv)

    with open(agg_json, "r") as f:
        rows = json.load(f)
    assert isinstance(rows, list) and len(rows) > 0

    # basic content check of the first row
    r0 = rows[0]
    assert "algo" in r0 and "k" in r0 and "latency_ms_avg" in r0 and "recall_at_k" in r0


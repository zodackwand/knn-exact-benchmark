## knnbench — Minimal KNN benchmarking environment

### Quickstart

1) Create a virtualenv and install deps
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Run a minimal config
```bash
python bench.py --config configs/minimal.yaml
```

This generates a new `results/run-*/` with:
- Per-(algo,dataset,k) artifacts under `<algo>__<dataset>/k<k>/`:
  - `latencies_ms.csv`, `latency.png`, `summary.json`
- Aggregated summaries: `aggregated_results.csv` and `aggregated_results.json`

3) View coarse progress in the terminal. You will see lines like:
```
[+] bruteforce_numpy @ toy_gaussian_... metric=l2 k=10
[~] progress 50/200 for bruteforce_numpy @ ... k=10
```

### Datasets

Toy datasets are addressed by a key and cached on disk:
```
toy_{dist}_N{N}_D{D}_nq{nq}_seed{seed}
```
If the key is missing, data are generated and saved under `data/` automatically.

Configure datasets via YAML:
```yaml
datasets:
  - toy_gaussian_N10000_D64_nq200_seed42
```

### Algorithm interface (stable)

Algorithms live in `algorithms/<name>.py` and expose a class `Algo` implementing:

```python
class Algo:
    def __init__(self, metric: str = "l2", **params): ...
    def build(self, xb: np.ndarray, metric: str | None = None) -> None: ...
    def query(self, xq: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...
    def stats(self) -> dict: ...  # optional extra metrics
```

See `algorithms/_template.py` for a scaffold and `algorithms/bruteforce_numpy.py` for a concrete example.

### Comparing runs

Use the utility to diff two runs by `(algo, dataset_key, metric, k)`:
```bash
python tools/compare_runs.py --a results/run-AAA --b results/run-BBB --out results/compare-AAA-vs-BBB
```
It writes a CSV diff and optional latency ratio plots.

### Metric alignment (important)

- Ground-truth is computed using the configured metric (e.g., `l2`, `ip`, `cos`).
- Your algorithm must use the same metric for Recall@k to be meaningful.
- The ground-truth cache key includes the metric, so different metrics produce distinct GT files.
- If your algorithm chooses the metric internally, record it in `stats()['used_metric']` for reproducibility/debugging. Note: the benchmark currently always computes ground-truth using the metric from the YAML config.

### Metrics collected

Framework-provided (automatic):
- Recall@k (against ground-truth computed with the active metric from config)
- Query latency: `latency_ms_avg`, `latency_ms_p95` (per‑query latencies saved to `latencies_ms.csv`)
- Build time: `build_time_s` (measured or taken from `stats()` if provided)
- Process memory RSS: `ram_rss_mb_before_build`, `ram_rss_mb_after_build`

Provided by algorithms via `Algo.stats()` (optional but recommended):
- Index-only memory footprint: `index_size_bytes` (size of the index structure itself)
- Traversal/graph stats: `visited_nodes_avg`, `visited_nodes_p95`, `visited_nodes_rel_avg`
- Graph shape: `edges` (total), `avg_out_degree`
- Effective metric used (if the algo owns metric selection): `used_metric`

Notes:
- If a stat key is not returned by the algorithm, it will appear as `null` in results.
- See `algorithms/_template.py` for a minimal example of reporting `stats()`.

### Config example

```yaml
outdir: results
metric: l2
warmup: 10
data_dir: data

algorithms:
  - name: bruteforce_numpy

datasets:
  - toy_gaussian_N10000_D64_nq200_seed42

k_values: [1, 10]
```

### Sanity-check an adapter (smoke test)

Before running a full benchmark, you can quickly validate that an adapter implements the expected interface and produces sensible results:

```bash
python tools/smoke_algo.py \
  --algo bruteforce_numpy \
  --dataset-key toy_gaussian_N10000_D64_nq200_seed42 \
  --metric l2 \
  --k 10
```
This will:
- build the index, run queries on the dataset key (auto-generated if missing),
- compute Recall@k vs cached ground truth,
- print a short summary and exit non-zero if you provide a failing `--min-recall` threshold.

### Contributing

Read `CONTRIBUTING.md` for adding a new algorithm or adapter in a few minutes.

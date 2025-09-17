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

### Available algorithms

- `bruteforce_numpy` — Exact brute-force search using numpy
- `faiss_flat` — Exact brute-force search using FAISS library (supports L2 and inner product metrics)
- `faiss_hnsw` — Approximate nearest neighbors using FAISS HNSW (supports L2 and inner product metrics)
- `faiss_ivf` — Fast approximate nearest neighbors using FAISS IVF (supports L2 and inner product metrics)
- `sklearn_knn` — Exact nearest neighbors using scikit-learn (supports L2 and cosine metrics)
- `annoy_algo` — Approximate nearest neighbors using Spotify's Annoy library (supports L2 and cosine metrics)

### Algorithm parameters

Some algorithms support configuration parameters. While the current framework doesn't pass YAML parameters, you can modify algorithm defaults directly in the algorithm files:

**Annoy parameters** (in `algorithms/annoy_algo.py`):
- `n_trees` (default: 50) — Number of trees built during indexing. More trees = better recall but slower build time and larger memory usage
- `search_k` (default: -1) — Search effort during queries. Higher values = better recall but slower queries. When -1, uses `n_trees * k` as default

**FAISS HNSW parameters** (in `algorithms/faiss_hnsw.py`):
- `M` (default: 16) — Number of bidirectional links for each element during construction. Higher values = better recall but larger memory usage and slower build
- `efConstruction` (default: 200) — Size of dynamic candidate list during index construction. Higher values = better quality index but slower build time
- `efSearch` (default: 64) — Size of dynamic candidate list during search. Higher values = better recall but slower queries

**FAISS IVF parameters** (in `algorithms/faiss_ivf.py`):
- `nlist` (default: 100) — Number of cluster centroids for partitioning. Higher values = better recall but slower build time and more memory
- `nprobe` (default: 10) — Number of clusters to search during queries. Higher values = better recall but slower queries (max: nlist)

**sklearn KNN parameters** (in `algorithms/sklearn_knn.py`):
- `algorithm` (default: "brute") — Algorithm choice: "auto", "ball_tree", "kd_tree", "brute"
- `leaf_size` (default: 30) — Leaf size for tree algorithms (ball_tree, kd_tree). Smaller values = more memory but potentially faster queries

Example modification:
```python
# In algorithms/annoy_algo.py, line ~12-13:
self.n_trees = params.get("n_trees", 100)  # Increase for better recall
self.search_k = params.get("search_k", 2000)  # Increase for better recall

# In algorithms/faiss_hnsw.py, line ~12-14:
self.M = params.get("M", 32)  # More connections for better recall
self.efConstruction = params.get("efConstruction", 400)  # Better build quality
self.efSearch = params.get("efSearch", 128)  # More thorough search

# In algorithms/faiss_ivf.py, line ~12-13:
self.nlist = params.get("nlist", 50)  # Fewer clusters for smaller datasets
self.nprobe = params.get("nprobe", 20)  # Search more clusters for better recall

# In algorithms/sklearn_knn.py, line ~12-13:
self.algorithm = params.get("algorithm", "auto")  # Let sklearn choose automatically
self.leaf_size = params.get("leaf_size", 20)  # Smaller leaf size for tree algorithms
```

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

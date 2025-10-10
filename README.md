## knnbench — Minimal KNN benchmarking environment

### Quickstart

1) Create a virtualenv and install deps
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Run a benchmark (choose one method):

**Option A: Command-line (single algorithm, quick start)**
```bash
python bench.py --algo bruteforce_numpy --n 10000 --dim 64 --nq 200 --k 10
```

**Option B: Config file (multiple algorithms, more control)**
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

knnbench supports both toy datasets and real-world benchmark datasets.

#### Toy Datasets
Toy datasets are addressed by a key and cached on disk:
```
toy_{dist}_N{N}_D{D}_nq{nq}_seed{seed}
```
If the key is missing, data are generated and saved under `data/` automatically.

#### Real Datasets
Real datasets are automatically downloaded and cached. Currently supported:

**SIFT Family** (128D SIFT features, L2 metric):
- `sift1m` — 1M base vectors, 10K queries (full SIFT1M dataset)
- `sift100k` — 100K base vectors, 10K queries (subset of SIFT1M)
- `sift10k` — 10K base vectors, 10K queries (subset of SIFT1M)

Real datasets are downloaded from their original sources (e.g., INRIA corpus-texmex for SIFT) and cached locally. The system handles format conversion and subset extraction automatically.

Configure datasets via YAML:
```yaml
datasets:
  - toy_gaussian_N10000_D64_nq200_seed42  # Toy dataset
  - sift10k                               # Real dataset
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

Algorithms support configuration parameters directly through YAML config files. Parameters are passed to the algorithm constructor automatically.

**Available parameters by algorithm:**

**Annoy** (`annoy_algo`):
- `n_trees` (default: 50) — Number of trees built during indexing. More trees = better recall but slower build time and larger memory usage
- `search_k` (default: -1) — Search effort during queries. Higher values = better recall but slower queries. When -1, uses `n_trees * k` as default

**FAISS HNSW** (`faiss_hnsw`):
- `M` (default: 16) — Number of bidirectional links for each element during construction. Higher values = better recall but larger memory usage and slower build
- `efConstruction` (default: 200) — Size of dynamic candidate list during index construction. Higher values = better quality index but slower build time
- `efSearch` (default: 64) — Size of dynamic candidate list during search. Higher values = better recall but slower queries

**FAISS IVF** (`faiss_ivf`):
- `nlist` (default: 100) — Number of cluster centroids for partitioning. Higher values = better recall but slower build time and more memory
- `nprobe` (default: 10) — Number of clusters to search during queries. Higher values = better recall but slower queries (max: nlist)

**sklearn KNN** (`sklearn_knn`):
- `algorithm` (default: "brute") — Algorithm choice: "auto", "ball_tree", "kd_tree", "brute"
- `leaf_size` (default: 30) — Leaf size for tree algorithms (ball_tree, kd_tree). Smaller values = more memory but potentially faster queries

**FAISS Flat** (`faiss_flat`): No configurable parameters (exact brute-force)
**Brute-force NumPy** (`bruteforce_numpy`): No configurable parameters (exact brute-force)

**Example configuration:**
```yaml
algorithms:
  - bruteforce_numpy                    # No parameters
  - faiss_flat                          # No parameters
  - annoy_algo:
      n_trees: 100                      # More trees for better recall
      search_k: 2000                    # Higher search effort
  - faiss_hnsw:
      M: 32                             # More connections
      efConstruction: 400               # Better build quality
      efSearch: 128                     # More thorough search
  - faiss_ivf:
      nlist: 50                         # Fewer clusters for smaller datasets
      nprobe: 20                        # Search more clusters
  - sklearn_knn:
      algorithm: auto                   # Let sklearn choose automatically
      leaf_size: 20                     # Smaller leaf size
```

### Metadata Filtering

knnbench supports metadata-based filtering to simulate selective vector search scenarios. This allows benchmarking performance when searching only a subset of vectors based on metadata criteria.

#### How it works
- Each vector gets assigned a random integer metadata value (0-100) using a deterministic seed based on the dataset
- Metadata is automatically generated via `utils/metadata_generator.py` and cached for reproducibility
- Filtering narrows the search space by only considering vectors with metadata in a specified range
- The `metadata_range` defines which vectors are eligible for search (e.g., `[0, 25]` means only vectors with metadata 0-25)

#### Configuration
Enable filtering in your YAML config:

```yaml
filtering:
  enabled: true
  metadata_range: [0, 25]    # Only search vectors with metadata 0-25 (≈25% selectivity)
```

When filtering is disabled, all vectors are searched (equivalent to `metadata_range: [0, 100]`).

#### Selectivity percentages
The `metadata_range` roughly corresponds to search selectivity:
- `[0, 100]` — 100% selectivity (no filtering, search all vectors)
- `[0, 50]` — ~50% selectivity (search roughly half the vectors)
- `[0, 25]` — ~25% selectivity (search roughly quarter of the vectors)
- `[0, 10]` — ~10% selectivity (search roughly 10% of the vectors)

#### Results tracking
When filtering is enabled, additional metrics are recorded:
- `filtering_enabled`: Whether filtering was active
- `filter_range`: The metadata range used
- `filter_selectivity_percent`: Calculated selectivity percentage
- `eligible_vectors`: Number of vectors that matched the filter

This enables analysis of how search performance scales with selectivity levels.

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

### Config examples

#### Basic configuration (toy dataset)
```yaml
outdir: results
metric: l2
warmup: 10
data_dir: data

algorithms:
  - bruteforce_numpy

datasets:
  - toy_gaussian_N10000_D64_nq200_seed42

k_values: [1, 10]
```

#### Real dataset with algorithm parameters
```yaml
outdir: results
metric: l2
warmup: 10
data_dir: data

algorithms:
  - faiss_flat                         # Exact baseline
  - faiss_hnsw:                        # Approximate with custom params
      M: 32
      efConstruction: 400
      efSearch: 128
  - annoy_algo:                        # Approximate with custom params
      n_trees: 100
      search_k: 2000

datasets:
  - sift10k                            # Real dataset

k_values: [1, 10, 100]
```

#### Filtering configuration
```yaml
outdir: results
metric: l2
warmup: 10
data_dir: data

algorithms:
  - faiss_flat
  - faiss_ivf:
      nlist: 100
      nprobe: 10

datasets:
  - sift100k

filtering:                             # Enable metadata filtering
  enabled: true
  metadata_range: [0, 25]              # Search ~25% of vectors

k_values: [10]
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

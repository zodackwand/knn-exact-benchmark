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

4) Optional web UI (simple log/progress)
```bash
python -m webui.server
# open http://127.0.0.1:5000
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

### Contributing

Read `CONTRIBUTING.md` for adding a new algorithm or adapter in a few minutes.


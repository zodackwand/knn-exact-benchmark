## Contributing: Adding or adapting algorithms

This project standardizes on a tiny adapter interface so any KNN implementation can plug in quickly.

### The `Algo` interface

Create `algorithms/<your_algo>.py` exposing a class named `Algo`:

```python
class Algo:
    def __init__(self, metric: str = "l2", **params):
        # store params; metric in {"l2","ip","cos"} where supported
        ...

    def build(self, xb: np.ndarray, metric: str | None = None) -> None:
        # prepare your index from xb (float32 array). Metric override is optional.
        # record timing and memory in self._stats if you can.
        ...

    def query(self, xq: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        # return (I, D) with shapes [nq, k], types int64 and float32
        ...

    def stats(self) -> dict:
        # optional extra metrics, e.g. {"visited_nodes": avg, "edges": total, ...}
        return {...}
```

See `algorithms/_template.py` and `algorithms/bruteforce_numpy.py` for examples.

### Visited nodes (optional but recommended)

If your algorithm traverses a structure, track nodes visited per query and expose aggregated values in `stats()` or return per-query counts via an attribute the bench can read.

Recommended stats keys:
- `visited_nodes_avg`, `visited_nodes_p95`
- `visited_nodes_rel_avg` = `visited_nodes_avg / N`
- `edges` (total), `avg_out_degree`
- `index_size_bytes`, `ram_rss_mb_after_build`

### Adding to a config

In YAML, reference your module by filename (without `.py`):
```yaml
algorithms:
  - name: your_algo_name
```

### Datasets

You can use built-in toy datasets via key strings, or extend loaders to support file-backed datasets. For toy:
```yaml
datasets:
  - toy_gaussian_N10000_D64_nq200_seed42
```

### Metric alignment

- The benchmark computes ground-truth using a metric (e.g., `l2`, `ip`, `cos`).
- Ensure your algorithm uses the same metric; otherwise Recall@k is not meaningful.
- If your algorithm decides the metric internally, expose it via `stats()` (e.g., `{ "used_metric": "cos" }`) so the bench can compute ground-truth with the correct metric and record it in results.

### Testing your adapter

Run a small config and ensure artifacts are generated:
```bash
python bench.py --config configs/minimal.yaml
```

### Style

- Prefer clear, readable code and explicit variable names.
- Avoid global state; keep all state in the `Algo` instance.
- Keep `stats()` stable and summarized; don’t return huge arrays.


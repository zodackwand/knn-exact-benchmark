# KNN Benchmark Suite: Development Plan

This document outlines the development plan for the KNN algorithm testing framework. It's structured to be easily understood by LLM agents for implementation.

## 1. Unified Algorithm Interface

All KNN algorithms must implement this standard interface:

```python
class KNNAlgorithm:
    def __init__(self, **params):
        """
        Initialize with configurable parameters

        Args:
            **params: Algorithm parameters
        """
        self.params = params
        self.index = None
        self.is_built = False

    def build(self, data):
        """
        Build the search index

        Args:
            data: Numpy array of shape (n, d) with n vectors of dimension d

        Returns:
            float: Build time in seconds
        """
        pass

    def query(self, queries, k):
        """
        Search for k nearest neighbors for each query

        Args:
            queries: Numpy array of shape (nq, d) with nq queries of dimension d
            k: Number of nearest neighbors to find

        Returns:
            Tuple[ndarray, ndarray]: (indices, distances)
                - indices: indexes of k nearest neighbors of shape (nq, k)
                - distances: distances to k nearest neighbors of shape (nq, k)
        """
        pass
```

## 2. Algorithm Template

Create `algorithms/template.py` with base class implementation and example:

```python
class KNNAlgorithm:
    """Base class for all KNN algorithms"""
    # Interface implementation as above

class ExampleKNN(KNNAlgorithm):
    """Example algorithm implementation"""
    def __init__(self, metric="l2", **params):
        super().__init__(**params)
        self.metric = metric

    def build(self, data):
        # Example build implementation
        pass

    def query(self, queries, k):
        # Example query implementation
        pass
```

## 3. Adapt Existing Algorithms

Refactor `bruteforce_numpy.py` to follow the unified interface.

## 4. Adapters for Allen's Algorithms

Create adapters for:
- Ball Pruning
- Exact efSearch

Each adapter should implement the `KNNAlgorithm` interface while using the original code from `allens_code/functions/`.

## 5. Algorithm Organization

Directory structure for `algorithms/`:
```
algorithms/
  __init__.py
  template.py
  bruteforce_numpy.py
  faiss_exact.py
  ball_prune_adapter.py
  exact_efsearch_adapter.py
```

## 6. Ground Truth System

For ground truth computation:
- Use bruteforce algorithm as the reference
- Cache results in the `data/ground_truth/` folder
- Generate automatically if missing

```python
def generate_ground_truth(dataset_name, xq, xb, k=100):
    """Generates and saves ground truth for a dataset"""
    # Implementation
```

## 7. YAML Configuration

Create a configuration system using YAML files:

```yaml
algorithms:
  - name: bruteforce_numpy
    params: {}
  - name: ball_prune_adapter
    params:
      leaf_size: 40
  - name: exact_efsearch_adapter
    params:
      ef: 200

datasets:
  - toy_gaussian_N100000_D128_nq1000_seed42

metrics:
  - latency
  - recall@1
  - recall@10
  - recall@100
  - memory
  - build_time

k_values: [1, 10, 100]
```

## 8. Extended Metrics

Add calculation of the following metrics:
- Recall@k (comparison with ground truth)
- Memory consumption (before and after index building)
- Index build time (separate from query time)

Save results in the following formats:
- CSV for detailed results
- JSON for aggregated results
- PNG for visualizations

## Component Integration

Update `bench.py` to support:
- Loading configuration
- Running specified algorithms on selected datasets
- Calculating all metrics
- Saving results in a structured format

```python
def run_benchmark(config_path):
    """Runs benchmark according to configuration"""
    # Implementation
```

This plan provides a complete structure for creating a professional KNN algorithm testing environment that is easily extensible and allows for rapid testing of new algorithms.
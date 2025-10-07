# utils/dataset_loader.py
import numpy as np
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

from .make_toy import load_by_key as load_toy_by_key, parse_key as parse_toy_key
from .dataset_registry import get_dataset_info, DatasetInfo, is_real_dataset, suggest_similar_datasets
from .dataset_downloader import get_downloader
from .format_loaders import load_vectors
from .metadata_generator import load_or_generate_metadata


class DatasetError(Exception):
    """Exception raised for dataset loading errors."""
    pass


def load_real_dataset(dataset_key: str, data_dir: str = "data", include_metadata: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load a real dataset by key.

    Args:
        dataset_key: Dataset identifier (e.g., 'sift1m', 'sift100k')
        data_dir: Directory for caching downloaded files
        include_metadata: Whether to include vector metadata for filtering

    Returns:
        Tuple of (xb, xq, metadata) where:
        - xb: base vectors (database)
        - xq: query vectors
        - metadata: dataset metadata dict (includes 'vector_metadata' if requested)
    """
    # Get dataset info from registry
    dataset_info = get_dataset_info(dataset_key)
    if dataset_info is None:
        suggestions = suggest_similar_datasets(dataset_key)
        suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise DatasetError(f"Unknown dataset: {dataset_key}.{suggestion_text}")

    # Download/get cached dataset files
    downloader = get_downloader(data_dir)
    file_paths = downloader.get_dataset_files(dataset_info)

    # Load base vectors
    if "base" not in file_paths:
        raise DatasetError(f"Dataset {dataset_key} missing base vectors")
    base_path = file_paths["base"]
    xb = load_vectors(str(base_path), format=dataset_info.files["base"].format)

    # Load query vectors
    if "query" not in file_paths:
        raise DatasetError(f"Dataset {dataset_key} missing query vectors")
    query_path = file_paths["query"]
    xq = load_vectors(str(query_path), format=dataset_info.files["query"].format)

    # Handle subsets (e.g., sift100k is subset of sift1m)
    if dataset_info.subset_info and "base_slice" in dataset_info.subset_info:
        base_slice = dataset_info.subset_info["base_slice"]
        xb = xb[base_slice]

    # Load ground truth if available (for validation/debugging)
    groundtruth = None
    if "groundtruth" in file_paths:
        try:
            gt_path = file_paths["groundtruth"]
            groundtruth = load_vectors(str(gt_path), format=dataset_info.files["groundtruth"].format)
            # Convert to int64 for indices
            groundtruth = groundtruth.astype(np.int64)
        except Exception as e:
            print(f"[warning] Failed to load ground truth for {dataset_key}: {e}")

    # Create metadata
    metadata = {
        "key": dataset_key,
        "name": dataset_info.name,
        "description": dataset_info.description,
        "n_base": xb.shape[0],
        "n_query": xq.shape[0],
        "dimension": xb.shape[1],
        "metric": dataset_info.metric,
        "has_groundtruth": groundtruth is not None,
        "dataset_type": "real",
        "parent_dataset": dataset_info.parent_dataset,
        "subset_info": dataset_info.subset_info,
    }

    if groundtruth is not None:
        metadata["groundtruth_shape"] = groundtruth.shape

    # Add vector metadata if requested
    if include_metadata:
        vector_metadata = load_or_generate_metadata(dataset_key, xb.shape[0], data_dir)
        metadata["vector_metadata"] = vector_metadata

    # Validate dimensions
    if xb.shape[1] != xq.shape[1]:
        raise DatasetError(f"Dimension mismatch: base={xb.shape[1]}, query={xq.shape[1]}")

    if dataset_info.dimension and xb.shape[1] != dataset_info.dimension:
        raise DatasetError(f"Expected dimension {dataset_info.dimension}, got {xb.shape[1]}")

    print(f"[dataset] Loaded {dataset_key}: base={xb.shape}, query={xq.shape}, metric={dataset_info.metric}")

    return xb, xq, metadata


def load_dataset(dataset_key: str, data_dir: str = "data", include_metadata: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load any dataset (toy or real) by key.

    This is the main entry point for dataset loading. It automatically
    routes to toy or real dataset loaders based on the key format.

    Args:
        dataset_key: Dataset identifier
                    - Toy: 'toy_gaussian_N10000_D64_nq200_seed42'
                    - Real: 'sift1m', 'sift100k', etc.
        data_dir: Directory for data storage/caching
        include_metadata: Whether to include vector metadata for filtering (real datasets only)

    Returns:
        Tuple of (xb, xq, metadata) where:
        - xb: base vectors (database) as float32 array
        - xq: query vectors as float32 array
        - metadata: dataset metadata dict (includes 'vector_metadata' if requested)

    Raises:
        DatasetError: If dataset cannot be loaded
    """
    try:
        if is_real_dataset(dataset_key):
            # Load real dataset
            return load_real_dataset(dataset_key, data_dir, include_metadata)
        elif dataset_key.startswith("toy_"):
            # Load toy dataset (existing functionality - no metadata support for now)
            xb, xq, metadata = load_toy_by_key(dataset_key, data_dir)
            # Note: Toy datasets don't support vector metadata yet
            if include_metadata:
                print(f"[warning] Vector metadata not supported for toy datasets (dataset: {dataset_key})")
            return xb, xq, metadata
        else:
            # Unknown format
            suggestions = suggest_similar_datasets(dataset_key)
            if suggestions:
                suggestion_text = f" Did you mean: {', '.join(suggestions)}?"
            else:
                suggestion_text = " Use format 'toy_gaussian_N1000_D64_nq100_seed42' for toy datasets."
            raise DatasetError(f"Unknown dataset key format: {dataset_key}.{suggestion_text}")

    except Exception as e:
        if isinstance(e, DatasetError):
            raise
        else:
            raise DatasetError(f"Failed to load dataset {dataset_key}: {e}")


def list_available_datasets() -> Dict[str, list]:
    """List all available datasets by category.

    Returns:
        Dict with keys 'real' and 'toy_example' containing dataset lists
    """
    from .dataset_registry import list_datasets

    result = {
        "real": list_datasets(),
        "toy_example": ["toy_gaussian_N10000_D64_nq200_seed42", "toy_uniform_N5000_D128_nq100_seed123"]
    }

    return result


def get_dataset_info_summary(dataset_key: str) -> Optional[Dict[str, Any]]:
    """Get summary information about a dataset without loading it.

    Args:
        dataset_key: Dataset identifier

    Returns:
        Dict with dataset summary or None if not found
    """
    if is_real_dataset(dataset_key):
        dataset_info = get_dataset_info(dataset_key)
        if dataset_info:
            return {
                "key": dataset_info.key,
                "name": dataset_info.name,
                "description": dataset_info.description,
                "dimension": dataset_info.dimension,
                "metric": dataset_info.metric,
                "n_base": dataset_info.n_base,
                "n_query": dataset_info.n_query,
                "has_groundtruth": dataset_info.has_groundtruth,
                "dataset_type": "real",
                "parent_dataset": dataset_info.parent_dataset,
            }
    elif dataset_key.startswith("toy_"):
        try:
            toy_params = parse_toy_key(dataset_key)
            return {
                "key": dataset_key,
                "name": f"Toy {toy_params['dist'].title()} Dataset",
                "description": f"Synthetic {toy_params['dist']} vectors",
                "dimension": toy_params["D"],
                "metric": "l2",  # Default for toy datasets
                "n_base": toy_params["N"],
                "n_query": toy_params["nq"],
                "has_groundtruth": False,  # Computed on demand
                "dataset_type": "toy",
                "seed": toy_params["seed"],
            }
        except ValueError:
            pass

    return None


def validate_dataset_for_benchmark(dataset_key: str, metric: str, k_values: list) -> None:
    """Validate that a dataset is suitable for benchmarking.

    Args:
        dataset_key: Dataset identifier
        metric: Distance metric to be used
        k_values: List of k values for k-NN

    Raises:
        DatasetError: If dataset is not suitable
    """
    info = get_dataset_info_summary(dataset_key)
    if info is None:
        raise DatasetError(f"Cannot get info for dataset: {dataset_key}")

    # Check metric compatibility
    if info["dataset_type"] == "real":
        dataset_metric = info.get("metric", "l2")
        if metric != dataset_metric:
            print(f"[warning] Metric mismatch: dataset expects {dataset_metric}, using {metric}")
            print(f"[warning] Recall@k calculations may not be meaningful")

    # Check k values vs dataset size
    max_k = max(k_values) if k_values else 0
    n_base = info.get("n_base")
    if n_base and max_k >= n_base:
        raise DatasetError(f"max(k)={max_k} >= dataset size {n_base}. Use smaller k values.")

    print(f"[validation] Dataset {dataset_key} is suitable for benchmarking")


# Backward compatibility: alias to maintain existing interface
load_by_key = load_dataset
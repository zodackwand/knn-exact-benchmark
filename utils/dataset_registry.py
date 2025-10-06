# utils/dataset_registry.py
from typing import Dict, List, Optional, Any, NamedTuple
from dataclasses import dataclass
import os


@dataclass
class DatasetFile:
    """Represents a single file in a dataset."""
    name: str                    # File identifier (e.g., 'base', 'query', 'groundtruth')
    url: str                     # Download URL
    filename: str                # Local filename
    format: str                  # File format ('fvecs', 'ivecs', 'hdf5', etc.)
    checksum: Optional[str] = None  # SHA256 checksum for verification
    size_bytes: Optional[int] = None  # File size in bytes


@dataclass
class DatasetInfo:
    """Complete dataset metadata."""
    key: str                     # Unique dataset identifier (e.g., 'sift1m')
    name: str                    # Human-readable name
    description: str             # Dataset description
    dimension: int               # Vector dimension
    metric: str                  # Distance metric ('l2', 'ip', 'cos')
    files: Dict[str, DatasetFile]  # File components
    n_base: Optional[int] = None    # Number of base vectors
    n_query: Optional[int] = None   # Number of query vectors
    has_groundtruth: bool = True    # Whether groundtruth is provided
    parent_dataset: Optional[str] = None  # Parent dataset for subsets
    subset_info: Optional[Dict[str, Any]] = None  # Subset-specific info


class DatasetRegistry:
    """Registry of available datasets."""

    def __init__(self):
        self._datasets: Dict[str, DatasetInfo] = {}
        self._register_builtin_datasets()

    def register(self, dataset: DatasetInfo) -> None:
        """Register a new dataset."""
        self._datasets[dataset.key] = dataset

    def get(self, key: str) -> Optional[DatasetInfo]:
        """Get dataset info by key."""
        return self._datasets.get(key)

    def list_datasets(self) -> List[str]:
        """Get list of all registered dataset keys."""
        return list(self._datasets.keys())

    def list_by_family(self, family: str) -> List[str]:
        """Get datasets from a specific family (e.g., 'sift')."""
        return [key for key in self._datasets.keys() if key.startswith(family)]

    def is_registered(self, key: str) -> bool:
        """Check if a dataset is registered."""
        return key in self._datasets

    def _register_builtin_datasets(self):
        """Register built-in datasets."""
        self._register_sift_family()

    def _register_sift_family(self):
        """Register SIFT dataset family."""

        # Base SIFT URL (from INRIA corpus-texmex)
        sift_url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

        # SIFT1M (full dataset)
        sift1m = DatasetInfo(
            key="sift1m",
            name="SIFT1M",
            description="1M SIFT features (128D) for approximate nearest neighbor search evaluation",
            dimension=128,
            metric="l2",
            n_base=1000000,
            n_query=10000,
            files={
                "base": DatasetFile(
                    name="base",
                    url=sift_url,
                    filename="sift_base.fvecs",
                    format="fvecs"
                ),
                "query": DatasetFile(
                    name="query",
                    url=sift_url,
                    filename="sift_query.fvecs",
                    format="fvecs"
                ),
                "groundtruth": DatasetFile(
                    name="groundtruth",
                    url=sift_url,
                    filename="sift_groundtruth.ivecs",
                    format="ivecs"
                )
            }
        )
        self.register(sift1m)

        # SIFT100K (subset of SIFT1M)
        sift100k = DatasetInfo(
            key="sift100k",
            name="SIFT100K",
            description="100K SIFT features (subset of SIFT1M)",
            dimension=128,
            metric="l2",
            n_base=100000,
            n_query=10000,
            parent_dataset="sift1m",
            subset_info={"base_slice": slice(0, 100000)},
            files=sift1m.files.copy()  # Uses same files, but loads subset
        )
        self.register(sift100k)

        # SIFT10K (subset of SIFT1M)
        sift10k = DatasetInfo(
            key="sift10k",
            name="SIFT10K",
            description="10K SIFT features (subset of SIFT1M)",
            dimension=128,
            metric="l2",
            n_base=10000,
            n_query=10000,
            parent_dataset="sift1m",
            subset_info={"base_slice": slice(0, 10000)},
            files=sift1m.files.copy()  # Uses same files, but loads subset
        )
        self.register(sift10k)

        # Alternative SIFT datasets (for future extension)
        # Could add SIFT128, SIFT256, etc. if available


def get_registry() -> DatasetRegistry:
    """Get the global dataset registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = DatasetRegistry()
    return _global_registry


# Global registry instance
_global_registry: Optional[DatasetRegistry] = None


# Convenience functions
def get_dataset_info(key: str) -> Optional[DatasetInfo]:
    """Get dataset info by key."""
    return get_registry().get(key)


def list_datasets() -> List[str]:
    """List all available dataset keys."""
    return get_registry().list_datasets()


def is_real_dataset(key: str) -> bool:
    """Check if a key corresponds to a real (non-toy) dataset."""
    return get_registry().is_registered(key)


def register_dataset(dataset: DatasetInfo) -> None:
    """Register a new dataset in the global registry."""
    get_registry().register(dataset)


# Dataset key validation
def validate_dataset_key(key: str) -> bool:
    """Validate that a dataset key is either a toy dataset or registered real dataset."""
    # Check if it's a toy dataset key
    if key.startswith("toy_"):
        return True

    # Check if it's a registered real dataset
    return is_real_dataset(key)


def suggest_similar_datasets(key: str) -> List[str]:
    """Suggest similar dataset keys for typos/misspellings."""
    available = list_datasets()

    # Simple similarity: datasets with common prefixes
    suggestions = []
    key_lower = key.lower()

    for dataset_key in available:
        dataset_lower = dataset_key.lower()
        # Check for common prefixes or substrings
        if (key_lower in dataset_lower or
            dataset_lower in key_lower or
            any(part in dataset_lower for part in key_lower.split('_') if len(part) > 2)):
            suggestions.append(dataset_key)

    return suggestions[:5]  # Return top 5 suggestions
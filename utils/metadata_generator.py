"""
Vector metadata generation utilities for filtering experiments.

This module provides functionality to generate consistent, reproducible metadata
for dataset vectors to enable filtering performance analysis.
"""

import os
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional


def load_or_generate_metadata(
    dataset_key: str,
    n_vectors: int,
    data_dir: str = "data"
) -> np.ndarray:
    """
    Load existing metadata or generate new metadata for a dataset.

    Generates random integer metadata values (0-100) for each vector in the dataset.
    Uses a deterministic seed based on dataset_key and n_vectors to ensure
    reproducibility across runs.

    Args:
        dataset_key: Unique identifier for the dataset (e.g., 'sift10k')
        n_vectors: Number of vectors in the dataset
        data_dir: Directory to store/load metadata files

    Returns:
        np.ndarray: Array of shape (n_vectors,) with dtype int8, values 0-100

    Example:
        >>> metadata = load_or_generate_metadata('sift10k', 10000)
        >>> print(f"Shape: {metadata.shape}, Range: {metadata.min()}-{metadata.max()}")
        Shape: (10000,), Range: 0-100
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Construct metadata file path
    metadata_path = Path(data_dir) / f"{dataset_key}_vector_metadata.npy"

    # Try to load existing metadata
    if metadata_path.exists():
        try:
            metadata = np.load(metadata_path)

            # Validate shape matches expected
            if metadata.shape[0] != n_vectors:
                print(f"[metadata] Warning: existing metadata has {metadata.shape[0]} vectors, expected {n_vectors}")
                print(f"[metadata] Regenerating metadata for {dataset_key}")
            else:
                print(f"[metadata] Loaded existing metadata for {dataset_key}: {metadata.shape}")
                return metadata

        except Exception as e:
            print(f"[metadata] Failed to load existing metadata: {e}")
            print(f"[metadata] Regenerating metadata for {dataset_key}")

    # Generate new metadata
    print(f"[metadata] Generating new metadata for {dataset_key} ({n_vectors} vectors)")

    # Create deterministic seed from dataset info
    seed_string = f"{dataset_key}_{n_vectors}"
    seed = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)

    # Generate random metadata values 0-100
    rng = np.random.RandomState(seed)
    metadata = rng.randint(0, 101, size=n_vectors, dtype=np.int8)

    # Save for future use
    try:
        np.save(metadata_path, metadata)
        print(f"[metadata] Saved metadata to {metadata_path}")
    except Exception as e:
        print(f"[metadata] Warning: Failed to save metadata: {e}")

    return metadata


if __name__ == "__main__":
    # Test script
    print("Testing metadata generator...")

    # Test with sample dataset
    test_dataset = "test_dataset"
    test_vectors = 1000

    # Generate twice to test consistency
    meta1 = load_or_generate_metadata(test_dataset, test_vectors)
    meta2 = load_or_generate_metadata(test_dataset, test_vectors)

    print(f"\nResults:")
    print(f"Shape: {meta1.shape}")
    print(f"Dtype: {meta1.dtype}")
    print(f"Range: {meta1.min()}-{meta1.max()}")
    print(f"Consistent: {np.array_equal(meta1, meta2)}")
    print(f"Sample values: {meta1[:10]}")

    # Clean up test file
    test_file = f"data/{test_dataset}_vector_metadata.npy"
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"\nCleaned up test file: {test_file}")
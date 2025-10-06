#!/usr/bin/env python3
"""
Test script for SIFT dataset loading functionality.
This tests the new real dataset infrastructure without downloading large files.
"""

import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_registry import get_registry, list_datasets, get_dataset_info
from utils.dataset_loader import list_available_datasets, get_dataset_info_summary
from utils.dataset_loader import load_dataset, DatasetError


def test_registry():
    """Test dataset registry functionality."""
    print("=== Testing Dataset Registry ===")

    registry = get_registry()
    datasets = list_datasets()

    print(f"Registered datasets: {datasets}")

    # Test SIFT dataset info
    for dataset_key in ['sift10k', 'sift100k', 'sift1m']:
        info = get_dataset_info(dataset_key)
        if info:
            print(f"\n{dataset_key}:")
            print(f"  Name: {info.name}")
            print(f"  Description: {info.description}")
            print(f"  Dimension: {info.dimension}")
            print(f"  Metric: {info.metric}")
            print(f"  Base vectors: {info.n_base}")
            print(f"  Query vectors: {info.n_query}")
            if info.subset_info:
                print(f"  Subset info: {info.subset_info}")
        else:
            print(f"ERROR: Could not find info for {dataset_key}")

    print("\n✓ Registry test completed")


def test_dataset_info():
    """Test dataset info functionality."""
    print("\n=== Testing Dataset Info ===")

    available = list_available_datasets()
    print(f"Available dataset categories: {list(available.keys())}")
    print(f"Real datasets: {available['real']}")
    print(f"Toy examples: {available['toy_example']}")

    # Test dataset summaries
    for dataset_key in ['sift10k', 'toy_gaussian_N1000_D64_nq100_seed42']:
        summary = get_dataset_info_summary(dataset_key)
        if summary:
            print(f"\n{dataset_key} summary:")
            for k, v in summary.items():
                print(f"  {k}: {v}")
        else:
            print(f"ERROR: Could not get summary for {dataset_key}")

    print("\n✓ Dataset info test completed")


def test_toy_dataset_loading():
    """Test that toy dataset loading still works."""
    print("\n=== Testing Toy Dataset Loading ===")

    try:
        # Load a small toy dataset
        xb, xq, meta = load_dataset("toy_gaussian_N1000_D64_nq100_seed42")

        print(f"Loaded toy dataset:")
        print(f"  Base vectors: {xb.shape}")
        print(f"  Query vectors: {xq.shape}")
        print(f"  Metadata keys: {list(meta.keys())}")
        print(f"  Dataset type: {meta.get('dataset_type', 'unknown')}")

        assert xb.shape == (1000, 64), f"Unexpected base shape: {xb.shape}"
        assert xq.shape == (100, 64), f"Unexpected query shape: {xq.shape}"

        print("✓ Toy dataset loading works correctly")

    except Exception as e:
        print(f"ERROR: Toy dataset loading failed: {e}")
        return False

    return True


def test_real_dataset_validation():
    """Test real dataset validation without actually downloading."""
    print("\n=== Testing Real Dataset Validation ===")

    try:
        # This should not download, just validate the dataset key exists
        info = get_dataset_info_summary("sift10k")
        if info:
            print(f"SIFT10K validation:")
            print(f"  Registered: True")
            print(f"  Name: {info['name']}")
            print(f"  Expected shape: {info['n_base']} x {info['dimension']}")
            print("✓ SIFT10K is properly registered")
        else:
            print("ERROR: SIFT10K not found in registry")
            return False

        # Test invalid dataset key
        try:
            load_dataset("invalid_dataset_key")
            print("ERROR: Should have failed for invalid key")
            return False
        except DatasetError as e:
            print(f"✓ Correctly rejected invalid key: {e}")

    except Exception as e:
        print(f"ERROR: Real dataset validation failed: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("KNN Benchmark - SIFT Dataset Loading Test")
    print("==========================================")

    all_passed = True

    try:
        test_registry()
        test_dataset_info()
        all_passed &= test_toy_dataset_loading()
        all_passed &= test_real_dataset_validation()

        print(f"\n{'='*50}")
        if all_passed:
            print("✓ ALL TESTS PASSED")
            print("\nThe SIFT dataset infrastructure is ready!")
            print("You can now use datasets like 'sift10k', 'sift100k', 'sift1m' in your configs.")
            print("\nExample usage:")
            print("  python bench.py --config configs/sift_example.yaml")
        else:
            print("✗ SOME TESTS FAILED")
            return 1

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
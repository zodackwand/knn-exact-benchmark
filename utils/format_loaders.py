# utils/format_loaders.py
import os
import numpy as np
from typing import Tuple, Optional
import struct


def read_fvecs(file_path: str) -> np.ndarray:
    """Read .fvecs file format (float32 vectors).

    Format: Each vector is stored as [d, v1, v2, ..., vd] where d is dimension (int32)
    and vi are float32 components.

    Returns:
        np.ndarray: Shape (n_vectors, dimension) with dtype float32
    """
    with open(file_path, 'rb') as f:
        # Read first 4 bytes to get dimension
        d_bytes = f.read(4)
        if len(d_bytes) != 4:
            raise ValueError(f"Invalid fvecs file: {file_path}")

        d = struct.unpack('<i', d_bytes)[0]  # little-endian int32
        f.seek(0)  # Reset to beginning

        # Read entire file
        data = np.fromfile(f, dtype=np.int32)

        # Each vector is (d+1) int32 values: [dim, v1, v2, ..., vd]
        n_vectors = len(data) // (d + 1)
        if len(data) % (d + 1) != 0:
            raise ValueError(f"Invalid fvecs file size: {file_path}")

        # Reshape and extract vectors (skip dimension column)
        vectors = data.reshape(n_vectors, d + 1)[:, 1:].copy().view(np.float32)

        return vectors


def read_bvecs(file_path: str) -> np.ndarray:
    """Read .bvecs file format (uint8 vectors).

    Format: Each vector is stored as [d, v1, v2, ..., vd] where d is dimension (int32)
    and vi are uint8 components.

    Returns:
        np.ndarray: Shape (n_vectors, dimension) with dtype uint8
    """
    with open(file_path, 'rb') as f:
        # Read first 4 bytes to get dimension
        d_bytes = f.read(4)
        if len(d_bytes) != 4:
            raise ValueError(f"Invalid bvecs file: {file_path}")

        d = struct.unpack('<i', d_bytes)[0]  # little-endian int32
        f.seek(0)  # Reset to beginning

        vectors = []
        while True:
            # Read dimension
            d_bytes = f.read(4)
            if len(d_bytes) != 4:
                break
            dim = struct.unpack('<i', d_bytes)[0]
            if dim != d:
                raise ValueError(f"Inconsistent dimension in bvecs file: {file_path}")

            # Read vector components
            vec_bytes = f.read(d)
            if len(vec_bytes) != d:
                raise ValueError(f"Incomplete vector in bvecs file: {file_path}")

            vectors.append(np.frombuffer(vec_bytes, dtype=np.uint8))

        if not vectors:
            raise ValueError(f"No vectors found in bvecs file: {file_path}")

        return np.array(vectors, dtype=np.uint8)


def read_ivecs(file_path: str) -> np.ndarray:
    """Read .ivecs file format (int32 vectors).

    Format: Each vector is stored as [d, v1, v2, ..., vd] where d is dimension (int32)
    and vi are int32 components.

    Returns:
        np.ndarray: Shape (n_vectors, dimension) with dtype int32
    """
    with open(file_path, 'rb') as f:
        # Read first 4 bytes to get dimension
        d_bytes = f.read(4)
        if len(d_bytes) != 4:
            raise ValueError(f"Invalid ivecs file: {file_path}")

        d = struct.unpack('<i', d_bytes)[0]  # little-endian int32
        f.seek(0)  # Reset to beginning

        # Read entire file as int32
        data = np.fromfile(f, dtype=np.int32)

        # Each vector is (d+1) int32 values: [dim, v1, v2, ..., vd]
        n_vectors = len(data) // (d + 1)
        if len(data) % (d + 1) != 0:
            raise ValueError(f"Invalid ivecs file size: {file_path}")

        # Reshape and extract vectors (skip dimension column)
        vectors = data.reshape(n_vectors, d + 1)[:, 1:].copy()

        return vectors


def write_fvecs(vectors: np.ndarray, file_path: str) -> None:
    """Write vectors to .fvecs file format.

    Args:
        vectors: np.ndarray with shape (n_vectors, dimension) and dtype float32
        file_path: Output file path
    """
    vectors = vectors.astype(np.float32)
    n_vectors, d = vectors.shape

    with open(file_path, 'wb') as f:
        for i in range(n_vectors):
            # Write dimension as int32
            f.write(struct.pack('<i', d))
            # Write vector as float32
            vectors[i].tofile(f)


def write_ivecs(vectors: np.ndarray, file_path: str) -> None:
    """Write vectors to .ivecs file format.

    Args:
        vectors: np.ndarray with shape (n_vectors, dimension) and dtype int32
        file_path: Output file path
    """
    vectors = vectors.astype(np.int32)
    n_vectors, d = vectors.shape

    with open(file_path, 'wb') as f:
        for i in range(n_vectors):
            # Write dimension as int32
            f.write(struct.pack('<i', d))
            # Write vector as int32
            vectors[i].tofile(f)


def detect_format(file_path: str) -> Optional[str]:
    """Detect file format based on extension.

    Returns:
        str: Format name ('fvecs', 'bvecs', 'ivecs', 'hdf5', 'npy') or None
    """
    ext = os.path.splitext(file_path)[1].lower()

    format_map = {
        '.fvecs': 'fvecs',
        '.bvecs': 'bvecs',
        '.ivecs': 'ivecs',
        '.hdf5': 'hdf5',
        '.h5': 'hdf5',
        '.npy': 'npy',
    }

    return format_map.get(ext)


def load_vectors(file_path: str, format: Optional[str] = None) -> np.ndarray:
    """Load vectors from file with automatic format detection.

    Args:
        file_path: Path to vector file
        format: Format override ('fvecs', 'bvecs', 'ivecs', 'hdf5', 'npy')
                If None, auto-detect from extension

    Returns:
        np.ndarray: Loaded vectors
    """
    if format is None:
        format = detect_format(file_path)

    if format == 'fvecs':
        return read_fvecs(file_path)
    elif format == 'bvecs':
        return read_bvecs(file_path).astype(np.float32)  # Convert to float32 for consistency
    elif format == 'ivecs':
        return read_ivecs(file_path).astype(np.float32)  # Convert to float32 for consistency
    elif format == 'npy':
        return np.load(file_path)
    elif format == 'hdf5':
        try:
            import h5py
            with h5py.File(file_path, 'r') as f:
                # Try common dataset names
                for key in ['vectors', 'data', 'train', 'test']:
                    if key in f:
                        return f[key][:]
                # If no common key found, use first dataset
                keys = list(f.keys())
                if keys:
                    return f[keys[0]][:]
                else:
                    raise ValueError(f"No datasets found in HDF5 file: {file_path}")
        except ImportError:
            raise ImportError("h5py is required for HDF5 support. Install with: pip install h5py")
    else:
        raise ValueError(f"Unsupported format: {format}")


# Convenience functions for common file patterns
def load_sift_base(data_dir: str, dataset_name: str = "sift") -> np.ndarray:
    """Load SIFT base vectors (database vectors)."""
    base_file = os.path.join(data_dir, f"{dataset_name}_base.fvecs")
    return load_vectors(base_file)


def load_sift_query(data_dir: str, dataset_name: str = "sift") -> np.ndarray:
    """Load SIFT query vectors."""
    query_file = os.path.join(data_dir, f"{dataset_name}_query.fvecs")
    return load_vectors(query_file)


def load_sift_groundtruth(data_dir: str, dataset_name: str = "sift") -> np.ndarray:
    """Load SIFT ground truth (nearest neighbor indices)."""
    gt_file = os.path.join(data_dir, f"{dataset_name}_groundtruth.ivecs")
    return load_vectors(gt_file).astype(np.int64)  # Ensure int64 for indices
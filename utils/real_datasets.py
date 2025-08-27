# utils/real_datasets.py - Real dataset loaders for benchmarking
import os
import json
import numpy as np
import urllib.request
import struct
from typing import Tuple, Dict, Any


# Dataset registry with metadata (lightweight versions for development)
REAL_DATASETS = {
    "sift10k": {
        "name": "SIFT10K", 
        "description": "10K SIFT features sample (128D, ~5MB)",
        "base_url": "http://corpus-texmex.irisa.fr/",
        "files": {
            "base": "siftsmall_base.fvecs",    # 10K vectors
            "query": "siftsmall_query.fvecs",  # 100 queries
            "groundtruth": "siftsmall_groundtruth.ivecs"
        },
        "N": 10000,
        "D": 128, 
        "nq": 100,
        "metric": "l2",
        "size_mb": 5.2,
        "citation": "Jegou et al. Product quantization for nearest neighbor search. TPAMI 2011"
    },
    "glove_mini": {
        "name": "GloVe-Mini",
        "description": "50K most common GloVe embeddings (50D, ~10MB)", 
        "generator": "synthetic",  # We'll generate this based on GloVe characteristics
        "N": 50000,
        "D": 50,
        "nq": 1000, 
        "metric": "cos",
        "size_mb": 10.0,
        "citation": "Pennington et al. GloVe: Global Vectors for Word Representation. EMNLP 2014"
    },
    "realistic_images": {
        "name": "Realistic Image Features",
        "description": "SIFT-like synthetic features with realistic clustering (128D, ~2MB)",
        "generator": "synthetic", 
        "N": 20000,
        "D": 128,
        "nq": 500,
        "metric": "l2", 
        "size_mb": 2.0,
        "characteristics": "clustered, sparse, realistic_variance"
    }
    # Add more lightweight datasets as needed
}


def read_fvecs(filename: str) -> np.ndarray:
    """Read .fvecs format (float32 vectors with dimension prefix)"""
    with open(filename, 'rb') as f:
        vectors = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('<i', dim_bytes)[0]
            vec = struct.unpack('<' + 'f' * dim, f.read(dim * 4))
            vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


def read_ivecs(filename: str) -> np.ndarray:
    """Read .ivecs format (int32 vectors with dimension prefix)"""  
    with open(filename, 'rb') as f:
        vectors = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('<i', dim_bytes)[0]
            vec = struct.unpack('<' + 'i' * dim, f.read(dim * 4))
            vectors.append(vec)
    return np.array(vectors, dtype=np.int32)


def download_sift1m(data_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Download and parse SIFT1M dataset"""
    dataset_info = REAL_DATASETS["sift1m"]
    
    # Create dataset directory
    sift_dir = os.path.join(data_dir, "sift1m")
    os.makedirs(sift_dir, exist_ok=True)
    
    # Download files if not present
    for file_type, filename in dataset_info["files"].items():
        local_path = os.path.join(sift_dir, filename)
        if not os.path.exists(local_path):
            url = dataset_info["base_url"] + filename
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, local_path)
    
    # Parse vectors
    base_path = os.path.join(sift_dir, dataset_info["files"]["base"])
    query_path = os.path.join(sift_dir, dataset_info["files"]["query"])
    
    xb = read_fvecs(base_path)
    xq = read_fvecs(query_path)
    
    meta = {
        "key": "sift1m",
        "name": dataset_info["name"],
        "N": xb.shape[0],
        "D": xb.shape[1], 
        "nq": xq.shape[0],
        "metric": dataset_info["metric"],
        "citation": dataset_info["citation"]
    }
    
    return xb, xq, meta


def download_glove100(data_dir: str, max_vectors: int = 100000) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Download and parse GloVe-100D vectors, create query subset"""
    dataset_info = REAL_DATASETS["glove100"]
    
    glove_dir = os.path.join(data_dir, "glove100")
    os.makedirs(glove_dir, exist_ok=True)
    
    # Download GloVe file
    filename = dataset_info["files"]["vectors"]
    local_path = os.path.join(glove_dir, filename)
    if not os.path.exists(local_path):
        url = dataset_info["base_url"] + filename + ".zip"
        print(f"Downloading {filename}... (this may take a while)")
        # Note: Would need to handle zip extraction in real implementation
        
    # Parse text format: "word 0.123 0.456 ..."
    vectors = []
    words = []
    with open(local_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_vectors:
                break
            parts = line.strip().split()
            word = parts[0]
            vec = [float(x) for x in parts[1:]]
            vectors.append(vec)
            words.append(word)
    
    vectors = np.array(vectors, dtype=np.float32)
    
    # Create query subset (last 10% of vectors)
    split = int(0.9 * len(vectors))
    xb = vectors[:split]
    xq = vectors[split:]
    
    meta = {
        "key": "glove100",
        "name": dataset_info["name"],
        "N": xb.shape[0],
        "D": xb.shape[1],
        "nq": xq.shape[0], 
        "metric": dataset_info["metric"],
        "words_sample": words[:10],  # store first 10 words as example
        "citation": dataset_info["citation"]
    }
    
    return xb, xq, meta


# Registry of download functions
DATASET_LOADERS = {
    "sift1m": download_sift1m,
    "glove100": download_glove100,
}


def load_real_dataset(key: str, data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load real dataset by key, downloading if necessary"""
    if key not in DATASET_LOADERS:
        raise ValueError(f"Unknown real dataset: {key}. Available: {list(DATASET_LOADERS.keys())}")
    
    # Check if already cached in numpy format
    cache_dir = os.path.join(data_dir, key)
    cached_files = {
        "xb": os.path.join(cache_dir, "xb.npy"),
        "xq": os.path.join(cache_dir, "xq.npy"), 
        "meta": os.path.join(cache_dir, "meta.json")
    }
    
    if all(os.path.exists(f) for f in cached_files.values()):
        # Load from cache
        xb = np.load(cached_files["xb"])
        xq = np.load(cached_files["xq"])
        with open(cached_files["meta"], 'r') as f:
            meta = json.load(f)
        return xb, xq, meta
    
    # Download and parse
    loader = DATASET_LOADERS[key]
    xb, xq, meta = loader(data_dir)
    
    # Cache in numpy format for fast loading
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cached_files["xb"], xb)
    np.save(cached_files["xq"], xq)
    with open(cached_files["meta"], 'w') as f:
        json.dump(meta, f, indent=2)
    
    return xb, xq, meta
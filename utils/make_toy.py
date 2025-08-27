# utils/make_toy.py
import os, json, re
import numpy as np


def make_toy(N=100_000, D=128, nq=1_000, dist="gaussian", seed=42):
    rng = np.random.default_rng(seed)
    if dist == "gaussian":
        xb = rng.standard_normal((N, D), dtype=np.float32)
        xq = rng.standard_normal((nq, D), dtype=np.float32)
    elif dist == "uniform":
        xb = rng.random((N, D), dtype=np.float32) - 0.5
        xq = rng.random((nq, D), dtype=np.float32) - 0.5
    else:
        raise ValueError(f"Unknown dist: {dist}")
    return xb, xq


def _key(N, D, nq, dist, seed):
    return f"toy_{dist}_N{N}_D{D}_nq{nq}_seed{seed}"


def _paths(data_dir, key):
    base = os.path.join(data_dir, key)
    return {
        "xb":  base + "_xb.npy",
        "xq":  base + "_xq.npy",
        "meta": base + "_meta.json"
    }


def load_or_make(N, D, nq, dist, seed=42, data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    key = _key(N, D, nq, dist, seed)
    p = _paths(data_dir, key)
    if all(os.path.exists(pth) for pth in p.values()):
        xb = np.load(p["xb"], mmap_mode=None)
        xq = np.load(p["xq"], mmap_mode=None)
        with open(p["meta"], "r") as f:
            meta = json.load(f)
        return xb, xq, meta

    # No cache — generate and save
    xb, xq = make_toy(N=N, D=D, nq=nq, dist=dist, seed=seed)
    np.save(p["xb"], xb)
    np.save(p["xq"], xq)
    meta = {"key": key, "N": N, "D": D, "nq": nq, "dist": dist, "seed": seed}
    with open(p["meta"], "w") as f:
        json.dump(meta, f)
    return xb, xq, meta


_KEY_RE = re.compile(r"^toy_(?P<dist>\w+)_N(?P<N>\d+)_D(?P<D>\d+)_nq(?P<nq>\d+)_seed(?P<seed>\d+)$")


def parse_key(key: str):
    m = _KEY_RE.match(key)
    if not m:
        raise ValueError(f"Bad dataset key: {key}")
    g = m.groupdict()
    return {
        "dist": g["dist"],
        "N": int(g["N"]),
        "D": int(g["D"]),
        "nq": int(g["nq"]),
        "seed": int(g["seed"]),
    }


def load_by_key(key: str, data_dir: str = "data"):
    """Load xb/xq/meta by string key, generate if missing."""
    # Check if it's a toy dataset
    if key.startswith("toy_"):
        p = _paths(data_dir, key)
        if all(os.path.exists(pth) for pth in p.values()):
            xb = np.load(p["xb"], mmap_mode=None)
            xq = np.load(p["xq"], mmap_mode=None)
            with open(p["meta"], "r") as f:
                meta = json.load(f)
            return xb, xq, meta
        # If absent — parse and generate
        params = parse_key(key)
        return load_or_make(**params, data_dir=data_dir)
    else:
        # Try to load as real dataset
        try:
            from .real_datasets import load_real_dataset
            return load_real_dataset(key, data_dir)
        except ImportError:
            raise ValueError(f"Real datasets not available. Unknown dataset key: {key}")
        except Exception as e:
            raise ValueError(f"Failed to load real dataset '{key}': {e}")

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_version() -> str:
    """Return project version from VERSION file located at repo root.
    Falls back to "0.0.0-unknown" if file is missing.
    """
    # Determine repo root as parent of this file's parent
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, os.pardir))
    path = os.path.join(root, "VERSION")
    try:
        with open(path, "r") as f:
            v = f.read().strip()
            return v or "0.0.0-unknown"
    except Exception:
        return "0.0.0-unknown"


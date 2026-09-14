import math
import numpy as np
from pathlib import Path

def load_idx(path):
    """Read an IDX file and return its contents as a uint8 array"""
    with open(path, "rb") as f:
        raw = f.read()

    if len(raw) < 4:
        raise ValueError(f"{path}: too short to hold an IDX header ({len(raw)} bytes)")

    dcode, ndims = raw[2], raw[3]
    if raw[0] != 0 or raw[1] != 0 or dcode != 8:
        raise ValueError(f"{path}: not a uint8 IDX file, header starts {raw[:4].hex()}")

    header_len = 4 + 4 * ndims
    # header fields are big-endian (4 bytes)
    dims = tuple(int.from_bytes(raw[4 + 4*i : 8 + 4*i], "big") for i in range(ndims))

    expected = math.prod(dims)
    actual = len(raw) - header_len
    if actual != expected:
        raise ValueError(
            f"{path}: truncated, expected {expected} bytes of data"
            f"for shape {dims}, found {actual}"
        )

    return np.frombuffer(raw, dtype=np.uint8, offset=header_len).reshape(dims)

def load_mnist(data_dir, n_val=5000, seed=0):
    """Return flattened, scaled MNIST as (X_train, y_train, X_val, y_val, X_test, y_test)"""

    d = Path(data_dir)
    
    def images(name):
        X = load_idx(d / name).reshape(-1, 28 * 28).astype(np.float32)
        X /= 255.0
        return X
    
    def labels(name):
        return load_idx(d / name).astype(np.int64)
    
    X_train = images("train-images.idx3-ubyte")
    y_train = labels("train-labels.idx1-ubyte")
    X_test  = images(d / "t10k-images.idx3-ubyte")
    y_test  = labels(d / "t10k-labels.idx1-ubyte")
    
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("image-label count mismatch")
    
    perm = np.random.default_rng(seed).permutation(len(X_train))
    val, train = perm[:n_val], perm[n_val:] # Keep disjoint
    return (
        X_train[train], y_train[train], X_train[val], y_train[val], X_test, y_test
    )
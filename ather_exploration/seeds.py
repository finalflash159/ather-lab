"""Stable independent RNG namespaces: never Python hash() or global random state."""

import hashlib
import json

import numpy as np


def derive_seed(seed: int, namespace: str, *parts) -> int:
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be integer 0..2**64-1")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be nonempty")
    raw = json.dumps(
        [seed, namespace, *parts], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def stage_rng(seed: int, namespace: str, *parts) -> np.random.Generator:
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(derive_seed(seed, namespace, *parts)))
    )

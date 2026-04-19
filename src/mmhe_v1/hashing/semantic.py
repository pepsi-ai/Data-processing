from __future__ import annotations

import random
from typing import Iterable


def semantic_hash_128(vector: Iterable[float], seed: int) -> str:
    values = list(vector)
    if not values:
        raise ValueError("vector must not be empty")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise TypeError("vector values must be numeric")

    rng = random.Random(seed)
    bits: list[str] = []
    for _ in range(128):
        projection = sum(float(value) * rng.uniform(-1.0, 1.0) for value in values)
        bits.append("1" if projection >= 0 else "0")
    return "".join(bits)

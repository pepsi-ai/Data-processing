from __future__ import annotations

import math
from typing import Iterable


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)

    if len(left_values) != len(right_values):
        raise ValueError("left and right vectors must have the same length")
    if not left_values:
        raise ValueError("vectors must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in left_values):
        raise TypeError("left vector values must be numeric")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in right_values):
        raise TypeError("right vector values must be numeric")

    dot = sum(float(a) * float(b) for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(float(value) * float(value) for value in left_values))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("cosine similarity is undefined for zero-norm vectors")

    return dot / (left_norm * right_norm)

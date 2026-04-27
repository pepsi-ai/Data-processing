from __future__ import annotations

import math
from typing import Any, Iterable


def normalize_embedding(values: Iterable[float]) -> list[float]:
    vector = list(values)
    if not vector:
        raise ValueError("embedding vector must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector):
        raise TypeError("embedding values must be numeric")

    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm == 0.0:
        raise ValueError("cannot normalize zero vector")

    return [float(value) / norm for value in vector]


def coerce_numeric_row(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "squeeze"):
        values = values.squeeze(0)
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
        values = values[0]
    if not isinstance(values, list):
        raise TypeError("encoder output must be convertible to a numeric list")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise TypeError("encoder output values must be numeric")
    return [float(value) for value in values]


def coerce_feature_row(values: Any) -> list[float]:
    values = coerce_numeric_row(values)
    return normalize_embedding(values)

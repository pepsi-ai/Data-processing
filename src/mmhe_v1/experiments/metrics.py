from __future__ import annotations

from typing import Iterable, Sequence


def recall_at_k(relevant_ids: Iterable[str], ranked_ids: Sequence[str], k: int) -> float:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    relevant = set(relevant_ids)
    top_k = list(ranked_ids[:k])
    if not relevant:
        raise ValueError("relevant_ids must not be empty")
    return 1.0 if any(item in relevant for item in top_k) else 0.0


def mean_absolute_error(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("left and right must have the same length")
    if not left:
        raise ValueError("left and right must not be empty")
    return sum(abs(float(a) - float(b)) for a, b in zip(left, right)) / len(left)


def rank_consistency(expected_ranked: Sequence[str], observed_ranked: Sequence[str], k: int) -> float:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    effective_k = min(k, len(expected_ranked), len(observed_ranked))
    if effective_k == 0:
        raise ValueError("k must overlap with non-empty ranked inputs")

    expected_top = set(expected_ranked[:effective_k])
    observed_top = set(observed_ranked[:effective_k])
    return len(expected_top & observed_top) / effective_k

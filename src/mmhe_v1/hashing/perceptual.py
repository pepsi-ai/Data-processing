from __future__ import annotations

import importlib
from pathlib import Path

from PIL import Image


def compute_phash(path: Path | str) -> str:
    try:
        imagehash = importlib.import_module("imagehash")
    except ModuleNotFoundError as error:
        raise RuntimeError("imagehash dependency is required for compute_phash") from error

    with Image.open(Path(path)) as image:
        digest = imagehash.phash(image.convert("RGB"))
    return str(digest)


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("left and right digests must have the same length")

    return (int(left, 16) ^ int(right, 16)).bit_count()

from __future__ import annotations

import hashlib
from pathlib import Path

from mmhe_v1.canonicalize.image import canonicalize_image
from mmhe_v1.canonicalize.text import canonicalize_text


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def raw_file_sha256(path: Path | str) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_image_sha256(path: Path | str) -> str:
    canonical = canonicalize_image(Path(path))
    header = f"RGB:{canonical.width}x{canonical.height}".encode("ascii")
    return sha256_bytes(header + canonical.pixel_bytes)


def canonical_text_sha256(text: str) -> str:
    canonical = canonicalize_text(text)
    return sha256_bytes(canonical.canonical_bytes)

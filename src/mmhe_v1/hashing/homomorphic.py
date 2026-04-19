from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from mmhe_v1.canonicalize.image import canonicalize_image
from mmhe_v1.canonicalize.text import canonicalize_text
from mmhe_v1.types import CanonicalImage, CanonicalText


HomomorphicDigest = tuple[int, ...]
_DEFAULT_MODULUS = (1 << 61) - 1


def _coerce_numeric_sequence(values: Iterable[float], *, field_name: str) -> list[float]:
    vector = list(values)
    if not vector:
        raise ValueError(f"{field_name} must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector):
        raise TypeError(f"{field_name} values must be numeric")
    return [float(value) for value in vector]


def quantize_vector(values: Iterable[float], *, scale: int = 1_000) -> tuple[int, ...]:
    if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
        raise ValueError("scale must be a positive integer")
    vector = _coerce_numeric_sequence(values, field_name="vector")
    return tuple(int(round(value * scale)) for value in vector)


def encode_quantized_vector_items(values: Iterable[float], *, scale: int = 1_000) -> list[bytes]:
    quantized = quantize_vector(values, scale=scale)
    return [f"dim={index}|value={value}".encode("utf-8") for index, value in enumerate(quantized)]


def _chunk_bytes(payload: bytes, *, chunk_size: int) -> list[bytes]:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not payload:
        return [b""]
    return [payload[offset : offset + chunk_size] for offset in range(0, len(payload), chunk_size)]


def encode_payload_items(
    payload: bytes,
    *,
    namespace: str,
    chunk_size: int = 1024,
) -> list[bytes]:
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("namespace must be a non-empty string")
    items = [f"{namespace}|length={len(payload)}".encode("utf-8")]
    for index, chunk in enumerate(_chunk_bytes(payload, chunk_size=chunk_size)):
        items.append(f"{namespace}|chunk={index}|".encode("utf-8") + chunk)
    return items


def encode_ranked_result_items(
    ranked_ids: Sequence[str],
    scores: Iterable[float],
    *,
    scale: int = 1_000,
) -> list[bytes]:
    ids = list(ranked_ids)
    if not ids:
        raise ValueError("ranked_ids must not be empty")
    if any(not isinstance(document_id, str) or not document_id.strip() for document_id in ids):
        raise ValueError("ranked_ids must contain non-empty strings")

    quantized_scores = quantize_vector(scores, scale=scale)
    if len(ids) != len(quantized_scores):
        raise ValueError("ranked_ids and scores must have the same length")

    return [
        f"rank={rank}|id={document_id}|score={score}".encode("utf-8")
        for rank, (document_id, score) in enumerate(zip(ids, quantized_scores))
    ]


@dataclass(frozen=True, slots=True)
class LtHash:
    output_size: int = 8
    modulus: int = _DEFAULT_MODULUS
    seed: int = 0

    def __post_init__(self) -> None:
        if self.output_size <= 0:
            raise ValueError("output_size must be > 0")
        if self.modulus <= 2:
            raise ValueError("modulus must be > 2")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")

    def zero(self) -> HomomorphicDigest:
        return tuple(0 for _ in range(self.output_size))

    def hash_single(self, data: bytes) -> HomomorphicDigest:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")

        raw = bytes(data)
        lanes: list[int] = []
        counter = 0
        seed_bytes = self.seed.to_bytes(8, byteorder="big", signed=False)
        while len(lanes) < self.output_size:
            digest = hashlib.sha256(seed_bytes + counter.to_bytes(4, "big") + raw).digest()
            for offset in range(0, len(digest), 8):
                if len(lanes) == self.output_size:
                    break
                chunk = digest[offset : offset + 8]
                lanes.append(int.from_bytes(chunk, byteorder="big", signed=False) % self.modulus)
            counter += 1
        return tuple(lanes)

    def combine(self, left: Sequence[int], right: Sequence[int]) -> HomomorphicDigest:
        self._validate_digest(left)
        self._validate_digest(right)
        return tuple((int(a) + int(b)) % self.modulus for a, b in zip(left, right))

    def remove(self, combined: Sequence[int], item_hash: Sequence[int]) -> HomomorphicDigest:
        self._validate_digest(combined)
        self._validate_digest(item_hash)
        return tuple((int(a) - int(b)) % self.modulus for a, b in zip(combined, item_hash))

    def hash_many(self, items: Iterable[bytes]) -> HomomorphicDigest:
        digest = self.zero()
        for item in items:
            digest = self.combine(digest, self.hash_single(item))
        return digest

    def hexdigest(self, digest: Sequence[int]) -> str:
        self._validate_digest(digest)
        return "".join(f"{int(value):016x}" for value in digest)

    def _validate_digest(self, digest: Sequence[int]) -> None:
        values = tuple(digest)
        if len(values) != self.output_size:
            raise ValueError("digest length does not match output_size")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("digest values must be integers")


def hash_quantized_vector(
    values: Iterable[float],
    *,
    lthash: LtHash | None = None,
    scale: int = 1_000,
) -> HomomorphicDigest:
    hasher = lthash or LtHash()
    return hasher.hash_many(encode_quantized_vector_items(values, scale=scale))


def _resolve_canonical_text(
    value: str | CanonicalText,
    *,
    unicode_form: str = "NFC",
    lowercase: bool = True,
    collapse_whitespace: bool = True,
) -> CanonicalText:
    if isinstance(value, CanonicalText):
        return value
    if not isinstance(value, str):
        raise TypeError("text must be a string or CanonicalText")
    return canonicalize_text(
        value,
        unicode_form=unicode_form,
        lowercase=lowercase,
        collapse_whitespace=collapse_whitespace,
    )


def _canonical_image_payload(image: CanonicalImage) -> bytes:
    header = f"RGB:{image.width}x{image.height}".encode("ascii")
    return header + image.pixel_bytes


def _resolve_canonical_image(
    value: str | Path | CanonicalImage,
    *,
    size: tuple[int, int] = (224, 224),
    resample: str = "bicubic",
) -> CanonicalImage:
    if isinstance(value, CanonicalImage):
        return value
    if isinstance(value, (str, Path)):
        return canonicalize_image(Path(value), size=size, resample=resample)
    raise TypeError("image must be a filesystem path or CanonicalImage")


def hash_payload_bytes(
    payload: bytes,
    *,
    namespace: str,
    lthash: LtHash | None = None,
    chunk_size: int = 1024,
) -> HomomorphicDigest:
    hasher = lthash or LtHash()
    return hasher.hash_many(encode_payload_items(payload, namespace=namespace, chunk_size=chunk_size))


def hash_canonical_text(
    value: str | CanonicalText,
    *,
    lthash: LtHash | None = None,
    chunk_size: int = 128,
    unicode_form: str = "NFC",
    lowercase: bool = True,
    collapse_whitespace: bool = True,
) -> HomomorphicDigest:
    canonical = _resolve_canonical_text(
        value,
        unicode_form=unicode_form,
        lowercase=lowercase,
        collapse_whitespace=collapse_whitespace,
    )
    return hash_payload_bytes(
        canonical.canonical_bytes,
        namespace="text",
        lthash=lthash,
        chunk_size=chunk_size,
    )


def hash_canonical_image(
    value: str | Path | CanonicalImage,
    *,
    lthash: LtHash | None = None,
    chunk_size: int = 4096,
    size: tuple[int, int] = (224, 224),
    resample: str = "bicubic",
) -> HomomorphicDigest:
    canonical = _resolve_canonical_image(value, size=size, resample=resample)
    return hash_payload_bytes(
        _canonical_image_payload(canonical),
        namespace="image",
        lthash=lthash,
        chunk_size=chunk_size,
    )


def hash_ranked_results(
    ranked_ids: Sequence[str],
    scores: Iterable[float],
    *,
    lthash: LtHash | None = None,
    scale: int = 1_000,
) -> HomomorphicDigest:
    hasher = lthash or LtHash()
    return hasher.hash_many(encode_ranked_result_items(ranked_ids, scores, scale=scale))

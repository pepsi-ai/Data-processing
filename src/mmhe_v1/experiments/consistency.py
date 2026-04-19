from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from mmhe_v1.hashing.homomorphic import (
    LtHash,
    hash_canonical_image,
    hash_canonical_text,
    hash_quantized_vector,
    hash_ranked_results,
)
from mmhe_v1.types import CanonicalImage, CanonicalText


@dataclass(frozen=True, slots=True)
class CommitmentCheck:
    plaintext_digest: str
    encrypted_digest: str
    consistent: bool


def _build_commitment_check(
    plain_digest: str,
    other_digest: str,
) -> CommitmentCheck:
    return CommitmentCheck(
        plaintext_digest=plain_digest,
        encrypted_digest=other_digest,
        consistent=plain_digest == other_digest,
    )


def verify_vector_commitment(
    plain_vector: Iterable[float],
    encrypted_then_decrypted_vector: Iterable[float],
    *,
    lthash: LtHash | None = None,
    scale: int = 1_000,
) -> CommitmentCheck:
    hasher = lthash or LtHash()
    plain_digest = hasher.hexdigest(hash_quantized_vector(plain_vector, lthash=hasher, scale=scale))
    encrypted_digest = hasher.hexdigest(
        hash_quantized_vector(encrypted_then_decrypted_vector, lthash=hasher, scale=scale)
    )
    return _build_commitment_check(plain_digest, encrypted_digest)


def verify_ranked_result_commitment(
    *,
    plain_ranked_ids: Sequence[str],
    plain_scores: Iterable[float],
    encrypted_ranked_ids: Sequence[str],
    encrypted_scores: Iterable[float],
    lthash: LtHash | None = None,
    scale: int = 1_000,
) -> CommitmentCheck:
    hasher = lthash or LtHash()
    plain_digest = hasher.hexdigest(
        hash_ranked_results(plain_ranked_ids, plain_scores, lthash=hasher, scale=scale)
    )
    encrypted_digest = hasher.hexdigest(
        hash_ranked_results(encrypted_ranked_ids, encrypted_scores, lthash=hasher, scale=scale)
    )
    return _build_commitment_check(plain_digest, encrypted_digest)


def verify_canonical_text_commitment(
    plain_text: str | CanonicalText,
    other_text: str | CanonicalText,
    *,
    lthash: LtHash | None = None,
    chunk_size: int = 128,
    unicode_form: str = "NFC",
    lowercase: bool = True,
    collapse_whitespace: bool = True,
    scale: int = 1_000,
) -> CommitmentCheck:
    del scale
    hasher = lthash or LtHash()
    plain_digest = hasher.hexdigest(
        hash_canonical_text(
            plain_text,
            lthash=hasher,
            chunk_size=chunk_size,
            unicode_form=unicode_form,
            lowercase=lowercase,
            collapse_whitespace=collapse_whitespace,
        )
    )
    other_digest = hasher.hexdigest(
        hash_canonical_text(
            other_text,
            lthash=hasher,
            chunk_size=chunk_size,
            unicode_form=unicode_form,
            lowercase=lowercase,
            collapse_whitespace=collapse_whitespace,
        )
    )
    return _build_commitment_check(plain_digest, other_digest)


def verify_canonical_image_commitment(
    plain_image: str | Path | CanonicalImage,
    other_image: str | Path | CanonicalImage,
    *,
    lthash: LtHash | None = None,
    chunk_size: int = 4096,
    size: tuple[int, int] = (224, 224),
    resample: str = "bicubic",
) -> CommitmentCheck:
    hasher = lthash or LtHash()
    plain_digest = hasher.hexdigest(
        hash_canonical_image(
            plain_image,
            lthash=hasher,
            chunk_size=chunk_size,
            size=size,
            resample=resample,
        )
    )
    other_digest = hasher.hexdigest(
        hash_canonical_image(
            other_image,
            lthash=hasher,
            chunk_size=chunk_size,
            size=size,
            resample=resample,
        )
    )
    return _build_commitment_check(plain_digest, other_digest)

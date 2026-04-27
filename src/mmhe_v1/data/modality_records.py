from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from mmhe_v1.canonicalize.image import canonicalize_image
from mmhe_v1.canonicalize.text import canonicalize_text
from mmhe_v1.canonicalize.video import canonical_video_payload, canonicalize_video
from mmhe_v1.embedding.chinese_clip_encoder import (
    encode_image as encode_chinese_clip_image,
    encode_text as encode_chinese_clip_text,
)
from mmhe_v1.embedding.openclip_encoder import (
    encode_image as encode_openclip_image,
    encode_text as encode_openclip_text,
)
from mmhe_v1.embedding.utils import normalize_embedding
from mmhe_v1.embedding.video_action_encoder import VideoAction, VideoActionEncoding, encode_video_action
from mmhe_v1.hashing.homomorphic import (
    LtHash,
    hash_canonical_image,
    hash_canonical_text,
    hash_quantized_vector,
)
from mmhe_v1.hashing.perceptual import compute_phash
from mmhe_v1.hashing.semantic import semantic_hash_128
from mmhe_v1.hashing.strict import canonical_image_sha256, canonical_text_sha256, raw_file_sha256, sha256_bytes
from mmhe_v1.types import AppConfig, CanonicalImage, CanonicalText


@dataclass(frozen=True, slots=True)
class TextModalityRecord:
    canonical_text: str
    canonical_sha256: str
    homomorphic_payload_digest: str
    semantic_hash_128: str
    homomorphic_embedding_digest: str
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class ImageModalityRecord:
    source_path: Path
    raw_sha256: str
    canonical_sha256: str
    phash: str
    homomorphic_payload_digest: str
    semantic_hash_128: str
    homomorphic_embedding_digest: str
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class VideoModalityRecord:
    source_path: Path
    raw_sha256: str
    canonical_sha256: str
    semantic_hash_128: str
    homomorphic_embedding_digest: str
    embedding: list[float]
    top_actions: list[VideoAction]


def _deterministic_embedding_from_bytes(
    payload: bytes,
    *,
    seed: int,
    embedding_dim: int,
    namespace: str,
) -> list[float]:
    values: list[float] = []
    counter = 0
    seed_prefix = f"{seed}:{namespace}:".encode("utf-8") + payload
    while len(values) < embedding_dim:
        digest = hashlib.sha256(seed_prefix + counter.to_bytes(4, "big")).digest()
        for offset in range(0, len(digest), 4):
            chunk = digest[offset : offset + 4]
            if len(chunk) < 4:
                continue
            scaled = int.from_bytes(chunk, "big") / 0xFFFFFFFF
            values.append((scaled * 2.0) - 1.0)
            if len(values) == embedding_dim:
                break
        counter += 1
    return normalize_embedding(values)


def _build_text_embedding(canonical: CanonicalText, config: AppConfig) -> list[float]:
    if config.embedding_backend == "deterministic_stub":
        return _deterministic_embedding_from_bytes(
            canonical.canonical_bytes,
            seed=config.seed,
            embedding_dim=config.embedding_dim,
            namespace="text",
        )
    if config.embedding_backend == "chinese_clip":
        return encode_chinese_clip_text(
            canonical.canonical_text,
            model_name=config.model_name,
            pretrained_tag=config.pretrained_tag,
            device="cpu",
        )
    if config.embedding_backend == "open_clip":
        return encode_openclip_text(
            canonical.canonical_text,
            model_name=config.model_name,
            pretrained_tag=config.pretrained_tag,
            device="cpu",
        )
    raise ValueError(f"unsupported embedding backend: {config.embedding_backend}")


def _build_image_embedding(canonical: CanonicalImage, config: AppConfig) -> list[float]:
    if config.embedding_backend == "deterministic_stub":
        header = f"RGB:{canonical.width}x{canonical.height}".encode("ascii")
        return _deterministic_embedding_from_bytes(
            header + canonical.pixel_bytes,
            seed=config.seed,
            embedding_dim=config.embedding_dim,
            namespace="image",
        )
    if config.embedding_backend == "chinese_clip":
        return encode_chinese_clip_image(
            canonical,
            model_name=config.model_name,
            pretrained_tag=config.pretrained_tag,
            device="cpu",
        )
    if config.embedding_backend == "open_clip":
        return encode_openclip_image(
            canonical,
            model_name=config.model_name,
            pretrained_tag=config.pretrained_tag,
            device="cpu",
        )
    raise ValueError(f"unsupported embedding backend: {config.embedding_backend}")


def build_text_modality_record(
    text: str,
    config: AppConfig,
    *,
    lthash: LtHash | None = None,
    payload_chunk_size: int = 128,
    embedding_scale: int = 1_000,
) -> TextModalityRecord:
    hasher = lthash or LtHash(seed=config.seed)
    canonical = canonicalize_text(
        text,
        unicode_form=config.text_unicode_form,
        lowercase=config.text_lowercase,
        collapse_whitespace=config.text_collapse_whitespace,
    )
    embedding = _build_text_embedding(canonical, config)
    return TextModalityRecord(
        canonical_text=canonical.canonical_text,
        canonical_sha256=canonical_text_sha256(text),
        homomorphic_payload_digest=hasher.hexdigest(
            hash_canonical_text(canonical, lthash=hasher, chunk_size=payload_chunk_size)
        ),
        semantic_hash_128=semantic_hash_128(embedding, seed=config.hash.projection_seed),
        homomorphic_embedding_digest=hasher.hexdigest(
            hash_quantized_vector(embedding, lthash=hasher, scale=embedding_scale)
        ),
        embedding=embedding,
    )


def build_image_modality_record(
    path: str | Path,
    config: AppConfig,
    *,
    lthash: LtHash | None = None,
    payload_chunk_size: int = 4096,
    embedding_scale: int = 1_000,
) -> ImageModalityRecord:
    resolved_path = Path(path)
    hasher = lthash or LtHash(seed=config.seed)
    canonical = canonicalize_image(
        resolved_path,
        size=config.canonical_image_size,
        resample=config.image_resample,
    )
    embedding = _build_image_embedding(canonical, config)
    return ImageModalityRecord(
        source_path=resolved_path,
        raw_sha256=raw_file_sha256(resolved_path),
        canonical_sha256=canonical_image_sha256(resolved_path),
        phash=compute_phash(resolved_path),
        homomorphic_payload_digest=hasher.hexdigest(
            hash_canonical_image(canonical, lthash=hasher, chunk_size=payload_chunk_size)
        ),
        semantic_hash_128=semantic_hash_128(embedding, seed=config.hash.projection_seed),
        homomorphic_embedding_digest=hasher.hexdigest(
            hash_quantized_vector(embedding, lthash=hasher, scale=embedding_scale)
        ),
        embedding=embedding,
    )


def build_video_modality_record(
    path: str | Path,
    config: AppConfig,
    *,
    lthash: LtHash | None = None,
    embedding_scale: int = 1_000,
) -> VideoModalityRecord:
    resolved_path = Path(path)
    hasher = lthash or LtHash(seed=config.seed)
    canonical = canonicalize_video(
        resolved_path,
        clip_len=config.video.clip_len,
        frame_sample_rate=config.video.frame_sample_rate,
        size=config.canonical_image_size,
    )
    encoded = encode_video_action(
        canonical,
        model_name=config.video.model_name,
        top_k=config.video.top_k,
        device="cpu",
    )
    return VideoModalityRecord(
        source_path=resolved_path,
        raw_sha256=raw_file_sha256(resolved_path),
        canonical_sha256=sha256_bytes(canonical_video_payload(canonical)),
        semantic_hash_128=semantic_hash_128(encoded.embedding, seed=config.hash.projection_seed),
        homomorphic_embedding_digest=hasher.hexdigest(
            hash_quantized_vector(encoded.embedding, lthash=hasher, scale=embedding_scale)
        ),
        embedding=encoded.embedding,
        top_actions=encoded.top_actions,
    )

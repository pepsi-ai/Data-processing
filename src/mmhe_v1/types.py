from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PathsConfig:
    repo_root: Path
    dataset_root: Path
    artifacts_root: Path
    raw_manifest: Path
    v1_input_manifest: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_root", Path(self.repo_root))
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "artifacts_root", Path(self.artifacts_root))
        object.__setattr__(self, "raw_manifest", Path(self.raw_manifest))
        object.__setattr__(self, "v1_input_manifest", Path(self.v1_input_manifest))
        for field_name in (
            "repo_root",
            "dataset_root",
            "artifacts_root",
            "raw_manifest",
            "v1_input_manifest",
        ):
            if str(getattr(self, field_name)).strip() == "":
                raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True, slots=True)
class HashConfig:
    semantic_bits: int = 128
    projection_seed: int = 7

    def __post_init__(self) -> None:
        if self.semantic_bits <= 0:
            raise ValueError("semantic_bits must be > 0")
        if self.projection_seed < 0:
            raise ValueError("projection_seed must be >= 0")


@dataclass(frozen=True, slots=True)
class CkksConfig:
    enabled: bool = False
    poly_modulus_degree: int = 8192
    scaling_mod_size: int = 40

    def __post_init__(self) -> None:
        if self.poly_modulus_degree <= 0:
            raise ValueError("poly_modulus_degree must be > 0")
        if self.scaling_mod_size <= 0:
            raise ValueError("scaling_mod_size must be > 0")


@dataclass(frozen=True, slots=True)
class AppConfig:
    seed: int
    embedding_backend: str
    model_name: str
    pretrained_tag: str
    embedding_dim: int
    paths: PathsConfig
    hash: HashConfig = HashConfig()
    ckks: CkksConfig = CkksConfig()
    text_unicode_form: str = "NFC"
    text_lowercase: bool = True
    text_collapse_whitespace: bool = True
    canonical_image_size: tuple[int, int] = (224, 224)
    image_resample: str = "bicubic"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.embedding_backend not in {"open_clip", "deterministic_stub"}:
            raise ValueError("embedding_backend must be one of: open_clip, deterministic_stub")
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if not self.pretrained_tag.strip():
            raise ValueError("pretrained_tag must not be empty")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        if len(self.canonical_image_size) != 2:
            raise ValueError("canonical_image_size must contain width and height")
        width, height = self.canonical_image_size
        if width <= 0 or height <= 0:
            raise ValueError("canonical_image_size values must be > 0")
        if self.image_resample not in {"nearest", "bilinear", "bicubic", "lanczos"}:
            raise ValueError("image_resample must be a supported value")


@dataclass(frozen=True, slots=True)
class CanonicalText:
    original_text: str
    canonical_text: str
    canonical_bytes: bytes


@dataclass(frozen=True, slots=True)
class CanonicalImage:
    source_path: Path
    width: int
    height: int
    pixel_bytes: bytes


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    output_dir: Path
    reports_dir: Path
    metric_snapshot: dict[str, float]
    runtime_metadata: dict[str, Any]
    summary_path: Path
    result_json_path: Path

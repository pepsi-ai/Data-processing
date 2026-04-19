from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .types import AppConfig, CkksConfig, HashConfig, PathsConfig

_TOP_LEVEL_KEYS = {
    "seed",
    "embedding_backend",
    "model_name",
    "pretrained_tag",
    "embedding_dim",
    "paths",
    "text_normalization",
    "image",
    "hash",
    "ckks",
}
_PATH_KEYS = {
    "repo_root",
    "dataset_root",
    "artifacts_root",
    "raw_manifest",
    "v1_input_manifest",
}
_TEXT_KEYS = {"unicode_form", "lowercase", "collapse_whitespace"}
_IMAGE_KEYS = {"width", "height", "resample"}
_HASH_KEYS = {"semantic_bits", "projection_seed"}
_CKKS_KEYS = {"enabled", "poly_modulus_degree", "scaling_mod_size"}


class ConfigValidationError(ValueError):
    pass


def _require_mapping(value: Any, *, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigValidationError(f"{section} must be a mapping")
    return value


def _reject_unknown_keys(raw: dict[str, Any], *, allowed: set[str], section: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigValidationError(f"{section} contains unknown keys: {', '.join(unknown)}")


def _require_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ConfigValidationError(f"{field_name} must be a boolean")


def _require_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(f"{field_name} must be an integer")
    return value


def _require_str(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ConfigValidationError(f"{field_name} must be a string")
    return value


def _resolve_path(raw: Any, *, field_name: str, base_dir: Path) -> Path:
    if isinstance(raw, Path):
        candidate = raw
    elif isinstance(raw, str):
        if raw.strip() == "":
            raise ConfigValidationError(f"{field_name} must not be empty")
        candidate = Path(raw)
    else:
        raise ConfigValidationError(f"{field_name} must be a string path")

    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def load_config(path: Path) -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigValidationError(f"config does not exist: {path}")
    base_dir = path.parent.resolve()

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = _require_mapping(payload, section="config")
    _reject_unknown_keys(raw, allowed=_TOP_LEVEL_KEYS, section="config")

    paths_raw = _require_mapping(raw.get("paths"), section="paths")
    _reject_unknown_keys(paths_raw, allowed=_PATH_KEYS, section="paths")
    missing_paths = _PATH_KEYS - set(paths_raw)
    if missing_paths:
        missing = ", ".join(sorted(missing_paths))
        raise ConfigValidationError(f"paths missing required keys: {missing}")

    text_raw = _require_mapping(raw.get("text_normalization", {}), section="text_normalization")
    _reject_unknown_keys(text_raw, allowed=_TEXT_KEYS, section="text_normalization")

    image_raw = _require_mapping(raw.get("image", {}), section="image")
    _reject_unknown_keys(image_raw, allowed=_IMAGE_KEYS, section="image")

    hash_raw = _require_mapping(raw.get("hash", {}), section="hash")
    _reject_unknown_keys(hash_raw, allowed=_HASH_KEYS, section="hash")

    ckks_raw = _require_mapping(raw.get("ckks", {}), section="ckks")
    _reject_unknown_keys(ckks_raw, allowed=_CKKS_KEYS, section="ckks")

    try:
        config = AppConfig(
            seed=_require_int(raw["seed"], field_name="seed"),
            embedding_backend=_require_str(
                raw.get("embedding_backend", "open_clip"),
                field_name="embedding_backend",
            ),
            model_name=_require_str(raw["model_name"], field_name="model_name"),
            pretrained_tag=_require_str(
                raw.get("pretrained_tag", "laion2b_s34b_b79k"),
                field_name="pretrained_tag",
            ),
            embedding_dim=_require_int(raw["embedding_dim"], field_name="embedding_dim"),
            paths=PathsConfig(
                repo_root=_resolve_path(
                    paths_raw["repo_root"], field_name="paths.repo_root", base_dir=base_dir
                ),
                dataset_root=_resolve_path(
                    paths_raw["dataset_root"], field_name="paths.dataset_root", base_dir=base_dir
                ),
                artifacts_root=_resolve_path(
                    paths_raw["artifacts_root"],
                    field_name="paths.artifacts_root",
                    base_dir=base_dir,
                ),
                raw_manifest=_resolve_path(
                    paths_raw["raw_manifest"], field_name="paths.raw_manifest", base_dir=base_dir
                ),
                v1_input_manifest=_resolve_path(
                    paths_raw["v1_input_manifest"],
                    field_name="paths.v1_input_manifest",
                    base_dir=base_dir,
                ),
            ),
            hash=HashConfig(
                semantic_bits=_require_int(
                    hash_raw.get("semantic_bits", 128), field_name="hash.semantic_bits"
                ),
                projection_seed=_require_int(
                    hash_raw.get("projection_seed", _require_int(raw["seed"], field_name="seed")),
                    field_name="hash.projection_seed",
                ),
            ),
            ckks=CkksConfig(
                enabled=_require_bool(ckks_raw.get("enabled", False), field_name="ckks.enabled"),
                poly_modulus_degree=_require_int(
                    ckks_raw.get("poly_modulus_degree", 8192),
                    field_name="ckks.poly_modulus_degree",
                ),
                scaling_mod_size=_require_int(
                    ckks_raw.get("scaling_mod_size", 40), field_name="ckks.scaling_mod_size"
                ),
            ),
            text_unicode_form=_require_str(
                text_raw.get("unicode_form", "NFC"), field_name="text_normalization.unicode_form"
            ),
            text_lowercase=_require_bool(
                text_raw.get("lowercase", True), field_name="text_normalization.lowercase"
            ),
            text_collapse_whitespace=_require_bool(
                text_raw.get("collapse_whitespace", True),
                field_name="text_normalization.collapse_whitespace",
            ),
            canonical_image_size=(
                _require_int(image_raw.get("width", 224), field_name="image.width"),
                _require_int(image_raw.get("height", 224), field_name="image.height"),
            ),
            image_resample=_require_str(
                image_raw.get("resample", "bicubic"), field_name="image.resample"
            ).lower(),
        )
    except KeyError as error:
        raise ConfigValidationError(f"missing required config key: {error.args[0]}") from error
    except (TypeError, ValueError) as error:
        raise ConfigValidationError(str(error)) from error

    return config

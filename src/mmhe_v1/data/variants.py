from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mmhe_v1.canonicalize.text import normalize_text


@dataclass(frozen=True, slots=True)
class ImageVariantSpec:
    name: str
    source_path: Path
    target_format: str
    size: tuple[int, int] | None
    apply_exif_orientation: bool


def build_text_variants(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    canonical_preserve_case = normalize_text(text, lowercase=False)
    canonical_lower = normalize_text(text, lowercase=True)

    variants = {
        text,
        canonical_preserve_case,
        canonical_lower,
    }
    return sorted(variants)


def build_image_variants(
    source_path: Path | str,
    *,
    canonical_size: tuple[int, int] = (224, 224),
) -> list[ImageVariantSpec]:
    if (
        not isinstance(canonical_size, tuple)
        or len(canonical_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in canonical_size)
    ):
        raise ValueError("canonical_size must be a (width, height) tuple of positive integers")

    path = Path(source_path)
    source_format = path.suffix.lstrip(".").upper() or "BIN"
    width, height = canonical_size
    canonical_name = f"canonical_rgb_{width}" if width == height else f"canonical_rgb_{width}x{height}"

    return [
        ImageVariantSpec(
            name="source_original",
            source_path=path,
            target_format=source_format,
            size=None,
            apply_exif_orientation=False,
        ),
        ImageVariantSpec(
            name=canonical_name,
            source_path=path,
            target_format="PNG",
            size=canonical_size,
            apply_exif_orientation=True,
        ),
    ]

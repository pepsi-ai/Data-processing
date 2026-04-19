from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from mmhe_v1.types import CanonicalImage

_RESAMPLE_MAP = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def canonicalize_image(
    path: Path,
    *,
    size: tuple[int, int] = (224, 224),
    resample: str = "bicubic",
) -> CanonicalImage:
    source_path = Path(path)
    if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
        raise ValueError("size must be (width, height) with positive integers")
    if resample not in _RESAMPLE_MAP:
        raise ValueError(f"unsupported resample mode: {resample}")

    with Image.open(source_path) as raw:
        oriented = ImageOps.exif_transpose(raw)
        canonical = oriented.convert("RGB").resize(size, _RESAMPLE_MAP[resample])
        pixel_bytes = canonical.tobytes()
        return CanonicalImage(
            source_path=source_path,
            width=canonical.width,
            height=canonical.height,
            pixel_bytes=pixel_bytes,
        )

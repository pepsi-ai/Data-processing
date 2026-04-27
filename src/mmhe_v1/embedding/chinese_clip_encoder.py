from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from mmhe_v1.canonicalize.image import canonicalize_image
from mmhe_v1.embedding.utils import coerce_feature_row
from mmhe_v1.types import CanonicalImage


@dataclass(frozen=True, slots=True)
class ChineseClipBackend:
    model: Any
    processor: Any
    model_name: str
    device: str


def _load_torch():
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError:
        return None


@lru_cache(maxsize=4)
def load_chinese_clip_backend(
    model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16",
    pretrained_tag: str = "",
    device: str = "cpu",
) -> ChineseClipBackend:
    try:
        transformers = importlib.import_module("transformers")
    except ModuleNotFoundError as error:
        raise RuntimeError("transformers dependency is required for Chinese-CLIP encoding") from error

    model = transformers.ChineseCLIPModel.from_pretrained(model_name)
    processor = transformers.ChineseCLIPProcessor.from_pretrained(model_name)
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return ChineseClipBackend(
        model=model,
        processor=processor,
        model_name=model_name,
        device=device,
    )


def _canonical_image_to_pil(image: CanonicalImage) -> Image.Image:
    return Image.frombytes("RGB", (image.width, image.height), image.pixel_bytes)


def _coerce_image(value: CanonicalImage | Path | str | Image.Image) -> Image.Image:
    if isinstance(value, CanonicalImage):
        return _canonical_image_to_pil(value)
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, Path)):
        canonical = canonicalize_image(Path(value))
        return _canonical_image_to_pil(canonical)
    raise TypeError("image must be a CanonicalImage, PIL image, or filesystem path")


def _move_inputs_to_device(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    return inputs


def _as_kwargs(inputs: Any) -> dict[str, Any]:
    if not isinstance(inputs, dict):
        raise TypeError("processor output must be a mapping")
    return dict(inputs)


def encode_text(
    text: str,
    *,
    backend: ChineseClipBackend | None = None,
    model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16",
    pretrained_tag: str = "",
    device: str = "cpu",
) -> list[float]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    resolved_backend = backend or load_chinese_clip_backend(
        model_name=model_name,
        pretrained_tag=pretrained_tag,
        device=device,
    )
    inputs = resolved_backend.processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )
    inputs = _move_inputs_to_device(inputs, resolved_backend.device)
    torch = _load_torch()
    if torch is not None and hasattr(torch, "inference_mode"):
        with torch.inference_mode():
            features = resolved_backend.model.get_text_features(**_as_kwargs(inputs))
    else:
        features = resolved_backend.model.get_text_features(**_as_kwargs(inputs))
    return coerce_feature_row(features)


def encode_image(
    image: CanonicalImage | Path | str | Image.Image,
    *,
    backend: ChineseClipBackend | None = None,
    model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16",
    pretrained_tag: str = "",
    device: str = "cpu",
) -> list[float]:
    resolved_backend = backend or load_chinese_clip_backend(
        model_name=model_name,
        pretrained_tag=pretrained_tag,
        device=device,
    )
    inputs = resolved_backend.processor(
        images=[_coerce_image(image)],
        return_tensors="pt",
    )
    inputs = _move_inputs_to_device(inputs, resolved_backend.device)
    torch = _load_torch()
    if torch is not None and hasattr(torch, "inference_mode"):
        with torch.inference_mode():
            features = resolved_backend.model.get_image_features(**_as_kwargs(inputs))
    else:
        features = resolved_backend.model.get_image_features(**_as_kwargs(inputs))
    return coerce_feature_row(features)

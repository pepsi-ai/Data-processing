from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from mmhe_v1.canonicalize.image import canonicalize_image
from mmhe_v1.types import CanonicalImage


@dataclass(frozen=True, slots=True)
class EncoderMetadata:
    model_name: str
    pretrained_tag: str
    embedding_dim: int


@dataclass(frozen=True, slots=True)
class OpenClipBackend:
    module: Any
    model: Any
    tokenizer: Any
    preprocess: Any
    metadata: EncoderMetadata
    device: str


def normalize_embedding(values: Iterable[float]) -> list[float]:
    vector = list(values)
    if not vector:
        raise ValueError("embedding vector must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector):
        raise TypeError("embedding values must be numeric")

    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm == 0.0:
        raise ValueError("cannot normalize zero vector")

    return [float(value) / norm for value in vector]


def _load_torch():
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise RuntimeError("torch dependency is required for OpenCLIP encoding") from error


def _coerce_feature_row(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "squeeze"):
        values = values.squeeze(0)
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list):
        raise TypeError("encoder output must be convertible to a numeric list")
    return normalize_embedding(values)


def _infer_embedding_dim(model: Any) -> int:
    projection = getattr(model, "text_projection", None)
    shape = getattr(projection, "shape", None)
    if shape and len(shape) >= 2:
        return int(shape[-1])
    output_dim = getattr(getattr(model, "visual", None), "output_dim", None)
    if isinstance(output_dim, int) and output_dim > 0:
        return output_dim
    raise RuntimeError("unable to infer OpenCLIP embedding dimension from model")


@lru_cache(maxsize=4)
def load_openclip_backend(
    model_name: str = "ViT-B-32",
    pretrained_tag: str = "laion2b_s34b_b79k",
    device: str = "cpu",
) -> OpenClipBackend:
    try:
        module = importlib.import_module("open_clip")
    except ModuleNotFoundError as error:
        raise RuntimeError("open_clip dependency is required for real encoder loading") from error

    model, _, preprocess = module.create_model_and_transforms(
        model_name,
        pretrained=pretrained_tag,
        device=device,
    )
    model.eval()
    tokenizer = module.get_tokenizer(model_name)
    metadata = EncoderMetadata(
        model_name=model_name,
        pretrained_tag=pretrained_tag,
        embedding_dim=_infer_embedding_dim(model),
    )
    return OpenClipBackend(
        module=module,
        model=model,
        tokenizer=tokenizer,
        preprocess=preprocess,
        metadata=metadata,
        device=device,
    )


def encode_text(
    text: str,
    *,
    backend: OpenClipBackend | None = None,
    model_name: str = "ViT-B-32",
    pretrained_tag: str = "laion2b_s34b_b79k",
    device: str = "cpu",
) -> list[float]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    resolved_backend = backend or load_openclip_backend(
        model_name=model_name,
        pretrained_tag=pretrained_tag,
        device=device,
    )
    torch = _load_torch()
    tokens = resolved_backend.tokenizer([text])
    if hasattr(tokens, "to"):
        tokens = tokens.to(resolved_backend.device)

    with torch.inference_mode():
        features = resolved_backend.model.encode_text(tokens)
    return _coerce_feature_row(features)


def _canonical_image_to_pil(image: CanonicalImage) -> Image.Image:
    return Image.frombytes("RGB", (image.width, image.height), image.pixel_bytes)


def _coerce_image(value: CanonicalImage | Path | str | Image.Image) -> Image.Image:
    if isinstance(value, CanonicalImage):
        return _canonical_image_to_pil(value)
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (str, Path)):
        canonical = canonicalize_image(Path(value))
        return _canonical_image_to_pil(canonical)
    raise TypeError("image must be a CanonicalImage, PIL image, or filesystem path")


def encode_image(
    image: CanonicalImage | Path | str | Image.Image,
    *,
    backend: OpenClipBackend | None = None,
    model_name: str = "ViT-B-32",
    pretrained_tag: str = "laion2b_s34b_b79k",
    device: str = "cpu",
) -> list[float]:
    resolved_backend = backend or load_openclip_backend(
        model_name=model_name,
        pretrained_tag=pretrained_tag,
        device=device,
    )
    torch = _load_torch()
    image_tensor = resolved_backend.preprocess(_coerce_image(image)).unsqueeze(0)
    if hasattr(image_tensor, "to"):
        image_tensor = image_tensor.to(resolved_backend.device)

    with torch.inference_mode():
        features = resolved_backend.model.encode_image(image_tensor)
    return _coerce_feature_row(features)

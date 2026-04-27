from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from PIL import Image

from mmhe_v1.embedding.utils import coerce_numeric_row, normalize_embedding
from mmhe_v1.types import CanonicalImage, CanonicalVideo


@dataclass(frozen=True, slots=True)
class VideoAction:
    label: str
    score: float


@dataclass(frozen=True, slots=True)
class VideoActionEncoding:
    embedding: list[float]
    top_actions: list[VideoAction]


@dataclass(frozen=True, slots=True)
class VideoActionBackend:
    model: Any
    processor: Any
    model_name: str
    device: str


@lru_cache(maxsize=2)
def load_video_action_backend(
    model_name: str = "nateraw/videomae-base-finetuned-ucf101",
    device: str = "cpu",
) -> VideoActionBackend:
    try:
        transformers = importlib.import_module("transformers")
    except ModuleNotFoundError as error:
        raise RuntimeError("transformers dependency is required for VideoMAE encoding") from error

    model = transformers.VideoMAEForVideoClassification.from_pretrained(model_name)
    processor = transformers.VideoMAEImageProcessor.from_pretrained(model_name)
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return VideoActionBackend(
        model=model,
        processor=processor,
        model_name=model_name,
        device=device,
    )


def _canonical_image_to_pil(image: CanonicalImage) -> Image.Image:
    return Image.frombytes("RGB", (image.width, image.height), image.pixel_bytes)


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
    if isinstance(inputs, dict):
        return dict(inputs)
    if hasattr(inputs, "keys") and hasattr(inputs, "__getitem__"):
        return {key: inputs[key] for key in inputs.keys()}
    raise TypeError("processor output must be a mapping")


def _softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _label_for(model: Any, index: int) -> str:
    id2label = getattr(getattr(model, "config", None), "id2label", None)
    if isinstance(id2label, dict):
        return str(id2label.get(index, str(index)))
    return str(index)


def encode_video_action(
    video: CanonicalVideo,
    *,
    backend: VideoActionBackend | None = None,
    model_name: str = "nateraw/videomae-base-finetuned-ucf101",
    device: str = "cpu",
    top_k: int = 5,
) -> VideoActionEncoding:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if not isinstance(video, CanonicalVideo):
        raise TypeError("video must be a CanonicalVideo")

    resolved_backend = backend or load_video_action_backend(model_name=model_name, device=device)
    frames = [_canonical_image_to_pil(frame) for frame in video.frames]
    inputs = resolved_backend.processor(images=frames, return_tensors="pt")
    inputs = _move_inputs_to_device(inputs, resolved_backend.device)
    outputs = resolved_backend.model(**_as_kwargs(inputs))
    logits = coerce_numeric_row(getattr(outputs, "logits"))
    embedding = normalize_embedding(logits)
    probabilities = _softmax(logits)
    ranked = sorted(enumerate(probabilities), key=lambda item: (-item[1], item[0]))
    actions = [
        VideoAction(label=_label_for(resolved_backend.model, index), score=float(score))
        for index, score in ranked[:top_k]
    ]
    return VideoActionEncoding(embedding=embedding, top_actions=actions)

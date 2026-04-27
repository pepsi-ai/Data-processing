from __future__ import annotations

import math
import sys
import types
from collections.abc import Iterator
from pathlib import Path

from mmhe_v1.types import CanonicalImage, CanonicalVideo


class _FakeInputs(dict):
    def to(self, device: str) -> "_FakeInputs":
        self["device"] = device
        return self


class _FakeBatchFeature:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def to(self, device: str) -> "_FakeBatchFeature":
        self._payload["device"] = device
        return self

    def __iter__(self) -> Iterator[str]:
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)

    def __getitem__(self, key: str) -> object:
        return self._payload[key]

    def keys(self):
        return self._payload.keys()


class _FakeProcessor:
    @classmethod
    def from_pretrained(cls, model_name: str) -> "_FakeProcessor":
        assert model_name == "fake/videomae"
        return cls()

    def __call__(self, images: list[object], return_tensors: str) -> _FakeInputs:
        assert len(images) == 3
        assert return_tensors == "pt"
        return _FakeBatchFeature({"pixel_values": "fake"})


class _FakeLogits:
    def detach(self) -> "_FakeLogits":
        return self

    def cpu(self) -> "_FakeLogits":
        return self

    def tolist(self) -> list[list[float]]:
        return [[1.0, 3.0, 2.0]]


class _FakeOutputs:
    logits = _FakeLogits()


class _FakeModel:
    config = types.SimpleNamespace(id2label={0: "ApplyEyeMakeup", 1: "Archery", 2: "BabyCrawling"})

    @classmethod
    def from_pretrained(cls, model_name: str) -> "_FakeModel":
        assert model_name == "fake/videomae"
        return cls()

    def to(self, device: str) -> "_FakeModel":
        self.device = device
        return self

    def eval(self) -> None:
        self.evaluated = True

    def __call__(self, **inputs: object) -> _FakeOutputs:
        assert inputs["pixel_values"] == "fake"
        return _FakeOutputs()


def test_encode_video_action_returns_normalized_logits_and_top_actions(monkeypatch) -> None:
    from mmhe_v1.embedding import video_action_encoder

    fake_transformers = types.SimpleNamespace(
        VideoMAEForVideoClassification=_FakeModel,
        VideoMAEImageProcessor=_FakeProcessor,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    video_action_encoder.load_video_action_backend.cache_clear()

    frame = CanonicalImage(
        source_path=Path("video.avi"),
        width=2,
        height=2,
        pixel_bytes=bytes([255, 0, 0] * 4),
    )
    canonical = CanonicalVideo(
        source_path=Path("video.avi"),
        frame_count=3,
        fps=30.0,
        duration_seconds=0.1,
        sampled_indices=(0, 1, 2),
        frames=(frame, frame, frame),
    )

    result = video_action_encoder.encode_video_action(
        canonical,
        model_name="fake/videomae",
        top_k=2,
    )

    assert result.embedding == [
        0.2672612419124244,
        0.8017837257372732,
        0.5345224838248488,
    ]
    assert [action.label for action in result.top_actions] == ["Archery", "BabyCrawling"]
    raw_softmax = [math.exp(value - 3.0) for value in [1.0, 3.0, 2.0]]
    total = sum(raw_softmax)
    assert result.top_actions[0].score == raw_softmax[1] / total
    assert result.top_actions[1].score == raw_softmax[2] / total

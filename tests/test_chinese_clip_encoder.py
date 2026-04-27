from __future__ import annotations

import sys
import types

from PIL import Image

from mmhe_v1.embedding import chinese_clip_encoder


class _FakeTensor:
    def __init__(self, values: list[list[float]]) -> None:
        self._values = values

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def squeeze(self, dim: int = 0) -> "_FakeTensor":
        if dim == 0 and len(self._values) == 1:
            return _FakeTensor([self._values[0]])
        return self

    def tolist(self) -> list[float]:
        if len(self._values) == 1:
            return self._values[0]
        raise AssertionError("expected a single feature row")


class _FakeInputs(dict):
    def to(self, device: str) -> "_FakeInputs":
        self["device"] = device
        return self


class _FakeProcessor:
    @classmethod
    def from_pretrained(cls, model_name: str) -> "_FakeProcessor":
        assert model_name == "fake/chinese-clip"
        return cls()

    def __call__(self, **kwargs: object) -> _FakeInputs:
        return _FakeInputs(kwargs)


class _FakeModel:
    @classmethod
    def from_pretrained(cls, model_name: str) -> "_FakeModel":
        assert model_name == "fake/chinese-clip"
        return cls()

    def to(self, device: str) -> "_FakeModel":
        self.device = device
        return self

    def eval(self) -> None:
        self.evaluated = True

    def get_text_features(self, **inputs: object) -> _FakeTensor:
        assert inputs["text"] == ["中文红色汽车"]
        return _FakeTensor([[3.0, 4.0]])

    def get_image_features(self, **inputs: object) -> _FakeTensor:
        assert len(inputs["images"]) == 1
        return _FakeTensor([[0.0, 5.0]])


def test_encode_text_uses_chinese_clip_and_normalizes(monkeypatch) -> None:
    fake_transformers = types.SimpleNamespace(
        ChineseCLIPModel=_FakeModel,
        ChineseCLIPProcessor=_FakeProcessor,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    chinese_clip_encoder.load_chinese_clip_backend.cache_clear()

    vector = chinese_clip_encoder.encode_text(
        "中文红色汽车",
        model_name="fake/chinese-clip",
    )

    assert vector == [0.6, 0.8]


def test_encode_image_uses_chinese_clip_and_normalizes(monkeypatch) -> None:
    fake_transformers = types.SimpleNamespace(
        ChineseCLIPModel=_FakeModel,
        ChineseCLIPProcessor=_FakeProcessor,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    chinese_clip_encoder.load_chinese_clip_backend.cache_clear()

    image = Image.new("RGB", (2, 2), color=(255, 0, 0))
    vector = chinese_clip_encoder.encode_image(image, model_name="fake/chinese-clip")

    assert vector == [0.0, 1.0]

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

from mmhe_v1.data.modality_records import VideoAction, VideoModalityRecord


class _FakeVideoCapture:
    frames: list[np.ndarray] = []

    def __init__(self, path: str) -> None:
        self.path = path
        self.position = 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == 7:
            return float(len(self.frames))
        if prop == 5:
            return 25.0
        return 0.0

    def set(self, prop: int, value: float) -> None:
        if prop == 1:
            self.position = int(value)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame.copy()

    def release(self) -> None:
        self.released = True


def _install_fake_cv2(monkeypatch) -> None:
    _FakeVideoCapture.frames = [
        np.full((2, 2, 3), index, dtype=np.uint8) for index in range(10)
    ]

    def resize(frame: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
        width, height = size
        return np.full((height, width, 3), int(frame[0, 0, 0]), dtype=np.uint8)

    fake_cv2 = types.SimpleNamespace(
        CAP_PROP_POS_FRAMES=1,
        CAP_PROP_FPS=5,
        CAP_PROP_FRAME_COUNT=7,
        COLOR_BGR2RGB=9,
        INTER_AREA=3,
        VideoCapture=_FakeVideoCapture,
        cvtColor=lambda frame, code: frame[..., ::-1],
        resize=resize,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)


def test_canonicalize_video_samples_fixed_stride_frames(monkeypatch, tmp_path: Path) -> None:
    _install_fake_cv2(monkeypatch)
    from mmhe_v1.canonicalize.video import canonicalize_video

    video_path = tmp_path / "sample.avi"
    video_path.write_bytes(b"fake video bytes")

    canonical = canonicalize_video(
        video_path,
        clip_len=4,
        frame_sample_rate=2,
        size=(2, 2),
    )

    assert canonical.frame_count == 10
    assert canonical.fps == 25.0
    assert canonical.duration_seconds == 0.4
    assert canonical.sampled_indices == (0, 2, 4, 6)
    assert len(canonical.frames) == 4
    assert canonical.frames[0].pixel_bytes == bytes([0, 0, 0] * 4)
    assert canonical.frames[-1].pixel_bytes == bytes([6, 6, 6] * 4)


def test_build_video_modality_record_uses_video_action_embedding(monkeypatch, tmp_path: Path) -> None:
    _install_fake_cv2(monkeypatch)
    from mmhe_v1.config import load_config
    from mmhe_v1.data import modality_records

    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 7",
                "embedding_backend: chinese_clip",
                "model_name: OFA-Sys/chinese-clip-vit-base-patch16",
                "embedding_dim: 512",
                "paths:",
                "  repo_root: .",
                "  dataset_root: dataset",
                "  artifacts_root: artifacts",
                "  raw_manifest: raw.json",
                "  v1_input_manifest: input.jsonl",
                "ckks:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )
    video_path = tmp_path / "sample.avi"
    video_path.write_bytes(b"fake video bytes")

    def fake_encode_video_action(canonical, **kwargs):
        return modality_records.VideoActionEncoding(
            embedding=[0.6, 0.8],
            top_actions=[
                VideoAction(label="ApplyEyeMakeup", score=0.9),
                VideoAction(label="Archery", score=0.1),
            ],
        )

    monkeypatch.setattr(modality_records, "encode_video_action", fake_encode_video_action)
    record = modality_records.build_video_modality_record(video_path, load_config(config_path))

    assert record.source_path == video_path
    assert record.embedding == [0.6, 0.8]
    assert record.top_actions[0].label == "ApplyEyeMakeup"
    assert len(record.semantic_hash_128) == 128
    assert record.homomorphic_embedding_digest


def test_run_video_flow_writes_json_and_summary(monkeypatch, tmp_path: Path) -> None:
    from mmhe_v1 import video_flow

    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 7",
                "embedding_backend: chinese_clip",
                "model_name: OFA-Sys/chinese-clip-vit-base-patch16",
                "embedding_dim: 512",
                "paths:",
                "  repo_root: .",
                "  dataset_root: dataset",
                "  artifacts_root: artifacts",
                "  raw_manifest: raw.json",
                "  v1_input_manifest: input.jsonl",
                "ckks:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )
    video_path = tmp_path / "sample.avi"
    video_path.write_bytes(b"fake video bytes")

    fake_record = VideoModalityRecord(
        source_path=video_path,
        raw_sha256="raw",
        canonical_sha256="canonical",
        semantic_hash_128="1" * 128,
        homomorphic_embedding_digest="abcd",
        embedding=[0.6, 0.8],
        top_actions=[VideoAction(label="ApplyEyeMakeup", score=0.9)],
    )
    monkeypatch.setattr(video_flow, "build_video_modality_record", lambda *args, **kwargs: fake_record)

    paths = video_flow.run_video_flow(
        config_path=config_path,
        video_path=video_path,
        output_dir=tmp_path / "out",
    )

    payload = json.loads(paths["result_json_path"].read_text(encoding="utf-8"))
    assert payload["_comment"] == "单视频动作识别流程结果。JSON 不支持真正注释，因此使用 _comment 和 _field_notes 字段保存说明。"
    assert payload["_field_notes"]["config"] == "本次运行使用的配置快照。"
    assert payload["video"]["_field_notes"]["embedding.preview"] == "归一化后视频动作特征向量的前 8 个数值，仅用于快速查看。"
    assert payload["ckks"]["_field_notes"]["digest_consistent"] == "解密后的 embedding 哈希是否与明文 embedding 哈希一致。"
    assert payload["video"]["top_actions"][0]["label"] == "ApplyEyeMakeup"
    assert payload["config"]["video_backend"] == "videomae_action"
    assert "Video Flow Summary" in paths["summary_path"].read_text(encoding="utf-8")

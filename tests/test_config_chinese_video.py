from __future__ import annotations

from pathlib import Path

import pytest

from mmhe_v1.config import ConfigValidationError, load_config


def _write_config(tmp_path: Path, extra: str = "") -> Path:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 7",
                "embedding_backend: chinese_clip",
                "model_name: OFA-Sys/chinese-clip-vit-base-patch16",
                "embedding_dim: 512",
                "",
                "paths:",
                "  repo_root: .",
                "  dataset_root: dataset",
                "  artifacts_root: artifacts",
                "  raw_manifest: raw.json",
                "  v1_input_manifest: input.jsonl",
                "",
                extra,
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_chinese_clip_backend_and_video_defaults_are_valid(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))

    assert config.embedding_backend == "chinese_clip"
    assert config.model_name == "OFA-Sys/chinese-clip-vit-base-patch16"
    assert config.embedding_dim == 512
    assert config.video.backend == "videomae_action"
    assert config.video.model_name == "nateraw/videomae-base-finetuned-ucf101"
    assert config.video.clip_len == 16
    assert config.video.frame_sample_rate == 4
    assert config.video.top_k == 5


def test_video_config_rejects_invalid_clip_len(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        "\n".join(
            [
                "video:",
                "  backend: videomae_action",
                "  model_name: nateraw/videomae-base-finetuned-ucf101",
                "  clip_len: 0",
                "  frame_sample_rate: 4",
                "  top_k: 5",
            ]
        ),
    )

    with pytest.raises(ConfigValidationError, match="clip_len"):
        load_config(path)

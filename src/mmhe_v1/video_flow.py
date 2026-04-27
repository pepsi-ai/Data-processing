from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mmhe_v1.config import load_config
from mmhe_v1.crypto.ckks import CKKSAdapter, detect_tenseal_unavailable_reason
from mmhe_v1.data.modality_records import VideoModalityRecord, build_video_modality_record
from mmhe_v1.hashing.homomorphic import LtHash, hash_quantized_vector


def _build_ckks_adapter(config) -> CKKSAdapter:
    tenseal_reason = detect_tenseal_unavailable_reason()
    if config.ckks.enabled:
        if tenseal_reason is not None:
            raise RuntimeError(f"TenSEAL is required but unavailable: {tenseal_reason}")
        return CKKSAdapter.from_tenseal(
            poly_modulus_degree=config.ckks.poly_modulus_degree,
            scaling_mod_size=config.ckks.scaling_mod_size,
        )
    return CKKSAdapter.make_test_double(reason=tenseal_reason or "backend disabled for video flow")


def _preview_vector(values: list[float], *, limit: int = 8) -> list[float]:
    return [round(value, 6) for value in values[:limit]]


def _record_payload(record: VideoModalityRecord) -> dict[str, Any]:
    return {
        "_comment": "视频样本处理后的特征、动作识别结果和哈希结果。",
        "_field_notes": {
            "source_path": "输入视频文件路径。",
            "raw_sha256": "原始视频文件字节级 SHA-256，用于确认源文件是否完全一致。",
            "canonical_sha256": "抽帧、RGB 转换、尺寸规范化后的规范视频内容 SHA-256。",
            "semantic_hash_128": "基于视频动作 embedding 生成的 128 位语义哈希。",
            "homomorphic_embedding_digest": "对量化后视频 embedding 做 LtHash 同态哈希得到的摘要。",
            "embedding.dimension": "视频动作特征向量维度；当前 VideoMAE/UCF101 输出 101 个动作类别维度。",
            "embedding.preview": "归一化后视频动作特征向量的前 8 个数值，仅用于快速查看。",
            "top_actions": "VideoMAE 给出的 top-k 动作类别和 softmax 置信度。",
        },
        "source_path": str(record.source_path),
        "raw_sha256": record.raw_sha256,
        "canonical_sha256": record.canonical_sha256,
        "semantic_hash_128": record.semantic_hash_128,
        "homomorphic_embedding_digest": record.homomorphic_embedding_digest,
        "embedding": {
            "dimension": len(record.embedding),
            "preview": _preview_vector(record.embedding),
        },
        "top_actions": [asdict(action) for action in record.top_actions],
    }


def _summary_text(payload: dict[str, Any]) -> str:
    top_actions = payload["video"]["top_actions"]
    lines = [
        "# Video Flow Summary",
        "",
        f"- Video: {payload['video']['source_path']}",
        f"- Video Backend: {payload['config']['video_backend']}",
        f"- Video Model: {payload['config']['video_model_name']}",
        f"- Embedding Dim: {payload['video']['embedding']['dimension']}",
        f"- CKKS Backend: {payload['ckks']['backend']}",
        f"- CKKS Backend Status: {payload['ckks']['status']}",
        "",
        "## Top Actions",
        "",
    ]
    for rank, action in enumerate(top_actions, start=1):
        lines.append(f"{rank}. {action['label']}: {action['score']:.6f}")
    return "\n".join(lines) + "\n"


def run_video_flow(
    *,
    config_path: str | Path,
    video_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    config = load_config(Path(config_path))
    resolved_output_dir = Path(output_dir).resolve()
    reports_dir = resolved_output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    hasher = LtHash(seed=config.seed)
    record = build_video_modality_record(video_path, config, lthash=hasher)
    adapter = _build_ckks_adapter(config)
    encrypted = adapter.encrypt(record.embedding)
    decrypted = adapter.decrypt(encrypted)
    decrypted_digest = hasher.hexdigest(hash_quantized_vector(decrypted, lthash=hasher))

    payload = {
        "_comment": "单视频动作识别流程结果。JSON 不支持真正注释，因此使用 _comment 和 _field_notes 字段保存说明。",
        "_field_notes": {
            "config": "本次运行使用的配置快照。",
            "video": "视频处理结果，包括特征、哈希和动作分类输出。",
            "ckks": "视频 embedding 进入 CKKS 加密/解密后的验证结果。",
        },
        "config": {
            "_comment": "配置字段说明本次流程如何抽帧、选择模型和生成 top-k 输出。",
            "_field_notes": {
                "config_path": "实验配置文件的绝对路径。",
                "embedding_backend": "图文 embedding 后端；当前视频动作流程不直接使用它做对齐。",
                "image_size": "视频帧规范化后的宽高。",
                "video_backend": "视频处理后端，当前为 VideoMAE 动作识别。",
                "video_model_name": "Hugging Face 上加载的视频动作识别模型。",
                "video_clip_len": "每个视频采样的帧数。",
                "video_frame_sample_rate": "抽帧步长；4 表示每隔 4 帧取一帧。",
                "video_top_k": "输出置信度最高的动作类别数量。",
            },
            "config_path": str(Path(config_path).resolve()),
            "embedding_backend": config.embedding_backend,
            "image_size": list(config.canonical_image_size),
            "video_backend": config.video.backend,
            "video_model_name": config.video.model_name,
            "video_clip_len": config.video.clip_len,
            "video_frame_sample_rate": config.video.frame_sample_rate,
            "video_top_k": config.video.top_k,
        },
        "video": _record_payload(record),
        "ckks": {
            "_comment": "CKKS 验证用于确认视频 embedding 加密再解密后仍能保持量化哈希一致。",
            "_field_notes": {
                "backend": "实际使用的 CKKS 后端名称。",
                "status": "CKKS 后端可用状态。",
                "unavailable_reason": "后端不可用时的原因；可用时为 null。",
                "decrypted_embedding_digest": "解密后 embedding 重新量化并哈希得到的摘要。",
                "digest_consistent": "解密后的 embedding 哈希是否与明文 embedding 哈希一致。",
            },
            "backend": adapter.backend_name.replace("-", "_"),
            "status": adapter.backend_status,
            "unavailable_reason": adapter.unavailable_reason,
            "decrypted_embedding_digest": decrypted_digest,
            "digest_consistent": decrypted_digest == record.homomorphic_embedding_digest,
        },
    }

    result_json_path = reports_dir / "video_result.json"
    summary_path = reports_dir / "video_summary.md"
    result_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path.write_text(_summary_text(payload), encoding="utf-8")
    return {
        "result_json_path": result_json_path,
        "summary_path": summary_path,
    }

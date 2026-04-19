from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmhe_v1.types import ExperimentResult


def render_summary_markdown(metrics_or_result: dict[str, Any] | ExperimentResult) -> str:
    if isinstance(metrics_or_result, ExperimentResult):
        metrics = metrics_or_result.metric_snapshot
        metadata = metrics_or_result.runtime_metadata
    else:
        metrics = metrics_or_result
        metadata = {}

    lines = [
        "# V1 Experiment Summary",
        "",
        f"- Recall@10: {metrics['recall_at_10']:.4f}",
        f"- Mean Absolute Error: {metrics['mae']:.4f}",
        f"- Rank Consistency: {metrics['rank_consistency']:.4f}",
    ]
    if "homomorphic_input_consistency" in metrics:
        lines.append(
            f"- Homomorphic Input Consistency: {metrics['homomorphic_input_consistency']:.4f}"
        )
    if "homomorphic_result_consistency" in metrics:
        lines.append(
            f"- Homomorphic Result Consistency: {metrics['homomorphic_result_consistency']:.4f}"
        )
    if metadata:
        lines.extend(
            [
                "",
                "## Runtime Metadata",
                "",
                f"- Seed: {metadata.get('seed')}",
                f"- Embedding Backend: {metadata.get('embedding_backend')}",
                f"- Model Name: {metadata.get('model_name')}",
                f"- Pretrained Tag: {metadata.get('pretrained_tag')}",
                f"- Embedding Dim: {metadata.get('embedding_dim')}",
                f"- Hash Semantic Bits: {metadata.get('hash.semantic_bits')}",
                f"- CKKS Backend: {metadata.get('ckks_backend')}",
                f"- CKKS Backend Status: {metadata.get('ckks_backend_status')}",
            ]
        )
        if metadata.get("homomorphic_hash_backend") is not None:
            lines.append(f"- Homomorphic Hash Backend: {metadata.get('homomorphic_hash_backend')}")
        if metadata.get("homomorphic_hash_scale") is not None:
            lines.append(f"- Homomorphic Hash Scale: {metadata.get('homomorphic_hash_scale')}")
    return "\n".join(lines) + "\n"


def write_result_artifacts(result: ExperimentResult) -> ExperimentResult:
    result.reports_dir.mkdir(parents=True, exist_ok=True)
    summary_text = render_summary_markdown(result)
    result.summary_path.write_text(summary_text, encoding="utf-8")
    payload = {
        "metric_snapshot": result.metric_snapshot,
        "runtime_metadata": result.runtime_metadata,
        "output_dir": str(result.output_dir),
        "reports_dir": str(result.reports_dir),
        "summary_path": str(result.summary_path),
        "result_json_path": str(result.result_json_path),
    }
    result.result_json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result

from __future__ import annotations

import hashlib
from pathlib import Path

from mmhe_v1.canonicalize.text import canonicalize_text
from mmhe_v1.config import load_config
from mmhe_v1.crypto.ckks import CKKSAdapter, detect_tenseal_unavailable_reason
from mmhe_v1.crypto.similarity import cosine_similarity
from mmhe_v1.embedding.openclip_encoder import encode_text, normalize_embedding
from mmhe_v1.experiments.consistency import verify_ranked_result_commitment, verify_vector_commitment
from mmhe_v1.experiments.metrics import mean_absolute_error, rank_consistency, recall_at_k
from mmhe_v1.hashing.homomorphic import LtHash
from mmhe_v1.hashing.semantic import semantic_hash_128
from mmhe_v1.hashing.strict import canonical_text_sha256
from mmhe_v1.reporting import write_result_artifacts
from mmhe_v1.types import ExperimentResult


def _deterministic_vector(text: str, *, seed: int, embedding_dim: int) -> list[float]:
    canonical = canonicalize_text(text).canonical_text
    values: list[float] = []
    counter = 0
    while len(values) < embedding_dim:
        digest = hashlib.sha256(f"{seed}:{canonical}:{counter}".encode("utf-8")).digest()
        for offset in range(0, len(digest), 4):
            chunk = digest[offset : offset + 4]
            if len(chunk) < 4:
                continue
            scaled = int.from_bytes(chunk, "big") / 0xFFFFFFFF
            values.append((scaled * 2.0) - 1.0)
            if len(values) == embedding_dim:
                break
        counter += 1
    return normalize_embedding(values)


def _build_text_embedding(text: str, *, config) -> list[float]:
    canonical = canonicalize_text(
        text,
        unicode_form=config.text_unicode_form,
        lowercase=config.text_lowercase,
        collapse_whitespace=config.text_collapse_whitespace,
    ).canonical_text
    if config.embedding_backend == "deterministic_stub":
        return _deterministic_vector(canonical, seed=config.seed, embedding_dim=config.embedding_dim)

    vector = encode_text(
        canonical,
        model_name=config.model_name,
        pretrained_tag=config.pretrained_tag,
        device="cpu",
    )
    if len(vector) != config.embedding_dim:
        raise ValueError(
            f"embedding dimension mismatch: expected {config.embedding_dim}, got {len(vector)}"
        )
    return vector


def _build_ckks_adapter(config) -> CKKSAdapter:
    tenseal_reason = detect_tenseal_unavailable_reason()
    if config.ckks.enabled:
        if tenseal_reason is not None:
            raise RuntimeError(f"TenSEAL is required but unavailable: {tenseal_reason}")
        return CKKSAdapter.from_tenseal(
            poly_modulus_degree=config.ckks.poly_modulus_degree,
            scaling_mod_size=config.ckks.scaling_mod_size,
        )
    return CKKSAdapter.make_test_double(reason=tenseal_reason or "backend disabled for v1")


def _rank_ids(query_vector: list[float], document_vectors: dict[str, list[float]]) -> tuple[list[str], list[float]]:
    scored = [
        (document_id, cosine_similarity(query_vector, vector))
        for document_id, vector in document_vectors.items()
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [item[0] for item in scored], [item[1] for item in scored]


def _rank_encrypted_ids(
    encrypted_query: object,
    encrypted_document_vectors: dict[str, object],
    *,
    adapter: CKKSAdapter,
) -> tuple[list[str], list[float]]:
    scored = [
        (document_id, adapter.encrypted_cosine_similarity(encrypted_query, encrypted_vector))
        for document_id, encrypted_vector in encrypted_document_vectors.items()
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [item[0] for item in scored], [item[1] for item in scored]


def run_experiment(config_path: str | Path, output_dir: str | Path | None = None) -> ExperimentResult:
    config = load_config(Path(config_path))
    resolved_output_dir = Path(output_dir) if output_dir is not None else config.paths.artifacts_root
    resolved_output_dir = resolved_output_dir.resolve()
    reports_dir = resolved_output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    sample_pairs = [
        ("a1", "A red car", "a   red car"),
        ("b2", "Blue ocean", "blue ocean"),
        ("c3", "Quiet forest", "quiet forest"),
    ]
    query_vectors = {
        sample_id: _build_text_embedding(query_text, config=config)
        for sample_id, query_text, _ in sample_pairs
    }
    document_vectors = {
        sample_id: _build_text_embedding(doc_text, config=config)
        for sample_id, _, doc_text in sample_pairs
    }

    adapter = _build_ckks_adapter(config)
    homomorphic_hasher = LtHash(seed=config.seed)
    commitment_scale = 1_000
    encrypted_document_vectors = {
        sample_id: adapter.encrypt(vector) for sample_id, vector in document_vectors.items()
    }

    plain_scores: list[float] = []
    ckks_scores: list[float] = []
    recalls: list[float] = []
    consistencies: list[float] = []
    input_commitments: list[float] = []
    result_commitments: list[float] = []
    example_vector_commitment: tuple[str, str] | None = None
    example_result_commitment: tuple[str, str] | None = None

    for sample_id, query_vector in query_vectors.items():
        plain_ranked_ids, plain_ranked_scores = _rank_ids(query_vector, document_vectors)
        encrypted_query = adapter.encrypt(query_vector)
        decrypted_query = adapter.decrypt(encrypted_query)
        ckks_ranked_ids, ckks_ranked_scores = _rank_encrypted_ids(
            encrypted_query,
            encrypted_document_vectors,
            adapter=adapter,
        )
        vector_commitment = verify_vector_commitment(
            query_vector,
            decrypted_query,
            lthash=homomorphic_hasher,
            scale=commitment_scale,
        )
        result_commitment = verify_ranked_result_commitment(
            plain_ranked_ids=plain_ranked_ids,
            plain_scores=plain_ranked_scores,
            encrypted_ranked_ids=ckks_ranked_ids,
            encrypted_scores=ckks_ranked_scores,
            lthash=homomorphic_hasher,
            scale=commitment_scale,
        )

        recalls.append(recall_at_k([sample_id], plain_ranked_ids, k=10))
        consistencies.append(rank_consistency(plain_ranked_ids, ckks_ranked_ids, k=10))
        plain_scores.extend(plain_ranked_scores)
        ckks_scores.extend(ckks_ranked_scores)
        input_commitments.append(1.0 if vector_commitment.consistent else 0.0)
        result_commitments.append(1.0 if result_commitment.consistent else 0.0)

        if example_vector_commitment is None:
            example_vector_commitment = (
                vector_commitment.plaintext_digest,
                vector_commitment.encrypted_digest,
            )
        if example_result_commitment is None:
            example_result_commitment = (
                result_commitment.plaintext_digest,
                result_commitment.encrypted_digest,
            )

    metric_snapshot = {
        "recall_at_10": round(sum(recalls) / len(recalls), 6),
        "mae": round(mean_absolute_error(plain_scores, ckks_scores), 6),
        "rank_consistency": round(sum(consistencies) / len(consistencies), 6),
        "homomorphic_input_consistency": round(sum(input_commitments) / len(input_commitments), 6),
        "homomorphic_result_consistency": round(sum(result_commitments) / len(result_commitments), 6),
    }

    first_sample_id, first_query_text, _ = sample_pairs[0]
    runtime_metadata = {
        "seed": config.seed,
        "model_name": config.model_name,
        "embedding_dim": config.embedding_dim,
        "embedding_backend": config.embedding_backend,
        "pretrained_tag": config.pretrained_tag,
        "hash.semantic_bits": config.hash.semantic_bits,
        "ckks_backend": adapter.backend_name.replace("-", "_"),
        "ckks_backend_status": adapter.backend_status,
        "ckks_backend_unavailable_reason": adapter.unavailable_reason,
        "config_path": str(Path(config_path).resolve()),
        "query_example_id": first_sample_id,
        "query_example_canonical_sha256": canonical_text_sha256(first_query_text),
        "query_example_semantic_hash": semantic_hash_128(
            query_vectors[first_sample_id], seed=config.hash.projection_seed
        ),
        "homomorphic_hash_backend": "lthash_style_additive",
        "homomorphic_hash_scale": commitment_scale,
        "homomorphic_hash_width": homomorphic_hasher.output_size,
        "query_example_plain_homomorphic_hash": example_vector_commitment[0] if example_vector_commitment else None,
        "query_example_ckks_homomorphic_hash": example_vector_commitment[1] if example_vector_commitment else None,
        "query_example_plain_result_hash": example_result_commitment[0] if example_result_commitment else None,
        "query_example_ckks_result_hash": example_result_commitment[1] if example_result_commitment else None,
        "metric_snapshot": metric_snapshot,
    }

    result = ExperimentResult(
        output_dir=resolved_output_dir,
        reports_dir=reports_dir,
        metric_snapshot=metric_snapshot,
        runtime_metadata=runtime_metadata,
        summary_path=reports_dir / "summary.md",
        result_json_path=reports_dir / "result.json",
    )
    return write_result_artifacts(result)

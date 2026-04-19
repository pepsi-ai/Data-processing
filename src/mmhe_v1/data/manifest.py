from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_ALLOWED_MODALITIES = {"image", "text", "audio", "video"}


class ManifestContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    sample_id: str
    modality: str
    source_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", self.sample_id.strip())
        object.__setattr__(self, "modality", self.modality.strip().lower())
        object.__setattr__(self, "source_path", Path(self.source_path))


def build_manifest_index(rows: Iterable[ManifestEntry]) -> dict[str, dict[str, Path]]:
    grouped: dict[str, dict[str, Path]] = {}
    for row in rows:
        if not isinstance(row, ManifestEntry):
            raise ManifestContractError("rows must contain ManifestEntry objects")
        if not row.sample_id:
            raise ManifestContractError("sample_id must not be empty")
        if not row.modality:
            raise ManifestContractError("modality must not be empty")
        if row.modality not in _ALLOWED_MODALITIES:
            raise ManifestContractError(
                f"modality must be one of {sorted(_ALLOWED_MODALITIES)}"
            )
        source_path_text = str(row.source_path).strip()
        if source_path_text in {"", "."}:
            raise ManifestContractError("source_path must not be empty or '.'")

        sample_bucket = grouped.setdefault(row.sample_id, {})
        if row.modality in sample_bucket:
            raise ManifestContractError(
                f"duplicate modality for sample_id={row.sample_id}: {row.modality}"
            )
        sample_bucket[row.modality] = row.source_path

    return grouped

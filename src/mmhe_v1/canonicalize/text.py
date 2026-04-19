from __future__ import annotations

import re
import unicodedata

from mmhe_v1.types import CanonicalText

_PUNCT_TRANSLATION = str.maketrans(
    {
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "：": ":",
        "；": ";",
        "“": "\"",
        "”": "\"",
        "‘": "'",
        "’": "'",
    }
)
_SPACE_BEFORE_ASCII_PUNCT = re.compile(r"\s+([,.;:!?])")


def normalize_text(
    text: str,
    *,
    unicode_form: str = "NFC",
    lowercase: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize(unicode_form, text)
    normalized = normalized.translate(_PUNCT_TRANSLATION)
    if collapse_whitespace:
        normalized = " ".join(normalized.split())
    normalized = _SPACE_BEFORE_ASCII_PUNCT.sub(r"\1", normalized)
    if lowercase:
        normalized = normalized.lower()
    return normalized


def canonicalize_text(
    text: str,
    *,
    unicode_form: str = "NFC",
    lowercase: bool = True,
    collapse_whitespace: bool = True,
) -> CanonicalText:
    normalized = normalize_text(
        text,
        unicode_form=unicode_form,
        lowercase=lowercase,
        collapse_whitespace=collapse_whitespace,
    )
    return CanonicalText(
        original_text=text,
        canonical_text=normalized,
        canonical_bytes=normalized.encode("utf-8"),
    )

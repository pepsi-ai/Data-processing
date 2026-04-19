from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Iterable


def _coerce_numeric_vector(vector: Iterable[float], *, field_name: str) -> list[float]:
    values = list(vector)
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise TypeError(f"{field_name} values must be numeric")
    return [float(value) for value in values]


@dataclass(slots=True)
class CKKSAdapter:
    backend_name: str
    backend_status: str = "available"
    unavailable_reason: str | None = None
    _tenseal: Any | None = field(default=None, repr=False)
    _context: Any | None = field(default=None, repr=False)

    @staticmethod
    def make_test_double(reason: str | None = None) -> "CKKSAdapter":
        status = "test double active"
        if reason:
            status = f"unavailable: {reason}"
        return CKKSAdapter(
            backend_name="test-double",
            backend_status=status,
            unavailable_reason=reason,
        )

    @classmethod
    def from_tenseal(
        cls,
        *,
        poly_modulus_degree: int = 8192,
        scaling_mod_size: int = 40,
    ) -> "CKKSAdapter":
        try:
            tenseal = importlib.import_module("tenseal")
        except Exception as error:
            raise RuntimeError("TenSEAL backend is unavailable") from error

        context = tenseal.context(
            tenseal.SCHEME_TYPE.CKKS,
            poly_modulus_degree,
            -1,
            [60, scaling_mod_size, scaling_mod_size, 60],
        )
        context.global_scale = 2**scaling_mod_size
        context.generate_galois_keys()
        context.generate_relin_keys()
        return cls(
            backend_name="tenseal",
            backend_status="available",
            unavailable_reason=None,
            _tenseal=tenseal,
            _context=context,
        )

    @property
    def is_test_double(self) -> bool:
        return self.backend_name == "test-double"

    def encrypt(self, vector: Iterable[float]) -> object:
        values = _coerce_numeric_vector(vector, field_name="encrypt payload")
        if self.is_test_double:
            return {
                "backend": self.backend_name,
                "payload": values,
            }
        if self._tenseal is None or self._context is None:
            raise RuntimeError("TenSEAL context is not initialized")
        return self._tenseal.ckks_vector(self._context, values)

    def decrypt(self, cipher: object) -> list[float]:
        if self.is_test_double:
            if not isinstance(cipher, dict):
                raise ValueError("cipher must be a mapping produced by CKKSAdapter.encrypt")
            if cipher.get("backend") != self.backend_name:
                raise ValueError("cipher backend does not match adapter backend")
            payload = cipher.get("payload")
            if not isinstance(payload, list) or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) for value in payload
            ):
                raise ValueError("cipher payload must be a numeric vector")
            return [float(value) for value in payload]

        if not hasattr(cipher, "decrypt"):
            raise ValueError("cipher must be a TenSEAL CKKS vector")
        plain = cipher.decrypt()
        return _coerce_numeric_vector(plain, field_name="decrypted payload")

    def encrypted_dot_product(self, left: object, right: object) -> object:
        if self.is_test_double:
            plain_left = self.decrypt(left)
            plain_right = self.decrypt(right)
            if len(plain_left) != len(plain_right):
                raise ValueError("vectors must have the same length")
            return self.encrypt([sum(a * b for a, b in zip(plain_left, plain_right))])

        if not hasattr(left, "dot") or not hasattr(right, "decrypt"):
            raise ValueError("left and right must be TenSEAL CKKS vectors")
        return left.dot(right)

    def encrypted_cosine_similarity(self, left: object, right: object) -> float:
        return float(self.decrypt(self.encrypted_dot_product(left, right))[0])


def detect_tenseal_unavailable_reason() -> str | None:
    try:
        importlib.import_module("tenseal")
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None

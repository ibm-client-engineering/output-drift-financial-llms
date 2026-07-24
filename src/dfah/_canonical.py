"""Canonical serialization, hashing, and artifact-safe redaction."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_SECRET_PATTERNS = (
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bsk-proj-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"),
)


def redact_text(value: str) -> str:
    """Replace common credential shapes without retaining a partial secret."""

    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return redacted


def to_jsonable(value: Any, *, redact: bool = True, path: str = "$") -> Any:
    """Convert a value to strict JSON data without a lossy ``repr`` fallback."""

    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(mode="python"), redact=redact, path=path)
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float at {path}")
        return value
    if isinstance(value, str):
        return redact_text(value) if redact else value
    if isinstance(value, Enum):
        return to_jsonable(value.value, redact=redact, path=path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string mapping key at {path}")
            result[key] = to_jsonable(item, redact=redact, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [
            to_jsonable(item, redact=redact, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"unsupported artifact value at {path}: {type(value).__name__}")


def canonical_bytes(value: Any, *, redact: bool = True) -> bytes:
    """Serialize strict JSON with stable key ordering and no NaN/Infinity."""

    return json.dumps(
        to_jsonable(value, redact=redact),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256(value: Any, *, redact: bool = False) -> str:
    """Hash a canonical value. Hashing is unredacted unless requested."""

    return hashlib.sha256(canonical_bytes(value, redact=redact)).hexdigest()


def contains_secret(value: Any) -> bool:
    """Return whether any serialized string matches a known credential shape."""

    raw = canonical_bytes(value, redact=False).decode("utf-8")
    return any(pattern.search(raw) is not None for pattern in _SECRET_PATTERNS)


def atomic_private_write(path: str | Path, payload: bytes) -> None:
    """Atomically write a mode-600 file without following a target symlink."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise OSError(f"unsafe artifact parent: {target.parent}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        parent_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()

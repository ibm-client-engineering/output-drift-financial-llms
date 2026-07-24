"""Recursively immutable JSON values for public records."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Annotated, Any, cast

from pydantic import AfterValidator, JsonValue, PlainSerializer


class FrozenDict(Mapping[str, Any]):
    """A small immutable mapping whose nested JSON containers are frozen too."""

    __slots__ = ("_data",)

    def __init__(self, values: Mapping[str, Any]):
        self._data = {key: freeze_json(value) for key, value in values.items()}

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"


def freeze_json(value: Any) -> Any:
    """Validate and recursively freeze a JSON value.

    Pydantic validates the public annotation first. This function is also used
    by :class:`FrozenDict` for nested values so callers cannot mutate a frozen
    model indirectly through a retained ``dict`` or ``list``.
    """

    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON values cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return FrozenDict(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(freeze_json(item) for item in value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def thaw_json(value: Any) -> JsonValue:
    """Convert immutable containers back to ordinary JSON containers."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return cast(JsonValue, value)


FrozenJson = Annotated[
    JsonValue,
    AfterValidator(freeze_json),
    PlainSerializer(thaw_json, return_type=JsonValue),
]

FrozenJsonMap = Annotated[
    Mapping[str, JsonValue],
    AfterValidator(freeze_json),
    PlainSerializer(thaw_json, return_type=dict[str, JsonValue]),
]

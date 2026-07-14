"""Strict UTF-8 JSON loading and deterministic serialization."""

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union


class StrictJsonError(ValueError):
    """Base class for project JSON errors."""


class DuplicateKeyError(StrictJsonError):
    """Raised when an object contains the same key more than once."""


class NonStandardConstantError(StrictJsonError):
    """Raised for NaN and infinity spellings."""


JsonInput = Union[str, bytes, bytearray]


def _reject_duplicates(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result = {}  # type: Dict[str, Any]
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError("duplicate JSON key: {0}".format(key))
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise NonStandardConstantError(
        "non-standard JSON numeric constant: {0}".format(value)
    )


def loads(data: JsonInput) -> Any:
    """Load strict JSON, rejecting invalid UTF-8, duplicate keys and constants."""

    if isinstance(data, (bytes, bytearray)):
        try:
            text = bytes(data).decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise StrictJsonError("JSON input is not valid UTF-8") from error
    elif isinstance(data, str):
        text = data
    else:
        raise TypeError("JSON input must be str, bytes, or bytearray")
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (DuplicateKeyError, NonStandardConstantError):
        raise
    except json.JSONDecodeError as error:
        raise StrictJsonError(str(error)) from error


def load(path: Union[str, Path]) -> Any:
    """Read a project JSON file as bytes and apply strict loading."""

    return loads(Path(path).read_bytes())


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StrictJsonError("nonfinite values cannot be serialized")
        return 0.0 if value == 0.0 else value
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        normalized = {}  # type: Dict[str, Any]
        for key, item in value.items():
            if not isinstance(key, str):
                raise StrictJsonError("JSON object keys must be strings")
            normalized[key] = _normalize(item)
        return normalized
    raise StrictJsonError(
        "unsupported deterministic JSON value: {0}".format(type(value).__name__)
    )


def dumps(value: Any) -> str:
    """Serialize using the repository deterministic JSON profile."""

    normalized = _normalize(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def dump_bytes(value: Any, trailing_newline: bool = True) -> bytes:
    """Return deterministic UTF-8 bytes, normally as a complete text file."""

    suffix = "\n" if trailing_newline else ""
    return (dumps(value) + suffix).encode("utf-8")

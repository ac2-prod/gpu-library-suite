"""SHA-256 helpers with exact-byte semantics."""

import hashlib
import stat
from pathlib import Path
from typing import Iterable, List, Union

from .strict_json import dump_bytes


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def deterministic_json_sha256(value: object, trailing_newline: bool = True) -> str:
    return sha256_bytes(dump_bytes(value, trailing_newline=trailing_newline))


SOURCE_SNAPSHOT_DOMAIN = b"gpu-library-suite-source-snapshot-v1\0"


def _length_prefix(value: int) -> bytes:
    if value < 0 or value >= 1 << 64:
        raise ValueError("source snapshot value is outside uint64 range")
    return value.to_bytes(8, byteorder="big", signed=False)


def source_snapshot_sha256(
    root: Union[str, Path], relative_paths: Iterable[str]
) -> str:
    """Hash a complete, ordered path/content/executable-bit source snapshot."""

    repository = Path(root).resolve()
    if not repository.is_dir() or repository.is_symlink():
        raise ValueError("source snapshot root must be a real directory")
    supplied = list(relative_paths)
    if len(supplied) != len(set(supplied)):
        raise ValueError("source snapshot paths contain duplicates")
    normalized = []  # type: List[str]
    for value in supplied:
        if not isinstance(value, str) or value == "" or "\x00" in value:
            raise ValueError("source snapshot path must be a nonempty string")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source snapshot path escapes the repository")
        canonical = relative.as_posix()
        if canonical in {"", "."}:
            raise ValueError("source snapshot path is invalid")
        canonical.encode("utf-8", errors="strict")
        normalized.append(canonical)
    if len(normalized) != len(set(normalized)):
        raise ValueError("source snapshot paths normalize to duplicates")
    normalized.sort()

    digest = hashlib.sha256()
    digest.update(SOURCE_SNAPSHOT_DOMAIN)
    digest.update(_length_prefix(len(normalized)))
    for relative_text in normalized:
        relative = Path(relative_text)
        path = repository
        for part in relative.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError(
                    "source snapshot cannot contain symbolic links: {0}".format(
                        relative_text
                    )
                )
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(
                "source snapshot entry is not a regular file: {0}".format(
                    relative_text
                )
            )
        content_digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                content_digest.update(block)
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or size != after.st_size:
            raise ValueError(
                "source snapshot entry changed while hashing: {0}".format(
                    relative_text
                )
            )
        path_bytes = relative_text.encode("utf-8")
        executable = 1 if before.st_mode & 0o111 else 0
        digest.update(_length_prefix(len(path_bytes)))
        digest.update(path_bytes)
        digest.update(bytes((executable,)))
        digest.update(_length_prefix(size))
        digest.update(content_digest.digest())
    return digest.hexdigest()

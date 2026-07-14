#!/usr/bin/env python3
"""Generate one deterministic executable manifest from CMake target metadata."""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from gpu_suite.hashing import sha256_bytes, sha256_file
from gpu_suite.manifest import artifact_id, validate_entry
from gpu_suite.strict_json import dump_bytes, load


class GenerationError(ValueError):
    pass


def _require_string(entry: Mapping[str, Any], name: str) -> str:
    value = entry.get(name)
    if not isinstance(value, str) or value == "":
        raise GenerationError("invalid target field: {0}".format(name))
    return value


def build_manifest(input_path: Path) -> Dict[str, Any]:
    descriptor = load(input_path)
    if not isinstance(descriptor, dict) or not isinstance(
        descriptor.get("entries"), list
    ):
        raise GenerationError("manifest input must contain an entries array")
    metadata_path_value = descriptor.get("build_metadata_path")
    if not isinstance(metadata_path_value, str):
        raise GenerationError("manifest input lacks build_metadata_path")
    metadata_path = Path(metadata_path_value)
    metadata_bytes = metadata_path.read_bytes()
    metadata = load(metadata_path)
    if not isinstance(metadata, dict):
        raise GenerationError("build metadata must be an object")
    metadata_sha256 = sha256_bytes(metadata_bytes)
    entries = []  # type: List[Dict[str, Any]]
    for source in descriptor["entries"]:
        if not isinstance(source, dict):
            raise GenerationError("target entry must be an object")
        path = Path(_require_string(source, "executable_path")).resolve()
        if not path.is_file():
            raise GenerationError("executable does not exist: {0}".format(path))
        language = _require_string(source, "compiler_language")
        compiler_metadata = metadata.get(language)
        if not isinstance(compiler_metadata, dict):
            raise GenerationError("missing compiler metadata for {0}".format(language))
        entry = {
            "artifact_id": "",
            "target_name": _require_string(source, "target_name"),
            "build_profile": _require_string(source, "build_profile"),
            "backend_variant": _require_string(source, "backend_variant"),
            "executable_path": str(path),
            "library": _require_string(source, "library"),
            "implementation": _require_string(source, "implementation"),
            "executable_role": _require_string(source, "executable_role"),
            "build_type": _require_string(metadata, "build_type"),
            "binary_sha256": sha256_file(path),
            "build_metadata_sha256": metadata_sha256,
            "compiler": _require_string(compiler_metadata, "compiler"),
            "compiler_language": language,
            "compiler_version": _require_string(
                compiler_metadata, "compiler_version"
            ),
            "git_commit": _require_string(metadata, "git_commit"),
            "git_dirty": metadata.get("git_dirty"),
            "supported_cpu_backends": source.get("cpu_backends", []),
        }
        if metadata.get("build_profile") != entry["build_profile"]:
            raise GenerationError("target and build metadata profiles differ")
        entry["artifact_id"] = artifact_id(entry)
        entries.append(validate_entry(entry))
    entries.sort(
        key=lambda item: (
            item["library"],
            item["implementation"],
            item["executable_role"],
            item["target_name"],
        )
    )
    return {
        "build_metadata_sha256": metadata_sha256,
        "entries": entries,
        "manifest_schema_version": 1,
    }


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(path))
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        print("usage: generate_partial_manifest.py INPUT OUTPUT", file=sys.stderr)
        return 2
    try:
        manifest = build_manifest(Path(arguments[0]))
        atomic_write(Path(arguments[1]), dump_bytes(manifest))
    except (OSError, ValueError) as error:
        print("partial manifest generation failed: {0}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

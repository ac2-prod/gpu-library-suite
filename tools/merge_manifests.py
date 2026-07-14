#!/usr/bin/env python3
"""Merge CPU/CUDA and OpenACC partial manifests without provider ambiguity."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from gpu_suite.hashing import sha256_bytes
from gpu_suite.manifest import ManifestError, merge_entries
from gpu_suite.strict_json import dump_bytes, load


def merge_manifests(paths: Sequence[Path]) -> Dict[str, Any]:
    partials = []  # type: List[List[Dict[str, Any]]]
    metadata_hashes = set()
    for path in paths:
        document = load(path)
        if not isinstance(document, dict) or document.get("manifest_schema_version") != 1:
            raise ManifestError("invalid partial manifest: {0}".format(path))
        entries = document.get("entries")
        if not isinstance(entries, list):
            raise ManifestError("partial manifest lacks entries")
        partials.append(entries)
        metadata_hash = document.get("build_metadata_sha256")
        if not isinstance(metadata_hash, str) or len(metadata_hash) != 64:
            raise ManifestError("invalid build metadata hash")
        metadata_hashes.add(metadata_hash)
    entries = merge_entries(partials)
    return {
        "build_metadata_sha256s": sorted(metadata_hashes),
        "entries": entries,
        "manifest_schema_version": 1,
    }


def exclusive_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("partials", nargs="+", type=Path)
    arguments = parser.parse_args(argv)
    try:
        document = merge_manifests(arguments.partials)
        content = dump_bytes(document)
        exclusive_write(arguments.output, content)
        print(sha256_bytes(content))
    except (OSError, ValueError) as error:
        print("manifest merge failed: {0}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Render cross-wave elapsed-time figures with raw Thrust version evidence."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from gpu_suite.hashing import sha256_bytes, sha256_file
from gpu_suite.plotting import build_plot_metadata, render_plots
from gpu_suite.schema import validate_raw_result
from gpu_suite.strict_json import dump_bytes, load, loads


def load_aggregate(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    records = []  # type: List[Dict[str, Any]]
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if line == "":
                raise ValueError(
                    "blank aggregate line in {0}:{1}".format(path, line_number)
                )
            value = loads(line)
            if not isinstance(value, dict):
                raise ValueError("aggregate record must be an object")
            records.append(value)
    if not records:
        raise ValueError("aggregate input is empty")
    return records


def load_raw_results(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    records = []  # type: List[Dict[str, Any]]
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line == "":
                raise ValueError(
                    "blank raw-result line in {0}:{1}".format(path, line_number)
                )
            value = loads(line)
            if not isinstance(value, dict):
                raise ValueError("raw-result record must be an object")
            records.append(validate_raw_result(value))
    if not records:
        raise ValueError("raw-result input is empty")
    return records


def exclusive_write(path: Path, content: bytes) -> None:
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("aggregate_results", nargs="+", type=Path)
    parser.add_argument("--raw-results", required=True, nargs="+", type=Path)
    parser.add_argument("--display-config", type=Path)
    parser.add_argument("--node-metadata", nargs="+", type=Path, default=[])
    arguments = parser.parse_args(argv)
    try:
        if not arguments.output_directory.is_dir():
            raise ValueError("output directory must already exist")
        if arguments.metadata_output.exists():
            raise ValueError("plot metadata output already exists")
        source = b"".join(path.read_bytes() for path in arguments.aggregate_results)
        records = load_aggregate(arguments.aggregate_results)
        raw_records = load_raw_results(arguments.raw_results)
        metadata = build_plot_metadata(
            records, sha256_bytes(source), raw_records=raw_records,
            display_configuration=(load(arguments.display_config)
                                   if arguments.display_config else None),
            node_metadata=[load(path) for path in arguments.node_metadata],
        )
        metadata["display_config_sha256"] = (
            sha256_file(arguments.display_config) if arguments.display_config else None
        )
        metadata["node_metadata_sha256s"] = [
            sha256_file(path) for path in arguments.node_metadata
        ]
        metadata = render_plots(metadata, arguments.output_directory)
        exclusive_write(arguments.metadata_output, dump_bytes(metadata))
        if metadata["matplotlib"]["status"] == "unexecuted":
            print(metadata["matplotlib"]["reason"], file=sys.stderr)
    except (OSError, ValueError) as error:
        print("plotting failed: {0}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

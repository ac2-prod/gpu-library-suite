#!/usr/bin/env python3
"""Create block, wave, cross-wave, and optional pooled aggregate records."""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from gpu_suite.aggregation import aggregate_results
from gpu_suite.config import load_config
from gpu_suite.hashing import sha256_file
from gpu_suite.results_io import load_raw_results
from gpu_suite.strict_json import dump_bytes, dumps


def exclusive_write(path: Path, content: bytes) -> None:
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--no-pooled", action="store_true")
    parser.add_argument("raw_results", nargs="+", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.output.exists() or arguments.metadata_output.exists():
            raise ValueError("aggregate output already exists")
        config = load_config(arguments.config)
        raw = []
        for path in arguments.raw_results:
            raw.extend(load_raw_results(path))
        run_ids = {record["run_id"] for record in raw}
        runtime_hashes = {record["runtime_environment_sha256"] for record in raw}
        config_hashes = {record["config_sha256"] for record in raw}
        if len(run_ids) != 1 or len(runtime_hashes) != 1 or len(config_hashes) != 1:
            raise ValueError("aggregate input has mixed campaign provenance")
        if config_hashes != {sha256_file(arguments.config)}:
            raise ValueError("aggregate input configuration hash mismatch")
        records, metadata = aggregate_results(raw, config, not arguments.no_pooled)
        content = "".join(dumps(record) + "\n" for record in records).encode("utf-8")
        exclusive_write(arguments.output, content)
        exclusive_write(arguments.metadata_output, dump_bytes(metadata))
    except (OSError, ValueError) as error:
        print("aggregation failed: {0}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

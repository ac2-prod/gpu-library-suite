#!/usr/bin/env python3
"""Validate node raw results and complete campaign/provenance invariants."""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from gpu_suite.config import load_config
from gpu_suite.hashing import sha256_file
from gpu_suite.runner import load_manifest
from gpu_suite.results_io import load_raw_results
from gpu_suite.strict_json import dump_bytes, dumps
from gpu_suite.validation import validate_campaign


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("raw_results", nargs="+", type=Path)
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config)
        manifest, _ = load_manifest(arguments.manifest)
        records = []
        for path in arguments.raw_results:
            records.extend(load_raw_results(path))
        report = validate_campaign(
            records, config, sha256_file(arguments.config), manifest
        )
        content = dump_bytes(report)
        if arguments.report_output is None:
            sys.stdout.buffer.write(content)
        else:
            descriptor = os.open(
                str(arguments.report_output),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
        return 0 if report["validation_status"] == "pass" else 1
    except (OSError, ValueError) as error:
        print("result validation failed: {0}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Run one node's interleaved benchmark suite or print a side-effect-free plan."""

import argparse
import socket
import sys
from pathlib import Path
from typing import Optional, Sequence

from gpu_suite.config import load_config
from gpu_suite.hashing import sha256_file
from gpu_suite.ordering import implementation_order
from gpu_suite.runner import (
    build_schedule,
    dry_run_document,
    execute_schedule,
    load_build_metadata,
    load_manifest,
    validate_execution_context,
    validate_manifest_artifacts,
    validate_sha256,
)
from gpu_suite.strict_json import dumps


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--build-metadata", required=True, action="append", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--system-label", required=True)
    parser.add_argument("--wave", required=True, type=int)
    parser.add_argument("--node-index", required=True, type=int)
    parser.add_argument("--hostname", default=socket.gethostname())
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runtime-environment-sha256", required=True)
    parser.add_argument("--git-diff-sha256")
    parser.add_argument("--source-snapshot-sha256")
    parser.add_argument("--scheduler")
    parser.add_argument("--scheduler-job-id")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        validate_execution_context(
            arguments.run_id, arguments.system_label, arguments.hostname,
            arguments.wave, arguments.node_index, arguments.device,
        )
        validate_sha256(
            arguments.runtime_environment_sha256,
            "runtime environment SHA-256",
        )
        if arguments.git_diff_sha256 is not None:
            validate_sha256(arguments.git_diff_sha256, "Git diff SHA-256")
        if arguments.source_snapshot_sha256 is not None:
            validate_sha256(
                arguments.source_snapshot_sha256, "source snapshot SHA-256"
            )
        config = load_config(arguments.config)
        manifest, manifest_sha256 = load_manifest(arguments.manifest)
        metadata = load_build_metadata(arguments.build_metadata)
        dirty_hash_available = bool(
            arguments.git_diff_sha256 or arguments.source_snapshot_sha256
        )
        validate_manifest_artifacts(
            manifest, metadata, config["run_mode"], dirty_hash_available
        )
        context = {
            "config": config,
            "cpu_threads": config["cpu_threads"],
            "device": arguments.device,
            "hostname": arguments.hostname,
            "implementation_order": implementation_order(
                arguments.node_index, arguments.wave
            ),
            "node_index": arguments.node_index,
            "run_id": arguments.run_id,
            "scheduler": arguments.scheduler,
            "scheduler_job_id": arguments.scheduler_job_id,
            "system_label": arguments.system_label,
            "wave": arguments.wave,
        }
        schedule = build_schedule(config, manifest, context)
        config_sha256 = sha256_file(arguments.config)
        if arguments.dry_run:
            document = dry_run_document(
                config, manifest, schedule, context, manifest_sha256,
                config_sha256,
            )
            sys.stdout.write(dumps(document) + "\n")
            return 0
        if arguments.output is None:
            raise ValueError("--output is required unless --dry-run is used")
        return execute_schedule(
            schedule, context, arguments.output, config_sha256,
            arguments.runtime_environment_sha256, metadata,
            arguments.git_diff_sha256, arguments.source_snapshot_sha256,
        )
    except (OSError, ValueError) as error:
        print("run_suite failed: {0}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

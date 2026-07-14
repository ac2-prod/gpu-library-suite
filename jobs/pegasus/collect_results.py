#!/usr/bin/env python3
"""Audit all expected node statuses and hash artifacts without rewriting raw data."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from gpu_suite.hashing import sha256_file  # noqa: E402
from gpu_suite.strict_json import dump_bytes, load  # noqa: E402


class CollectionError(ValueError):
    pass


def _validate_node_status(
    status: Mapping[str, Any], run_id: str, wave: int, rank: int, hostname: str,
) -> None:
    expected = {
        "node_status_schema_version": 1,
        "run_id": run_id,
        "wave": wave,
        "node_index": rank,
        "hostname": hostname,
        "block_id": "{0}|{1}|{2}".format(run_id, wave, hostname),
    }
    for name, value in expected.items():
        if status.get(name) != value:
            raise CollectionError(
                "node status mismatch for {0}/{1}".format(hostname, name)
            )
    for name in (
        "benchmark_status", "verification_status", "collection_status",
        "raw_result_collection_status", "log_collection_status",
    ):
        if not isinstance(status.get(name), str):
            raise CollectionError("node status lacks " + name)


def _artifact_records(nodes_directory: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    artifacts = []
    failures = []
    for path in sorted(nodes_directory.rglob("*")):
        relative = str(path.relative_to(nodes_directory))
        if path.is_symlink():
            failures.append("symbolic link is not an accepted artifact: " + relative)
        elif path.is_file():
            artifacts.append({
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            })
    return artifacts, failures


def collect_results(
    wave_metadata_path: Path, nodes_directory: Path,
) -> Tuple[Dict[str, Any], bool]:
    wave_metadata = load(wave_metadata_path)
    if (
        not isinstance(wave_metadata, dict)
        or wave_metadata.get("wave_metadata_schema_version") != 1
    ):
        raise CollectionError("invalid wave metadata")
    mapping = wave_metadata.get("rank_host_mapping")
    if not isinstance(mapping, list):
        raise CollectionError("wave metadata lacks rank-host mapping")
    run_id = wave_metadata["run_id"]
    wave = wave_metadata["wave"]
    expected_hosts = {item["hostname"] for item in mapping}
    failures = []  # type: List[str]
    node_summaries = []
    if not nodes_directory.is_dir() or nodes_directory.is_symlink():
        raise CollectionError("nodes directory is unavailable")
    observed_hosts = {
        path.name for path in nodes_directory.iterdir()
        if path.is_dir() and not path.is_symlink()
    }
    for hostname in sorted(observed_hosts.difference(expected_hosts)):
        failures.append("unexpected node output: " + hostname)
    for item in sorted(mapping, key=lambda value: value["rank"]):
        rank = item["rank"]
        hostname = item["hostname"]
        node_directory = nodes_directory / hostname
        status_path = node_directory / "node-status.json"
        summary = {"hostname": hostname, "node_index": rank, "status": None}
        if not node_directory.is_dir() or node_directory.is_symlink():
            failures.append("missing node directory: " + hostname)
            summary["status"] = "missing"
            node_summaries.append(summary)
            continue
        if not status_path.is_file() or status_path.is_symlink():
            failures.append("missing node status: " + hostname)
            summary["status"] = "missing-status"
            node_summaries.append(summary)
            continue
        try:
            status = load(status_path)
            if not isinstance(status, dict):
                raise CollectionError("node status is not an object")
            _validate_node_status(status, run_id, wave, rank, hostname)
        except (OSError, ValueError) as error:
            failures.append("invalid node status {0}: {1}".format(hostname, error))
            summary["status"] = "invalid-status"
            node_summaries.append(summary)
            continue
        summary.update({
            "benchmark_status": status["benchmark_status"],
            "collection_status": status["collection_status"],
            "status": status.get("status"),
            "telemetry_status": status.get("telemetry_status"),
            "tool_status": status.get("tool_status"),
            "verification_status": status["verification_status"],
        })
        if status["benchmark_status"] != "success":
            failures.append("benchmark failure: " + hostname)
        if status["verification_status"] != "success":
            failures.append("verification failure: " + hostname)
        if (
            status["collection_status"] != "success"
            or status["raw_result_collection_status"] != "success"
            or status["log_collection_status"] != "success"
        ):
            failures.append("artifact collection failure: " + hostname)
        raw_files = [
            path for path in (
                node_directory / "raw-results.jsonl",
                node_directory / "raw-results.csv",
            ) if path.is_file() and not path.is_symlink()
        ]
        if len(raw_files) != 1:
            failures.append("node must contain exactly one raw result file: " + hostname)
        node_summaries.append(summary)
    artifacts, artifact_failures = _artifact_records(nodes_directory)
    failures.extend(artifact_failures)
    success = not failures
    document = {
        "artifact_count": len(artifacts),
        "artifact_manifest_schema_version": 1,
        "artifacts": artifacts,
        "collection_status": "success" if success else "failure",
        "expected_node_count": len(mapping),
        "failures": failures,
        "node_summaries": node_summaries,
        "run_id": run_id,
        "wave": wave,
    }
    return document, success


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave-metadata", required=True, type=Path)
    parser.add_argument("--nodes-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        document, success = collect_results(
            arguments.wave_metadata, arguments.nodes_directory
        )
        with arguments.output.open("xb") as stream:
            stream.write(dump_bytes(document))
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, ValueError) as error:
        print("result collection failed: {0}".format(error), file=sys.stderr)
        return 1
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Validate campaign provenance and exclusively create one Pegasus wave."""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from gpu_suite.config import load_config  # noqa: E402
from gpu_suite.hashing import sha256_file  # noqa: E402
from gpu_suite.ordering import (  # noqa: E402
    assignment_counts,
    implementation_order,
    permutation_index,
    size_order_index,
)
from gpu_suite.runner import (  # noqa: E402
    load_build_metadata,
    load_manifest,
    utc_timestamp,
    validate_execution_context,
    validate_manifest_artifacts,
    validate_sha256,
)
from gpu_suite.scheduler import validate_scheduler_identity  # noqa: E402
from gpu_suite.strict_json import dump_bytes, load  # noqa: E402


class WavePreparationError(ValueError):
    pass


HOSTNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")


def _exclusive_write(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _require_deterministic(path: Path, document: Any, name: str) -> bytes:
    content = path.read_bytes()
    if content != dump_bytes(document):
        raise WavePreparationError(name + " does not use deterministic JSON")
    return content


def parse_rank_host_mapping(path: Path, expected_nodes: int) -> List[Dict[str, Any]]:
    mapping = []
    seen_ranks = set()
    seen_hosts = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 2:
            raise WavePreparationError(
                "invalid rank-host mapping line {0}".format(line_number)
            )
        try:
            rank = int(fields[0])
        except ValueError as error:
            raise WavePreparationError("mapping rank is not an integer") from error
        hostname = fields[1]
        if rank < 0 or rank in seen_ranks:
            raise WavePreparationError("duplicate or negative mapping rank")
        if HOSTNAME_RE.fullmatch(hostname) is None or hostname.startswith("."):
            raise WavePreparationError("unsafe mapped hostname")
        if hostname in seen_hosts:
            raise WavePreparationError("one node appears at more than one rank")
        seen_ranks.add(rank)
        seen_hosts.add(hostname)
        mapping.append({"hostname": hostname, "rank": rank})
    if seen_ranks != set(range(expected_nodes)):
        raise WavePreparationError("rank-host mapping is incomplete")
    mapping.sort(key=lambda item: item["rank"])
    return mapping


def _git_provenance(
    manifest: Mapping[str, Any], git_diff_sha256: Optional[str],
    source_snapshot_sha256: Optional[str], run_mode: str,
) -> Dict[str, Any]:
    availability_values = {
        entry["git_metadata_available"] for entry in manifest["entries"]
    }
    commits = {entry["git_commit"] for entry in manifest["entries"]}
    dirty_values = {entry["git_dirty"] for entry in manifest["entries"]}
    if (len(availability_values) != 1 or len(commits) != 1 or
            len(dirty_values) != 1):
        raise WavePreparationError("manifest has mixed Git provenance")
    available = next(iter(availability_values))
    dirty = next(iter(dirty_values))
    if git_diff_sha256 is not None and source_snapshot_sha256 is not None:
        raise WavePreparationError("select one authoritative dirty-source hash")
    if not available:
        if run_mode == "production":
            raise WavePreparationError(
                "production requires available Git metadata"
            )
        if git_diff_sha256 is not None:
            raise WavePreparationError(
                "Git-unavailable source cannot use a Git-diff hash"
            )
        if source_snapshot_sha256 is None:
            raise WavePreparationError(
                "Git-unavailable source requires a source-snapshot hash"
            )
        dirty_source = {
            "kind": "source-snapshot",
            "sha256": source_snapshot_sha256,
        }
    elif dirty:
        if run_mode == "production":
            raise WavePreparationError("production requires a clean worktree")
        if git_diff_sha256 is None and source_snapshot_sha256 is None:
            raise WavePreparationError("dirty source requires an authoritative hash")
        dirty_source = {
            "kind": "git-diff" if git_diff_sha256 is not None else "source-snapshot",
            "sha256": git_diff_sha256 or source_snapshot_sha256,
        }
    else:
        if git_diff_sha256 is not None or source_snapshot_sha256 is not None:
            raise WavePreparationError("clean source cannot carry a dirty-source hash")
        dirty_source = None
    return {
        "dirty_source_provenance": dirty_source,
        "git_metadata_available": available,
        "git_commit": next(iter(commits)),
        "git_dirty": dirty,
    }


def _immutable_run_metadata(
    run_id: str, system_label: str, config: Mapping[str, Any],
    config_sha256: str, manifest_sha256: str, runtime_sha256: str,
    git: Mapping[str, Any], submission_host: str, timestamp: str,
) -> Dict[str, Any]:
    return {
        "campaign_creation_timestamp": timestamp,
        "dirty_source_provenance": git["dirty_source_provenance"],
        "effective_config_sha256": config_sha256,
        "executables_manifest_sha256": manifest_sha256,
        "git_metadata_available": git["git_metadata_available"],
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
        "launcher": {"name": "prepare_wave.py", "schema_version": 1},
        "run_id": run_id,
        "run_metadata_schema_version": 1,
        "run_mode": config["run_mode"],
        "runtime_environment_sha256": runtime_sha256,
        "submission_host": submission_host,
        "system_label": system_label,
    }


def _validate_existing_campaign(
    run_root: Path, expected: Mapping[str, Any], config_sha256: str,
    manifest_sha256: str, runtime_sha256: str,
) -> None:
    metadata = load(run_root / "run-metadata.json")
    for name, value in expected.items():
        if name == "campaign_creation_timestamp":
            continue
        if metadata.get(name) != value:
            suffix = (
                "; runtime environment changed and requires a new run ID"
                if name == "runtime_environment_sha256" else ""
            )
            raise WavePreparationError(
                "existing campaign provenance mismatch for {0}{1}".format(name, suffix)
            )
    hashes = {
        "effective-config.json": config_sha256,
        "executables-manifest.json": manifest_sha256,
        "runtime-environment.json": runtime_sha256,
    }
    for name, digest in hashes.items():
        path = run_root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise WavePreparationError("existing campaign artifact mismatch: " + name)


def prepare_wave(
    result_root: Path, run_id: str, wave: int, expected_nodes: int,
    mapping_path: Path, config_path: Path, manifest_path: Path,
    build_metadata_paths: Sequence[Path], runtime_environment_path: Path,
    system_label: str, scheduler: str, scheduler_job_id: str,
    submission_host: str, git_diff_sha256: Optional[str] = None,
    source_snapshot_sha256: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    validate_execution_context(run_id, system_label, submission_host, wave, 0, 0)
    if (
        not isinstance(expected_nodes, int)
        or isinstance(expected_nodes, bool)
        or expected_nodes <= 0
        or expected_nodes > 150
    ):
        raise WavePreparationError("expected node count must be positive")
    try:
        validate_scheduler_identity(scheduler, scheduler_job_id)
    except ValueError as error:
        raise WavePreparationError(str(error)) from error
    if not result_root.is_dir() or result_root.is_symlink():
        raise WavePreparationError("shared result root must be an existing real directory")
    mapping = parse_rank_host_mapping(mapping_path, expected_nodes)
    config = load_config(config_path)
    config_content = _require_deterministic(config_path, config, "effective config")
    manifest, manifest_sha256 = load_manifest(manifest_path)
    manifest_content = _require_deterministic(
        manifest_path, manifest, "executable manifest"
    )
    runtime_environment = load(runtime_environment_path)
    if (
        not isinstance(runtime_environment, dict)
        or runtime_environment.get("runtime_environment_schema_version") != 1
    ):
        raise WavePreparationError("invalid runtime environment document")
    runtime_content = _require_deterministic(
        runtime_environment_path, runtime_environment, "runtime environment"
    )
    if runtime_environment.get("executables_manifest_sha256") != manifest_sha256:
        raise WavePreparationError(
            "runtime environment was collected for a different manifest"
        )
    config_sha256 = sha256_file(config_path)
    runtime_sha256 = sha256_file(runtime_environment_path)
    if git_diff_sha256 is not None:
        validate_sha256(git_diff_sha256, "Git diff SHA-256")
    if source_snapshot_sha256 is not None:
        validate_sha256(source_snapshot_sha256, "source snapshot SHA-256")
    dirty_hash_available = bool(git_diff_sha256 or source_snapshot_sha256)
    metadata_by_hash = load_build_metadata(build_metadata_paths)
    runtime_metadata_hashes = {
        item.get("sha256")
        for item in runtime_environment.get("build_metadata", [])
        if isinstance(item, dict)
    }
    if runtime_metadata_hashes != set(metadata_by_hash):
        raise WavePreparationError(
            "runtime environment build metadata hashes do not match"
        )
    validate_manifest_artifacts(
        manifest, metadata_by_hash, config["run_mode"], dirty_hash_available
    )
    git = _git_provenance(
        manifest, git_diff_sha256, source_snapshot_sha256, config["run_mode"]
    )
    now = timestamp or utc_timestamp()
    expected_run_metadata = _immutable_run_metadata(
        run_id, system_label, config, config_sha256, manifest_sha256,
        runtime_sha256, git, submission_host, now,
    )
    run_root = result_root / run_id
    if run_root.exists():
        if not run_root.is_dir() or run_root.is_symlink():
            raise WavePreparationError("run path is not a real directory")
        _validate_existing_campaign(
            run_root, expected_run_metadata, config_sha256, manifest_sha256,
            runtime_sha256,
        )
    else:
        run_root.mkdir()
        _exclusive_write(run_root / "effective-config.json", config_content)
        _exclusive_write(run_root / "executables-manifest.json", manifest_content)
        _exclusive_write(run_root / "runtime-environment.json", runtime_content)
        _exclusive_write(
            run_root / "run-metadata.json", dump_bytes(expected_run_metadata)
        )
        (run_root / "waves").mkdir()

    waves_root = run_root / "waves"
    if not waves_root.is_dir() or waves_root.is_symlink():
        raise WavePreparationError("campaign waves directory is invalid")
    wave_root = waves_root / str(wave)
    if wave_root.exists():
        raise WavePreparationError("wave output already exists")
    wave_root.mkdir()
    (wave_root / "nodes").mkdir()
    implementation_assignments = []
    size_assignments = []
    for item in mapping:
        rank = item["rank"]
        implementation_assignments.append({
            "hostname": item["hostname"],
            "implementation_order": list(implementation_order(rank, wave)),
            "permutation_index": permutation_index(rank, wave),
            "rank": rank,
        })
        size_assignments.append({
            "hostname": item["hostname"],
            "rank": rank,
            "size_order_index": size_order_index(rank),
        })
    counts = assignment_counts(expected_nodes, wave)
    wave_metadata = {
        "effective_config_sha256": config_sha256,
        "executables_manifest_sha256": manifest_sha256,
        "expected_node_count": expected_nodes,
        "implementation_assignments": implementation_assignments,
        "observed_node_count": len(mapping),
        "output_collision_validation": "pass",
        "permutation_assignment_counts": counts["permutation_assignment_counts"],
        "rank_host_mapping": mapping,
        "run_id": run_id,
        "runtime_environment_sha256": runtime_sha256,
        "scheduler": scheduler,
        "scheduler_job_id": scheduler_job_id,
        "size_order_assignment_counts": counts["size_order_assignment_counts"],
        "size_order_assignments": size_assignments,
        "timestamp": now,
        "wave": wave,
        "wave_metadata_schema_version": 1,
    }
    _exclusive_write(wave_root / "wave-metadata.json", dump_bytes(wave_metadata))
    return wave_metadata


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--wave", required=True, type=int)
    parser.add_argument("--expected-nodes", required=True, type=int)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--build-metadata", required=True, action="append", type=Path)
    parser.add_argument("--runtime-environment", required=True, type=Path)
    parser.add_argument("--system-label", required=True)
    parser.add_argument("--scheduler", required=True)
    parser.add_argument("--scheduler-job-id", required=True)
    parser.add_argument("--submission-host", required=True)
    parser.add_argument("--git-diff-sha256")
    parser.add_argument("--source-snapshot-sha256")
    arguments = parser.parse_args(argv)
    try:
        prepare_wave(
            arguments.result_root, arguments.run_id, arguments.wave,
            arguments.expected_nodes, arguments.mapping, arguments.config,
            arguments.manifest, arguments.build_metadata,
            arguments.runtime_environment, arguments.system_label,
            arguments.scheduler, arguments.scheduler_job_id,
            arguments.submission_host, arguments.git_diff_sha256,
            arguments.source_snapshot_sha256,
        )
    except (OSError, ValueError) as error:
        print("wave preparation failed: {0}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

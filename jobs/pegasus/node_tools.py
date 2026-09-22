#!/usr/bin/env python3
"""Create deterministic node metadata, raw classification, and final status."""

import argparse
import csv
import os
import shutil
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from gpu_suite.config import BENCHMARKS, load_config  # noqa: E402
from gpu_suite.ordering import (  # noqa: E402
    implementation_order,
    permutation_index,
    size_order_index,
)
from gpu_suite.results_io import load_raw_results  # noqa: E402
from gpu_suite.runner import (  # noqa: E402
    load_manifest,
    utc_timestamp,
    validate_execution_context,
    validate_sha256,
)
from gpu_suite.scheduler import validate_scheduler_identity  # noqa: E402
from gpu_suite.strict_json import dump_bytes, load, loads  # noqa: E402
from collect_runtime_environment import command_record, CommandExecutor  # noqa: E402
from cuda_runtime_probe import (  # noqa: E402
    CUDA_RUNTIME_PROBE_KEYS,
    CudaRuntimeProbe,
    probe_node_cuda_runtime,
)
from job_config import CPU_ENVIRONMENT_KEYS  # noqa: E402


class NodeToolError(ValueError):
    pass


def _exclusive_write(path: Path, document: Mapping[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(dump_bytes(document))


def _cpu_environment(environment: Mapping[str, str]) -> Dict[str, str]:
    result = {}
    for name in CPU_ENVIRONMENT_KEYS:
        value = environment.get(name)
        if value is None or value == "":
            raise NodeToolError("missing CPU runtime variable " + name)
        result[name] = value
    return result


def build_node_metadata(
    config_path: Path, manifest_path: Path, run_id: str, wave: int,
    node_index: int, hostname: str, runtime_environment_sha256: str,
    environment: Mapping[str, str] = os.environ,
    cuda_runtime_probe: CudaRuntimeProbe = probe_node_cuda_runtime,
    scheduler: Optional[str] = None,
    scheduler_job_id: Optional[str] = None,
) -> Dict[str, Any]:
    validate_execution_context(run_id, "node-metadata", hostname, wave, node_index, 0)
    validate_sha256(runtime_environment_sha256, "runtime environment SHA-256")
    validate_scheduler_identity(scheduler, scheduler_job_id)
    config = load_config(config_path)
    manifest, manifest_sha256 = load_manifest(manifest_path)
    order_index = size_order_index(node_index)
    size_order = {}
    series = {}
    cpu_series = []
    for benchmark in BENCHMARKS:
        definition = config["benchmarks"][benchmark]
        cases = list(definition["cases"])
        if order_index == 1:
            cases.reverse()
        size_order[benchmark] = [dict(case["parameters"]) for case in cases]
        series[benchmark] = [dict(item) for item in definition["series"]]
        for item in definition["series"]:
            backend = item["cpu_backend"]
            if item["implementation"] == "cpu":
                cpu_series.append({
                    "benchmark": benchmark,
                    "cpu_backend": backend,
                    "cpu_parallelism": item["cpu_parallelism"],
                    "effective_threads": item["cpu_threads_effective"],
                    "requested_threads": config["cpu_threads"],
                    "series_role": item["series_role"],
                })
    artifacts = [
        {
            "artifact_id": entry["artifact_id"],
            "binary_sha256": entry["binary_sha256"],
            "build_profile": entry["build_profile"],
            "implementation": entry["implementation"],
            "library": entry["library"],
            "target_name": entry["target_name"],
        }
        for entry in manifest["entries"]
        if entry["executable_role"] == "benchmark"
    ]
    artifacts.sort(key=lambda item: item["artifact_id"])
    curand_parameters = config["benchmarks"]["curand"]["cases"][0]["parameters"]
    cuda_runtime_identity = dict(cuda_runtime_probe())
    if set(cuda_runtime_identity) != CUDA_RUNTIME_PROBE_KEYS:
        raise NodeToolError("invalid node-local CUDA runtime probe result")
    if cuda_runtime_identity["query_status"] not in {
        "success", "failure", "unavailable",
    }:
        raise NodeToolError("invalid node-local CUDA runtime probe status")
    return {
        "artifacts": artifacts,
        "block_id": "{0}|{1}|{2}".format(run_id, wave, hostname),
        "cpu_runtime_environment": _cpu_environment(environment),
        "cpu_series_threading": cpu_series,
        "cuda_runtime_identity": cuda_runtime_identity,
        "gpu_identity": {
            "diagnostic": environment.get(
                "GPU_SUITE_NODE_GPU_QUERY_DIAGNOSTIC"
            ) or None,
            "name": environment.get("GPU_SUITE_GPU_NAME") or None,
            "nvidia_driver_version": environment.get(
                "GPU_SUITE_NVIDIA_DRIVER_VERSION"
            ) or None,
            "query_status": environment.get(
                "GPU_SUITE_NODE_GPU_QUERY_STATUS", "unavailable"
            ),
            "uuid": environment.get("GPU_SUITE_GPU_UUID") or None,
        },
        "curand": {
            "cpu_engine": "std::mt19937_64",
            "cuda_generator": curand_parameters["generator"],
            "distribution": curand_parameters["distribution"],
            "offset": curand_parameters["offset"],
            "order": curand_parameters["order"],
            "seed": curand_parameters["seed"],
        },
        "executables_manifest_sha256": manifest_sha256,
        "hostname": hostname,
        "implementation_order": list(implementation_order(node_index, wave)),
        "node_index": node_index,
        "node_metadata_schema_version": 1,
        "permutation_index": permutation_index(node_index, wave),
        "requested_cpu_threads": config["cpu_threads"],
        "run_id": run_id,
        "runtime_environment_sha256": runtime_environment_sha256,
        "scheduler": scheduler,
        "scheduler_job_id": scheduler_job_id,
        "series": series,
        "size_order": size_order,
        "size_order_index": order_index,
        "wave": wave,
    }


def _local_executor(arguments: Sequence[str]):
    environment = dict(os.environ)
    environment["LC_ALL"] = "C"
    completed = subprocess.run(
        list(arguments), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=environment, check=False,
    )
    return completed.returncode, completed.stdout.decode("utf-8", errors="strict")


def build_local_node_metadata(
    config_path: Path, manifest_path: Path, run_id: str, wave: int,
    runtime_environment_sha256: str, gpu_uuid: Optional[str] = None,
    environment: Mapping[str, str] = os.environ,
    executor: CommandExecutor = _local_executor,
    which=shutil.which,
    cuda_runtime_probe: CudaRuntimeProbe = probe_node_cuda_runtime,
) -> Dict[str, Any]:
    """Observe one local worker, without scheduler or MPI identity fabrication."""
    selected_environment = dict(environment)
    for name in (
        "GPU_SUITE_GPU_NAME", "GPU_SUITE_GPU_UUID", "GPU_SUITE_NVIDIA_DRIVER_VERSION",
        "GPU_SUITE_NODE_GPU_QUERY_STATUS", "GPU_SUITE_NODE_GPU_QUERY_DIAGNOSTIC",
    ):
        selected_environment.pop(name, None)
    cpu_probe = command_record(["lscpu", "--json"], executor, which)
    cpu_name = None
    if cpu_probe["status"] == "success":
        try:
            cpu = loads("\n".join(cpu_probe["output"]))
            names = [item["data"] for item in cpu["lscpu"]
                     if item.get("field", "").strip().rstrip(":") == "Model name"]
            if len(names) == 1 and isinstance(names[0], str) and names[0].strip():
                cpu_name = names[0].strip()
        except (ValueError, KeyError, TypeError, AttributeError):
            pass
    gpu_probe = command_record([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader",
    ], executor, which)
    chosen = None
    if gpu_probe["status"] == "success":
        rows = [[part.strip() for part in row]
                for row in csv.reader(gpu_probe["output"]) if row]
        if any(len(row) != 3 or not all(row) for row in rows):
            raise NodeToolError("malformed local GPU identity response")
        if gpu_uuid is not None:
            if environment.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
                raise NodeToolError("local GPU UUID requires the same CUDA_VISIBLE_DEVICES UUID")
            selected = [row for row in rows if row[1] == gpu_uuid]
            if len(selected) != 1:
                raise NodeToolError("local GPU UUID is not uniquely observed")
            chosen = selected[0]
        elif len(rows) == 1:
            visible = environment.get("CUDA_VISIBLE_DEVICES")
            if visible not in {None, "0", rows[0][1]}:
                raise NodeToolError("local GPU visibility differs from the observed single GPU")
            chosen = rows[0]
        elif len(rows) > 1:
            raise NodeToolError("multiple local GPUs require --gpu-uuid and CUDA_VISIBLE_DEVICES")
    if chosen is not None:
        selected_environment.update({
            "GPU_SUITE_GPU_NAME": chosen[0], "GPU_SUITE_GPU_UUID": chosen[1],
            "GPU_SUITE_NVIDIA_DRIVER_VERSION": chosen[2],
            "GPU_SUITE_NODE_GPU_QUERY_STATUS": "success",
        })
    else:
        selected_environment.update({
            "GPU_SUITE_NODE_GPU_QUERY_STATUS": "unavailable",
            "GPU_SUITE_NODE_GPU_QUERY_DIAGNOSTIC": "local GPU identity was not observed",
        })
    document = build_node_metadata(
        config_path, manifest_path, run_id, wave, 0, socket.gethostname(),
        runtime_environment_sha256, selected_environment, cuda_runtime_probe,
    )
    manifest, _ = load_manifest(manifest_path)
    if any(entry["implementation"] in {"cuda", "openacc"} for entry in manifest["entries"]):
        if chosen is None or document["cuda_runtime_identity"]["query_status"] != "success":
            raise NodeToolError("local GPU artifacts require observed GPU and CUDA runtime identity")
    document["cpu_identity"] = {
        "name": cpu_name,
        "source": "lscpu" if cpu_name is not None else "unknown",
        "query_status": "success" if cpu_name is not None else "unavailable",
    }
    document["local_machine_probes"] = {"cpu": cpu_probe, "gpu": gpu_probe}
    document["execution_mode"] = "local"
    return document


def classify_raw(
    path: Path, node_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    records = load_raw_results(path)
    if node_metadata is not None:
        expected_scheduler = node_metadata.get("scheduler")
        expected_scheduler_job_id = node_metadata.get("scheduler_job_id")
        validate_scheduler_identity(
            expected_scheduler, expected_scheduler_job_id
        )
        identity = node_metadata.get("gpu_identity")
        if not isinstance(identity, Mapping):
            raise NodeToolError("node metadata lacks gpu_identity")
        expected_name = identity.get("name")
        expected_uuid = identity.get("uuid")
        runtime_identity = node_metadata.get("cuda_runtime_identity")
        if not isinstance(runtime_identity, Mapping):
            raise NodeToolError("node metadata lacks cuda_runtime_identity")
        expected_driver = runtime_identity.get("cuda_driver_api_version")
        expected_runtime = runtime_identity.get("cuda_runtime_version")
        for record in records:
            if (
                record["scheduler"] != expected_scheduler
                or record["scheduler_job_id"] != expected_scheduler_job_id
            ):
                raise NodeToolError(
                    "raw scheduler identity differs from node metadata"
                )
            if record["implementation"] not in {"cuda", "openacc"}:
                continue
            comparisons = (
                ("gpu_name", expected_name, "GPU name"),
                ("gpu_uuid", expected_uuid, "GPU UUID"),
                ("cuda_driver_version", expected_driver,
                 "CUDA Driver API version"),
                ("cuda_runtime_version", expected_runtime,
                 "CUDA Runtime version"),
            )
            for field, expected, label in comparisons:
                observed = record[field]
                if expected is not None and observed is not None and observed != expected:
                    raise NodeToolError(
                        "raw {0} differs from node metadata".format(label)
                    )
                if record["status"] == "success" and (
                    expected is None or observed is None
                ):
                    raise NodeToolError(
                        "successful raw row lacks complete node-matched {0}"
                        .format(label)
                    )
    statuses = Counter(record["status"] for record in records)
    verification = Counter(record["verification_status"] for record in records)
    origins = Counter(
        record["failure_origin"] for record in records
        if record["failure_origin"] is not None
    )
    benchmark_status = (
        "failure"
        if statuses.get("failure", 0) or statuses.get("skipped", 0)
        else "success"
    )
    verification_status = (
        "failure"
        if verification.get("failure", 0) or verification.get("nonfinite", 0)
        else "success"
    )
    return {
        "benchmark_status": benchmark_status,
        "failure_origin_counts": dict(sorted(origins.items())),
        "raw_classification_schema_version": 1,
        "record_count": len(records),
        "status_counts": dict(sorted(statuses.items())),
        "verification_counts": dict(sorted(verification.items())),
        "verification_status": verification_status,
    }


def build_node_status(
    run_id: str, wave: int, node_index: int, hostname: str,
    runner_exit_code: int, process_exit_code: int,
    collection_status: str, raw_collection_status: str,
    log_collection_status: str, tool_status: str,
    classification: Optional[Mapping[str, Any]],
    telemetry: Optional[Mapping[str, Any]], messages: Sequence[str],
    termination_signal: Optional[str],
    scheduler: Optional[str] = None,
    scheduler_job_id: Optional[str] = None,
) -> Dict[str, Any]:
    validate_execution_context(run_id, "node-status", hostname, wave, node_index, 0)
    validate_scheduler_identity(scheduler, scheduler_job_id)
    benchmark_status = (
        classification.get("benchmark_status", "failure")
        if classification is not None else "failure"
    )
    if runner_exit_code != 0:
        benchmark_status = "failure"
    verification_status = (
        classification.get("verification_status", "not-run")
        if classification is not None else "not-run"
    )
    telemetry_status = (
        telemetry.get("telemetry_status", "unavailable")
        if telemetry is not None else "unavailable"
    )
    fatal = collection_status != "success"
    overall = (
        "success"
        if benchmark_status == "success"
        and verification_status == "success"
        and not fatal
        else "failure"
    )
    return {
        "artifact_collection_timestamp": utc_timestamp(),
        "benchmark_status": benchmark_status,
        "block_id": "{0}|{1}|{2}".format(run_id, wave, hostname),
        "collection_status": collection_status,
        "fatal_infrastructure_failure": fatal,
        "hostname": hostname,
        "log_collection_status": log_collection_status,
        "messages": list(messages),
        "node_index": node_index,
        "node_status_schema_version": 1,
        "process_exit_code_before_collection": process_exit_code,
        "raw_classification": dict(classification) if classification else None,
        "raw_result_collection_status": raw_collection_status,
        "run_id": run_id,
        "runner_exit_code": runner_exit_code,
        "scheduler": scheduler,
        "scheduler_job_id": scheduler_job_id,
        "status": overall,
        "telemetry_status": telemetry_status,
        "termination_signal": termination_signal,
        "tool_status": tool_status,
        "verification_status": verification_status,
        "wave": wave,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    metadata = subparsers.add_parser("metadata")
    metadata.add_argument("--config", required=True, type=Path)
    metadata.add_argument("--manifest", required=True, type=Path)
    metadata.add_argument("--run-id", required=True)
    metadata.add_argument("--wave", required=True, type=int)
    metadata.add_argument("--node-index", required=True, type=int)
    metadata.add_argument("--hostname", required=True)
    metadata.add_argument("--scheduler")
    metadata.add_argument("--scheduler-job-id")
    metadata.add_argument("--runtime-environment-sha256", required=True)
    metadata.add_argument("--output", required=True, type=Path)

    classify = subparsers.add_parser("classify")
    classify.add_argument("--raw", required=True, type=Path)
    classify.add_argument("--node-metadata", type=Path)
    classify.add_argument("--output", required=True, type=Path)

    raw_name = subparsers.add_parser("raw-name")
    raw_name.add_argument("--config", required=True, type=Path)

    subparsers.add_parser("timestamp")

    status = subparsers.add_parser("status")
    status.add_argument("--run-id", required=True)
    status.add_argument("--wave", required=True, type=int)
    status.add_argument("--node-index", required=True, type=int)
    status.add_argument("--hostname", required=True)
    status.add_argument("--scheduler")
    status.add_argument("--scheduler-job-id")
    status.add_argument("--runner-exit-code", required=True, type=int)
    status.add_argument("--process-exit-code", required=True, type=int)
    status.add_argument("--collection-status", required=True,
                        choices=("success", "failure"))
    status.add_argument("--raw-collection-status", required=True,
                        choices=("success", "failure", "missing"))
    status.add_argument("--log-collection-status", required=True,
                        choices=("success", "failure"))
    status.add_argument("--tool-status", required=True,
                        choices=("success", "failure", "warning"))
    status.add_argument("--classification", type=Path)
    status.add_argument("--telemetry-metadata", type=Path)
    status.add_argument("--termination-signal")
    status.add_argument("--message", action="append", default=[])
    status.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "metadata":
            document = build_node_metadata(
                arguments.config, arguments.manifest, arguments.run_id,
                arguments.wave, arguments.node_index, arguments.hostname,
                arguments.runtime_environment_sha256,
                scheduler=arguments.scheduler,
                scheduler_job_id=arguments.scheduler_job_id,
            )
            _exclusive_write(arguments.output, document)
        elif arguments.command == "classify":
            metadata_document = (
                load(arguments.node_metadata)
                if arguments.node_metadata is not None else None
            )
            _exclusive_write(
                arguments.output,
                classify_raw(arguments.raw, metadata_document),
            )
        elif arguments.command == "raw-name":
            output_format = load_config(arguments.config)["output_format"]
            print("raw-results.csv" if output_format == "csv" else "raw-results.jsonl")
        elif arguments.command == "timestamp":
            print(utc_timestamp())
        else:
            classification = (
                load(arguments.classification)
                if arguments.classification is not None
                and arguments.classification.is_file() else None
            )
            telemetry = (
                load(arguments.telemetry_metadata)
                if arguments.telemetry_metadata is not None
                and arguments.telemetry_metadata.is_file() else None
            )
            document = build_node_status(
                arguments.run_id, arguments.wave, arguments.node_index,
                arguments.hostname, arguments.runner_exit_code,
                arguments.process_exit_code, arguments.collection_status,
                arguments.raw_collection_status,
                arguments.log_collection_status, arguments.tool_status,
                classification, telemetry, arguments.message,
                arguments.termination_signal,
                scheduler=arguments.scheduler,
                scheduler_job_id=arguments.scheduler_job_id,
            )
            _exclusive_write(arguments.output, document)
    except (OSError, ValueError) as error:
        print("node metadata/status failed: {0}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Create deterministic node metadata, raw classification, and final status."""

import argparse
import ctypes
import ctypes.util
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence


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
from gpu_suite.strict_json import dump_bytes, load  # noqa: E402
from job_config import CPU_ENVIRONMENT_KEYS  # noqa: E402


class NodeToolError(ValueError):
    pass


CudaRuntimeProbe = Callable[[], Mapping[str, Any]]


def _format_cuda_version(value: int) -> str:
    return "{0}.{1}.{2}".format(
        value // 1000, (value % 1000) // 10, value % 10,
    )


def probe_node_cuda_runtime() -> Dict[str, Any]:
    """Query the node-local CUDA Runtime without requiring a compiler."""

    candidates = []
    located = ctypes.util.find_library("cudart")
    if located:
        candidates.append(located)
    candidates.append("libcudart.so")
    for directory in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if not directory:
            continue
        try:
            candidates.extend(
                str(path) for path in sorted(Path(directory).glob("libcudart.so*"))
                if path.is_file()
            )
        except OSError:
            continue
    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)

    runtime = None
    load_errors = []
    loaded_library = None
    for candidate in unique_candidates:
        try:
            runtime = ctypes.CDLL(candidate)
            loaded_library = candidate
            break
        except OSError as error:
            load_errors.append("{0}: {1}".format(candidate, error))
    if runtime is None:
        return {
            "cuda_driver_api_version": None,
            "cuda_runtime_version": None,
            "diagnostic": "; ".join(load_errors) or "libcudart was not found",
            "loaded_library": None,
            "query_status": "unavailable",
        }

    try:
        driver_function = runtime.cudaDriverGetVersion
        runtime_function = runtime.cudaRuntimeGetVersion
        for function in (driver_function, runtime_function):
            function.argtypes = [ctypes.POINTER(ctypes.c_int)]
            function.restype = ctypes.c_int
        driver_value = ctypes.c_int()
        runtime_value = ctypes.c_int()
        driver_status = int(driver_function(ctypes.byref(driver_value)))
        runtime_status = int(runtime_function(ctypes.byref(runtime_value)))
    except (AttributeError, TypeError, ValueError) as error:
        return {
            "cuda_driver_api_version": None,
            "cuda_runtime_version": None,
            "diagnostic": "CUDA version API lookup failed: {0}".format(error),
            "loaded_library": loaded_library,
            "query_status": "failure",
        }
    if driver_status != 0 or runtime_status != 0:
        return {
            "cuda_driver_api_version": None,
            "cuda_runtime_version": None,
            "diagnostic": (
                "cudaDriverGetVersion/cudaRuntimeGetVersion returned {0}/{1}"
                .format(driver_status, runtime_status)
            ),
            "loaded_library": loaded_library,
            "query_status": "failure",
        }
    return {
        "cuda_driver_api_version": _format_cuda_version(driver_value.value),
        "cuda_runtime_version": _format_cuda_version(runtime_value.value),
        "diagnostic": None,
        "loaded_library": loaded_library,
        "query_status": "success",
    }


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
) -> Dict[str, Any]:
    validate_execution_context(run_id, "node-metadata", hostname, wave, node_index, 0)
    validate_sha256(runtime_environment_sha256, "runtime environment SHA-256")
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
    required_runtime_fields = {
        "cuda_driver_api_version", "cuda_runtime_version", "diagnostic",
        "loaded_library", "query_status",
    }
    if set(cuda_runtime_identity) != required_runtime_fields:
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
        "series": series,
        "size_order": size_order,
        "size_order_index": order_index,
        "wave": wave,
    }


def classify_raw(
    path: Path, node_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    records = load_raw_results(path)
    if node_metadata is not None:
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
                if record["status"] == "success" and expected is not None and observed is None:
                    raise NodeToolError(
                        "successful raw row lacks node-matched {0}".format(label)
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
) -> Dict[str, Any]:
    validate_execution_context(run_id, "node-status", hostname, wave, node_index, 0)
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
            )
            _exclusive_write(arguments.output, document)
    except (OSError, ValueError) as error:
        print("node metadata/status failed: {0}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

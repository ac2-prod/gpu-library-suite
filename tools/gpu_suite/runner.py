"""Deterministic single-node suite scheduling and failure materialization."""

import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import BENCHMARKS
from .hashing import sha256_bytes, sha256_file
from .manifest import merge_entries, validate_build_metadata
from .ordering import permutation_index, size_order_index
from .results_io import ExclusiveRawWriter, parse_stdout_prefix
from .schema import RUN_ID_RE, SHA256_RE, validate_raw_result
from .strict_json import dump_bytes, dumps, load


ZERO_SHA256 = "0" * 64
STEMS = {
    "cufft": "fft",
    "cublas": "blas",
    "cusparse": "sparse",
    "cusolver": "solver",
    "curand": "rand",
    "thrust": "reduce",
}
SERIAL_CPU_BACKENDS = {
    "cpu-fftw-serial",
    "cpu-std-random-serial",
    "cpu-stl-serial",
    "cpu-reference-csr",
}
CORE_PARAMETER_KEYS = {
    "cufft": ("nfft", "batch", "transform"),
    "cublas": ("m", "n", "k", "alpha", "beta"),
    "cusparse": ("nx", "ny", "alpha", "beta"),
    "cusolver": ("n", "nrhs"),
    "curand": ("size", "generator", "distribution", "seed", "offset", "order"),
    "thrust": ("size", "operation"),
}


class RunnerError(ValueError):
    pass


def utc_timestamp() -> str:
    now = datetime.now(timezone.utc)
    milliseconds = now.microsecond // 1000
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + "{0:03d}Z".format(milliseconds)


def validate_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise RunnerError("invalid {0}".format(name))
    return value


def validate_execution_context(
    run_id: str, system_label: str, hostname: str, wave: int,
    node_index: int, device: int,
) -> None:
    if RUN_ID_RE.fullmatch(run_id) is None or run_id.startswith(".") or ".." in run_id:
        raise RunnerError("invalid run ID")
    for name, value in (("system label", system_label), ("hostname", hostname)):
        if (
            not isinstance(value, str)
            or value == ""
            or "|" in value
            or any(ord(character) < 32 for character in value)
        ):
            raise RunnerError("invalid {0}".format(name))
    for name, value in (("wave", wave), ("node index", node_index), ("device", device)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RunnerError("invalid {0}".format(name))


def load_manifest(path: Path) -> Tuple[Dict[str, Any], str]:
    document = load(path)
    if not isinstance(document, dict) or document.get("manifest_schema_version") != 1:
        raise RunnerError("invalid executable manifest")
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise RunnerError("manifest entries must be an array")
    validated = merge_entries([entries])
    result = dict(document)
    result["entries"] = validated
    return result, sha256_file(path)


def load_build_metadata(paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    metadata_by_hash = {}  # type: Dict[str, Dict[str, Any]]
    for path in paths:
        content = path.read_bytes()
        document = load(path)
        if not isinstance(document, dict):
            raise RunnerError("build metadata must be an object")
        digest = sha256_bytes(content)
        if digest in metadata_by_hash and metadata_by_hash[digest] != document:
            raise RunnerError("build metadata hash collision")
        metadata_by_hash[digest] = document
    return metadata_by_hash


def validate_manifest_artifacts(
    manifest: Mapping[str, Any], metadata_by_hash: Mapping[str, Mapping[str, Any]],
    run_mode: str, dirty_hash_available: bool,
) -> None:
    for entry in manifest["entries"]:
        metadata = metadata_by_hash.get(entry["build_metadata_sha256"])
        if metadata is None:
            raise RunnerError("manifest entry has no supplied build metadata")
        validate_build_metadata(entry, metadata)
        path = Path(entry["executable_path"])
        if not path.is_file():
            raise RunnerError("manifest executable does not exist: {0}".format(path))
        if sha256_file(path) != entry["binary_sha256"]:
            raise RunnerError("binary hash mismatch: {0}".format(path))
        if run_mode == "production" and entry["git_dirty"]:
            raise RunnerError("production execution requires clean build metadata")
        if run_mode == "production" and entry["build_type"] != "Release":
            raise RunnerError("production execution requires Release artifacts")
        if entry["git_dirty"] and not dirty_hash_available:
            raise RunnerError("dirty build requires a diff or source-snapshot hash")


def normalized_parameters(benchmark: str, parameters: Mapping[str, Any]) -> Dict[str, Any]:
    if benchmark == "cufft":
        return {
            "nfft": parameters["nfft"],
            "batch": parameters["batch"],
            "transform": parameters["transform"],
        }
    if benchmark == "cublas":
        return {
            "m": parameters["size"],
            "n": parameters["size"],
            "k": parameters["size"],
            "alpha": parameters["alpha"],
            "beta": parameters["beta"],
        }
    if benchmark == "cusparse":
        side = math.isqrt(parameters["size"])
        return {
            "nx": side,
            "ny": side,
            "alpha": parameters["alpha"],
            "beta": parameters["beta"],
        }
    if benchmark == "cusolver":
        return {"n": parameters["size"], "nrhs": parameters["nrhs"]}
    return dict(parameters)


def core_raw_parameters(
    benchmark: str, parameters: Mapping[str, Any]
) -> Dict[str, Any]:
    """Select comparison-defining parameters while ignoring backend metadata."""

    if benchmark not in CORE_PARAMETER_KEYS:
        raise RunnerError("unknown benchmark parameter family")
    missing = [name for name in CORE_PARAMETER_KEYS[benchmark] if name not in parameters]
    if missing:
        raise RunnerError(
            "raw parameters lack comparison keys: {0}".format(", ".join(missing))
        )
    return {name: parameters[name] for name in CORE_PARAMETER_KEYS[benchmark]}


def problem_sizes(benchmark: str, parameters: Mapping[str, Any]) -> Tuple[int, Optional[int]]:
    if benchmark == "cufft":
        return parameters["nfft"], parameters["batch"]
    if benchmark == "cusparse":
        side = math.isqrt(parameters["size"])
        nnz = 5 * parameters["size"] - 4 * side
        return parameters["size"], nnz
    if benchmark == "cusolver":
        return parameters["size"], parameters["nrhs"]
    return parameters["size"], None


def _option(arguments: List[str], name: str, value: Any) -> None:
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    else:
        rendered = str(value)
    arguments.extend([name, rendered])


def build_command(
    entry: Mapping[str, Any], benchmark: str, parameters: Mapping[str, Any],
    scope_name: str, scope: Mapping[str, Any], series: Mapping[str, Any],
    context: Mapping[str, Any], verification: Mapping[str, Any], output_format: str,
) -> List[str]:
    arguments = [entry["executable_path"]]
    if benchmark == "cufft":
        _option(arguments, "--size", parameters["nfft"])
        _option(arguments, "--batch", parameters["batch"])
        _option(arguments, "--transform", parameters["transform"])
    else:
        _option(arguments, "--size", parameters["size"])
    for name in ("alpha", "beta", "nrhs", "generator", "distribution", "offset", "order", "operation"):
        if name in parameters:
            _option(arguments, "--" + name.replace("_", "-"), parameters[name])
    if "seed" in parameters:
        _option(arguments, "--seed", parameters["seed"])
    _option(arguments, "--warmup", scope["warmup"])
    _option(arguments, "--repeat", scope["repeat"])
    _option(arguments, "--trials", scope["trials"])
    _option(arguments, "--scope", scope_name)
    _option(arguments, "--verify", True)
    _option(arguments, "--output", "-")
    _option(arguments, "--format", output_format)
    _option(arguments, "--device", context["device"])
    _option(arguments, "--run-id", context["run_id"])
    _option(arguments, "--system-label", context["system_label"])
    _option(arguments, "--node-index", context["node_index"])
    _option(arguments, "--wave", context["wave"])
    _option(arguments, "--cpu-threads", context["cpu_threads"])
    _option(arguments, "--implementation-order", ",".join(context["implementation_order"]))
    if series["implementation"] == "cpu":
        _option(arguments, "--cpu-backend", series["cpu_backend"])
    for name, value in verification.items():
        _option(arguments, "--" + name.replace("_", "-"), value)
    return arguments


def build_schedule(
    config: Mapping[str, Any], manifest: Mapping[str, Any], context: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    benchmark_entries = {}
    for entry in manifest["entries"]:
        if entry["executable_role"] == "benchmark":
            benchmark_entries[(entry["library"], entry["implementation"])] = entry
    order = tuple(context["implementation_order"])
    schedule = []  # type: List[Dict[str, Any]]
    for benchmark in BENCHMARKS:
        definition = config["benchmarks"][benchmark]
        if not definition["enabled"]:
            continue
        indexed_cases = list(enumerate(definition["cases"]))
        if size_order_index(context["node_index"]) == 1:
            indexed_cases.reverse()
        primary = {
            item["implementation"]: item
            for item in definition["series"]
            if item["series_role"] == "primary"
        }
        auxiliaries = [
            item for item in definition["series"] if item["series_role"] == "auxiliary"
        ]
        ordered_series = [primary[implementation] for implementation in order] + auxiliaries
        for case_index, case in indexed_cases:
            for scope_name in ("compute", "end-to-end"):
                scope = case["scopes"][scope_name]
                for series in ordered_series:
                    entry = benchmark_entries.get((benchmark, series["implementation"]))
                    reason = None
                    if entry is None:
                        reason = "benchmark executable is absent from manifest"
                    elif series["implementation"] == "cpu" and series["cpu_backend"] not in entry["supported_cpu_backends"]:
                        reason = "CPU backend is not supported by selected executable"
                    command = None
                    if entry is not None and reason is None:
                        command = build_command(
                            entry, benchmark, case["parameters"], scope_name,
                            scope, series, context, definition["verification"],
                            config["output_format"],
                        )
                    schedule.append(
                        {
                            "artifact_id": None if entry is None else entry["artifact_id"],
                            "argv": command,
                            "benchmark": benchmark,
                            "case_index": case_index,
                            "entry": entry,
                            "parameters": dict(case["parameters"]),
                            "prerequisite_failure": reason,
                            "scope": scope_name,
                            "scope_settings": dict(scope),
                            "series": dict(series),
                        }
                    )
    return schedule


def dry_run_document(
    config: Mapping[str, Any], manifest: Mapping[str, Any], schedule: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any], manifest_sha256: str, config_sha256: str,
) -> Dict[str, Any]:
    selected_ids = sorted(
        {item["artifact_id"] for item in schedule if item["artifact_id"] is not None}
    )
    commands = []
    for item in schedule:
        commands.append(
            {
                "argv": item["argv"],
                "artifact_id": item["artifact_id"],
                "benchmark": item["benchmark"],
                "case_index": item["case_index"],
                "parameters": item["parameters"],
                "prerequisite_failure": item["prerequisite_failure"],
                "scope": item["scope"],
                "scope_settings": item["scope_settings"],
                "series": item["series"],
            }
        )
    return {
        "commands": commands,
        "config": config,
        "config_sha256": config_sha256,
        "dry_run_schema_version": 1,
        "manifest": {
            "entries": [
                entry for entry in manifest["entries"] if entry["artifact_id"] in selected_ids
            ],
            "manifest_schema_version": 1,
        },
        "manifest_sha256": manifest_sha256,
        "node": {
            "implementation_order": list(context["implementation_order"]),
            "node_index": context["node_index"],
            "permutation_index": permutation_index(context["node_index"], context["wave"]),
            "size_order_index": size_order_index(context["node_index"]),
            "wave": context["wave"],
        },
    }


def _metadata_for_entry(
    entry: Mapping[str, Any], metadata_by_hash: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any]:
    metadata = metadata_by_hash[entry["build_metadata_sha256"]]
    compiler = metadata[entry["compiler_language"]]
    return compiler


def synthetic_result(
    item: Mapping[str, Any], context: Mapping[str, Any], config_sha256: str,
    runtime_environment_sha256: str, metadata_by_hash: Mapping[str, Mapping[str, Any]],
    trial: int, attempted: bool, origin: str, message: str,
    exit_code: Optional[int], git_diff_sha256: Optional[str],
    source_snapshot_sha256: Optional[str],
) -> Dict[str, Any]:
    entry = item["entry"]
    series = item["series"]
    parameters = item["parameters"]
    benchmark = item["benchmark"]
    primary_size, secondary_size = problem_sizes(benchmark, parameters)
    if entry is None:
        compiler = {"compiler": "unavailable", "compiler_version": "unavailable", "compiler_flags": ""}
        binary_sha256 = ZERO_SHA256
        git_commit = "unavailable"
        git_dirty = False
    else:
        compiler = _metadata_for_entry(entry, metadata_by_hash)
        binary_sha256 = entry["binary_sha256"]
        git_commit = entry["git_commit"]
        git_dirty = entry["git_dirty"]
    cpu_backend = series["cpu_backend"] if series["implementation"] == "cpu" else None
    serial = cpu_backend in SERIAL_CPU_BACKENDS
    record = {
        "result_schema_version": 1,
        "run_id": context["run_id"],
        "record_timestamp": utc_timestamp(),
        "measurement_start_timestamp": None,
        "measurement_end_timestamp": None,
        "system_label": context["system_label"],
        "wave": context["wave"],
        "node_index": context["node_index"],
        "hostname": context["hostname"],
        "block_id": "{0}|{1}|{2}".format(context["run_id"], context["wave"], context["hostname"]),
        "scheduler": context["scheduler"],
        "scheduler_job_id": context["scheduler_job_id"],
        "implementation_order": list(context["implementation_order"]),
        "benchmark": benchmark,
        "implementation": series["implementation"],
        "scope": item["scope"],
        "problem_size": primary_size,
        "secondary_size": secondary_size,
        "parameters": normalized_parameters(benchmark, parameters),
        "precision": context["config"]["benchmarks"][benchmark]["precision"],
        "cpu_backend": cpu_backend,
        "cpu_backend_role": series["cpu_backend_role"] if cpu_backend else None,
        "series_role": series["series_role"],
        "cpu_threads_requested": context["cpu_threads"] if cpu_backend else None,
        "cpu_threads_effective": 1 if serial else None,
        "cpu_parallelism": ("serial" if serial else "unknown") if cpu_backend else None,
        "warmup": item["scope_settings"]["warmup"],
        "repeat": item["scope_settings"]["repeat"],
        "trial": trial,
        "attempted": attempted,
        "failure_origin": origin,
        "elapsed_total_sec": None,
        "elapsed_sec": None,
        "clock_id": "CLOCK_MONOTONIC",
        "clock_resolution_sec": time.get_clock_info("monotonic").resolution,
        "verification_metrics": {},
        "verification_thresholds": {},
        "verification_primary_metric": None,
        "verification_status": "skipped",
        "getrf_info": None,
        "getrs_info": None,
        "device_id": context["device"] if series["implementation"] != "cpu" else None,
        "gpu_name": None,
        "gpu_uuid": None,
        "cuda_driver_version": None,
        "compiler": compiler["compiler"],
        "compiler_version": compiler["compiler_version"],
        "compiler_flags": compiler.get("compiler_flags", ""),
        "library_name": cpu_backend if cpu_backend else benchmark,
        "library_version": None,
        "cuda_runtime_version": None,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "git_diff_sha256": git_diff_sha256 if git_dirty else None,
        "source_snapshot_sha256": source_snapshot_sha256 if git_dirty else None,
        "config_sha256": config_sha256,
        "runtime_environment_sha256": runtime_environment_sha256,
        "binary_sha256": binary_sha256,
        "exit_code": exit_code if attempted else None,
        "status": "failure" if attempted else "skipped",
        "message": message,
    }
    return validate_raw_result(record)


def _core_parameters_match(
    benchmark: str, expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> bool:
    normalized = normalized_parameters(benchmark, expected)
    try:
        selected = core_raw_parameters(benchmark, actual)
    except RunnerError:
        return False
    return selected == normalized


def validate_subprocess_record(
    record: Mapping[str, Any], item: Mapping[str, Any], context: Mapping[str, Any],
    config_sha256: str, runtime_environment_sha256: str,
) -> Dict[str, Any]:
    validated = validate_raw_result(record)
    entry = item["entry"]
    series = item["series"]
    expected_values = {
        "run_id": context["run_id"],
        "system_label": context["system_label"],
        "wave": context["wave"],
        "node_index": context["node_index"],
        "hostname": context["hostname"],
        "implementation_order": list(context["implementation_order"]),
        "benchmark": item["benchmark"],
        "implementation": series["implementation"],
        "scope": item["scope"],
        "warmup": item["scope_settings"]["warmup"],
        "repeat": item["scope_settings"]["repeat"],
        "cpu_backend": series["cpu_backend"],
        "cpu_backend_role": series["cpu_backend_role"],
        "series_role": series["series_role"],
        "config_sha256": config_sha256,
        "runtime_environment_sha256": runtime_environment_sha256,
        "binary_sha256": entry["binary_sha256"],
        "compiler": entry["compiler"],
        "compiler_version": entry["compiler_version"],
        "git_commit": entry["git_commit"],
        "git_dirty": entry["git_dirty"],
    }
    for name, expected in expected_values.items():
        if validated[name] != expected:
            raise RunnerError("subprocess record mismatch for {0}".format(name))
    if not _core_parameters_match(item["benchmark"], item["parameters"], validated["parameters"]):
        raise RunnerError("subprocess record parameters do not match command")
    return validated


def _process_exit_code(returncode: int) -> int:
    return returncode if returncode >= 0 else 128 + abs(returncode)


def execute_schedule(
    schedule: Sequence[Mapping[str, Any]], context: Mapping[str, Any], output_path: Path,
    config_sha256: str, runtime_environment_sha256: str,
    metadata_by_hash: Mapping[str, Mapping[str, Any]],
    git_diff_sha256: Optional[str], source_snapshot_sha256: Optional[str],
) -> int:
    any_failure = False
    stop_remaining = False
    output_format = context["config"]["output_format"]
    with ExclusiveRawWriter(output_path, output_format) as writer:
        for item in schedule:
            trials = item["scope_settings"]["trials"]
            if stop_remaining or item["prerequisite_failure"] is not None:
                message = "prior suite failure" if stop_remaining else item["prerequisite_failure"]
                origin = "prior-failure" if stop_remaining else "prerequisite"
                for trial in range(trials):
                    writer.write(
                        synthetic_result(
                            item, context, config_sha256, runtime_environment_sha256,
                            metadata_by_hash, trial, False, origin, message,
                            None, git_diff_sha256, source_snapshot_sha256,
                        )
                    )
                any_failure = True
                if not context["config"]["continue_on_failure"]:
                    stop_remaining = True
                continue

            entry = item["entry"]
            environment = os.environ.copy()
            environment.update(
                {
                    "GPU_SUITE_BINARY_SHA256": entry["binary_sha256"],
                    "GPU_SUITE_CONFIG_SHA256": config_sha256,
                    "GPU_SUITE_RUNTIME_ENVIRONMENT_SHA256": runtime_environment_sha256,
                }
            )
            if context["scheduler"] is not None:
                environment["GPU_SUITE_SCHEDULER"] = context["scheduler"]
            if context["scheduler_job_id"] is not None:
                environment["GPU_SUITE_SCHEDULER_JOB_ID"] = context["scheduler_job_id"]
            if git_diff_sha256 is not None:
                environment["GPU_SUITE_GIT_DIFF_SHA256"] = git_diff_sha256
            if source_snapshot_sha256 is not None:
                environment["GPU_SUITE_SOURCE_SNAPSHOT_SHA256"] = source_snapshot_sha256
            completed = subprocess.run(
                item["argv"], text=True, capture_output=True, check=False,
                env=environment,
            )
            if completed.stderr:
                sys.stderr.write(
                    "[{0}/{1}/{2}] {3}".format(
                        item["benchmark"], item["scope"],
                        item["series"]["implementation"], completed.stderr,
                    )
                )
                if not completed.stderr.endswith("\n"):
                    sys.stderr.write("\n")
            parsed, parse_error = parse_stdout_prefix(completed.stdout, output_format)
            accepted = []  # type: List[Dict[str, Any]]
            contamination = parse_error
            fatal_failure_seen = False
            for record in parsed:
                try:
                    expected_trial = len(accepted)
                    if record["trial"] != expected_trial or record["trial"] >= trials:
                        raise RunnerError("duplicate, out-of-order, or out-of-range trial")
                    validated = validate_subprocess_record(
                        record, item, context, config_sha256,
                        runtime_environment_sha256,
                    )
                    if fatal_failure_seen and not (
                        validated["status"] == "skipped"
                        and validated["failure_origin"] == "prior-failure"
                    ):
                        raise RunnerError(
                            "trials after a failure must be prior-failure skips"
                        )
                    if (
                        validated["status"] == "failure"
                        and validated["failure_origin"] != "verification"
                    ):
                        fatal_failure_seen = True
                    accepted.append(validated)
                except ValueError as error:
                    contamination = str(error)
                    break
            for record in accepted:
                writer.write(record)
            unresolved = len(accepted)
            abnormal = completed.returncode != 0
            if contamination is not None:
                sys.stderr.write(
                    "[{0}] stdout contamination: {1}\n".format(
                        item["benchmark"], contamination
                    )
                )
            failure_needed = unresolved < trials
            if failure_needed:
                origin = (
                    "subprocess"
                    if abnormal and contamination is None
                    else "output-validation"
                )
                message = contamination or (
                    "subprocess exited with status {0}".format(completed.returncode)
                    if abnormal
                    else "subprocess did not emit every expected trial"
                )
                writer.write(
                    synthetic_result(
                        item, context, config_sha256, runtime_environment_sha256,
                        metadata_by_hash, unresolved, True, origin, message,
                        _process_exit_code(completed.returncode) if abnormal else 1,
                        git_diff_sha256, source_snapshot_sha256,
                    )
                )
                for trial in range(unresolved + 1, trials):
                    writer.write(
                        synthetic_result(
                            item, context, config_sha256, runtime_environment_sha256,
                            metadata_by_hash, trial, False, "prior-failure",
                            "prior subprocess/output failure", None,
                            git_diff_sha256, source_snapshot_sha256,
                        )
                    )
                any_failure = True
            if abnormal or contamination is not None:
                any_failure = True
            if any(record["status"] != "success" for record in accepted):
                any_failure = True
            if any_failure and not context["config"]["continue_on_failure"]:
                stop_remaining = True
    return 1 if any_failure else 0

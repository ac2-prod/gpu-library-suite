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
        if run_mode == "production" and not entry["git_metadata_available"]:
            raise RunnerError("production execution requires available Git metadata")
        if run_mode == "production" and entry["git_dirty"]:
            raise RunnerError("production execution requires clean build metadata")
        if run_mode == "production" and entry["build_type"] != "Release":
            raise RunnerError("production execution requires Release artifacts")
        if entry["git_dirty"] is True and not dirty_hash_available:
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
            "alpha": float(parameters["alpha"]),
            "beta": float(parameters["beta"]),
        }
    if benchmark == "cusparse":
        side = math.isqrt(parameters["size"])
        return {
            "nx": side,
            "ny": side,
            "alpha": float(parameters["alpha"]),
            "beta": float(parameters["beta"]),
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
    result = {
        name: parameters[name] for name in CORE_PARAMETER_KEYS[benchmark]
    }
    for name in ("alpha", "beta"):
        if name in result:
            result[name] = float(result[name])
    return result


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
        _option(arguments, "--cpu-backend-role", series["cpu_backend_role"])
        _option(arguments, "--series-role", series["series_role"])
        _option(arguments, "--cpu-parallelism", series["cpu_parallelism"])
        if series["cpu_threads_effective"] is not None:
            _option(
                arguments, "--cpu-threads-effective",
                series["cpu_threads_effective"],
            )
    else:
        _option(arguments, "--series-role", series["series_role"])
    for name, value in verification.items():
        _option(arguments, "--" + name.replace("_", "-"), value)
    return arguments


def build_schedule(
    config: Mapping[str, Any], manifest: Mapping[str, Any], context: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    benchmark_entries = {}
    for entry in manifest["entries"]:
        if entry["executable_role"] == "benchmark":
            key = (entry["library"], entry["implementation"])
            if key in benchmark_entries:
                raise RunnerError(
                    "duplicate benchmark manifest entry: {0}/{1}".format(*key)
                )
            benchmark_entries[key] = entry
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
        compiler = {
            "compiler": "unavailable",
            "compiler_version": "unavailable",
            "global_configure_flags": "",
        }
        binary_sha256 = ZERO_SHA256
        git_metadata_available = False
        git_commit = None
        git_dirty = None
    else:
        compiler = _metadata_for_entry(entry, metadata_by_hash)
        binary_sha256 = entry["binary_sha256"]
        git_metadata_available = entry["git_metadata_available"]
        git_commit = entry["git_commit"]
        git_dirty = entry["git_dirty"]
    cpu_backend = series["cpu_backend"] if series["implementation"] == "cpu" else None
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
        "cpu_threads_effective": (
            series["cpu_threads_effective"] if cpu_backend else None
        ),
        "cpu_parallelism": series["cpu_parallelism"] if cpu_backend else None,
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
        "verification_status": (
            "failure" if attempted and origin == "verification" else "skipped"
        ),
        "getrf_info": None,
        "getrs_info": None,
        "device_id": context["device"] if series["implementation"] != "cpu" else None,
        "gpu_name": None,
        "gpu_uuid": None,
        "cuda_driver_version": None,
        "compiler": compiler["compiler"],
        "compiler_version": compiler["compiler_version"],
        "global_configure_flags": compiler.get("global_configure_flags", ""),
        "library_name": cpu_backend if cpu_backend else benchmark,
        "library_version": None,
        "cuda_runtime_version": None,
        "git_metadata_available": git_metadata_available,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "git_diff_sha256": git_diff_sha256 if git_dirty is True else None,
        "source_snapshot_sha256": (
            source_snapshot_sha256 if git_dirty is True else None
        ),
        "config_sha256": config_sha256,
        "runtime_environment_sha256": runtime_environment_sha256,
        "binary_sha256": binary_sha256,
        "exit_code": exit_code if attempted else None,
        "status": "failure" if attempted else "skipped",
        "message": message,
    }
    return validate_raw_result(record)


def expected_raw_parameters(
    benchmark: str, parameters: Mapping[str, Any], implementation: str,
) -> Dict[str, Any]:
    """Reconstruct the exact parameter object a selected executable must emit."""

    expected = normalized_parameters(benchmark, parameters)
    if benchmark == "curand":
        expected = dict(expected)
        expected["verification_sample_count"] = expected["size"]
        if implementation == "cpu":
            expected["cpu_engine"] = "std::mt19937_64"
            expected["distribution_interval"] = "[0,1)"
        else:
            expected["generator_algorithm"] = "CURAND_RNG_PSEUDO_DEFAULT"
            expected["distribution_interval"] = "(0,1]"
    return expected


def _absolute_plus_relative(
    reference_scale: float, verification: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "method": "absolute-plus-relative",
        "reference_scale": reference_scale,
        "abs_tolerance": verification["abs_tolerance"],
        "rel_tolerance": verification["rel_tolerance"],
    }


def _expected_verification(
    item: Mapping[str, Any], context: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Tuple[str, ...], str]:
    benchmark = item["benchmark"]
    parameters = normalized_parameters(benchmark, item["parameters"])
    verification = context["config"]["benchmarks"][benchmark]["verification"]
    repeat = item["scope_settings"]["repeat"]
    updates = repeat if item["scope"] == "compute" else 1
    if benchmark == "cufft":
        return (
            {
                "dc_relative_error": {
                    "method": "relative-upper-bound",
                    "upper_bound": verification["rel_tolerance"],
                },
                "non_dc_max_abs_error": {
                    "method": "absolute-upper-bound",
                    "upper_bound": verification["abs_tolerance"],
                },
            },
            ("dc_relative_error", "non_dc_max_abs_error"),
            "non_dc_max_abs_error",
        )
    if benchmark == "cublas":
        expected = 1.0
        for _ in range(updates):
            expected = (
                parameters["alpha"] * parameters["k"]
                + parameters["beta"] * expected
            )
        return (
            {"max_abs_error": _absolute_plus_relative(abs(expected), verification)},
            ("max_abs_error",),
            "max_abs_error",
        )
    if benchmark == "cusparse":
        def neighbor_contributions(length: int) -> Tuple[int, ...]:
            if length == 1:
                return (0,)
            if length == 2:
                return (1,)
            return (1, 2)

        horizontal = neighbor_contributions(parameters["nx"])
        vertical = neighbor_contributions(parameters["ny"])
        bases = tuple(
            float(4 - horizontal_count - vertical_count)
            for horizontal_count in horizontal
            for vertical_count in vertical
        )
        reference_scale = 0.0
        for base in bases:
            expected = 1.0
            for _ in range(updates):
                expected = (
                    parameters["alpha"] * base
                    + parameters["beta"] * expected
                )
            reference_scale = max(reference_scale, abs(expected))
        return (
            {
                "max_abs_error": _absolute_plus_relative(
                    reference_scale, verification
                )
            },
            ("max_abs_error",),
            "max_abs_error",
        )
    if benchmark == "cusolver":
        threshold = _absolute_plus_relative(1.0, verification)
        return (
            {
                "solution_relative_error": dict(threshold),
                "relative_residual": dict(threshold),
            },
            ("solution_relative_error", "relative_residual"),
            "relative_residual",
        )
    if benchmark == "curand":
        count = float(parameters["size"])
        sigma = verification["sigma_multiplier"]
        interval = (
            "[0,1)" if item["series"]["implementation"] == "cpu" else "(0,1]"
        )
        return (
            {
                "observed_range": {
                    "method": "inclusive-range",
                    "lower_bound": 0.0,
                    "upper_bound": 1.0,
                    "backend_interval": interval,
                },
                "sample_mean": {
                    "method": "uniform-mean-sigma-bound",
                    "expected_mean": verification["expected_mean"],
                    "sigma_multiplier": sigma,
                    "absolute_bound": sigma * math.sqrt(1.0 / (12.0 * count)),
                },
                "second_central_moment_about_half": {
                    "method": "uniform-second-central-moment-sigma-bound",
                    "expected_second_central_moment": verification[
                        "expected_second_central_moment"
                    ],
                    "sigma_multiplier": sigma,
                    "absolute_bound": sigma * math.sqrt(1.0 / (180.0 * count)),
                },
            },
            (
                "observed_min",
                "observed_max",
                "sample_mean",
                "second_central_moment_about_half",
            ),
            "sample_mean",
        )
    if benchmark == "thrust":
        return (
            {
                "absolute_error": _absolute_plus_relative(
                    float(parameters["size"]), verification
                )
            },
            ("absolute_error",),
            "absolute_error",
        )
    raise RunnerError("unknown benchmark verification family")


def _threshold_values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(
                _threshold_values_match(expected[name], actual[name])
                for name in expected
            )
        )
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and math.isclose(float(actual), expected, rel_tol=1e-15, abs_tol=0.0)
        )
    return actual == expected


def _validate_exact_config_threshold_operands(
    validated: Mapping[str, Any], item: Mapping[str, Any],
    context: Mapping[str, Any],
) -> None:
    """Require configuration-owned operands to survive stdout exactly.

    Derived reference scales and statistical bounds may differ by a final
    host-library rounding step, so the structural comparison above permits a
    one-ulp-scale tolerance for those values.  Values copied directly from the
    effective configuration do not need that tolerance and must compare equal.
    """

    benchmark = item["benchmark"]
    actual = validated["verification_thresholds"]
    verification = context["config"]["benchmarks"][benchmark]["verification"]
    operands = []  # type: List[Tuple[str, str, Any]]
    if benchmark == "cufft":
        operands.extend((
            ("dc_relative_error", "upper_bound",
             verification["rel_tolerance"]),
            ("non_dc_max_abs_error", "upper_bound",
             verification["abs_tolerance"]),
        ))
    elif benchmark in {"cublas", "cusparse"}:
        operands.extend((
            ("max_abs_error", "abs_tolerance",
             verification["abs_tolerance"]),
            ("max_abs_error", "rel_tolerance",
             verification["rel_tolerance"]),
        ))
    elif benchmark == "cusolver":
        for metric in ("solution_relative_error", "relative_residual"):
            operands.extend((
                (metric, "abs_tolerance", verification["abs_tolerance"]),
                (metric, "rel_tolerance", verification["rel_tolerance"]),
            ))
    elif benchmark == "curand":
        operands.extend((
            ("observed_range", "lower_bound", 0.0),
            ("observed_range", "upper_bound", 1.0),
            ("sample_mean", "expected_mean", verification["expected_mean"]),
            ("sample_mean", "sigma_multiplier",
             verification["sigma_multiplier"]),
            ("second_central_moment_about_half",
             "expected_second_central_moment",
             verification["expected_second_central_moment"]),
            ("second_central_moment_about_half", "sigma_multiplier",
             verification["sigma_multiplier"]),
        ))
    elif benchmark == "thrust":
        operands.extend((
            ("absolute_error", "abs_tolerance",
             verification["abs_tolerance"]),
            ("absolute_error", "rel_tolerance",
             verification["rel_tolerance"]),
        ))
    else:
        raise RunnerError("unknown benchmark verification family")

    for metric, operand, expected in operands:
        if actual[metric][operand] != expected:
            raise RunnerError(
                "subprocess record verification threshold operand mismatch: "
                "{0}.{1}".format(metric, operand)
            )


def _validate_verification(
    validated: Mapping[str, Any], item: Mapping[str, Any],
    context: Mapping[str, Any],
) -> None:
    if validated["verification_status"] == "skipped":
        if validated["status"] == "success":
            raise RunnerError("subprocess skipped requested verification")
        if (
            validated["verification_metrics"] != {}
            or validated["verification_thresholds"] != {}
            or validated["verification_primary_metric"] is not None
        ):
            raise RunnerError("skipped verification contains verification data")
        return
    thresholds, metric_names, primary = _expected_verification(item, context)
    if not _threshold_values_match(
        thresholds, validated["verification_thresholds"]
    ):
        raise RunnerError("subprocess record verification thresholds mismatch")
    _validate_exact_config_threshold_operands(validated, item, context)
    if set(validated["verification_metrics"]) != set(metric_names):
        raise RunnerError("subprocess record verification metrics mismatch")
    if validated["verification_primary_metric"] != primary:
        raise RunnerError("subprocess record verification primary metric mismatch")


def validate_subprocess_record(
    record: Mapping[str, Any], item: Mapping[str, Any], context: Mapping[str, Any],
    config_sha256: str, runtime_environment_sha256: str,
    git_diff_sha256: Optional[str], source_snapshot_sha256: Optional[str],
) -> Dict[str, Any]:
    validated = validate_raw_result(record)
    entry = item["entry"]
    series = item["series"]
    benchmark = item["benchmark"]
    primary_size, secondary_size = problem_sizes(benchmark, item["parameters"])
    cpu = series["implementation"] == "cpu"
    expected_git_diff = git_diff_sha256 if entry["git_dirty"] is True else None
    expected_source_snapshot = (
        source_snapshot_sha256 if entry["git_dirty"] is True else None
    )
    expected_values = {
        "run_id": context["run_id"],
        "system_label": context["system_label"],
        "wave": context["wave"],
        "node_index": context["node_index"],
        "hostname": context["hostname"],
        "block_id": "{0}|{1}|{2}".format(
            context["run_id"], context["wave"], context["hostname"]
        ),
        "scheduler": context["scheduler"],
        "scheduler_job_id": context["scheduler_job_id"],
        "implementation_order": list(context["implementation_order"]),
        "benchmark": benchmark,
        "implementation": series["implementation"],
        "scope": item["scope"],
        "problem_size": primary_size,
        "secondary_size": secondary_size,
        "parameters": expected_raw_parameters(
            benchmark, item["parameters"], series["implementation"]
        ),
        "precision": context["config"]["benchmarks"][benchmark]["precision"],
        "warmup": item["scope_settings"]["warmup"],
        "repeat": item["scope_settings"]["repeat"],
        "cpu_backend": series["cpu_backend"] if cpu else None,
        "cpu_backend_role": series["cpu_backend_role"] if cpu else None,
        "series_role": series["series_role"],
        "cpu_threads_requested": context["cpu_threads"] if cpu else None,
        "cpu_threads_effective": (
            series["cpu_threads_effective"]
            if cpu else None
        ),
        "cpu_parallelism": (
            series["cpu_parallelism"]
            if cpu else None
        ),
        "device_id": None if cpu else context["device"],
        "library_name": series["cpu_backend"] if cpu else benchmark,
        "config_sha256": config_sha256,
        "runtime_environment_sha256": runtime_environment_sha256,
        "binary_sha256": entry["binary_sha256"],
        "compiler": entry["compiler"],
        "compiler_version": entry["compiler_version"],
        "global_configure_flags": entry["global_configure_flags"],
        "git_metadata_available": entry["git_metadata_available"],
        "git_commit": entry["git_commit"],
        "git_dirty": entry["git_dirty"],
        "git_diff_sha256": expected_git_diff,
        "source_snapshot_sha256": expected_source_snapshot,
    }
    for name, expected in expected_values.items():
        if validated[name] != expected:
            raise RunnerError("subprocess record mismatch for {0}".format(name))
    _validate_verification(validated, item, context)
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
            # The explicit launch context owns scheduler provenance. A local
            # run must not inherit stale identity from a parent job shell.
            environment.pop("GPU_SUITE_SCHEDULER", None)
            environment.pop("GPU_SUITE_SCHEDULER_JOB_ID", None)
            if context["scheduler"] is not None:
                environment["GPU_SUITE_SCHEDULER"] = context["scheduler"]
            if context["scheduler_job_id"] is not None:
                environment["GPU_SUITE_SCHEDULER_JOB_ID"] = context["scheduler_job_id"]
            if entry["git_dirty"] is True and git_diff_sha256 is not None:
                environment["GPU_SUITE_GIT_DIFF_SHA256"] = git_diff_sha256
            if entry["git_dirty"] is True and source_snapshot_sha256 is not None:
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
                        runtime_environment_sha256, git_diff_sha256,
                        source_snapshot_sha256,
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

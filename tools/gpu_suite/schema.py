"""Validation for initial result schema version 1."""

import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .scheduler import validate_scheduler_identity


class SchemaError(ValueError):
    pass


RAW_FIELDS = (
    "result_schema_version",
    "run_id",
    "record_timestamp",
    "measurement_start_timestamp",
    "measurement_end_timestamp",
    "system_label",
    "wave",
    "node_index",
    "hostname",
    "block_id",
    "scheduler",
    "scheduler_job_id",
    "implementation_order",
    "benchmark",
    "implementation",
    "scope",
    "problem_size",
    "secondary_size",
    "parameters",
    "precision",
    "cpu_backend",
    "cpu_backend_role",
    "series_role",
    "cpu_threads_requested",
    "cpu_threads_effective",
    "cpu_parallelism",
    "warmup",
    "repeat",
    "trial",
    "attempted",
    "failure_origin",
    "elapsed_total_sec",
    "elapsed_sec",
    "clock_id",
    "clock_resolution_sec",
    "verification_metrics",
    "verification_thresholds",
    "verification_primary_metric",
    "verification_status",
    "getrf_info",
    "getrs_info",
    "device_id",
    "gpu_name",
    "gpu_uuid",
    "cuda_driver_version",
    "compiler",
    "compiler_version",
    "global_configure_flags",
    "library_name",
    "library_version",
    "cuda_runtime_version",
    "git_metadata_available",
    "git_commit",
    "git_dirty",
    "git_diff_sha256",
    "source_snapshot_sha256",
    "config_sha256",
    "runtime_environment_sha256",
    "binary_sha256",
    "exit_code",
    "status",
    "message",
)

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
BENCHMARKS = {"cufft", "cublas", "cusparse", "cusolver", "curand", "thrust"}
IMPLEMENTATIONS = {"cpu", "cuda", "openacc"}
SCOPES = {"compute", "end-to-end"}
ORDERS = {
    ("cpu", "cuda", "openacc"),
    ("cpu", "openacc", "cuda"),
    ("cuda", "cpu", "openacc"),
    ("cuda", "openacc", "cpu"),
    ("openacc", "cpu", "cuda"),
    ("openacc", "cuda", "cpu"),
}
FAILURE_ORIGINS = {
    "benchmark",
    "verification",
    "subprocess",
    "output-validation",
    "prior-failure",
    "prerequisite",
}


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def _optional_string(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _optional_integer(value: Any) -> bool:
    return value is None or _is_integer(value)


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _safe_hostname_component(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value not in {"", ".", ".."}
        and "/" not in value
        and "\\" not in value
        and not any(ord(character) < 32 or ord(character) == 127
                    for character in value)
    )


def _require_finite_json_numbers(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        _require(math.isfinite(value), "nonfinite number at {0}".format(path))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json_numbers(item, "{0}[{1}]".format(path, index))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require(isinstance(key, str),
                     "non-string JSON key at {0}".format(path))
            _require_finite_json_numbers(item, "{0}.{1}".format(path, key))
        return
    raise SchemaError("non-JSON value at {0}".format(path))


def validate_raw_result(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate one exact schema-v1 raw record and return a plain copy."""

    _require(isinstance(record, Mapping), "raw result must be an object")
    missing = set(RAW_FIELDS).difference(record.keys())
    extra = set(record.keys()).difference(RAW_FIELDS)
    _require(not missing, "missing raw fields: {0}".format(sorted(missing)))
    _require(not extra, "unknown raw fields: {0}".format(sorted(extra)))
    _require(record["result_schema_version"] == 1, "unsupported result schema")
    run_id = record["run_id"]
    _require(isinstance(run_id, str) and RUN_ID_RE.fullmatch(run_id) is not None,
             "invalid run_id")
    _require(not run_id.startswith(".") and ".." not in run_id,
             "unsafe run_id")
    _require(isinstance(record["record_timestamp"], str) and
             TIMESTAMP_RE.fullmatch(record["record_timestamp"]) is not None,
             "invalid record timestamp")
    for name in ("measurement_start_timestamp", "measurement_end_timestamp"):
        value = record[name]
        _require(value is None or
                 (isinstance(value, str) and TIMESTAMP_RE.fullmatch(value) is not None),
                 "invalid {0}".format(name))
    _require(record["measurement_end_timestamp"] is None or
             record["measurement_start_timestamp"] is not None,
             "measurement end requires a start")
    for name in ("system_label", "hostname", "compiler", "compiler_version",
                 "global_configure_flags", "library_name", "message"):
        _require(isinstance(record[name], str), "{0} must be a string".format(name))
    _require(_safe_hostname_component(record["hostname"]),
             "hostname must be a safe directory component")
    for name in ("wave", "node_index", "warmup", "repeat", "trial"):
        _require(_is_integer(record[name]) and record[name] >= 0,
                 "invalid {0}".format(name))
    _require(record["repeat"] > 0, "repeat must be positive")
    _require(record["block_id"] == "{0}|{1}|{2}".format(
        run_id, record["wave"], record["hostname"]), "invalid block_id")
    try:
        validate_scheduler_identity(
            record["scheduler"], record["scheduler_job_id"]
        )
    except ValueError as error:
        raise SchemaError("invalid scheduler identity: {0}".format(error)) from error
    order = record["implementation_order"]
    _require(isinstance(order, list) and tuple(order) in ORDERS,
             "invalid implementation_order")
    _require(record["benchmark"] in BENCHMARKS, "invalid benchmark")
    _require(record["implementation"] in IMPLEMENTATIONS,
             "invalid implementation")
    _require(record["scope"] in SCOPES, "invalid scope")
    for name in ("problem_size", "secondary_size"):
        value = record[name]
        _require(value is None or (_is_integer(value) and value > 0),
                 "invalid {0}".format(name))
    _require(isinstance(record["parameters"], dict), "parameters must be an object")
    _require_finite_json_numbers(record["parameters"], "parameters")
    if "source_language" in record["parameters"]:
        _require(record["parameters"]["source_language"] == "fortran",
                 "source_language extension must identify Fortran")
    _require(isinstance(record["precision"], str), "precision must be a string")
    for name in ("cpu_backend", "cpu_backend_role", "cpu_parallelism",
                 "verification_primary_metric", "gpu_name", "gpu_uuid",
                 "cuda_driver_version", "library_version", "cuda_runtime_version"):
        _require(_optional_string(record[name]), "invalid {0}".format(name))
    _require(record["series_role"] in {"primary", "auxiliary"},
             "invalid series_role")
    for name in ("cpu_threads_requested", "cpu_threads_effective", "getrf_info",
                 "getrs_info", "device_id", "exit_code"):
        _require(_optional_integer(record[name]), "invalid {0}".format(name))
    _require(record["device_id"] is None or record["device_id"] >= 0,
             "device_id must be non-negative or null")
    if record["implementation"] == "cpu":
        _require(isinstance(record["cpu_backend"], str) and
                 record["cpu_backend"] != "", "CPU backend is required")
        _require(record["cpu_backend_role"] in {"production", "reference"},
                 "invalid CPU backend role")
        _require(record["cpu_parallelism"] in {"serial", "threaded", "unknown"},
                 "invalid CPU parallelism")
        _require(_is_integer(record["cpu_threads_requested"]) and
                 record["cpu_threads_requested"] > 0,
                 "CPU requested threads must be positive")
        _require(record["cpu_threads_effective"] is None or
                 record["cpu_threads_effective"] > 0,
                 "CPU effective threads must be positive or null")
    else:
        _require(record["cpu_backend"] is None and
                 record["cpu_backend_role"] is None and
                 record["cpu_threads_requested"] is None and
                 record["cpu_threads_effective"] is None and
                 record["cpu_parallelism"] is None,
                 "GPU row contains CPU metadata")
        if record["status"] == "success":
            for name in (
                "gpu_name", "gpu_uuid", "cuda_driver_version",
                "cuda_runtime_version", "library_version",
            ):
                _require(isinstance(record[name], str) and record[name] != "",
                         "successful GPU row requires {0}".format(name))
    _require(isinstance(record["attempted"], bool), "attempted must be boolean")
    origin = record["failure_origin"]
    _require(origin is None or origin in FAILURE_ORIGINS, "invalid failure_origin")
    for name in ("elapsed_total_sec", "elapsed_sec"):
        _require(record[name] is None or _finite_nonnegative(record[name]),
                 "invalid {0}".format(name))
    _require((record["elapsed_total_sec"] is None) ==
             (record["elapsed_sec"] is None), "elapsed fields must be paired")
    _require(record["elapsed_total_sec"] is None or
             (record["measurement_start_timestamp"] is not None and
              record["measurement_end_timestamp"] is not None),
             "elapsed values require measurement timestamps")
    _require(record["clock_id"] == "CLOCK_MONOTONIC", "invalid clock_id")
    _require(_finite_nonnegative(record["clock_resolution_sec"]) and
             record["clock_resolution_sec"] > 0.0, "invalid clock resolution")
    _require(isinstance(record["verification_metrics"], dict) and
             isinstance(record["verification_thresholds"], dict),
             "verification fields must be objects")
    _require_finite_json_numbers(record["verification_metrics"],
                                 "verification_metrics")
    _require_finite_json_numbers(record["verification_thresholds"],
                                 "verification_thresholds")
    for metric_name, metric_value in record["verification_metrics"].items():
        _require(metric_value is None or
                 (isinstance(metric_value, (int, float)) and
                  not isinstance(metric_value, bool) and
                  math.isfinite(float(metric_value))),
                 "invalid verification metric {0}".format(metric_name))
    _require(record["verification_status"] in
             {"pass", "failure", "skipped", "nonfinite"},
             "invalid verification_status")
    primary_metric = record["verification_primary_metric"]
    _require(primary_metric is None or
             primary_metric in record["verification_metrics"],
             "verification primary metric is absent from metrics")
    if record["verification_status"] == "nonfinite":
        _require(primary_metric is not None and
                 record["verification_metrics"][primary_metric] is None,
                 "nonfinite verification requires a null primary metric")
    available = record["git_metadata_available"]
    _require(isinstance(available, bool),
             "git_metadata_available must be boolean")
    if available:
        _require(isinstance(record["git_commit"], str) and
                 record["git_commit"] != "",
                 "available Git metadata requires a commit")
        _require(isinstance(record["git_dirty"], bool),
                 "available Git metadata requires dirty state")
    else:
        _require(record["git_commit"] is None and record["git_dirty"] is None,
                 "unavailable Git metadata requires null commit and dirty")
    for name in ("config_sha256", "runtime_environment_sha256", "binary_sha256"):
        _require(isinstance(record[name], str) and
                 SHA256_RE.fullmatch(record[name]) is not None,
                 "invalid {0}".format(name))
    for name in ("git_diff_sha256", "source_snapshot_sha256"):
        value = record[name]
        _require(value is None or
                 (isinstance(value, str) and SHA256_RE.fullmatch(value) is not None),
                 "invalid {0}".format(name))
    status = record["status"]
    _require(status in {"success", "failure", "skipped"}, "invalid status")
    if status == "success":
        _require(record["attempted"] and origin is None, "invalid success attempt")
        _require(record["elapsed_total_sec"] is not None, "success requires timing")
        _require(record["exit_code"] == 0, "success exit_code must be zero")
        _require(record["message"] == "", "success message must be empty")
        _require(record["verification_status"] in {"pass", "skipped"},
                 "success has invalid verification status")
    elif status == "failure":
        _require(record["attempted"] and origin is not None,
                 "failure must be attempted with an origin")
        _require(origin not in {"prior-failure", "prerequisite"},
                 "attempted failure has an unstarted origin")
    else:
        _require(not record["attempted"] and
                 origin in {"prior-failure", "prerequisite"},
                 "skipped trial must be unattempted with an origin")
        _require(record["measurement_start_timestamp"] is None and
                 record["measurement_end_timestamp"] is None and
                 record["elapsed_total_sec"] is None and
                 record["elapsed_sec"] is None and record["exit_code"] is None,
                 "skipped trial cannot contain execution data")
        _require(record["verification_status"] == "skipped",
                 "skipped trial verification must be skipped")
    if (origin == "verification" or
            record["verification_status"] in {"failure", "nonfinite"}):
        _require(status == "failure" and record["attempted"] and
                 origin == "verification" and
                 record["verification_status"] in {"failure", "nonfinite"},
                 "inconsistent verification failure state")
    if record["benchmark"] == "cusolver":
        _require(record["repeat"] == 1, "cuSOLVER repeat must be one")
    return dict(record)


def validate_trial_indices(records: Iterable[Mapping[str, Any]], trials: int) -> None:
    """Reject duplicate/out-of-range trials and require every expected index."""

    _require(_is_integer(trials) and trials > 0, "trials must be positive")
    seen = set()
    for record in records:
        validate_raw_result(record)
        index = record["trial"]
        _require(index < trials, "trial index out of range: {0}".format(index))
        _require(index not in seen, "duplicate trial index: {0}".format(index))
        seen.add(index)
    expected = set(range(trials))
    _require(seen == expected, "missing trial indices: {0}".format(sorted(expected - seen)))

"""Cross-record campaign validation beyond the single-row schema."""

from collections import Counter, defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple

from .config import SUPPORTED_BENCHMARKS
from .families import AMD_BENCHMARKS
from .ordering import implementation_order
from .runner import (core_raw_parameters, cpu_threads_effective_matches,
                     normalized_parameters, problem_sizes)
from .schema import validate_raw_result
from .strict_json import dumps


class CampaignValidationError(ValueError):
    pass


def _invocation_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    parameters = core_raw_parameters(record["benchmark"], record["parameters"])
    return (
        record["block_id"],
        record["benchmark"],
        record["implementation"],
        record["cpu_backend"],
        record["scope"],
        dumps(parameters),
    )


def _configured_invocations(config: Mapping[str, Any]) -> Dict[Tuple[Any, ...], Mapping[str, Any]]:
    expected = {}
    for benchmark in SUPPORTED_BENCHMARKS:
        if benchmark not in config["benchmarks"]:
            continue
        definition = config["benchmarks"][benchmark]
        if not definition["enabled"]:
            continue
        for case in definition["cases"]:
            parameters = normalized_parameters(benchmark, case["parameters"])
            if config.get("source_language", "c-cpp") == "fortran":
                parameters["source_language"] = "fortran"
            signature = dumps(parameters)
            primary_size, secondary_size = problem_sizes(
                benchmark, case["parameters"]
            )
            for scope_name, scope in case["scopes"].items():
                for series in definition["series"]:
                    key = (
                        benchmark,
                        series["implementation"],
                        series["cpu_backend"],
                        scope_name,
                        signature,
                    )
                    if key in expected:
                        raise CampaignValidationError("duplicate configured invocation")
                    expected[key] = {
                        "parameters": parameters,
                        "precision": definition["precision"],
                        "problem_size": primary_size,
                        "secondary_size": secondary_size,
                        "scope": scope,
                        "series": series,
                    }
    return expected


def validate_rocfft_provenance(records: Sequence[Mapping[str, Any]]) -> None:
    """Runtime is not a mathematical key, but cannot vary inside a campaign."""
    amd = [row for row in records if row["benchmark"] in AMD_BENCHMARKS]
    if not amd:
        return
    if len(amd) != len(records):
        raise CampaignValidationError("AMD and NVIDIA require separate campaigns")
    for field in ("run_id", "config_sha256", "runtime_environment_sha256",
                  "git_metadata_available", "git_commit", "git_dirty",
                  "git_diff_sha256", "source_snapshot_sha256"):
        if len({row[field] for row in amd}) != 1:
            raise CampaignValidationError("mixed AMD provenance: " + field)
    artifacts = {}
    runtimes = {}
    aocl_versions = set()
    aocl_environments = set()
    gpu_versions = set()
    for row in amd:
        runtime = row["parameters"].get("runtime")
        if (row["benchmark"] == "rocsparse" and row["implementation"] == "cpu"
                and row["cpu_backend"] == "cpu-aocl-sparse"):
            # Failed invocations retain known observations. Missing observations
            # (including synthetic-output failures) must not become invented ones.
            version = row["library_version"]
            if isinstance(version, str) and version:
                aocl_versions.add(version)
            if runtime is not None and "aocl_sparse_environment" in runtime:
                aocl_environments.add(dumps(runtime["aocl_sparse_environment"]))
            if len(aocl_versions) > 1 or len(aocl_environments) > 1:
                raise CampaignValidationError("mixed AOCL Sparse observed runtime provenance")
        if runtime is None:
            if row["status"] == "success":
                raise CampaignValidationError("missing AMD runtime provenance")
            continue
        identity = (row["benchmark"], row["implementation"])
        signature = tuple(row[field] for field in
                          ("binary_sha256", "compiler", "compiler_version", "global_configure_flags"))
        signature += (runtime["build_metadata_sha256"],)
        if artifacts.setdefault(identity, signature) != signature:
            raise CampaignValidationError("mixed AMD artifact/compiler/build provenance")
        if row["status"] == "success":
            versions = (dumps(runtime), row["library_name"], row["library_version"])
            # Serial/threaded CPU backends may be explicit auxiliary series.
            key = (identity, row["cpu_backend"])
            if runtimes.setdefault(key, versions) != versions:
                raise CampaignValidationError("mixed AMD resolved-library/runtime provenance")
            if row["implementation"] in {"hip", "openmp"}:
                gpu_versions.add((runtime.get("hip_runtime_version"), runtime.get("hip_driver_version")))
                version_key = (row["benchmark"], "gpu-library")
                if runtimes.setdefault(version_key, row["library_version"]) != row["library_version"]:
                    raise CampaignValidationError("HIP/OpenMP library versions differ")
    if len(gpu_versions) > 1:
        raise CampaignValidationError("HIP/OpenMP runtime or library versions differ")


def validate_campaign(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
    config_sha256: str, manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    if not records:
        raise CampaignValidationError("campaign has no raw records")
    validated = [validate_raw_result(record) for record in records]
    validate_rocfft_provenance(validated)
    if len({record["run_id"] for record in validated}) != 1:
        raise CampaignValidationError("mixed run IDs")
    if len({record["runtime_environment_sha256"] for record in validated}) != 1:
        raise CampaignValidationError("mixed runtime environment hashes")
    if {record["config_sha256"] for record in validated} != {config_sha256}:
        raise CampaignValidationError("raw configuration hash mismatch")

    entries = {}
    for entry in manifest["entries"]:
        language = "fortran" if entry["compiler_language"] == "fortran" else "c-cpp"
        if language != config.get("source_language", "c-cpp"):
            raise CampaignValidationError("configuration and manifest source language differ")
        if entry["executable_role"] != "benchmark":
            continue
        key = (entry["library"], entry["implementation"])
        if key in entries:
            raise CampaignValidationError("duplicate benchmark manifest entry")
        entries[key] = entry
    expected = _configured_invocations(config)
    groups = defaultdict(list)  # type: DefaultDict[Tuple[Any, ...], List[Mapping[str, Any]]]
    blocks = {}  # type: Dict[str, Mapping[str, Any]]
    wave_node_blocks = {}  # type: Dict[Tuple[int, int], str]
    for record in validated:
        representative = blocks.setdefault(record["block_id"], record)
        for name in (
            "run_id", "wave", "node_index", "hostname", "block_id",
            "scheduler", "scheduler_job_id", "implementation_order",
            "runtime_environment_sha256",
        ):
            if record[name] != representative[name]:
                raise CampaignValidationError(
                    "inconsistent block metadata for {0}".format(name)
                )
        expected_order = list(
            implementation_order(record["node_index"], record["wave"], config.get("vendor", "nvidia"))
        )
        if record["implementation_order"] != expected_order:
            raise CampaignValidationError("implementation order formula mismatch")
        wave_node = (record["wave"], record["node_index"])
        previous_block = wave_node_blocks.setdefault(wave_node, record["block_id"])
        if previous_block != record["block_id"]:
            raise CampaignValidationError("one wave/node index maps to multiple blocks")
        key = _invocation_key(record)
        groups[key].append(record)
        entry = entries.get((record["benchmark"], record["implementation"]))
        if entry is None:
            if record["status"] != "skipped" or record["failure_origin"] != "prerequisite":
                raise CampaignValidationError("unmanifested executable has attempted rows")
        else:
            if (record["cpu_backend"] == "cpu-aocl-sparse" and record["attempted"]
                    and entry["supported_cpu_backends"] != ["cpu-aocl-sparse"]):
                raise CampaignValidationError("AOCL Sparse backend differs from manifest")
            if record["benchmark"] in AMD_BENCHMARKS:
                runtime = record["parameters"].get("runtime")
                if runtime is not None and runtime.get("build_metadata_sha256") != entry["build_metadata_sha256"]:
                    raise CampaignValidationError("AMD build metadata hash differs from manifest")
            if record["binary_sha256"] != entry["binary_sha256"]:
                raise CampaignValidationError("raw binary hash mismatch")
            if record["compiler"] != entry["compiler"] or record["compiler_version"] != entry["compiler_version"]:
                raise CampaignValidationError("raw compiler metadata mismatch")
            if record["global_configure_flags"] != entry["global_configure_flags"]:
                raise CampaignValidationError("raw configure flags mismatch")
            if (record["git_metadata_available"] != entry["git_metadata_available"] or
                    record["git_commit"] != entry["git_commit"] or
                    record["git_dirty"] != entry["git_dirty"]):
                raise CampaignValidationError("raw Git metadata mismatch")

    for block_id in blocks:
        for configured_key, definition in expected.items():
            benchmark, implementation, cpu_backend, scope_name, signature = configured_key
            key = (block_id, benchmark, implementation, cpu_backend, scope_name, signature)
            invocation = groups.get(key)
            if invocation is None:
                raise CampaignValidationError("missing invocation: {0}".format(key))
            scope = definition["scope"]
            indices = [record["trial"] for record in invocation]
            if len(indices) != len(set(indices)):
                raise CampaignValidationError("duplicate trial index")
            if set(indices) != set(range(scope["trials"])):
                raise CampaignValidationError("missing or out-of-range trial index")
            fatal_failure_seen = False
            skipped_origin = None
            for record in sorted(invocation, key=lambda item: item["trial"]):
                if record["warmup"] != scope["warmup"] or record["repeat"] != scope["repeat"]:
                    raise CampaignValidationError("scope settings differ across implementation")
                series = definition["series"]
                if (record["cpu_backend_role"] != series["cpu_backend_role"] or
                        record["series_role"] != series["series_role"] or
                        record["cpu_parallelism"] != series["cpu_parallelism"] or
                        not cpu_threads_effective_matches(record, series)):
                    raise CampaignValidationError("series classification mismatch")
                if record["precision"] != definition["precision"]:
                    raise CampaignValidationError("precision differs from configuration")
                if (
                    record["problem_size"] != definition["problem_size"]
                    or record["secondary_size"] != definition["secondary_size"]
                ):
                    raise CampaignValidationError("problem size differs from configuration")
                if core_raw_parameters(benchmark, record["parameters"]) != definition["parameters"]:
                    raise CampaignValidationError("parameters differ from configuration")
                if fatal_failure_seen and not (
                    record["status"] == "skipped"
                    and record["failure_origin"] == "prior-failure"
                ):
                    raise CampaignValidationError(
                        "trials after failure must be prior-failure skips"
                    )
                if (
                    record["status"] == "failure"
                    and record["failure_origin"] != "verification"
                ):
                    fatal_failure_seen = True
                if record["status"] == "skipped":
                    if skipped_origin is None:
                        skipped_origin = record["failure_origin"]
                    elif record["failure_origin"] != skipped_origin:
                        raise CampaignValidationError(
                            "one invocation mixes skipped origins"
                        )

    statuses = Counter(record["status"] for record in validated)
    origins = Counter(
        record["failure_origin"]
        for record in validated
        if record["failure_origin"] is not None
    )
    return {
        "block_count": len(blocks),
        "failure_origin_counts": dict(sorted(origins.items())),
        "record_count": len(validated),
        "run_id": validated[0]["run_id"],
        "runtime_environment_sha256": validated[0]["runtime_environment_sha256"],
        "status_counts": dict(sorted(statuses.items())),
        "validation_status": (
            "pass"
            if set(statuses).issubset({"success"})
            else "failure"
        ),
        "validation_summary_schema_version": 1,
        "wave_count": len({record["wave"] for record in validated}),
    }

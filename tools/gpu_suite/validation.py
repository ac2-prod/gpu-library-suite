"""Cross-record campaign validation beyond the single-row schema."""

from collections import Counter, defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple

from .config import BENCHMARKS
from .ordering import implementation_order
from .runner import core_raw_parameters, normalized_parameters, problem_sizes
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
    for benchmark in BENCHMARKS:
        definition = config["benchmarks"][benchmark]
        if not definition["enabled"]:
            continue
        for case in definition["cases"]:
            parameters = normalized_parameters(benchmark, case["parameters"])
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


def validate_campaign(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
    config_sha256: str, manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    if not records:
        raise CampaignValidationError("campaign has no raw records")
    validated = [validate_raw_result(record) for record in records]
    if len({record["run_id"] for record in validated}) != 1:
        raise CampaignValidationError("mixed run IDs")
    if len({record["runtime_environment_sha256"] for record in validated}) != 1:
        raise CampaignValidationError("mixed runtime environment hashes")
    if {record["config_sha256"] for record in validated} != {config_sha256}:
        raise CampaignValidationError("raw configuration hash mismatch")

    entries = {}
    for entry in manifest["entries"]:
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
            implementation_order(record["node_index"], record["wave"])
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
                        record["cpu_threads_effective"] != series["cpu_threads_effective"]):
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

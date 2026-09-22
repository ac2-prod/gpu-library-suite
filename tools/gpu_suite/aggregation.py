"""Hierarchical raw-trial aggregation with paired production-CPU speedups."""

import statistics
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import BENCHMARKS
from .ordering import permutation_index, size_order_index
from .runner import core_raw_parameters
from .strict_json import dumps


def statistical_summary(values: Sequence[float]) -> Dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "aggregate_status": "no_valid_samples",
            "iqr": None,
            "maximum": None,
            "median": None,
            "minimum": None,
            "q1": None,
            "q3": None,
        }
    median = statistics.median(ordered)
    if len(ordered) == 1:
        return {
            "aggregate_status": "insufficient_sample_count",
            "iqr": None,
            "maximum": ordered[0],
            "median": median,
            "minimum": ordered[0],
            "q1": None,
            "q3": None,
        }
    q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "aggregate_status": "success",
        "iqr": q3 - q1,
        "maximum": ordered[-1],
        "median": median,
        "minimum": ordered[0],
        "q1": q1,
        "q3": q3,
    }


def _raw_identity(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    parameters = core_raw_parameters(record["benchmark"], record["parameters"])
    return (
        record["run_id"],
        record["runtime_environment_sha256"],
        record["wave"],
        record["hostname"],
        record["block_id"],
        record["benchmark"],
        record["implementation"],
        record["scope"],
        record["problem_size"],
        record["secondary_size"],
        dumps(parameters),
        record["cpu_backend"],
        record["cpu_backend_role"],
        record["series_role"],
    )


def _counts(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    valid = 0
    attempted_failure = 0
    skipped = 0
    exclusions = {}  # type: Dict[str, int]
    for record in records:
        if (
            record["status"] == "success"
            and record["verification_status"] in {"pass", "skipped"}
            and record["elapsed_sec"] is not None
        ):
            valid += 1
        elif record["attempted"]:
            attempted_failure += 1
            origin = record["failure_origin"] or "verification"
            exclusions[origin] = exclusions.get(origin, 0) + 1
        else:
            skipped += 1
            origin = record["failure_origin"] or "unknown"
            exclusions[origin] = exclusions.get(origin, 0) + 1
    return {
        "attempted_failure_count": attempted_failure,
        "exclusion_counts_by_failure_origin": exclusions,
        "unattempted_skipped_count": skipped,
        "valid_success_count": valid,
    }


def _base_record(
    summary_level: str, run_id: str, runtime_hash: str, benchmark: str,
    implementation: str, comparison: Optional[str], scope: str,
    problem_size: Any, secondary_size: Any, parameter_signature: str,
    parameters: Mapping[str, Any], cpu_backend: Optional[str],
    series_role: str, summary_input_statistic: str,
) -> Dict[str, Any]:
    return {
        "aggregate_schema_version": 1,
        "aggregate_status": "no_valid_samples",
        "attempted_failure_count": 0,
        "benchmark": benchmark,
        "block_id": None,
        "comparison": comparison,
        "cpu_backend": cpu_backend,
        "exclusion_counts_by_failure_origin": {},
        "hostname": None,
        "implementation": implementation,
        "iqr": None,
        "maximum": None,
        "median": None,
        "minimum": None,
        "parameter_signature": parameter_signature,
        "parameters": dict(parameters),
        "problem_size": problem_size,
        "q1": None,
        "q3": None,
        "run_id": run_id,
        "runtime_environment_sha256": runtime_hash,
        "scope": scope,
        "secondary_size": secondary_size,
        "series_role": series_role,
        "summary_input_statistic": summary_input_statistic,
        "summary_level": summary_level,
        "unattempted_skipped_count": 0,
        "valid_success_count": 0,
        "wave": None,
    }


def _performance_values(records: Sequence[Mapping[str, Any]]) -> List[float]:
    return [
        float(record["elapsed_sec"])
        for record in records
        if record["status"] == "success"
        and record["verification_status"] in {"pass", "skipped"}
        and record["elapsed_sec"] is not None
    ]


def block_records(raw_records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups = defaultdict(list)  # type: DefaultDict[Tuple[Any, ...], List[Mapping[str, Any]]]
    for record in raw_records:
        groups[_raw_identity(record)].append(record)
    results = []  # type: List[Dict[str, Any]]
    for identity in sorted(groups, key=lambda value: tuple("" if item is None else str(item) for item in value)):
        records = groups[identity]
        first = records[0]
        parameters = core_raw_parameters(first["benchmark"], first["parameters"])
        parameter_signature = dumps(parameters)
        aggregate = _base_record(
            "block", first["run_id"], first["runtime_environment_sha256"],
            first["benchmark"], first["implementation"], None, first["scope"],
            first["problem_size"], first["secondary_size"], parameter_signature,
            parameters, first["cpu_backend"], first["series_role"],
            "trial_elapsed_sec",
        )
        aggregate.update(_counts(records))
        aggregate.update(statistical_summary(_performance_values(records)))
        aggregate["wave"] = first["wave"]
        aggregate["hostname"] = first["hostname"]
        aggregate["block_id"] = first["block_id"]
        results.append(aggregate)
    return results


def _series_identity(record: Mapping[str, Any], include_wave: bool) -> Tuple[Any, ...]:
    values = (
        record["run_id"],
        record["runtime_environment_sha256"],
        record["benchmark"],
        record["implementation"],
        record["comparison"],
        record["scope"],
        record["problem_size"],
        record["secondary_size"],
        record["parameter_signature"],
        record["cpu_backend"],
        record["series_role"],
    )
    return values + ((record["wave"],) if include_wave else ())


def add_block_speedups(
    blocks: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    results = [dict(record) for record in blocks]
    omissions = []  # type: List[Dict[str, Any]]
    lookup = {}
    for record in blocks:
        key = (
            record["run_id"], record["runtime_environment_sha256"], record["wave"],
            record["hostname"], record["benchmark"], record["scope"],
            record["problem_size"], record["secondary_size"],
            record["parameter_signature"], record["implementation"],
            record["cpu_backend"], record["series_role"],
        )
        lookup[key] = record
    for gpu in blocks:
        if gpu["implementation"] not in {"cuda", "openacc"} or gpu["series_role"] != "primary":
            continue
        backend = config["benchmarks"][gpu["benchmark"]]["default_speedup_cpu_backend"]
        cpu_key = (
            gpu["run_id"], gpu["runtime_environment_sha256"], gpu["wave"],
            gpu["hostname"], gpu["benchmark"], gpu["scope"], gpu["problem_size"],
            gpu["secondary_size"], gpu["parameter_signature"], "cpu", backend,
            "primary",
        )
        cpu = lookup.get(cpu_key)
        if cpu is None or cpu["median"] is None or gpu["median"] is None or gpu["median"] <= 0.0:
            omissions.append(
                {
                    "benchmark": gpu["benchmark"],
                    "block_id": gpu["block_id"],
                    "comparison": "cpu/{0}".format(gpu["implementation"]),
                    "problem_size": gpu["problem_size"],
                    "reason": "configured production CPU backend is unavailable or invalid",
                    "scope": gpu["scope"],
                }
            )
            continue
        speedup = float(cpu["median"]) / float(gpu["median"])
        result = _base_record(
            "block", gpu["run_id"], gpu["runtime_environment_sha256"],
            gpu["benchmark"], "speedup", "cpu/{0}".format(gpu["implementation"]),
            gpu["scope"], gpu["problem_size"], gpu["secondary_size"],
            gpu["parameter_signature"], gpu["parameters"], backend, "primary",
            "paired_block_median_ratio",
        )
        result.update(statistical_summary([speedup]))
        result["valid_success_count"] = 1
        result["wave"] = gpu["wave"]
        result["hostname"] = gpu["hostname"]
        result["block_id"] = gpu["block_id"]
        results.append(result)
    return results, omissions


def summarize_children(
    children: Sequence[Mapping[str, Any]], summary_level: str,
    summary_input_statistic: str, include_wave: bool,
) -> List[Dict[str, Any]]:
    groups = defaultdict(list)  # type: DefaultDict[Tuple[Any, ...], List[Mapping[str, Any]]]
    for child in children:
        groups[_series_identity(child, include_wave)].append(child)
    results = []  # type: List[Dict[str, Any]]
    for identity in sorted(groups, key=lambda value: tuple("" if item is None else str(item) for item in value)):
        group = groups[identity]
        first = group[0]
        values = [float(record["median"]) for record in group if record["median"] is not None]
        result = _base_record(
            summary_level, first["run_id"], first["runtime_environment_sha256"],
            first["benchmark"], first["implementation"], first["comparison"],
            first["scope"], first["problem_size"], first["secondary_size"],
            first["parameter_signature"], first["parameters"],
            first["cpu_backend"], first["series_role"], summary_input_statistic,
        )
        result.update(statistical_summary(values))
        result["valid_success_count"] = len(values)
        result["attempted_failure_count"] = sum(
            1
            for record in group
            if record["median"] is None and record["attempted_failure_count"] > 0
        )
        result["unattempted_skipped_count"] = sum(
            1
            for record in group
            if record["median"] is None
            and record["attempted_failure_count"] == 0
            and record["unattempted_skipped_count"] > 0
        )
        exclusions = {}  # type: Dict[str, int]
        for record in group:
            for origin, count in record["exclusion_counts_by_failure_origin"].items():
                exclusions[origin] = exclusions.get(origin, 0) + count
        result["exclusion_counts_by_failure_origin"] = exclusions
        if include_wave:
            result["wave"] = first["wave"]
        results.append(result)
    return results


def aggregate_results(
    raw_records: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
    include_pooled: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    blocks = block_records(raw_records)
    blocks_with_speedups, omissions = add_block_speedups(blocks, config)
    waves = summarize_children(
        blocks_with_speedups, "wave", "block_median", include_wave=True
    )
    cross_wave = summarize_children(
        waves, "cross-wave", "wave_median", include_wave=False
    )
    pooled = []
    if include_pooled:
        pooled = summarize_children(
            blocks_with_speedups, "pooled-exploratory", "block_median",
            include_wave=False,
        )
    records = blocks_with_speedups + waves + cross_wave + pooled
    records.sort(
        key=lambda record: (
            {"block": 0, "wave": 1, "cross-wave": 2, "pooled-exploratory": 3}[record["summary_level"]],
            -1 if record["wave"] is None else record["wave"],
            "" if record["hostname"] is None else record["hostname"],
            record["benchmark"], record["scope"],
            -1 if record["problem_size"] is None else record["problem_size"],
            record["implementation"],
            "" if record["comparison"] is None else record["comparison"],
        )
    )
    hosts = {record["hostname"] for record in raw_records}
    waves_seen = {record["wave"] for record in raw_records}
    blocks_seen = {record["block_id"] for record in raw_records}
    block_representatives = {}  # type: Dict[str, Mapping[str, Any]]
    for record in raw_records:
        block_representatives.setdefault(record["block_id"], record)
    permutation_counts = {str(index): 0 for index in range(6)}
    size_order_counts = {str(index): 0 for index in range(2)}
    joint_counts = {
        "{0}:{1}".format(permutation, size_order): 0
        for permutation in range(6)
        for size_order in range(2)
    }
    for record in block_representatives.values():
        permutation = permutation_index(record["node_index"], record["wave"])
        size_order = size_order_index(record["node_index"])
        permutation_counts[str(permutation)] += 1
        size_order_counts[str(size_order)] += 1
        joint_counts["{0}:{1}".format(permutation, size_order)] += 1
    metadata = {
        "aggregate_schema_version": 1,
        "block_count": len(blocks_seen),
        "blocks_per_wave": {
            str(wave): len({record["block_id"] for record in raw_records if record["wave"] == wave})
            for wave in sorted(waves_seen)
        },
        "curand_comparison_note": "Same distribution and output type task; different RNG algorithms.",
        "permutation_assignment_counts": permutation_counts,
        "permutation_size_order_counts": joint_counts,
        "size_order_assignment_counts": size_order_counts,
        "speedup_omissions": omissions,
        "unique_hostname_count": len(hosts),
        "wave_count": len(waves_seen),
    }
    return records, metadata

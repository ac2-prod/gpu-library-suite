"""Pure six-family smoke contracts; the caller supplies pinned suite imports.

This module never changes sys.path, discovers a checkout, launches a process or
writes results. Config paths, manifest and independently captured run context
must come from the caller's verified source/artifact snapshot.
"""

import copy
import hashlib
import math
import re
from collections import Counter

from gpu_suite.config import validate_config
from gpu_suite.families import AMD_BENCHMARKS
from gpu_suite.manifest import merge_entries
from gpu_suite.runner import (
    AOCL_SPARSE_ENVIRONMENT_KEYS, _expected_verification,
    aocl_sparse_environment_overrides, build_schedule, core_raw_parameters,
    normalized_parameters, validate_subprocess_record,
)
from gpu_suite.strict_json import dump_bytes, dumps, load
from gpu_suite.validation import validate_campaign


CAMPAIGN_THREADS = {"serial": 1, "threaded": 2}
FORMATS = ("jsonl", "csv")
CAMPAIGNS = tuple(name + "-" + fmt for name in CAMPAIGN_THREADS for fmt in FORMATS)
CPU_BACKENDS = {
    "rocblas": "cpu-generic-cblas", "rocsparse": "cpu-aocl-sparse",
    "rocsolver": "cpu-generic-lapacke", "rocrand": "cpu-std-random-serial",
    "rocthrust": "cpu-stl-serial",
}
SECOND_CASE_PARAMETERS = {
    "rocblas": {"size": 31, "alpha": 1.25, "beta": 0.5},
    "rocsparse": {"size": 100, "alpha": 1.25, "beta": 0.5},
    "rocsolver": {"size": 32, "nrhs": 2},
    "rocrand": {"size": 8192, "generator": "pseudo-default", "distribution": "uniform-double",
                "seed": 1234, "offset": 0, "order": "default"},
    "rocthrust": {"size": 8192, "operation": "transform-reduce-square-sum"},
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def selected_libraries(library):
    if library is None:
        return tuple(AMD_BENCHMARKS)
    values = (library,) if isinstance(library, str) else tuple(library)
    _require(values and len(values) == len(set(values))
             and set(values).issubset(SECOND_CASE_PARAMETERS), "unsupported focused smoke libraries")
    return values


def _hash(value, name, optional=False):
    _require((optional and value is None) or
             (isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)),
             "invalid " + name)


def _config(config, campaign, library=None):
    _require(campaign in CAMPAIGN_THREADS, "unknown smoke campaign")
    libraries = selected_libraries(library)
    result = validate_config(copy.deepcopy(config))
    threads = CAMPAIGN_THREADS[campaign]
    _require(result.get("vendor") == "amd" and result.get("source_language", "c-cpp") == "c-cpp"
             and result["run_mode"] == "smoke" and result["continue_on_failure"] is (library is None),
             "smoke requires AMD C/C++ and " + ("stop on failure" if library else "diagnostic continuation"))
    _require(result["cpu_threads"] == threads, "campaign CPU thread request differs")
    _require(set(result["benchmarks"]) == set(libraries),
             "smoke library selection differs")
    case_count = 2 if library else 1
    for library, definition in result["benchmarks"].items():
        _require(definition["enabled"] and len(definition["cases"]) == case_count,
                 "smoke case coverage differs")
        series = definition["series"]
        _require(len(series) == 3 and {item["implementation"] for item in series} == {"cpu", "hip", "openmp"}
                 and all(item["series_role"] == "primary" for item in series),
                 "each family requires exactly CPU/HIP/OpenMP primary series")
        cpu = next(item for item in series if item["implementation"] == "cpu")
        backend = "cpu-fftw-" + ("serial" if threads == 1 else "threaded") if library == "rocfft" else CPU_BACKENDS[library]
        serial = library in {"rocrand", "rocthrust"} or (library == "rocfft" and threads == 1)
        effective = (1 if serial else threads) if library in {"rocfft", "rocrand", "rocthrust"} else None
        _require(cpu["cpu_backend"] == backend and definition["default_speedup_cpu_backend"] == backend
                 and cpu["cpu_backend_role"] == "production"
                 and cpu["cpu_parallelism"] == ("serial" if serial else "threaded")
                 and cpu["cpu_threads_effective"] == effective,
                 "smoke CPU provider/thread policy differs for " + library)
        for case in definition["cases"]:
            scopes = case["scopes"]
            _require(set(scopes) == {"compute", "end-to-end"}, "both timing scopes are required")
            for scope in scopes.values():
                _require(scope == {"warmup": 1, "repeat": 1 if library == "rocsolver" else 2, "trials": 2},
                         "smoke requires the reviewed warm-up/repeat/two-trial matrix")
    return result


def load_smoke_config(config_path, *, campaign, output_format):
    """Load one explicitly supplied pinned config; change only its output format."""
    _require(output_format in FORMATS, "unknown smoke output format")
    result = _config(load(config_path), campaign)
    result["output_format"] = output_format
    return result


def campaign_configs(serial_path, threaded_path, library=None):
    """Return four independent effective config documents without writing them."""
    paths = {"serial": serial_path, "threaded": threaded_path}
    definitions = {name + "-" + fmt: load_smoke_config(paths[name], campaign=name, output_format=fmt)
                   for name in CAMPAIGN_THREADS for fmt in FORMATS}
    libraries = selected_libraries(library)
    if library:
        for label, config in definitions.items():
            config["benchmarks"] = {name: config["benchmarks"][name] for name in libraries}
            config["continue_on_failure"] = False
            # Retain each pinned first case and add one small, explicit case.
            for name in libraries:
                cases = config["benchmarks"][name]["cases"]
                second = copy.deepcopy(cases[0])
                second["parameters"] = copy.deepcopy(SECOND_CASE_PARAMETERS[name])
                cases.append(second)
            definitions[label] = _config(config, label.split("-")[0], library)
    return definitions


def _metric(value, name):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value) and value >= 0, "invalid smoke numerical metric: " + name)
    return value


def _numerical_pass(row, item, context):
    # Reference scales and statistical bounds are owned by the canonical runner.
    # Here we only apply its declared threshold methods to observed metrics.
    thresholds, _, _ = _expected_verification(item, context)
    metrics = row["verification_metrics"]
    for name, threshold in thresholds.items():
        method = threshold["method"]
        if method == "inclusive-range":
            lower = _metric(metrics["observed_min"], "observed_min")
            upper = _metric(metrics["observed_max"], "observed_max")
            passed = threshold["lower_bound"] <= lower <= upper <= threshold["upper_bound"]
        else:
            value = _metric(metrics[name], name)
            if method in {"relative-upper-bound", "absolute-upper-bound"}:
                passed = value <= threshold["upper_bound"]
            elif method == "absolute-plus-relative":
                passed = value <= threshold["abs_tolerance"] + threshold["rel_tolerance"] * threshold["reference_scale"]
            elif method in {"uniform-mean-sigma-bound", "uniform-second-central-moment-sigma-bound"}:
                expected = threshold["expected_mean" if method == "uniform-mean-sigma-bound" else "expected_second_central_moment"]
                passed = abs(value - expected) <= threshold["absolute_bound"]
            else:
                raise ValueError("unsupported canonical verification threshold method: " + method)
        _require(passed, "smoke numerical metric exceeds canonical threshold: " + name)
    if item["benchmark"] == "rocsolver":
        _require(row["getrf_info"] == 0 and row["getrs_info"] == 0,
                 "successful solver smoke requires zero factorization/solve info")


def validate_smoke_rows(rows, config, manifest, *, context, config_hash,
                        runtime_hash, source_hash, git_diff_hash, library=None):
    """Require 72 actual, verified rows from the exact independent run context.

    context contains the normal runner context and aocl_sparse_environment, the
    four startup controls independently captured for this campaign. Validation
    does not infer expected identity or environment from the rows under test.
    """
    name = next((name for name, count in CAMPAIGN_THREADS.items() if count == config.get("cpu_threads")), None)
    effective = _config(config, name, library)
    libraries = selected_libraries(library)
    invocations = len(libraries) * (12 if library else 6)
    row_count = invocations * 2
    _require(effective["output_format"] in FORMATS, "unknown smoke output format")
    for value, label in ((config_hash, "config hash"), (runtime_hash, "runtime hash"),
                         (source_hash, "source hash")):
        _hash(value, label)
    _hash(git_diff_hash, "Git diff hash", optional=True)
    _require(context["config"] == effective and context["cpu_threads"] == effective["cpu_threads"],
             "runner context differs from smoke configuration")
    _require(context["wave"] == 0 and context["node_index"] == 0,
             "smoke requires one node block in wave zero")
    _require(manifest.get("manifest_schema_version") == 1, "unsupported smoke manifest")
    checked_manifest = dict(manifest, entries=merge_entries([manifest["entries"]]))
    expected_artifacts = {(family, implementation) for family in libraries
                          for implementation in ("cpu", "hip", "openmp")}
    _require({(entry["library"], entry["implementation"]) for entry in checked_manifest["entries"]
              if entry["executable_role"] == "benchmark"} == expected_artifacts,
             "all selected benchmark artifacts are required")
    environment = context.get("aocl_sparse_environment")
    _require(isinstance(environment, dict) and set(environment) == set(AOCL_SPARSE_ENVIRONMENT_KEYS)
             and all(value is None or isinstance(value, str) for value in environment.values()),
             "independent AOCL Sparse startup environment is required")
    schedule = build_schedule(effective, checked_manifest, context)
    _require(len(schedule) == invocations and all(item["entry"] is not None and not item["prerequisite_failure"] for item in schedule),
             "all smoke invocations require their selected artifact/backend")
    expected = {}
    for item in schedule:
        overrides = aocl_sparse_environment_overrides(item, context)
        if overrides:
            _require(all(environment[key] == value for key, value in overrides.items()),
                     "AOCL Sparse startup controls differ from campaign")
            item["aocl_sparse_environment"] = dict(environment)
        prefix = (item["benchmark"], item["series"]["implementation"], item["scope"],
                  dumps(normalized_parameters(item["benchmark"], item["parameters"])))
        for trial in range(item["scope_settings"]["trials"]):
            key = prefix + (trial,)
            _require(key not in expected, "duplicate configured smoke row")
            expected[key] = item
    _require(len(rows) == len(expected) == row_count, "smoke row count differs")
    report = validate_campaign(rows, effective, config_hash, checked_manifest)
    _require(report["validation_status"] == "pass" and report["block_count"] == report["wave_count"] == 1,
             "smoke requires one complete successful campaign")
    seen = set()
    for row in rows:
        _require(row["status"] == "success" and row["verification_status"] == "pass"
                 and row["attempted"] and row["failure_origin"] is None,
                 "every smoke row must be an actual verified success")
        if library and row["implementation"] == "cpu" and row["benchmark"] in {"rocblas", "rocsparse", "rocsolver"}:
            _require(row["cpu_threads_effective"] is None,
                     "provider smoke must not claim an observed worker count")
        key = (row["benchmark"], row["implementation"], row["scope"],
               dumps(core_raw_parameters(row["benchmark"], row["parameters"])), row["trial"])
        _require(key in expected and key not in seen, "unexpected or duplicate smoke row")
        seen.add(key)
        item = expected[key]
        validated = validate_subprocess_record(row, item, context, config_hash, runtime_hash,
                                               git_diff_hash, source_hash)
        _numerical_pass(validated, item, context)
    _require(seen == set(expected), "missing smoke trial")
    return {"status": "PASS", "campaign": name + "-" + effective["output_format"],
            "run_id": context["run_id"], "row_count": len(rows), "invocation_count": len(schedule),
            "cpu_threads": effective["cpu_threads"], "output_format": effective["output_format"],
            "config_sha256": config_hash, "runtime_environment_sha256": runtime_hash,
            "source_snapshot_sha256": source_hash,
            "manifest_sha256": hashlib.sha256(dump_bytes(checked_manifest)).hexdigest(),
            "library_counts": dict(sorted(Counter(row["benchmark"] for row in rows).items())),
            "validation": report}


def validate_campaign_set(campaigns, library=None):
    """Combine only four already validated, separately identified campaigns."""
    _require(set(campaigns) == set(CAMPAIGNS), "all four smoke campaigns are required")
    libraries = selected_libraries(library)
    expected_counts = dict.fromkeys(libraries, 24 if library else 12)
    invocations = len(libraries) * (12 if library else 6)
    row_count = invocations * 2
    for label, report in campaigns.items():
        name, fmt = label.split("-")
        _require(report["status"] == "PASS" and report["campaign"] == label
                 and report["row_count"] == row_count and report["invocation_count"] == invocations
                 and report["cpu_threads"] == CAMPAIGN_THREADS[name] and report["output_format"] == fmt
                 and report["library_counts"] == expected_counts
                 and report["validation"]["validation_status"] == "pass"
                 and report["validation"]["status_counts"] == {"success": row_count}
                 and report["validation"]["run_id"] == report["run_id"],
                 "incomplete or mismatched smoke campaign report")
    for field in ("run_id", "config_sha256", "runtime_environment_sha256"):
        _require(len({report[field] for report in campaigns.values()}) == 4,
                 "smoke campaigns must have distinct " + field)
    for field in ("source_snapshot_sha256", "manifest_sha256"):
        _require(len({report[field] for report in campaigns.values()}) == 1,
                 "smoke campaigns must retain the same " + field)
    return {"status": "PASS", "campaign_count": 4, "row_count": 4 * row_count,
            "campaigns": copy.deepcopy(campaigns)}

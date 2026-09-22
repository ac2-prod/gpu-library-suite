"""Provenance-aware plot labels; no machine or thread-count defaults."""

import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


DISPLAY_KEYS = {
    "plot_display_schema_version", "run_id", "runtime_environment_sha256",
    "cpu_model", "gpu_model", "thread_label_mode",
}
BACKEND_LABELS = {
    "cpu-fftw-threaded": "FFTW",
    "cpu-fftw-serial": "FFTW serial",
    "cpu-onemkl": "oneMKL",
    "cpu-openblas": "OpenBLAS",
    "cpu-generic-cblas": "CBLAS",
    "cpu-generic-lapacke": "LAPACKE",
    "cpu-std-random-serial": "std::mt19937_64",
    "cpu-stl-serial": "STL",
    "cpu-openmp": "OpenMP",
    "cpu-reference": "reference",
}


def _name(value: Any) -> Optional[str]:
    if value is None:
        return None
    if (not isinstance(value, str) or not value.strip() or len(value) > 160
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
            or len(value.splitlines()) != 1):
        raise ValueError("plot model name must be a nonempty single-line string or null")
    return value.strip()


def _comparable_name(value: str) -> str:
    return " ".join(re.sub(r"\((?:R|TM)\)", "", value).split())


def _model(
    names: Sequence[Optional[str]], supplied: Optional[str], observed_source: str,
) -> Dict[str, Any]:
    known = sorted({_name(value) for value in names if value is not None})
    if len({_comparable_name(value) for value in known}) > 1:
        raise ValueError("plot input contains different machine models; separate the comparisons")
    if supplied is not None:
        if known and _comparable_name(supplied) != _comparable_name(known[0]):
            raise ValueError("explicit plot model contradicts observed machine identity")
        return {"value": supplied, "source": "user-specified", "observed_values": known}
    complete = bool(names) and all(value is not None for value in names)
    return {
        "value": known[0] if complete else None,
        "source": observed_source if complete else "unknown",
        "observed_values": known,
    }


def build_display_context(
    provenance: Tuple[Any, Any], raw_records: Sequence[Mapping[str, Any]],
    configuration: Optional[Mapping[str, Any]] = None,
    node_metadata: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    if configuration is not None:
        if not isinstance(configuration, Mapping) or set(configuration) != DISPLAY_KEYS:
            raise ValueError("invalid plot display configuration keys")
        if (type(configuration["plot_display_schema_version"]) is not int
                or configuration["plot_display_schema_version"] != 1):
            raise ValueError("unsupported plot display schema")
        if (configuration["run_id"], configuration["runtime_environment_sha256"]) != provenance:
            raise ValueError("plot display configuration provenance differs from data")
        mode = configuration["thread_label_mode"]
        if (not isinstance(mode, str)
                or mode not in {"requested-and-effective", "compact-requested"}):
            raise ValueError("invalid plot thread label mode")
        cpu_model = _name(configuration["cpu_model"])
        gpu_model = _name(configuration["gpu_model"])
    else:
        mode, cpu_model, gpu_model = "requested-and-effective", None, None
    if any((row.get("run_id"), row.get("runtime_environment_sha256")) != provenance
           for row in raw_records):
        raise ValueError("raw-result and aggregate provenance differ")

    by_host = {}
    for node in node_metadata:
        if not isinstance(node, Mapping):
            raise ValueError("plot node metadata must be an object")
        if (node.get("run_id"), node.get("runtime_environment_sha256")) != provenance:
            raise ValueError("plot node metadata provenance differs from data")
        identity = node.get("cpu_identity", {})
        if not isinstance(identity, Mapping):
            raise ValueError("plot CPU identity must be an object")
        value = identity.get("name") if identity.get("query_status") == "success" else None
        hostname = node.get("hostname")
        if not isinstance(hostname, str) or not hostname:
            raise ValueError("plot node metadata lacks hostname")
        if hostname in by_host and by_host[hostname] != value:
            raise ValueError("plot node metadata has inconsistent CPU identities")
        by_host[hostname] = value
    hosts = {row.get("hostname") for row in raw_records}
    cpu_names = [by_host.get(host) for host in hosts]
    gpu_names = [row.get("gpu_name") for row in raw_records
                 if row.get("implementation") in {"cuda", "openacc"}
                 and row.get("status") == "success"]
    context = {
        "cpu_model": _model(cpu_names, cpu_model, "node-metadata"),
        "gpu_model": _model(gpu_names, gpu_model, "raw-results"),
        "thread_label_mode": mode,
        "thread_label_semantics": (
            "C denotes requested CPU count, not measured utilization or effective threads"
            if mode == "compact-requested" else "requested and reported effective counts are distinct"
        ),
    }
    if mode == "compact-requested" and any(
        context[key]["value"] is None for key in ("cpu_model", "gpu_model")
    ):
        raise ValueError("compact plot labels require explicit or observed machine models")
    return context


def _counts(rows: Sequence[Mapping[str, Any]], key: str) -> Sequence[Optional[int]]:
    values = {row.get(key) for row in rows}
    for value in values:
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError("invalid raw CPU thread count for plot labels")
    return sorted(values, key=lambda value: -1 if value is None else value)


def _count_text(values: Sequence[Optional[int]]) -> str:
    if not values or values == [None]:
        return "unknown"
    if len(values) == 1:
        return str(values[0])
    return "mixed"


def series_display(
    record: Mapping[str, Any], raw_records: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    implementation = record["implementation"]
    if implementation in {"cuda", "openacc"}:
        model = context["gpu_model"]["value"] or "GPU model unknown"
        return {"label": model + ", " + ("CUDA" if implementation == "cuda" else "OpenACC")}
    if implementation != "cpu":
        raise ValueError("publication plots accept only CPU, CUDA, and OpenACC")
    backend = record.get("cpu_backend")
    if not isinstance(backend, str) or not backend:
        raise ValueError("CPU plot series requires an explicit backend")
    rows = [row for row in raw_records
            if row.get("benchmark") == record["benchmark"]
            and row.get("scope") == record["scope"]
            and row.get("implementation") == "cpu"
            and row.get("cpu_backend") == backend
            and row.get("series_role", "primary") == "primary"
            and row.get("status") == "success"]
    requested = _counts(rows, "cpu_threads_requested")
    effective = _counts(rows, "cpu_threads_effective")
    if effective == [1]:
        threads = "single thread"
    elif context["thread_label_mode"] == "compact-requested":
        if len(requested) != 1 or requested[0] is None:
            raise ValueError("compact plot labels require one recorded requested CPU count")
        threads = "{0} C".format(requested[0])
    else:
        threads = "requested {0}; effective {1}".format(
            _count_text(requested), _count_text(effective)
        )
    model = context["cpu_model"]["value"] or "CPU model unknown"
    return {
        "label": "{0}, {1} ({2})".format(model, BACKEND_LABELS.get(backend, backend), threads),
        "cpu_backend_source": "aggregate-results",
        "cpu_threads": {
            "requested_values": list(requested), "effective_values": list(effective),
            "source": "raw-results" if rows else "unknown",
        },
    }

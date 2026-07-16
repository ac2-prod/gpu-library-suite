"""Deterministic elapsed-time publication plots from cross-wave summaries."""

import importlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple


CURAND_COMPARISON_NOTE = (
    "Same uniform-double task; the serial CPU and GPU generators are not "
    "algorithm-equivalent."
)

PUBLICATION_BENCHMARKS = (
    "cufft", "cublas", "cusparse", "cusolver", "curand", "thrust",
)
REQUIRED_PUBLICATION_BENCHMARKS = PUBLICATION_BENCHMARKS[:-1]
PUBLICATION_SCOPES = ("compute", "end-to-end")
PUBLICATION_IMPLEMENTATIONS = ("cpu", "cuda", "openacc")

FIGURE_FILENAMES = {
    benchmark: "{0}-elapsed-time.png".format(benchmark)
    for benchmark in PUBLICATION_BENCHMARKS
}
X_LABELS = {
    "cufft": "1-D FFT length (nfft)",
    "cublas": "Matrix order (N)",
    "cusparse": "Matrix rows (N)",
    "cusolver": "Matrix order (N)",
    "curand": "Output elements (N)",
    "thrust": "Input elements (N)",
}
PANEL_TITLES = {
    "compute": "Library kernel execution time",
    "end-to-end": "End-to-end execution time",
}
CPU_SERIES_LABELS = {
    ("cufft", "cpu-fftw-threaded"): (
        "Intel Xeon Platinum 8468, FFTW (48 C)"
    ),
    ("cublas", "cpu-onemkl"): (
        "Intel Xeon Platinum 8468, oneMKL (48 C)"
    ),
    ("cusparse", "cpu-onemkl"): (
        "Intel Xeon Platinum 8468, oneMKL (48 C)"
    ),
    ("cusolver", "cpu-onemkl"): (
        "Intel Xeon Platinum 8468, oneMKL (48 C)"
    ),
    ("curand", "cpu-std-random-serial"): (
        "Intel Xeon Platinum 8468, std::mt19937_64 (single thread)"
    ),
    ("thrust", "cpu-stl-serial"): (
        "Intel Xeon Platinum 8468, STL (single thread)"
    ),
}


class PlotError(ValueError):
    pass


def _series_label(record: Mapping[str, Any]) -> str:
    implementation = record["implementation"]
    if implementation == "cpu":
        key = (record["benchmark"], record.get("cpu_backend"))
        try:
            return CPU_SERIES_LABELS[key]
        except KeyError as error:
            raise PlotError(
                "unexpected publication CPU series: {0}/{1}".format(*key)
            ) from error
    if implementation == "cuda":
        return "NVIDIA H100 PCIe, CUDA"
    if implementation == "openacc":
        return "NVIDIA H100 PCIe, OpenACC"
    raise PlotError("publication plots accept only CPU, CUDA, and OpenACC")


def _binary_tick_label(value: float, _position: Optional[int] = None) -> str:
    if not math.isfinite(value):
        return ""
    integer = int(round(value))
    if not math.isclose(value, integer, rel_tol=0.0, abs_tol=1.0e-9):
        return "{0:g}".format(value)
    if abs(integer) < 1024:
        return str(integer)
    if integer % (1024 * 1024) == 0:
        return "{0}M".format(integer // (1024 * 1024))
    if integer % 1024 == 0:
        return "{0}K".format(integer // 1024)
    return str(integer)


def primary_plot_series(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Select per-operation elapsed-time cross-wave records."""

    grouped = defaultdict(list)  # type: DefaultDict[Tuple[str, ...], List[Mapping[str, Any]]]
    provenance = set()
    for record in records:
        if record.get("aggregate_schema_version") != 1:
            raise PlotError("unsupported aggregate schema")
        provenance.add(
            (record.get("run_id"), record.get("runtime_environment_sha256"))
        )
        if (
            record.get("summary_level") != "cross-wave"
            or record.get("series_role") != "primary"
            or record.get("median") is None
            or record.get("implementation") not in PUBLICATION_IMPLEMENTATIONS
        ):
            continue
        if record.get("scope") not in PUBLICATION_SCOPES:
            raise PlotError("unsupported publication scope")
        median = record["median"]
        if not isinstance(median, (int, float)) or not math.isfinite(float(median)):
            raise PlotError("publication elapsed time must be finite")
        if float(median) < 0.0:
            raise PlotError("publication elapsed time must be non-negative")
        key = (
            str(record.get("benchmark")),
            str(record.get("scope")),
            str(record.get("implementation")),
            str(record.get("cpu_backend")),
        )
        grouped[key].append(record)
    if len(provenance) != 1:
        raise PlotError("plot input must contain one campaign provenance")

    series = []  # type: List[Dict[str, Any]]
    for key in sorted(grouped):
        group = sorted(
            grouped[key],
            key=lambda item: (
                item["problem_size"],
                -1 if item.get("secondary_size") is None else item["secondary_size"],
                item["parameter_signature"],
            ),
        )
        point_keys = set()
        points = []
        for record in group:
            point_key = (
                record["problem_size"],
                record.get("secondary_size"),
                record["parameter_signature"],
            )
            if point_key in point_keys:
                raise PlotError("duplicate primary plot point")
            point_keys.add(point_key)
            points.append(
                {
                    "median": record["median"],
                    "parameter_signature": record["parameter_signature"],
                    "problem_size": record["problem_size"],
                    "q1": record.get("q1"),
                    "q3": record.get("q3"),
                    "secondary_size": record.get("secondary_size"),
                }
            )
        first = group[0]
        series.append(
            {
                "benchmark": first["benchmark"],
                "comparison": None,
                "cpu_backend": first.get("cpu_backend"),
                "implementation": first["implementation"],
                "label": _series_label(first),
                "metric": "elapsed_sec",
                "points": points,
                "scope": first["scope"],
            }
        )
    return series


def _validate_series_contract(series: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    available = {str(item["benchmark"]) for item in series}
    unexpected = available.difference(PUBLICATION_BENCHMARKS)
    if unexpected:
        raise PlotError(
            "unexpected publication benchmark: {0}".format(sorted(unexpected))
        )
    missing = set(REQUIRED_PUBLICATION_BENCHMARKS).difference(available)
    if missing:
        raise PlotError(
            "missing publication benchmark: {0}".format(sorted(missing))
        )
    benchmarks = tuple(
        benchmark for benchmark in PUBLICATION_BENCHMARKS
        if benchmark in available
    )
    for benchmark in benchmarks:
        benchmark_series = [
            item for item in series if item["benchmark"] == benchmark
        ]
        scopes = {str(item["scope"]) for item in benchmark_series}
        if scopes != set(PUBLICATION_SCOPES):
            raise PlotError(
                "{0} requires compute and end-to-end panels".format(benchmark)
            )
        for scope in PUBLICATION_SCOPES:
            scope_series = [
                item for item in benchmark_series if item["scope"] == scope
            ]
            implementations = {
                str(item["implementation"]) for item in scope_series
            }
            if implementations != set(PUBLICATION_IMPLEMENTATIONS):
                raise PlotError(
                    "{0}/{1} requires CPU, CUDA, and OpenACC series".format(
                        benchmark, scope
                    )
                )
            if len(scope_series) != len(PUBLICATION_IMPLEMENTATIONS):
                raise PlotError("duplicate publication implementation series")
            point_sets = {
                tuple(
                    (
                        point["problem_size"],
                        point.get("secondary_size"),
                        point["parameter_signature"],
                    )
                    for point in item["points"]
                )
                for item in scope_series
            }
            if len(point_sets) != 1:
                raise PlotError("publication series have mismatched problem cases")
            only_points = next(iter(point_sets))
            if len(only_points) != 3:
                raise PlotError("publication series require exactly three cases")
    return benchmarks


def _validate_thrust_versions(
    benchmarks: Sequence[str], raw_records: Optional[Sequence[Mapping[str, Any]]],
    provenance: Tuple[Any, Any],
) -> None:
    if "thrust" not in benchmarks:
        return
    if raw_records is None:
        raise PlotError(
            "Thrust publication figure requires raw-result library_version evidence"
        )
    raw_provenance = {
        (record.get("run_id"), record.get("runtime_environment_sha256"))
        for record in raw_records
    }
    if raw_provenance != {provenance}:
        raise PlotError("raw-result and aggregate provenance differ")
    versions = {"cuda": set(), "openacc": set()}
    for record in raw_records:
        implementation = record.get("implementation")
        if (
            record.get("benchmark") == "thrust"
            and implementation in versions
            and record.get("status") == "success"
        ):
            version = record.get("library_version")
            if isinstance(version, str) and version:
                versions[str(implementation)].add(version)
    for implementation in ("cuda", "openacc"):
        if len(versions[implementation]) != 1:
            raise PlotError(
                "Thrust {0} requires exactly one successful library_version: {1}".format(
                    implementation, sorted(versions[implementation])
                )
            )
    cuda_version = next(iter(versions["cuda"]))
    openacc_version = next(iter(versions["openacc"]))
    if cuda_version != openacc_version:
        raise PlotError(
            "Thrust CUDA/OpenACC library_version mismatch: CUDA={0}, OpenACC={1}".format(
                cuda_version, openacc_version
            )
        )


def build_plot_metadata(
    records: Sequence[Mapping[str, Any]], source_sha256: str,
    raw_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    provenance = sorted(
        {
            (record["run_id"], record["runtime_environment_sha256"])
            for record in records
        }
    )
    if len(provenance) != 1:
        raise PlotError("plot input must contain one campaign provenance")
    series = primary_plot_series(records)
    benchmarks = _validate_series_contract(series)
    _validate_thrust_versions(benchmarks, raw_records, provenance[0])
    run_id, runtime_hash = provenance[0]
    return {
        "curand_comparison_note": CURAND_COMPARISON_NOTE,
        "generated_files": [],
        "matplotlib": {"status": "not-attempted", "version": None},
        "plot_metadata_schema_version": 1,
        "primary_summary_level": "cross-wave",
        "run_id": run_id,
        "runtime_environment_sha256": runtime_hash,
        "series": series,
        "source_aggregate_sha256": source_sha256,
    }


def render_plots(
    metadata: Dict[str, Any], output_directory: Path,
    importer: Callable[[str], Any] = importlib.import_module,
) -> Dict[str, Any]:
    """Render one two-panel elapsed-time figure per selected library."""

    benchmarks = _validate_series_contract(metadata["series"])
    filenames = [FIGURE_FILENAMES[benchmark] for benchmark in benchmarks]
    for filename in filenames:
        path = output_directory / filename
        if path.exists():
            raise PlotError("plot output already exists: {0}".format(path))

    try:
        matplotlib = importer("matplotlib")
        matplotlib.use("Agg")
        pyplot = importer("matplotlib.pyplot")
        ticker = importer("matplotlib.ticker")
    except (ImportError, ModuleNotFoundError) as error:
        metadata["matplotlib"] = {
            "reason": "matplotlib is unavailable: {0}".format(error),
            "status": "unexecuted",
            "version": None,
        }
        return metadata

    metadata["matplotlib"] = {
        "status": "available",
        "version": str(matplotlib.__version__),
    }
    implementation_order = {
        name: index for index, name in enumerate(PUBLICATION_IMPLEMENTATIONS)
    }
    for benchmark in benchmarks:
        figure, axes = pyplot.subplots(1, 2, figsize=(12.0, 4.8))
        for panel_index, scope in enumerate(PUBLICATION_SCOPES):
            axis = axes[panel_index]
            selected = [
                item for item in metadata["series"]
                if item["benchmark"] == benchmark and item["scope"] == scope
            ]
            for item in sorted(
                selected,
                key=lambda value: implementation_order[value["implementation"]],
            ):
                x_values = [point["problem_size"] for point in item["points"]]
                y_values = [
                    float(point["median"]) * 1000.0 for point in item["points"]
                ]
                axis.plot(x_values, y_values, marker="o", label=item["label"])
            axis.set_xscale("log", base=2)
            if benchmark == "cusolver":
                axis.set_xticks((4096, 8192, 12288))
            axis.xaxis.set_major_formatter(
                ticker.FuncFormatter(_binary_tick_label)
            )
            axis.set_xlabel(X_LABELS[benchmark])
            axis.set_ylabel("Elapsed time [ms]")
            axis.set_title(PANEL_TITLES[scope])
            axis.grid(True, which="both", alpha=0.25)
        legend_handles, legend_labels = axes[0].get_legend_handles_labels()
        figure.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
        )
        figure.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
        figure.savefig(str(output_directory / FIGURE_FILENAMES[benchmark]), dpi=150)
        pyplot.close(figure)
    metadata["generated_files"] = filenames
    metadata["matplotlib"]["status"] = "rendered" if filenames else "no-data"
    return metadata

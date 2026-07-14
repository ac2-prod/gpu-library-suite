"""Deterministic primary-plot selection and optional matplotlib rendering."""

import importlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, List, Mapping, Sequence, Tuple


CURAND_COMPARISON_NOTE = (
    "Same distribution and output type task; different RNG algorithms."
)


class PlotError(ValueError):
    pass


def _series_label(record: Mapping[str, Any]) -> str:
    if record["implementation"] == "cpu":
        if record["benchmark"] in {"curand", "thrust"}:
            return "Serial CPU baseline"
        return "CPU ({0})".format(record["cpu_backend"])
    if record["implementation"] == "cuda":
        return "CUDA"
    if record["implementation"] == "openacc":
        return "OpenACC"
    comparison = record.get("comparison")
    if comparison == "cpu/cuda":
        return "CPU/CUDA speedup"
    if comparison == "cpu/openacc":
        return "CPU/OpenACC speedup"
    raise PlotError("unknown aggregate implementation")


def primary_plot_series(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Select cross-wave primary records and return deterministic plot data."""

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
        ):
            continue
        implementation = record.get("implementation")
        metric = "speedup" if implementation == "speedup" else "elapsed_sec"
        key = (
            str(record.get("benchmark")),
            str(record.get("scope")),
            metric,
            str(implementation),
            str(record.get("comparison")),
            str(record.get("cpu_backend")),
        )
        grouped[key].append(record)
    if len(provenance) > 1:
        raise PlotError("plot input has mixed campaign provenance")

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
                "comparison": first.get("comparison"),
                "cpu_backend": first.get("cpu_backend"),
                "implementation": first["implementation"],
                "label": _series_label(first),
                "metric": key[2],
                "points": points,
                "scope": first["scope"],
            }
        )
    return series


def build_plot_metadata(
    records: Sequence[Mapping[str, Any]], source_sha256: str
) -> Dict[str, Any]:
    provenance = sorted(
        {
            (record["run_id"], record["runtime_environment_sha256"])
            for record in records
        }
    )
    if len(provenance) != 1:
        raise PlotError("plot input must contain one campaign provenance")
    run_id, runtime_hash = provenance[0]
    return {
        "curand_comparison_note": CURAND_COMPARISON_NOTE,
        "generated_files": [],
        "matplotlib": {"status": "not-attempted", "version": None},
        "plot_metadata_schema_version": 1,
        "primary_summary_level": "cross-wave",
        "run_id": run_id,
        "runtime_environment_sha256": runtime_hash,
        "series": primary_plot_series(records),
        "source_aggregate_sha256": source_sha256,
    }


def render_plots(
    metadata: Dict[str, Any], output_directory: Path,
    importer: Callable[[str], Any] = importlib.import_module,
) -> Dict[str, Any]:
    """Render selected series, or record why optional plotting was skipped."""

    try:
        matplotlib = importer("matplotlib")
        matplotlib.use("Agg")
        pyplot = importer("matplotlib.pyplot")
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
    plot_groups = defaultdict(list)  # type: DefaultDict[Tuple[str, str, str], List[Mapping[str, Any]]]
    for series in metadata["series"]:
        plot_groups[(series["benchmark"], series["scope"], series["metric"])].append(series)
    filenames = []
    for key in sorted(plot_groups):
        benchmark, scope, metric = key
        filename = "{0}-{1}-{2}.png".format(benchmark, scope, metric)
        path = output_directory / filename
        if path.exists():
            raise PlotError("plot output already exists: {0}".format(path))
        figure, axis = pyplot.subplots()
        for series in sorted(plot_groups[key], key=lambda item: item["label"]):
            x_values = [point["problem_size"] for point in series["points"]]
            y_values = [point["median"] for point in series["points"]]
            axis.plot(x_values, y_values, marker="o", label=series["label"])
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Problem size")
        axis.set_ylabel("Speedup" if metric == "speedup" else "Elapsed seconds")
        axis.set_title("{0} {1} {2}".format(benchmark, scope, metric))
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(str(path), dpi=150)
        pyplot.close(figure)
        filenames.append(filename)
    metadata["generated_files"] = filenames
    metadata["matplotlib"]["status"] = "rendered" if filenames else "no-data"
    return metadata

import tempfile
import unittest
from pathlib import Path

from gpu_suite.plotting import (
    CURAND_COMPARISON_NOTE,
    FIGURE_FILENAMES,
    PANEL_TITLES,
    PUBLICATION_BENCHMARKS,
    X_LABELS,
    PlotError,
    build_plot_metadata,
    render_plots,
)

from support import ZERO_HASH


SIZES = {
    "cufft": (256, 4096, 16384),
    "cublas": (512, 2048, 4096),
    "cusparse": (65536, 1048576, 4194304),
    "cusolver": (4096, 8192, 12288),
    "curand": (1048576, 16777216, 67108864),
    "thrust": (1048576, 16777216, 67108864),
}
CPU_BACKENDS = {
    "cufft": "cpu-fftw-threaded",
    "cublas": "cpu-onemkl",
    "cusparse": "cpu-onemkl",
    "cusolver": "cpu-onemkl",
    "curand": "cpu-std-random-serial",
    "thrust": "cpu-stl-serial",
}
EXPECTED_LEGEND_LABELS = {
    "cufft": (
        "Intel Xeon Platinum 8468, FFTW (48 C)",
        "NVIDIA H100 PCIe, CUDA",
        "NVIDIA H100 PCIe, OpenACC",
    ),
    "cublas": (
        "Intel Xeon Platinum 8468, oneMKL (48 C)",
        "NVIDIA H100 PCIe, CUDA",
        "NVIDIA H100 PCIe, OpenACC",
    ),
    "cusparse": (
        "Intel Xeon Platinum 8468, oneMKL (48 C)",
        "NVIDIA H100 PCIe, CUDA",
        "NVIDIA H100 PCIe, OpenACC",
    ),
    "cusolver": (
        "Intel Xeon Platinum 8468, oneMKL (48 C)",
        "NVIDIA H100 PCIe, CUDA",
        "NVIDIA H100 PCIe, OpenACC",
    ),
    "curand": (
        "Intel Xeon Platinum 8468, std::mt19937_64 (single thread)",
        "NVIDIA H100 PCIe, CUDA",
        "NVIDIA H100 PCIe, OpenACC",
    ),
    "thrust": (
        "Intel Xeon Platinum 8468, STL (single thread)",
        "NVIDIA H100 PCIe, CUDA",
        "NVIDIA H100 PCIe, OpenACC",
    ),
}
BINARY_TICKS = (
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    1048576,
    2097152,
    4194304,
    8388608,
    16777216,
    33554432,
    67108864,
)
BINARY_TICK_LABELS = (
    "512", "1K", "2K", "4K", "8K", "16K", "32K", "64K",
    "1M", "2M", "4M", "8M", "16M", "32M", "64M",
)


def aggregate_record(benchmark, scope, implementation, size, index):
    cpu_backend = CPU_BACKENDS[benchmark] if implementation == "cpu" else None
    elapsed = 0.001 * float(index + 1)
    if scope == "end-to-end":
        elapsed *= 10.0
    return {
        "aggregate_schema_version": 1,
        "benchmark": benchmark,
        "comparison": None,
        "cpu_backend": cpu_backend,
        "implementation": implementation,
        "median": elapsed,
        "parameter_signature": '{{"size":{0}}}'.format(size),
        "problem_size": size,
        "q1": elapsed * 0.9,
        "q3": elapsed * 1.1,
        "run_id": "run-1",
        "runtime_environment_sha256": ZERO_HASH,
        "scope": scope,
        "secondary_size": 4096 if benchmark == "cufft" else None,
        "series_role": "primary",
        "summary_level": "cross-wave",
    }


def publication_records():
    records = []
    for benchmark in PUBLICATION_BENCHMARKS:
        for scope in ("compute", "end-to-end"):
            for implementation in ("cpu", "cuda", "openacc"):
                for index, size in enumerate(SIZES[benchmark]):
                    records.append(
                        aggregate_record(
                            benchmark, scope, implementation, size, index
                        )
                    )
    speedup = aggregate_record("cublas", "compute", "cuda", 512, 0)
    speedup.update({
        "comparison": "cpu/cuda",
        "implementation": "speedup",
        "median": 20.0,
    })
    records.append(speedup)
    block = aggregate_record("cublas", "compute", "cuda", 512, 0)
    block["summary_level"] = "block"
    records.append(block)
    return records


def raw_version_records(cuda="300001", openacc="300001"):
    return [
        {
            "benchmark": "thrust",
            "implementation": implementation,
            "library_version": version,
            "run_id": "run-1",
            "runtime_environment_sha256": ZERO_HASH,
            "status": "success",
        }
        for implementation, version in (("cuda", cuda), ("openacc", openacc))
    ]


class FakeFormatter:
    def __init__(self, function):
        self.function = function

    def __call__(self, value, position=None):
        return self.function(value, position)


class FakeTicker:
    def __init__(self):
        self.formatters = []

    def FuncFormatter(self, function):
        formatter = FakeFormatter(function)
        self.formatters.append(formatter)
        return formatter


class FakeXAxis:
    def __init__(self):
        self.major_formatter = None
        self.major_formatter_calls = 0

    def set_major_formatter(self, formatter):
        self.major_formatter = formatter
        self.major_formatter_calls += 1


class FakeAxis:
    def __init__(self):
        self.lines = []
        self.xlabel = None
        self.ylabel = None
        self.title = None
        self.xaxis = FakeXAxis()
        self.initial_xticks = tuple(BINARY_TICKS)
        self.xticks = tuple(BINARY_TICKS)
        self.set_xticks_calls = []
        self.xscale_calls = []
        self.yscale_calls = []
        self.ylim_calls = []
        self.grid_calls = []
        self.legend_calls = []
        self.errorbar_calls = []
        self.set_position_calls = []

    def plot(self, x_values, y_values, **kwargs):
        line = {
            "x_values": list(x_values),
            "y_values": list(y_values),
            "kwargs": dict(kwargs),
        }
        self.lines.append(line)
        return [line]

    def get_legend_handles_labels(self):
        return self.lines, [line["kwargs"]["label"] for line in self.lines]

    def set_xscale(self, scale, base):
        self.xscale_calls.append((scale, base))

    def set_xticks(self, ticks):
        self.set_xticks_calls.append(tuple(ticks))
        self.xticks = tuple(ticks)

    def set_xlabel(self, label):
        self.xlabel = label

    def set_ylabel(self, label):
        self.ylabel = label

    def set_title(self, title):
        self.title = title

    def grid(self, enabled, which, alpha):
        self.grid_calls.append((enabled, which, alpha))

    def set_yscale(self, scale):
        self.yscale_calls.append(scale)

    def set_ylim(self, *limits):
        self.ylim_calls.append(limits)

    def legend(self, *args, **kwargs):
        self.legend_calls.append((args, kwargs))

    def errorbar(self, *args, **kwargs):
        self.errorbar_calls.append((args, kwargs))

    def set_position(self, position):
        self.set_position_calls.append(position)


class FakeFigure:
    def __init__(self, axes):
        self.axes = axes
        self.saved_path = None
        self.suptitle_calls = []
        self.legend_calls = []
        self.tight_layout_calls = []

    def suptitle(self, title):
        self.suptitle_calls.append(title)

    def legend(self, handles, labels, **kwargs):
        self.legend_calls.append({
            "handles": list(handles),
            "labels": list(labels),
            "kwargs": dict(kwargs),
        })

    def tight_layout(self, **kwargs):
        self.tight_layout_calls.append(dict(kwargs))

    def savefig(self, path, dpi):
        self.saved_path = path
        self.dpi = dpi


class FakePyplot:
    def __init__(self):
        self.figures = []
        self.subplots_calls = []

    def subplots(self, rows, columns, figsize, **kwargs):
        self.subplots_calls.append((rows, columns, figsize, dict(kwargs)))
        axes = [FakeAxis() for _ in range(rows * columns)]
        figure = FakeFigure(axes)
        self.figures.append(figure)
        return figure, axes

    def close(self, figure):
        figure.closed = True


class FakeMatplotlib:
    __version__ = "test"

    def use(self, backend):
        self.backend = backend


class PlottingTests(unittest.TestCase):
    def build_metadata(self, raw=None):
        return build_plot_metadata(
            publication_records(), ZERO_HASH,
            raw_records=raw if raw is not None else raw_version_records(),
        )

    def test_publication_series_are_cross_wave_elapsed_only(self):
        metadata = self.build_metadata()
        self.assertEqual(len(metadata["series"]), 6 * 2 * 3)
        self.assertEqual(metadata["plot_metadata_schema_version"], 1)
        self.assertEqual(metadata["primary_summary_level"], "cross-wave")
        self.assertEqual(metadata["curand_comparison_note"], CURAND_COMPARISON_NOTE)
        self.assertTrue(all(
            series["metric"] == "elapsed_sec" for series in metadata["series"]
        ))
        self.assertFalse(any(
            series["implementation"] == "speedup"
            for series in metadata["series"]
        ))
        labels = {
            benchmark: {
                series["label"] for series in metadata["series"]
                if series["benchmark"] == benchmark
            }
            for benchmark in PUBLICATION_BENCHMARKS
        }
        for benchmark in PUBLICATION_BENCHMARKS:
            self.assertEqual(
                labels[benchmark], set(EXPECTED_LEGEND_LABELS[benchmark])
            )

    def test_render_creates_exactly_six_two_panel_elapsed_ms_figures(self):
        metadata = self.build_metadata()
        pyplot = FakePyplot()
        matplotlib = FakeMatplotlib()
        ticker = FakeTicker()

        def importer(name):
            return {
                "matplotlib": matplotlib,
                "matplotlib.pyplot": pyplot,
                "matplotlib.ticker": ticker,
            }[name]

        with tempfile.TemporaryDirectory() as temporary:
            rendered = render_plots(
                metadata, Path(temporary), importer=importer
            )

        expected = [
            FIGURE_FILENAMES[benchmark]
            for benchmark in PUBLICATION_BENCHMARKS
        ]
        self.assertEqual(rendered["generated_files"], expected)
        self.assertEqual(len(pyplot.figures), 6)
        self.assertEqual(
            pyplot.subplots_calls,
            [(1, 2, (12.0, 4.8), {}) for _ in PUBLICATION_BENCHMARKS],
        )
        self.assertEqual(len(ticker.formatters), 6 * 2)
        self.assertFalse(any(
            word in filename
            for filename in expected
            for word in ("speedup", "throughput", "reuse", "amortized")
        ))
        for benchmark, figure in zip(PUBLICATION_BENCHMARKS, pyplot.figures):
            self.assertEqual(len(figure.axes), 2)
            self.assertEqual(figure.suptitle_calls, [])
            self.assertEqual(
                [axis.title for axis in figure.axes],
                [
                    "Library kernel execution time",
                    "End-to-end execution time",
                ],
            )
            self.assertEqual(
                PANEL_TITLES,
                {
                    "compute": "Library kernel execution time",
                    "end-to-end": "End-to-end execution time",
                },
            )
            self.assertEqual(len(figure.legend_calls), 1)
            legend = figure.legend_calls[0]
            self.assertEqual(
                legend["labels"], list(EXPECTED_LEGEND_LABELS[benchmark])
            )
            self.assertEqual(legend["handles"], figure.axes[0].lines)
            self.assertEqual(
                legend["kwargs"],
                {
                    "loc": "lower center",
                    "bbox_to_anchor": (0.5, 0.01),
                    "ncol": 3,
                },
            )
            self.assertEqual(
                figure.tight_layout_calls,
                [{"rect": (0.0, 0.12, 1.0, 1.0)}],
            )
            self.assertEqual(
                Path(figure.saved_path).name, FIGURE_FILENAMES[benchmark]
            )
            self.assertEqual(figure.dpi, 150)
            self.assertTrue(figure.closed)
            self.assertTrue(all(
                axis.ylabel == "Elapsed time [ms]" for axis in figure.axes
            ))
            self.assertTrue(all(len(axis.lines) == 3 for axis in figure.axes))
            self.assertEqual(
                figure.axes[0].lines[0]["y_values"][0], 1.0
            )
            self.assertEqual(
                figure.axes[1].lines[0]["y_values"][0], 10.0
            )
            for panel_index, axis in enumerate(figure.axes):
                self.assertEqual(axis.xlabel, X_LABELS[benchmark])
                self.assertEqual(axis.xscale_calls, [("log", 2)])
                if benchmark == "cusolver":
                    expected_ticks = (4096, 8192, 12288)
                    expected_tick_labels = ("4K", "8K", "12K")
                    self.assertEqual(
                        axis.set_xticks_calls, [expected_ticks]
                    )
                else:
                    expected_ticks = axis.initial_xticks
                    expected_tick_labels = BINARY_TICK_LABELS
                    self.assertEqual(axis.set_xticks_calls, [])
                self.assertEqual(axis.xticks, expected_ticks)
                self.assertEqual(axis.xaxis.major_formatter_calls, 1)
                self.assertEqual(
                    tuple(
                        axis.xaxis.major_formatter(value, index)
                        for index, value in enumerate(axis.xticks)
                    ),
                    expected_tick_labels,
                )
                self.assertEqual(axis.yscale_calls, [])
                self.assertEqual(axis.ylim_calls, [])
                self.assertEqual(axis.grid_calls, [(True, "both", 0.25)])
                self.assertEqual(axis.legend_calls, [])
                self.assertEqual(axis.errorbar_calls, [])
                self.assertEqual(axis.set_position_calls, [])
                self.assertEqual(
                    [line["kwargs"]["label"] for line in axis.lines],
                    list(EXPECTED_LEGEND_LABELS[benchmark]),
                )
                for line in axis.lines:
                    self.assertEqual(
                        set(line["kwargs"]), {"marker", "label"}
                    )
                    self.assertEqual(line["kwargs"]["marker"], "o")
                    self.assertEqual(
                        line["x_values"], list(SIZES[benchmark])
                    )
                expected_scale = 1.0 if panel_index == 0 else 10.0
                expected_y_values = [
                    expected_scale * float(index + 1)
                    for index in range(len(SIZES[benchmark]))
                ]
                for line in axis.lines:
                    self.assertEqual(line["y_values"], expected_y_values)

    def test_thrust_version_mismatch_is_rejected_before_rendering(self):
        with self.assertRaisesRegex(
            PlotError, "Thrust CUDA/OpenACC library_version mismatch"
        ):
            self.build_metadata(raw_version_records("300001", "200801"))

    def test_thrust_version_evidence_is_required(self):
        with self.assertRaisesRegex(PlotError, "requires raw-result"):
            build_plot_metadata(publication_records(), ZERO_HASH)

    def test_explicit_thrust_exclusion_renders_the_other_five_figures(self):
        records = [
            record for record in publication_records()
            if record["benchmark"] != "thrust"
        ]
        metadata = build_plot_metadata(records, ZERO_HASH)
        pyplot = FakePyplot()
        matplotlib = FakeMatplotlib()
        ticker = FakeTicker()

        def importer(name):
            return {
                "matplotlib": matplotlib,
                "matplotlib.pyplot": pyplot,
                "matplotlib.ticker": ticker,
            }[name]

        with tempfile.TemporaryDirectory() as temporary:
            rendered = render_plots(
                metadata, Path(temporary), importer=importer
            )
        self.assertEqual(len(rendered["generated_files"]), 5)
        self.assertNotIn("thrust-elapsed-time.png", rendered["generated_files"])

    def test_missing_thrust_implementation_is_not_silently_plotted(self):
        records = [
            record for record in publication_records()
            if not (
                record["benchmark"] == "thrust"
                and record["implementation"] == "openacc"
            )
        ]
        with self.assertRaisesRegex(PlotError, "requires CPU, CUDA, and OpenACC"):
            build_plot_metadata(
                records, ZERO_HASH, raw_records=raw_version_records()
            )

    def test_missing_matplotlib_is_reported_without_output(self):
        metadata = self.build_metadata()

        def missing(_name):
            raise ImportError("not installed")

        with tempfile.TemporaryDirectory() as temporary:
            rendered = render_plots(metadata, Path(temporary), importer=missing)
            self.assertEqual(rendered["matplotlib"]["status"], "unexecuted")
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from gpu_suite.plotting import (
    CURAND_COMPARISON_NOTE,
    FIGURE_FILENAMES,
    PANEL_TITLES,
    PUBLICATION_BENCHMARKS,
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


class FakeAxis:
    def __init__(self):
        self.lines = []
        self.xlabel = None
        self.ylabel = None
        self.title = None

    def plot(self, x_values, y_values, marker, label):
        self.lines.append((list(x_values), list(y_values), marker, label))

    def set_xscale(self, scale, base):
        self.xscale = (scale, base)

    def set_xlabel(self, label):
        self.xlabel = label

    def set_ylabel(self, label):
        self.ylabel = label

    def set_title(self, title):
        self.title = title

    def grid(self, enabled, which, alpha):
        self.grid_settings = (enabled, which, alpha)

    def legend(self):
        self.has_legend = True


class FakeFigure:
    def __init__(self, axes):
        self.axes = axes
        self.title = None
        self.saved_path = None

    def suptitle(self, title):
        self.title = title

    def tight_layout(self):
        self.tight = True

    def savefig(self, path, dpi):
        self.saved_path = path
        self.dpi = dpi


class FakePyplot:
    def __init__(self):
        self.figures = []

    def subplots(self, rows, columns, figsize):
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
            (series["benchmark"], series["label"])
            for series in metadata["series"]
        }
        self.assertIn(("cufft", "CPU: FFTW threaded, 48 threads"), labels)
        self.assertIn(("cublas", "CPU: oneMKL, 48 threads"), labels)
        self.assertIn(("curand", "CPU serial reference"), labels)
        self.assertIn(("thrust", "CPU serial reference"), labels)

    def test_render_creates_exactly_six_two_panel_elapsed_ms_figures(self):
        metadata = self.build_metadata()
        pyplot = FakePyplot()
        matplotlib = FakeMatplotlib()

        def importer(name):
            return matplotlib if name == "matplotlib" else pyplot

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
        self.assertFalse(any(
            word in filename
            for filename in expected
            for word in ("speedup", "throughput", "reuse", "amortized")
        ))
        for figure in pyplot.figures:
            self.assertEqual(len(figure.axes), 2)
            self.assertEqual(
                [axis.title for axis in figure.axes],
                [PANEL_TITLES["compute"], PANEL_TITLES["end-to-end"]],
            )
            self.assertTrue(all(
                axis.ylabel == "Elapsed time [ms]" for axis in figure.axes
            ))
            self.assertTrue(all(len(axis.lines) == 3 for axis in figure.axes))
            self.assertEqual(figure.axes[0].lines[0][1][0], 1.0)
            self.assertEqual(figure.axes[1].lines[0][1][0], 10.0)
            self.assertIn("lower is better", figure.title)

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

        def importer(name):
            return matplotlib if name == "matplotlib" else pyplot

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

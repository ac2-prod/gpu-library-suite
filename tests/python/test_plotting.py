import tempfile
import unittest
from pathlib import Path

from gpu_suite.aggregation import aggregate_results
from gpu_suite.plotting import (
    CURAND_COMPARISON_NOTE,
    build_plot_metadata,
    render_plots,
)

from support import ZERO_HASH, pilot_config, raw_success


class PlottingTests(unittest.TestCase):
    def test_serial_baseline_labels_and_curand_note(self):
        raw = [
            raw_success(benchmark="curand"),
            raw_success(benchmark="thrust"),
        ]
        records, _ = aggregate_results(raw, pilot_config())
        metadata = build_plot_metadata(records, ZERO_HASH)
        labels = {
            (series["benchmark"], series["label"])
            for series in metadata["series"]
        }
        self.assertIn(("curand", "Serial CPU baseline"), labels)
        self.assertIn(("thrust", "Serial CPU baseline"), labels)
        self.assertEqual(metadata["curand_comparison_note"], CURAND_COMPARISON_NOTE)
        self.assertTrue(all(
            series["metric"] == "elapsed_sec" for series in metadata["series"]
        ))

    def test_missing_matplotlib_is_reported_without_output(self):
        records, _ = aggregate_results([raw_success()], pilot_config())
        metadata = build_plot_metadata(records, ZERO_HASH)

        def missing(_name):
            raise ImportError("not installed")

        with tempfile.TemporaryDirectory() as temporary:
            rendered = render_plots(metadata, Path(temporary), importer=missing)
            self.assertEqual(rendered["matplotlib"]["status"], "unexecuted")
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()

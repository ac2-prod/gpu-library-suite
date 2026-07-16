import unittest

from gpu_suite.aggregation import aggregate_results, statistical_summary

from support import pilot_config, raw_success


class AggregationTests(unittest.TestCase):
    def test_inclusive_quantiles_for_zero_one_and_multiple_samples(self):
        zero = statistical_summary([])
        self.assertEqual(zero["aggregate_status"], "no_valid_samples")
        self.assertIsNone(zero["median"])

        one = statistical_summary([4.0])
        self.assertEqual(one["aggregate_status"], "insufficient_sample_count")
        self.assertEqual(one["median"], 4.0)
        self.assertIsNone(one["q1"])

        for count in (2, 5, 6, 8):
            values = list(range(1, count + 1))
            summary = statistical_summary(values)
            self.assertEqual(summary["aggregate_status"], "success")
            self.assertLessEqual(summary["minimum"], summary["q1"])
            self.assertLessEqual(summary["q1"], summary["median"])
            self.assertLessEqual(summary["median"], summary["q3"])
            self.assertLessEqual(summary["q3"], summary["maximum"])
        two = statistical_summary([1.0, 2.0])
        self.assertEqual(two["q1"], 1.25)
        self.assertEqual(two["q3"], 1.75)

    def test_cross_wave_uses_one_wave_median_not_pooled_blocks(self):
        raw = []
        for node, elapsed in enumerate((1.0, 1.0, 1.0)):
            raw.append(raw_success(wave=0, node_index=node,
                                   hostname="w0n{0}".format(node), elapsed=elapsed))
        for node, elapsed in enumerate((100.0, 100.0)):
            raw.append(raw_success(wave=1, node_index=node,
                                   hostname="w1n{0}".format(node), elapsed=elapsed))
        records, metadata = aggregate_results(raw, pilot_config())
        cross = next(record for record in records if
                     record["summary_level"] == "cross-wave" and
                     record["benchmark"] == "cufft" and
                     record["implementation"] == "cpu")
        pooled = next(record for record in records if
                      record["summary_level"] == "pooled-exploratory" and
                      record["benchmark"] == "cufft" and
                      record["implementation"] == "cpu")
        self.assertEqual(cross["summary_input_statistic"], "wave_median")
        self.assertEqual(cross["median"], 50.5)
        self.assertEqual(pooled["median"], 1.0)
        self.assertEqual(metadata["blocks_per_wave"], {"0": 3, "1": 2})
        self.assertEqual(sum(metadata["permutation_assignment_counts"].values()), 5)
        self.assertEqual(sum(metadata["size_order_assignment_counts"].values()), 5)

    def test_speedup_uses_only_configured_primary_cpu_backend(self):
        auxiliary = raw_success(elapsed=0.5, series_role="auxiliary")
        auxiliary.update({
            "cpu_backend": "cpu-fftw-serial",
            "cpu_backend_role": "reference",
            "cpu_parallelism": "serial",
            "cpu_threads_effective": 1,
            "library_name": "cpu-fftw-serial",
        })
        raw = [
            raw_success(elapsed=2.0, cpu_backend="cpu-fftw-threaded"),
            auxiliary,
            raw_success(implementation="cuda", elapsed=1.0),
        ]
        records, metadata = aggregate_results(raw, pilot_config())
        block_speedup = next(record for record in records if
                             record["summary_level"] == "block" and
                             record["comparison"] == "cpu/cuda")
        self.assertEqual(block_speedup["median"], 2.0)
        self.assertEqual(metadata["speedup_omissions"], [])

        records, metadata = aggregate_results(raw[1:], pilot_config())
        self.assertFalse(any(record["comparison"] == "cpu/cuda" for record in records))
        self.assertEqual(len(metadata["speedup_omissions"]), 1)

    def test_curand_backend_metadata_does_not_break_pairing(self):
        raw = [
            raw_success(benchmark="curand", elapsed=2.0),
            raw_success(benchmark="curand", implementation="cuda", elapsed=1.0),
        ]
        records, metadata = aggregate_results(raw, pilot_config())
        speedup = next(record for record in records if
                       record["summary_level"] == "block" and
                       record["comparison"] == "cpu/cuda")
        self.assertEqual(speedup["median"], 2.0)
        self.assertIn("different RNG algorithms", metadata["curand_comparison_note"])


if __name__ == "__main__":
    unittest.main()

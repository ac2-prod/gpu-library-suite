import math
import os
import subprocess
import unittest

from gpu_suite.schema import validate_raw_result, validate_trial_indices
from gpu_suite.strict_json import loads


def run_benchmark(path, arguments):
    completed = subprocess.run(
        [path] + arguments + ["--output", "-", "--format", "jsonl"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    records = [loads(line) for line in completed.stdout.splitlines()]
    for record in records:
        validate_raw_result(record)
    return records, completed.stderr


class Phase3FixtureRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("GPU_SUITE_CBLAS_FIXTURE_BENCH"), "fixture path absent")
    def test_cblas_compute_repeat_updates_c_continuously(self):
        records, stderr = run_benchmark(
            os.environ["GPU_SUITE_CBLAS_FIXTURE_BENCH"],
            ["--size", "8", "--warmup", "1", "--repeat", "2", "--trials", "2",
             "--scope", "compute", "--verify", "true", "--cpu-backend", "cpu-onemkl"],
        )
        self.assertEqual(stderr, "")
        validate_trial_indices(records, 2)
        for record in records:
            self.assertEqual(record["verification_metrics"]["max_abs_error"], 0.0)
            self.assertEqual(record["verification_thresholds"]["max_abs_error"]["reference_scale"], 17.0)

    @unittest.skipUnless(os.environ.get("GPU_SUITE_SPARSE_FIXTURE_BENCH"), "fixture path absent")
    def test_sparse_csr_and_repeat_verification(self):
        records, _ = run_benchmark(
            os.environ["GPU_SUITE_SPARSE_FIXTURE_BENCH"],
            ["--size", "16", "--warmup", "1", "--repeat", "2", "--trials", "2",
             "--scope", "compute", "--verify", "true", "--cpu-backend", "cpu-onemkl"],
        )
        validate_trial_indices(records, 2)
        self.assertTrue(all(record["verification_metrics"]["max_abs_error"] == 0.0 for record in records))

    @unittest.skipUnless(os.environ.get("GPU_SUITE_SOLVER_FIXTURE_BENCH"), "fixture path absent")
    def test_solver_staged_api_and_dense_system(self):
        records, _ = run_benchmark(
            os.environ["GPU_SUITE_SOLVER_FIXTURE_BENCH"],
            ["--size", "8", "--nrhs", "2", "--warmup", "1", "--repeat", "1",
             "--trials", "2", "--scope", "end-to-end", "--verify", "true",
             "--cpu-backend", "cpu-onemkl"],
        )
        validate_trial_indices(records, 2)
        for record in records:
            self.assertEqual(record["getrf_info"], 0)
            self.assertEqual(record["getrs_info"], 0)
            self.assertLessEqual(record["verification_metrics"]["solution_relative_error"], 1e-12)
            self.assertLessEqual(record["verification_metrics"]["relative_residual"], 1e-12)

    @unittest.skipUnless(os.environ.get("GPU_SUITE_RAND_CPU_BENCH"), "fixture path absent")
    def test_random_statistical_contract(self):
        records, _ = run_benchmark(
            os.environ["GPU_SUITE_RAND_CPU_BENCH"],
            ["--size", "65536", "--warmup", "1", "--repeat", "2", "--trials", "1",
             "--scope", "compute", "--verify", "true",
             "--cpu-backend", "cpu-std-random-serial"],
        )
        record = records[0]
        metrics = record["verification_metrics"]
        self.assertEqual(set(metrics), {"observed_min", "observed_max", "sample_mean", "second_central_moment_about_half"})
        self.assertGreaterEqual(metrics["observed_min"], 0.0)
        self.assertLessEqual(metrics["observed_max"], 1.0)
        mean_bound = 6.0 * math.sqrt(1.0 / (12.0 * 65536.0))
        moment_bound = 6.0 * math.sqrt(1.0 / (180.0 * 65536.0))
        self.assertAlmostEqual(record["verification_thresholds"]["sample_mean"]["absolute_bound"], mean_bound)
        self.assertAlmostEqual(record["verification_thresholds"]["second_central_moment_about_half"]["absolute_bound"], moment_bound)
        self.assertEqual(record["parameters"]["distribution_interval"], "[0,1)")

    @unittest.skipUnless(os.environ.get("GPU_SUITE_REDUCE_CPU_BENCH"), "fixture path absent")
    def test_reduction_expected_value(self):
        records, _ = run_benchmark(
            os.environ["GPU_SUITE_REDUCE_CPU_BENCH"],
            ["--size", "4096", "--warmup", "1", "--repeat", "1", "--trials", "2",
             "--scope", "end-to-end", "--verify", "true", "--cpu-backend", "cpu-stl-serial"],
        )
        validate_trial_indices(records, 2)
        self.assertTrue(all(record["verification_metrics"]["absolute_error"] == 0.0 for record in records))
        self.assertTrue(all(record["cpu_threads_requested"] == 1 for record in records))
        self.assertTrue(all(record["cpu_threads_effective"] == 1 for record in records))


if __name__ == "__main__":
    unittest.main()

"""Synthetic FFTW provider tests; no ROCm runtime or device is exercised."""
import csv
import io
import os
import subprocess
import unittest

from gpu_suite.schema import validate_raw_result, validate_trial_indices
from gpu_suite.strict_json import loads


@unittest.skipUnless(os.environ.get("GPU_SUITE_ROCFFT_FFTW_FIXTURE_BENCH"),
                     "synthetic rocFFT CPU benchmark path is not set")
class RocfftFixtureRuntimeTests(unittest.TestCase):
    def command(self, scope="compute", output_format="jsonl", threaded=True):
        return [os.environ["GPU_SUITE_ROCFFT_FFTW_FIXTURE_BENCH"],
                "--size", "8", "--batch", "3", "--warmup", "1", "--repeat", "2",
                "--trials", "2", "--scope", scope, "--verify", "true", "--output", "-",
                "--format", output_format, "--cpu-backend",
                "cpu-fftw-threaded" if threaded else "cpu-fftw-serial",
                "--cpu-threads", "4", "--cpu-threads-effective", "4" if threaded else "1",
                "--cpu-backend-role", "production" if threaded else "reference",
                "--series-role", "primary" if threaded else "auxiliary",
                "--cpu-parallelism", "threaded" if threaded else "serial"]

    def test_both_scopes_and_backends_validate_multibatch_jsonl(self):
        for scope in ("compute", "end-to-end"):
            for threaded in (True, False):
                with self.subTest(scope=scope, threaded=threaded):
                    completed = subprocess.run(self.command(scope, threaded=threaded),
                                               text=True, capture_output=True, check=False)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    records = [loads(line) for line in completed.stdout.splitlines()]
                    validate_trial_indices(records, 2)
                    self.assertEqual(len(records), 2)
                    for record in records:
                        validate_raw_result(record)
                        self.assertEqual(record["benchmark"], "rocfft")
                        self.assertEqual(record["implementation"], "cpu")
                        self.assertEqual(record["verification_status"], "pass")
                        self.assertEqual(record["parameters"]["batch"], 3)
                        self.assertEqual(record["cuda_runtime_version"], None)
                        self.assertEqual(record["elapsed_sec"], record["elapsed_total_sec"] / 2)

    def test_segmented_compute_restores_destructive_input_between_repeats(self):
        environment = os.environ.copy()
        environment["GPU_SUITE_TEST_DESTRUCTIVE_FFT_INPUT"] = "1"
        command = self.command("compute")
        command[command.index("--warmup") + 1] = "2"
        command[command.index("--repeat") + 1] = "3"
        completed = subprocess.run(command, env=environment,
                                   text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        records = [loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual(len(records), 2)
        for record in records:
            validate_raw_result(record)
            self.assertEqual(record["verification_status"], "pass")
            self.assertEqual(record["elapsed_sec"], record["elapsed_total_sec"] / 3)

    def test_csv_single_header_and_rocfft_identity(self):
        completed = subprocess.run(self.command("end-to-end", "csv"),
                                   text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.count("result_schema_version"), 1)
        rows = list(csv.DictReader(io.StringIO(completed.stdout)))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["benchmark"] for row in rows}, {"rocfft"})

    def test_nonfinite_output_is_failure(self):
        for scope in ("compute", "end-to-end"):
            for special in ("nan", "inf", "-inf"):
                environment = os.environ.copy()
                environment["GPU_SUITE_TEST_NONFINITE"] = special
                completed = subprocess.run(self.command(scope), env=environment,
                                           text=True, capture_output=True, check=False)
                with self.subTest(scope=scope, special=special):
                    self.assertNotEqual(completed.returncode, 0)
                    for line in completed.stdout.splitlines():
                        record = loads(line)
                        validate_raw_result(record)
                        self.assertEqual(record["verification_status"], "nonfinite")
                        self.assertEqual(record["failure_origin"], "verification")

    def test_dimension_overflow_rejected_before_allocation(self):
        command = self.command()
        command[command.index("--size") + 1] = "2147483648"
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("overflow", completed.stderr)
        records = [loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual(len(records), 2)
        for record in records:
            validate_raw_result(record)
            self.assertFalse(record["attempted"])
            self.assertIsNone(record["elapsed_sec"])


if __name__ == "__main__":
    unittest.main()

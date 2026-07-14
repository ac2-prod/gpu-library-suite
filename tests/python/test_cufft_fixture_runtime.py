import csv
import io
import os
import subprocess
import unittest

from gpu_suite.schema import validate_raw_result, validate_trial_indices
from gpu_suite.strict_json import loads


class CufftFixtureRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("GPU_SUITE_FFTW_FIXTURE_BENCH"),
        "synthetic FFTW benchmark path is not set",
    )
    def test_jsonl_stdout_contains_only_valid_records(self):
        command = [
            os.environ["GPU_SUITE_FFTW_FIXTURE_BENCH"],
            "--size",
            "8",
            "--batch",
            "2",
            "--warmup",
            "1",
            "--repeat",
            "2",
            "--trials",
            "2",
            "--scope",
            "compute",
            "--verify",
            "true",
            "--output",
            "-",
            "--format",
            "jsonl",
            "--cpu-backend",
            "cpu-fftw-threaded",
            "--cpu-threads",
            "4",
            "--cpu-threads-effective",
            "4",
            "--cpu-backend-role",
            "production",
            "--series-role",
            "primary",
            "--cpu-parallelism",
            "threaded",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        records = [loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual(len(records), 2)
        validate_trial_indices(records, 2)
        for record in records:
            self.assertEqual(record["status"], "success")
            self.assertEqual(record["cpu_backend"], "cpu-fftw-threaded")
            self.assertEqual(record["cpu_threads_effective"], 4)
            self.assertIsNotNone(record["measurement_start_timestamp"])
            self.assertIsNotNone(record["measurement_end_timestamp"])

    @unittest.skipUnless(
        os.environ.get("GPU_SUITE_FFTW_FIXTURE_BENCH"),
        "synthetic FFTW benchmark path is not set",
    )
    def test_csv_has_one_header(self):
        command = [
            os.environ["GPU_SUITE_FFTW_FIXTURE_BENCH"],
            "--size",
            "8",
            "--batch",
            "2",
            "--warmup",
            "0",
            "--repeat",
            "1",
            "--trials",
            "2",
            "--scope",
            "end-to-end",
            "--verify",
            "true",
            "--output",
            "-",
            "--format",
            "csv",
            "--cpu-backend",
            "cpu-fftw-serial",
            "--cpu-threads",
            "48",
            "--cpu-threads-effective",
            "1",
            "--cpu-backend-role",
            "reference",
            "--series-role",
            "auxiliary",
            "--cpu-parallelism",
            "serial",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        rows = list(csv.DictReader(io.StringIO(completed.stdout)))
        self.assertEqual(len(rows), 2)
        self.assertEqual(completed.stdout.count("result_schema_version"), 1)
        self.assertEqual({row["trial"] for row in rows}, {"0", "1"})

    @unittest.skipUnless(
        os.environ.get("GPU_SUITE_FFTW_FIXTURE_BENCH"),
        "synthetic FFTW benchmark path is not set",
    )
    def test_nonfinite_fft_output_is_a_verification_failure(self):
        base_command = [
            os.environ["GPU_SUITE_FFTW_FIXTURE_BENCH"],
            "--size", "8", "--batch", "1", "--warmup", "0",
            "--repeat", "1", "--trials", "1", "--scope", "compute",
            "--verify", "true", "--output", "-", "--format", "jsonl",
            "--cpu-backend", "cpu-fftw-threaded", "--cpu-threads", "2",
            "--cpu-threads-effective", "2", "--cpu-backend-role", "production",
            "--series-role", "primary", "--cpu-parallelism", "threaded",
        ]
        for special in ("nan", "inf", "-inf"):
            environment = os.environ.copy()
            environment["GPU_SUITE_TEST_NONFINITE"] = special
            completed = subprocess.run(
                base_command, text=True, capture_output=True, check=False,
                env=environment,
            )
            with self.subTest(special=special):
                self.assertNotEqual(completed.returncode, 0)
                self.assertNotIn("NaN", completed.stdout)
                self.assertNotIn("Infinity", completed.stdout)
                record = loads(completed.stdout.strip())
                validate_raw_result(record)
                self.assertEqual(record["status"], "failure")
                self.assertEqual(record["failure_origin"], "verification")
                self.assertEqual(record["verification_status"], "nonfinite")
                self.assertIsNone(
                    record["verification_metrics"][
                        record["verification_primary_metric"]
                    ]
                )


if __name__ == "__main__":
    unittest.main()

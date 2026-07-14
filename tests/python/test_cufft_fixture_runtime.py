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
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        rows = list(csv.DictReader(io.StringIO(completed.stdout)))
        self.assertEqual(len(rows), 2)
        self.assertEqual(completed.stdout.count("result_schema_version"), 1)
        self.assertEqual({row["trial"] for row in rows}, {"0", "1"})


if __name__ == "__main__":
    unittest.main()

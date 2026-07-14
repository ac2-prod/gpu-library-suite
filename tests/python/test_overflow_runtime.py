import os
import subprocess
import unittest

from gpu_suite.schema import validate_raw_result
from gpu_suite.strict_json import loads


def run_rejected(path, arguments):
    completed = subprocess.run(
        [path] + arguments + ["--warmup", "0", "--repeat", "1",
                              "--trials", "1", "--scope", "compute",
                              "--verify", "true", "--output", "-",
                              "--format", "jsonl"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        raise AssertionError("overflow request unexpectedly succeeded")
    lines = completed.stdout.splitlines()
    if len(lines) != 1:
        raise AssertionError("overflow request did not emit exactly one row")
    record = validate_raw_result(loads(lines[0]))
    if record["attempted"] or record["status"] != "skipped":
        raise AssertionError("pre-execution overflow was not skipped")
    if record["failure_origin"] != "prerequisite":
        raise AssertionError("overflow has the wrong failure origin")
    if record["verification_status"] != "skipped":
        raise AssertionError("overflow unexpectedly ran verification")


class OverflowRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(
        all(os.environ.get(name) for name in (
            "GPU_SUITE_FFTW_FIXTURE_BENCH",
            "GPU_SUITE_CBLAS_FIXTURE_BENCH",
            "GPU_SUITE_SPARSE_FIXTURE_BENCH",
            "GPU_SUITE_SOLVER_FIXTURE_BENCH",
            "GPU_SUITE_RAND_CPU_BENCH",
            "GPU_SUITE_REDUCE_CPU_BENCH",
        )),
        "CPU benchmark fixture paths absent",
    )
    def test_all_cpu_benchmarks_reject_api_or_byte_overflow(self):
        threaded = ["--cpu-backend", "cpu-onemkl", "--cpu-threads", "2",
                    "--cpu-backend-role", "production", "--series-role",
                    "primary", "--cpu-parallelism", "threaded"]
        serial = ["--cpu-threads", "48", "--cpu-threads-effective", "1",
                  "--cpu-backend-role", "production", "--series-role",
                  "primary", "--cpu-parallelism", "serial"]
        cases = (
            (os.environ["GPU_SUITE_FFTW_FIXTURE_BENCH"],
             ["--size", "2147483648", "--batch", "1", "--cpu-backend",
              "cpu-fftw-threaded", "--cpu-threads", "2",
              "--cpu-threads-effective", "2", "--cpu-backend-role",
              "production", "--series-role", "primary", "--cpu-parallelism",
              "threaded"]),
            (os.environ["GPU_SUITE_CBLAS_FIXTURE_BENCH"],
             ["--size", "2147483648"] + threaded),
            (os.environ["GPU_SUITE_SPARSE_FIXTURE_BENCH"],
             ["--nx", "2147483648", "--ny", "1"] + threaded),
            (os.environ["GPU_SUITE_SOLVER_FIXTURE_BENCH"],
             ["--size", "2147483648", "--nrhs", "1"] + threaded),
            (os.environ["GPU_SUITE_RAND_CPU_BENCH"],
             ["--size", "2305843009213693952", "--cpu-backend",
              "cpu-std-random-serial"] + serial),
            (os.environ["GPU_SUITE_REDUCE_CPU_BENCH"],
             ["--size", "2305843009213693952", "--cpu-backend",
              "cpu-stl-serial"] + serial),
        )
        for path, arguments in cases:
            with self.subTest(executable=path):
                run_rejected(path, arguments)


if __name__ == "__main__":
    unittest.main()

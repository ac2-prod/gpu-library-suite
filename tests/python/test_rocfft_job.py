"""Run the Sirius body with a fake allocation and inert Python payload only."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BODY = ROOT / "jobs/sirius/rocfft_job_body.sh"


class RocfftJobTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rocfft-job-fake-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.source = self.base / "source"
        self.run_id = "fake-run"
        self.run = self.source / "runs" / self.run_id
        self.run.mkdir(parents=True)
        tools = self.base / "fake-tools"
        tools.mkdir()
        for name, output in (("id", "fake-user"), ("date", "UTC=fake-clock"),
                             ("taskset", "fake-affinity")):
            tool = tools / name
            tool.write_text("#!/bin/bash\nprintf '%s\\n' '" + output + "'\n")
            tool.chmod(0o700)
        self.environment = {
            "PATH": str(tools) + ":/usr/bin:/bin", "HOME": "/fake/home",
            "HOSTNAME": "fake-compute", "PYTHONDONTWRITEBYTECODE": "1",
            "PBS_JOBID": "fake.pbs", "STAGE31_SOURCE_ROOT": str(self.source),
            "STAGE31_RUN_ROOT": str(self.run), "STAGE31_RUN_ID": self.run_id,
            "STAGE31_PYTHON": sys.executable,
        }
        self.payload()

    def payload(self, lines=None, exit_code=0):
        if lines is None:
            lines = ["GPU_SUITE_SIRIUS_ROCFFT_SMOKE PASS " + self.run_id]
        (self.run / "compute_smoke.py").write_text(
            "import json, pathlib, sys\n"
            "pathlib.Path(__file__).with_name('fake-argv.json').write_text("
            "json.dumps(sys.argv[1:]))\n"
            "print('fake helper stderr', file=sys.stderr)\n" +
            "".join("print(" + repr(line) + ")\n" for line in lines) +
            "sys.exit(" + str(exit_code) + ")\n")

    def execute(self):
        return subprocess.run(["/bin/bash", str(BODY)], env=self.environment,
                              capture_output=True, text=True, check=False)

    def rejected(self, reason):
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(reason, result.stdout)
        self.assertFalse((self.run / "fake-argv.json").exists())
        return result

    def test_allocation_required_before_payload(self):
        self.environment.pop("PBS_JOBID")
        self.rejected("MISSING_PBS_JOBID")

    def test_external_run_root_rejected(self):
        outside = self.base / "external"
        outside.mkdir()
        self.environment["STAGE31_RUN_ROOT"] = str(outside)
        self.rejected("RUN_ROOT_LAYOUT")

    def test_nested_run_directory_rejected(self):
        nested = self.run / self.run_id
        nested.mkdir()
        self.environment["STAGE31_RUN_ROOT"] = str(nested)
        self.rejected("RUN_ROOT_LAYOUT")

    def test_run_id_directory_mismatch_rejected(self):
        self.environment["STAGE31_RUN_ID"] = "different-run"
        self.rejected("RUN_ROOT_LAYOUT")

    def test_noncanonical_path_rejected_without_symlink_creation(self):
        self.environment["STAGE31_RUN_ROOT"] = str(self.run) + "/."
        self.rejected("NONCANONICAL_ROOT")

    def test_existing_logs_preserved_without_starting_payload(self):
        stderr = self.run / "compute-process.stderr"
        stderr.write_bytes(b"previous evidence\n")
        self.rejected("OUTPUT_ALREADY_EXISTS")
        self.assertEqual(stderr.read_bytes(), b"previous evidence\n")
        self.assertFalse((self.run / "compute-process.stdout").exists())

    def test_runs_child_accepted_with_exact_payload_arguments_and_separate_logs(self):
        result = self.execute()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads((self.run / "fake-argv.json").read_text()), [
            "--source-root", str(self.source), "--run-root", str(self.run),
            "--run-id", self.run_id, "--scheduler", "PBS", "--job-id", "fake.pbs",
            "--allocated-cpus", "24", "--allocated-gpus", "1", "--cpu-threads", "1"])
        self.assertIn("USER=fake-user HOSTNAME=fake-compute HOME=/fake/home PATH=", result.stdout)
        self.assertEqual(result.stderr, "fake helper stderr\n")
        self.assertNotIn("fake helper stderr", (self.run / "compute-process.stdout").read_text())
        self.assertEqual((self.run / "compute-process.stderr").read_text(), result.stderr)
        self.assertEqual(result.stdout.count("GPU_SUITE_SIRIUS_ROCFFT_SMOKE PASS"), 1)

    def test_explicit_thread_count_is_forwarded_to_helper_and_recorded(self):
        self.environment["STAGE31_CPU_THREADS"] = "2"
        result = self.execute()
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = json.loads((self.run / "fake-argv.json").read_text())
        self.assertEqual(argv[argv.index("--cpu-threads") + 1], "2")
        self.assertIn("CPU_THREADS=2 BUILD_PARALLELISM=1", result.stdout)

    def test_invalid_or_overallocated_thread_count_rejected_before_payload(self):
        for count in ("", "0", "25", "02", "-1", "2;anything", "99999999999999999999"):
            with self.subTest(count=count):
                self.environment["STAGE31_CPU_THREADS"] = count
                self.rejected("CPU_THREADS_EXCEED_ALLOCATION")

    def test_serial_campaign_is_explicitly_forwarded_with_threaded_request(self):
        self.environment.update(STAGE31_CPU_THREADS="2", STAGE31_INCLUDE_SERIAL="yes")
        result = self.execute()
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = json.loads((self.run / "fake-argv.json").read_text())
        self.assertEqual(argv.count("--include-serial-campaign"), 1)
        self.assertEqual(argv[argv.index("--cpu-threads") + 1], "2")

    def test_invalid_serial_campaign_selection_stops_before_payload(self):
        self.environment["STAGE31_INCLUDE_SERIAL"] = "true"
        self.rejected("INCLUDE_SERIAL_VALUE")
        self.environment["STAGE31_INCLUDE_SERIAL"] = "yes"
        self.rejected("INCLUDE_SERIAL_REQUIRES_THREADS")

    def test_helper_failure_marker_and_exit_preserved(self):
        self.payload(["GPU_SUITE_SIRIUS_ROCFFT_SMOKE FAIL fake-run COMPILE 7"], 7)
        result = self.execute()
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout.count("GPU_SUITE_SIRIUS_ROCFFT_SMOKE FAIL"), 1)
        self.assertIn("FAIL fake-run COMPILE 7", result.stdout)

    def test_duplicate_completion_is_failure(self):
        self.payload(["GPU_SUITE_SIRIUS_ROCFFT_SMOKE PASS fake-run"] * 2)
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAIL fake-run HELPER_COMPLETION", result.stdout)
        self.assertNotIn("SMOKE PASS", result.stdout)

    def test_missing_completion_is_failure(self):
        self.payload(["incomplete helper output"])
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAIL fake-run HELPER_COMPLETION", result.stdout)

    def test_valid_completion_with_wrong_run_marker_is_failure(self):
        self.payload(["GPU_SUITE_SIRIUS_ROCFFT_SMOKE PASS another-run",
                      "GPU_SUITE_SIRIUS_ROCFFT_SMOKE PASS fake-run"])
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAIL fake-run HELPER_COMPLETION", result.stdout)
        self.assertNotIn("SMOKE PASS", result.stdout)
        self.assertIn("PASS another-run", (self.run / "compute-process.stdout").read_text())

    def test_valid_completion_with_malformed_marker_is_failure(self):
        self.payload(["GPU_SUITE_SIRIUS_ROCFFT_SMOKE PASS fake-run",
                      "GPU_SUITE_SIRIUS_ROCFFT_SMOKE-malformed"])
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAIL fake-run HELPER_COMPLETION", result.stdout)
        self.assertNotIn("SMOKE PASS", result.stdout)
        self.assertIn("SMOKE-malformed", (self.run / "compute-process.stdout").read_text())

    def test_success_marker_with_nonzero_exit_is_failure(self):
        self.payload(exit_code=9)
        result = self.execute()
        self.assertEqual(result.returncode, 9)
        self.assertIn("FAIL fake-run HELPER_COMPLETION 9", result.stdout)
        self.assertNotIn("SMOKE PASS", result.stdout)

    def test_job_body_shell_syntax(self):
        result = subprocess.run(["/bin/bash", "-n", str(BODY)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("shellcheck"), "ShellCheck dependency unavailable")
    def test_job_body_shellcheck(self):
        result = subprocess.run([shutil.which("shellcheck"), str(BODY)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

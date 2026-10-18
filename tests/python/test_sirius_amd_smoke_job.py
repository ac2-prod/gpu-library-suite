"""Allocated shell gates using fake commands; never enter a login/module shell."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "jobs/sirius/amd_smoke_job.sh"


class SiriusAmdSmokeJobTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="amd-smoke-job-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.source = self.base / "source"
        self.run_id = "fake-smoke-run"
        self.run = self.source / "runs" / self.run_id
        self.run.mkdir(parents=True)
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.calls = self.base / "fake-calls.jsonl"
        self.packet = self.run / "packet.json"
        self.packet.write_text('{"purpose":"unexecuted fake packet"}\n')
        self.controller = self.run / "amd_smoke.py"
        self.controller.write_text("raise AssertionError('fake controller must not execute')\n")
        self.modules = self.run / "rocfft_compute_modules.sh"
        self.modules.write_text("printf 'ERROR: real sourcing is forbidden in this test' >&2; return 99\n")
        # Each command is a regular file, not a symlink. timeout deliberately
        # records but never executes the /bin/bash --login -i argument vector.
        dispatcher = "#!" + sys.executable + "\n" + '''import hashlib, json, os
from pathlib import Path
import sys
name = Path(sys.argv[0]).name
record = {"command": name, "argv": sys.argv[1:]}
if name == "timeout":
    record["environment"] = {key: os.environ.get(key) for key in (
        "STAGE31_SOURCE_ROOT", "STAGE31_RUN_ROOT", "STAGE31_RUN_ID",
        "GPU_SUITE_SIRIUS_COMPUTE_APPROVED", "PYTHONDONTWRITEBYTECODE", "TERM",
        "AMD_SMOKE_PYTHON", "AMD_SMOKE_PACKET_SHA256", "BASH_ENV", "ENV")}
with Path(os.environ["FAKE_CALLS"]).open("a") as stream:
    stream.write(json.dumps(record) + "\\n")
if name == "sha256sum":
    print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() + "  " + sys.argv[1])
elif name == "date": print("2026-10-17T00:00:00Z")
elif name == "id": print("fixture-user")
elif name == "hostname": print("fixture-compute")
elif name == "timeout":
    print("fake child stdout")
    print("fake child stderr", file=sys.stderr)
    raise SystemExit(int(os.environ.get("FAKE_TIMEOUT_EXIT", "0")))
else: raise AssertionError("unapproved fake command: " + name)
'''
        for name in ("sha256sum", "date", "id", "hostname", "timeout"):
            executable = self.bin / name
            executable.write_text(dispatcher)
            executable.chmod(0o700)
        self.environment = {
            "PATH": str(self.bin), "HOME": str(self.base), "FAKE_CALLS": str(self.calls),
            "GPU_SUITE_SIRIUS_AMD_SMOKE_APPROVED": "yes", "PBS_JOBID": "42.fixture",
            "PBS_ENVIRONMENT": "PBS_BATCH", "AMD_SMOKE_SOURCE_ROOT": str(self.source),
            "AMD_SMOKE_RUN_ROOT": str(self.run), "AMD_SMOKE_RUN_ID": self.run_id,
            "AMD_SMOKE_PYTHON": sys.executable, "AMD_SMOKE_TIMEOUT_SECONDS": "120",
            "AMD_SMOKE_PACKET_SHA256": self.sha(self.packet),
            "AMD_SMOKE_CONTROLLER_SHA256": self.sha(self.controller),
            "AMD_SMOKE_MODULES_SHA256": self.sha(self.modules),
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def invoke(self, environment=None):
        return subprocess.run(["/bin/bash", str(SCRIPT)], env=environment or self.environment,
                              cwd=self.base, capture_output=True, text=True, timeout=15, check=False)

    def records(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def assert_rejected_without_dispatch(self, environment, expected):
        before = [item for item in self.records() if item["command"] == "timeout"]
        result = self.invoke(environment)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn(expected, result.stderr)
        self.assertEqual([item for item in self.records() if item["command"] == "timeout"], before)
        self.assertFalse((self.run / "amd-smoke-job-start.txt").exists())

    def test_approval_and_pbs_batch_are_required_before_commands(self):
        for key, value in (("GPU_SUITE_SIRIUS_AMD_SMOKE_APPROVED", "no"), ("PBS_JOBID", ""),
                           ("PBS_JOBID", "malformed"), ("PBS_ENVIRONMENT", "PBS_INTERACTIVE"),
                           ("PBS_ENVIRONMENT", "")):
            with self.subTest(key=key, value=value):
                environment = dict(self.environment, **{key: value})
                self.assert_rejected_without_dispatch(environment, "approved PBS batch")
        self.assertEqual(self.records(), [])

    def test_required_values_and_canonical_run_paths_are_checked(self):
        for key in ("AMD_SMOKE_SOURCE_ROOT", "AMD_SMOKE_RUN_ROOT", "AMD_SMOKE_RUN_ID",
                    "AMD_SMOKE_PYTHON", "AMD_SMOKE_TIMEOUT_SECONDS", "AMD_SMOKE_PACKET_SHA256",
                    "AMD_SMOKE_CONTROLLER_SHA256", "AMD_SMOKE_MODULES_SHA256"):
            environment = dict(self.environment)
            environment.pop(key)
            with self.subTest(key=key): self.assert_rejected_without_dispatch(environment, "missing " + key)
        for overrides, message in (({"AMD_SMOKE_RUN_ID": "bad..id"}, "invalid run ID"),
                                   ({"AMD_SMOKE_RUN_ID": "x" * 82}, "invalid run ID"),
                                   ({"AMD_SMOKE_RUN_ROOT": str(self.base)}, "invalid source/run paths"),
                                   ({"AMD_SMOKE_SOURCE_ROOT": str(self.source) + "/.",
                                     "AMD_SMOKE_RUN_ROOT": str(self.source) + "/./runs/" + self.run_id}, "canonical directories")):
            with self.subTest(overrides=overrides):
                self.assert_rejected_without_dispatch(dict(self.environment, **overrides), message)

    def test_timeout_and_python_must_be_explicit_valid_values(self):
        for timeout in ("0", "29", "1801", "030", "1+30", "-1", "abc"):
            with self.subTest(timeout=timeout):
                self.assert_rejected_without_dispatch(dict(self.environment, AMD_SMOKE_TIMEOUT_SECONDS=timeout), "invalid timeout")
        nonexecutable = self.base / "not-executable"
        nonexecutable.write_text("fake")
        for python in ("python3", str(self.base / "missing"), str(nonexecutable)):
            with self.subTest(python=python):
                self.assert_rejected_without_dispatch(dict(self.environment, AMD_SMOKE_PYTHON=python), "Python executable")

    def test_packet_controller_and_modules_hashes_are_checked_before_attempt(self):
        for key, label in (("AMD_SMOKE_PACKET_SHA256", "packet"),
                           ("AMD_SMOKE_CONTROLLER_SHA256", "controller"),
                           ("AMD_SMOKE_MODULES_SHA256", "module helper")):
            for digest in ("0" * 64, "not-a-hash"):
                with self.subTest(key=key, digest=digest):
                    self.assert_rejected_without_dispatch(dict(self.environment, **{key: digest}), label + " hash mismatch")

    def test_changed_packet_stops_without_module_or_controller_execution(self):
        self.packet.write_text('{"changed":true}\n')
        self.assert_rejected_without_dispatch(self.environment, "packet hash mismatch")
        self.assertTrue(all(item["command"] == "sha256sum" for item in self.records()))

    def test_existing_log_or_controller_attempt_is_preserved_without_dispatch(self):
        existing = self.run / "amd-smoke-job.stderr"
        existing.write_bytes(b"preserved historical evidence\n")
        before = existing.read_bytes()
        self.assert_rejected_without_dispatch(self.environment, "existing attempt")
        self.assertEqual(existing.read_bytes(), before)
        # A second independent fixture checks the controller output gate without
        # removing or modifying the historical log created above.
        second = self.source / "runs" / "second-run"
        second.mkdir()
        for path in (self.packet, self.controller, self.modules):
            (second / path.name).write_bytes(path.read_bytes())
        (second / "smoke").mkdir()
        environment = dict(self.environment, AMD_SMOKE_RUN_ID="second-run", AMD_SMOKE_RUN_ROOT=str(second))
        result = self.invoke(environment)
        self.assertEqual(result.returncode, 2)
        self.assertIn("existing controller attempt", result.stderr)
        self.assertFalse((second / "amd-smoke-job-start.txt").exists())
        self.assertFalse(any(item["command"] == "timeout" for item in self.records()))

    def test_dispatch_is_one_owned_timeout_with_pinned_arguments_and_adapter(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        dispatch = [item for item in self.records() if item["command"] == "timeout"]
        self.assertEqual(len(dispatch), 1)
        argv = dispatch[0]["argv"]
        self.assertEqual(argv[:8], ["--signal=TERM", "--kill-after=15", "120", "/bin/bash", "--login", "-i", "-c", argv[7]])
        self.assertEqual(argv[8:], ["bash", str(self.modules), str(self.controller)])
        self.assertIn('source "$1" || exit $?', argv[7])
        self.assertIn('exec "$AMD_SMOKE_PYTHON" -B "$1"', argv[7])
        self.assertIn('--packet "$AMD_SMOKE_RUN_ROOT/packet.json"', argv[7])
        self.assertIn('--expected-packet-sha256 "$AMD_SMOKE_PACKET_SHA256"', argv[7])
        environment = dispatch[0]["environment"]
        self.assertEqual(environment["STAGE31_SOURCE_ROOT"], str(self.source))
        self.assertEqual(environment["STAGE31_RUN_ROOT"], str(self.run))
        self.assertEqual(environment["STAGE31_RUN_ID"], self.run_id)
        self.assertEqual(environment["GPU_SUITE_SIRIUS_COMPUTE_APPROVED"], "yes")
        self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(environment["TERM"], "dumb")
        self.assertIsNone(environment["BASH_ENV"])
        self.assertIsNone(environment["ENV"])

    def test_stdout_stderr_exit_metadata_and_completion_are_distinct(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0)
        stdout = (self.run / "amd-smoke-job.stdout").read_text()
        stderr = (self.run / "amd-smoke-job.stderr").read_text()
        self.assertIn("USER=fixture-user HOSTNAME=fixture-compute", stdout)
        self.assertIn("HOME=" + str(self.base), stdout)
        self.assertIn("PATH=" + str(self.bin), stdout)
        self.assertIn("PBS_JOBID=42.fixture", stdout)
        self.assertIn("fake child stdout", stdout)
        self.assertEqual(stderr, "fake child stderr\n")
        self.assertEqual((self.run / "amd-smoke-job-exit.txt").read_text(), "0\n")
        self.assertEqual(result.stdout, "GPU_SUITE_SIRIUS_AMD_SMOKE_END " + self.run_id + " 0\n")
        self.assertEqual(result.stderr, "")
        for name in ("start", "end"):
            self.assertEqual((self.run / ("amd-smoke-job-" + name + ".txt")).read_text(), "2026-10-17T00:00:00Z\n")
        for path in self.run.glob("amd-smoke-job*"):
            self.assertEqual(path.stat().st_mode & 0o077, 0)

    def test_child_failure_or_owned_timeout_never_retries(self):
        environment = dict(self.environment, FAKE_TIMEOUT_EXIT="124")
        result = self.invoke(environment)
        self.assertEqual(result.returncode, 124)
        self.assertEqual((self.run / "amd-smoke-job-exit.txt").read_text(), "124\n")
        self.assertIn(" " + self.run_id + " 124\n", result.stdout)
        logs_before = {path.name: path.read_bytes() for path in self.run.glob("amd-smoke-job*")}
        repeated = self.invoke(environment)
        self.assertEqual(repeated.returncode, 2)
        self.assertIn("existing attempt", repeated.stderr)
        self.assertEqual({path.name: path.read_bytes() for path in self.run.glob("amd-smoke-job*")}, logs_before)
        self.assertEqual(len([item for item in self.records() if item["command"] == "timeout"]), 1)

    def test_body_has_no_remote_submission_or_build_boundary(self):
        text = SCRIPT.read_text()
        self.assertLess(text.index("packet hash mismatch"), text.index("date -u"))
        self.assertLess(text.index("controller hash mismatch"), text.index("source \"$1\""))
        self.assertIn("set -C", text)
        self.assertIn("</dev/null >&3 2>&4", text)
        for token in ("ssh ", "scp ", "rsync ", "qsub", "qstat", "qdel", "rocminfo",
                      "rocm-smi", "cmake ", "ctest", "hipcc", "module load", "pkill", "killall"):
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()

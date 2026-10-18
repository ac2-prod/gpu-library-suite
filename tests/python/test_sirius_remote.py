"""Sirius transport tests: only injected fake children; never execute ssh."""

import hashlib
import copy
import importlib.util
import io
import json
import multiprocessing
import os
from pathlib import Path
import shlex
import stat
import subprocess
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("sirius_remote", ROOT / "jobs/sirius/remote.py")
remote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remote)
PRODUCTION_TEMPORARY_ROOT = remote.is_temporary_state_root
VALIDATE_SSH_EXECUTABLE = remote.validate_ssh_executable
VALIDATE_CONTROL_SOCKET = remote.validate_control_socket


class FakeChild:
    def __init__(self, argv, stdin, stdout, stderr, code=0, proof=True, error=b"",
                 timeout=False, ready=None, release=None, crash=False, env=None):
        self.argv, self.stdout, self.env = argv, stdout, env
        self.stderr = io.BytesIO() if stderr == subprocess.PIPE else stderr
        self.code, self.proof, self.error = code, proof, error
        self.timeout, self.ready, self.release, self.crash = timeout, ready, release, crash
        self.returncode = None
        self.killed = False
        self.sent = None

    def communicate(self, input=None, timeout=None):
        if input is None:
            return (None, self.stderr.getvalue() if isinstance(self.stderr, io.BytesIO) else None)
        self.sent = input
        if self.crash:
            os._exit(24)
        if self.ready is not None:
            self.ready.set()
            if not self.release.wait(10):
                raise RuntimeError("test worker release timeout")
        self.stdout.write(b"probe output\n")
        self.stderr.write(self.error)
        self.stdout.flush()
        self.stderr.flush()
        if self.timeout:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        if self.proof:
            token = shlex.split(self.argv[-1])[-1]
            self.stderr.write("GPU_SUITE_PROBE_END {0} {1}\n".format(token, self.code).encode())
            self.stderr.flush()
        self.returncode = self.code
        return (None, self.stderr.getvalue() if isinstance(self.stderr, io.BytesIO) else None)

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def worker_probe(state, coordination, mode, ready=None, release=None, results=None):
    store = remote.Store(state, coordination)

    def fake(*args, **kwargs):
        if mode == "crash-before-observed":
            os._exit(23)
        return FakeChild(*args, **kwargs, ready=ready, release=release, crash=mode == "crash-after-observed")

    try:
        value = store.probe("approval", "worker", popen=fake)
        if results is not None:
            results.put(value["status"])
    except remote.SafetyError as error:
        if results is not None:
            results.put(str(error))


class SiriusRemoteTests(unittest.TestCase):
    def setUp(self):
        # Respect the caller's dedicated validation TMPDIR. No symlinks are created.
        self.temp = tempfile.TemporaryDirectory(prefix="sirius-remote-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        # Python-only mocks keep every fixture disposable. The production CLI
        # cannot disable either check or select a different SSH executable.
        temporary_root = mock.patch.object(remote, "is_temporary_state_root", return_value=False)
        temporary_root.start()
        self.addCleanup(temporary_root.stop)
        executable = mock.patch.object(remote, "validate_ssh_executable")
        self.executable_check = executable.start()
        self.addCleanup(executable.stop)
        popen = mock.patch.object(remote.subprocess, "Popen",
                                 side_effect=AssertionError("real subprocess must not start"))
        self.subprocess_popen = popen.start()
        self.addCleanup(popen.stop)
        run = mock.patch.object(remote.subprocess, "run",
                               side_effect=AssertionError("real subprocess must not start"))
        self.subprocess_run = run.start()
        self.addCleanup(run.stop)
        self.store = remote.Store(self.base / "state", self.base / "coordination")
        self.store.initialize()
        self.digest = hashlib.sha256(remote.SCRIPT.read_bytes()).hexdigest()
        self.store.approve("approval", "approved-alias", "/approved/repository", self.digest)
        self.children = []
        self.context = multiprocessing.get_context("fork")

    def launch(self, **options):
        def fake(*args, **kwargs):
            child = FakeChild(*args, **kwargs, **options)
            self.children.append(child)
            return child
        return fake

    def read_ledger(self):
        return remote.read_json(self.store.root / "ledger.json")

    def legacy_fixture(self, store=None, status="CONNECTION_STOP"):
        """Build an isolated v1 record with the old argv, never read real state."""
        store = store or self.store
        approval_id, execution_id = "sirius-env-probe-v1", "sirius-env-probe-run1"
        store.approve(approval_id, "approved-alias", "/approved/repository", self.digest)
        ledger = remote.read_json(store.root / "ledger.json")
        approval = ledger["approvals"][approval_id]
        token = execution_id + "." + "1" * 32
        # Keep this historical argv independent of the current command builder.
        argv = ["/usr/bin/ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                "-o", "ConnectionAttempts=1", "-o", "ConnectTimeout=20",
                "-o", "ControlMaster=no", "-o", "ControlPath=none", "-o", "ControlPersist=no",
                "-o", "ClearAllForwardings=yes", "-o", "ForwardAgent=no", "-o", "ForwardX11=no",
                "-o", "PermitLocalCommand=no", "-o", "ForkAfterAuthentication=no", "-o", "StdinNull=no",
                "-o", "AddKeysToAgent=no", "-o", "UpdateHostKeys=no", "-o", "CheckHostIP=no",
                "--", "approved-alias", "bash -s -- /approved/repository " + token]
        logs = store.root / "executions" / execution_id
        logs.mkdir(mode=0o700)
        stdout = b""
        stderr = b"Permission denied (keyboard-interactive).\n" if status == "CONNECTION_STOP" else b""
        remote.exclusive_bytes(logs / "stdout.bin", stdout)
        remote.exclusive_bytes(logs / "stderr.bin", stderr)
        record = dict(approval_id=approval_id, execution_id=execution_id, approval=approval,
                      argv=argv, completion_token=token, status=status,
                      started_at="2026-10-16T17:41:23+00:00",
                      ended_at=None if status in ("RESERVED", "RUNNING") else "2026-10-16T17:41:24+00:00",
                      reserved_connections=1, ssh_calls=None if status == "RESERVED" else 1,
                      ssh_exit=255 if status == "CONNECTION_STOP" else None,
                      remote_exit=None, completion_proof=False,
                      reason="authentication/host key/connection limit" if status == "CONNECTION_STOP"
                      else "interrupted invocation; remote outcome unknown",
                      stdout_sha256=hashlib.sha256(stdout).hexdigest(),
                      stderr_sha256=hashlib.sha256(stderr).hexdigest())
        ledger["executions"][execution_id] = record
        store.save(ledger)
        return approval_id, execution_id

    def file_snapshot(self):
        return {str(path): (path.read_bytes(), path.stat().st_mtime_ns,
                            stat.S_IMODE(path.stat().st_mode), path.stat().st_uid)
                for path in self.base.rglob("*") if path.is_file()}

    def test_dry_run_never_launches_or_changes_files(self):
        before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.base.rglob("*") if p.is_file()}
        launcher = mock.Mock(side_effect=AssertionError("must not spawn"))
        result = self.store.probe("approval", "run", dry_run=True, popen=launcher)
        self.assertTrue(result["dry_run"])
        launcher.assert_not_called()
        self.executable_check.assert_not_called()
        self.subprocess_popen.assert_not_called()
        self.subprocess_run.assert_not_called()
        after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in self.base.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_hash_mismatch_stops_before_reservation_and_spawn(self):
        self.store.approve("changed", "approved-alias", "/approved/repository", "0" * 64)
        launcher = mock.Mock()
        with self.assertRaisesRegex(remote.SafetyError, "SHA-256"):
            self.store.probe("changed", "run", popen=launcher)
        launcher.assert_not_called()
        self.executable_check.assert_not_called()
        self.subprocess_popen.assert_not_called()
        self.subprocess_run.assert_not_called()
        self.assertEqual(self.read_ledger()["executions"], {})

    def test_success_has_proof_exact_bytes_and_distinct_logs(self):
        result = self.store.probe("approval", "run", popen=self.launch())
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual((result["ssh_exit"], result["remote_exit"], result["ssh_calls"],
                          result["reserved_connections"]), (0, 0, 1, 1))
        self.assertTrue(result["completion_proof"])
        self.assertLessEqual(result["started_at"], result["ended_at"])
        self.assertEqual(self.children[0].sent, remote.SCRIPT.read_bytes())
        self.assertEqual((self.store.root / "executions/run/stdout.bin").read_bytes(), b"probe output\n")
        self.assertIn(b"GPU_SUITE_PROBE_END", (self.store.root / "executions/run/stderr.bin").read_bytes())
        with self.store.locked():
            pass

    def test_script_is_read_once_and_same_bytes_are_sent(self):
        original = Path.read_bytes
        script_reads = []

        def read(path):
            if path == remote.SCRIPT:
                script_reads.append(path)
            return original(path)

        with mock.patch.object(Path, "read_bytes", read):
            self.store.probe("approval", "run", popen=self.launch())
        self.assertEqual(len(script_reads), 1)
        self.assertEqual(hashlib.sha256(self.children[0].sent).hexdigest(), self.digest)

    def test_script_path_change_at_spawn_does_not_change_sent_bytes(self):
        script = self.base / "probe.sh"
        approved = remote.SCRIPT.read_bytes()
        script.write_bytes(approved)

        def fake(*args, **kwargs):
            script.write_bytes(b"changed after approval check")
            child = FakeChild(*args, **kwargs)
            self.children.append(child)
            return child

        with mock.patch.object(remote, "SCRIPT", script):
            self.store.probe("approval", "run", popen=fake)
        self.assertEqual(self.children[0].sent, approved)

    def test_ssh_options_do_not_touch_existing_connections(self):
        self.store.probe("approval", "run", popen=self.launch(timeout=True))
        child = self.children[0]
        for option in ("ConnectionAttempts=1", "BatchMode=yes", "StrictHostKeyChecking=yes",
                       "ControlMaster=no", "ControlPath=none", "ControlPersist=no", "ConnectTimeout=20",
                       "ClearAllForwardings=yes", "ForwardAgent=no", "ForwardX11=no",
                       "PermitLocalCommand=no", "ForkAfterAuthentication=no", "StdinNull=no",
                       "AddKeysToAgent=no", "UpdateHostKeys=no", "CheckHostIP=no"):
            self.assertIn(option, child.argv)
        self.assertNotIn("-O", child.argv)
        self.assertIn("-T", child.argv)
        self.assertEqual(child.argv[0], "/usr/bin/ssh")
        self.assertFalse(any(value.startswith(("ProxyCommand=", "ProxyJump=", "-O"))
                             for value in child.argv))
        self.executable_check.assert_called_once_with()
        self.assertEqual(len(self.children), 1)
        self.assertTrue(child.killed)

    def test_publickey_only_argv_records_local_agent_without_forwarding_or_key_additions(self):
        expected = ("PreferredAuthentications=publickey", "PubkeyAuthentication=yes",
                    "KbdInteractiveAuthentication=no", "PasswordAuthentication=no",
                    "GSSAPIAuthentication=no", "HostbasedAuthentication=no", "IdentitiesOnly=yes",
                    "IdentityAgent=SSH_AUTH_SOCK", "LogLevel=DEBUG1", "ForwardAgent=no", "AddKeysToAgent=no")
        forbidden = ("PreferredAuthentications=keyboard-interactive", "KbdInteractiveAuthentication=yes",
                     "PasswordAuthentication=yes", "PubkeyAcceptedAlgorithms=+ssh-rsa",
                     "HostKeyAlgorithms=+ssh-rsa")
        dry_run = self.store.probe("approval", "run", dry_run=True, popen=mock.Mock())
        result = self.store.probe("approval", "run", popen=self.launch())
        for argv in (dry_run["argv"], result["argv"]):
            with self.subTest(argv=argv):
                self.assertEqual(argv[0], "/usr/bin/ssh")
                for option in expected:
                    self.assertEqual(argv.count(option), 1)
                    self.assertEqual(argv[argv.index(option) - 1], "-o")
                for option in forbidden:
                    self.assertNotIn(option, argv)
                self.assertFalse(any(value.startswith(("IdentityFile=", "PubkeyAcceptedAlgorithms=",
                                                       "HostKeyAlgorithms=", "ProxyCommand=", "ProxyJump=", "-O"))
                                     for value in argv))
        self.assertEqual(result["argv"], self.children[0].argv)
        self.assertEqual(result["argv"], self.read_ledger()["executions"]["run"]["argv"])
        self.assertEqual(len(self.children), 1)
        self.subprocess_popen.assert_not_called()
        self.subprocess_run.assert_not_called()

    def test_legacy_connection_stop_state_is_readable_without_conversion(self):
        approval_id, execution_id = self.legacy_fixture()
        before = self.file_snapshot()
        with self.store.locked() as ledger:
            self.store.recover(ledger)
            record = ledger["executions"][execution_id]
            self.assertEqual(record["status"], "CONNECTION_STOP")
            self.assertEqual((record["ssh_calls"], record["reserved_connections"]), (1, 1))
            self.assertIsNone(record["remote_exit"])
            self.assertFalse(record["completion_proof"])
            self.assertNotIn("IdentityAgent=SSH_AUTH_SOCK", record["argv"])
            self.assertEqual(ledger["approvals"][approval_id]["connection_budget"], 1)
        self.assertEqual(self.file_snapshot(), before)
        self.executable_check.assert_not_called()
        self.subprocess_popen.assert_not_called()
        self.subprocess_run.assert_not_called()

    def test_debug_fingerprints_are_redacted_without_losing_completion_evidence(self):
        sha256 = b"SHA256:" + b"A" * 43
        md5 = b"MD5:" + b":".join([b"ab"] * 16)
        diagnostics = (b"debug1: get_agent_identities: agent returned 1 keys\n"
                       b"debug1: Will attempt key: /fake/home/.ssh/id_rsa RSA " + sha256 + b" agent\n"
                       b"debug1: Offering public key: /fake/home/.ssh/id_rsa RSA " + sha256 + b" agent\n"
                       b"debug1: Server accepts key: /fake/home/.ssh/id_rsa RSA " + sha256 + b" agent\n"
                       b"debug1: synthetic legacy fingerprint " + md5 + b"\n")
        result = self.store.probe("approval", "run", popen=self.launch(error=diagnostics))
        stdout = (self.store.root / "executions/run/stdout.bin").read_bytes()
        stderr = (self.store.root / "executions/run/stderr.bin").read_bytes()
        self.assertEqual(stdout, b"probe output\n")
        self.assertNotIn(stdout, stderr)
        self.assertNotIn(sha256, stderr)
        self.assertNotIn(md5, stderr)
        self.assertEqual(stderr.count(b"[REDACTED_FINGERPRINT]"), 4)
        for event in (b"agent returned 1 keys", b"Will attempt key:", b"Offering public key:", b"Server accepts key:"):
            self.assertIn(event, stderr)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertTrue(result["completion_proof"])
        self.assertEqual(remote.completion_codes(stderr, result["completion_token"]), ["0"])
        self.assertEqual(result["stdout_sha256"], hashlib.sha256(stdout).hexdigest())
        self.assertEqual(result["stderr_sha256"], hashlib.sha256(stderr).hexdigest())
        with self.store.locked():
            pass

    def test_debug_authentication_evidence_is_kept_without_fingerprints(self):
        sha256 = b"SHA256:" + b"B" * 43
        diagnostics = (b"debug1: Offering public key: /fake/home/.ssh/id_rsa RSA " + sha256 + b" agent\n"
                       b"debug1: Authenticated using publickey with partial success.\n"
                       b"debug1: Authentications that can continue: keyboard-interactive\n"
                       b"Permission denied (publickey).\n")
        result = self.store.probe("approval", "run", popen=self.launch(code=255, proof=False, error=diagnostics))
        stderr = (self.store.root / "executions/run/stderr.bin").read_bytes()
        self.assertNotIn(sha256, stderr)
        self.assertIn(b"[REDACTED_FINGERPRINT]", stderr)
        self.assertIn(b"Authenticated using publickey with partial success", stderr)
        self.assertIn(b"Authentications that can continue: keyboard-interactive", stderr)
        self.assertIn(b"Permission denied (publickey)", stderr)
        self.assertEqual(result["status"], "CONNECTION_STOP")
        self.assertIsNone(result["remote_exit"])
        self.assertFalse(result["completion_proof"])
        self.assertEqual(result["stderr_sha256"], hashlib.sha256(stderr).hexdigest())

    def test_new_human_approval_after_legacy_connection_stop_preserves_evidence(self):
        approval_id, execution_id = self.legacy_fixture()
        before_ledger = self.read_ledger()
        before_files = self.file_snapshot()
        # Models a separately approved registration in a disposable fixture only.
        self.store.approve("human-approved-next", "approved-alias", "/approved/repository", self.digest)
        after_ledger = self.read_ledger()
        self.assertEqual(after_ledger["executions"], before_ledger["executions"])
        for old_id, old_approval in before_ledger["approvals"].items():
            self.assertEqual(after_ledger["approvals"][old_id], old_approval)
        after_files = self.file_snapshot()
        ledger_path = str(self.store.root / "ledger.json")
        self.assertEqual({p: value for p, value in before_files.items() if p != ledger_path},
                         {p: value for p, value in after_files.items() if p != ledger_path})
        launcher = mock.Mock(side_effect=AssertionError("must not spawn"))
        plan = self.store.probe("human-approved-next", "next-run", dry_run=True, popen=launcher)
        self.assertTrue(plan["dry_run"])
        with self.assertRaisesRegex(remote.SafetyError, "budget"):
            self.store.probe(approval_id, "other-run", popen=launcher)
        self.assertEqual(self.read_ledger()["executions"][execution_id]["status"], "CONNECTION_STOP")
        self.assertEqual(self.file_snapshot(), after_files)
        launcher.assert_not_called()
        self.executable_check.assert_not_called()
        self.subprocess_popen.assert_not_called()
        self.subprocess_run.assert_not_called()

    def test_unresolved_states_block_new_approval_and_probe_without_spawn(self):
        for status in ("UNKNOWN", "RESERVED", "RUNNING"):
            with self.subTest(status=status):
                store = remote.Store(self.base / (status + "-state"), self.base / (status + "-coordination"))
                store.initialize()
                approval_id, execution_id = self.legacy_fixture(store, status)
                launcher = mock.Mock(side_effect=AssertionError("must not spawn"))
                before = self.file_snapshot()
                with self.assertRaisesRegex(remote.SafetyError, "unresolved"):
                    store.probe(approval_id, "next-run", dry_run=True, popen=launcher)
                self.assertEqual(self.file_snapshot(), before)
                with self.assertRaisesRegex(remote.SafetyError, "unresolved"):
                    store.approve("human-approved-next", "approved-alias", "/approved/repository", self.digest)
                with self.assertRaisesRegex(remote.SafetyError, "unresolved"):
                    store.probe(approval_id, "next-run", popen=launcher)
                ledger = remote.read_json(store.root / "ledger.json")
                self.assertNotIn("human-approved-next", ledger["approvals"])
                self.assertEqual(set(ledger["executions"]), {execution_id})
                self.assertEqual(ledger["executions"][execution_id]["status"], "UNKNOWN")
                self.assertEqual(ledger["executions"][execution_id]["reserved_connections"], 1)
                launcher.assert_not_called()
                self.executable_check.assert_not_called()
                self.subprocess_popen.assert_not_called()
                self.subprocess_run.assert_not_called()

    def test_executable_validation_uses_fixed_file_metadata_without_subprocess(self):
        self.assertEqual(remote.SSH_EXECUTABLE, Path("/usr/bin/ssh"))
        with mock.patch.object(Path, "lstat", autospec=True,
                               return_value=SimpleNamespace(st_mode=stat.S_IFREG | 0o755)) as lstat, \
                mock.patch.object(remote.os, "access", return_value=True) as access, \
                mock.patch.object(remote.subprocess, "Popen") as popen, \
                mock.patch.object(remote.subprocess, "run") as run:
            VALIDATE_SSH_EXECUTABLE()
        lstat.assert_called_once_with(Path("/usr/bin/ssh"))
        access.assert_called_once_with("/usr/bin/ssh", os.X_OK)
        popen.assert_not_called()
        run.assert_not_called()

    def test_unavailable_ssh_executable_stops_before_reservation(self):
        original_lstat = Path.lstat
        before = self.read_ledger()
        for kind in ("missing", "directory", "symlink", "not-executable"):
            with self.subTest(kind=kind):
                def lstat(path):
                    if path != remote.SSH_EXECUTABLE:
                        return original_lstat(path)
                    if kind == "missing":
                        raise FileNotFoundError("mock missing executable")
                    mode = {"directory": stat.S_IFDIR, "symlink": stat.S_IFLNK,
                            "not-executable": stat.S_IFREG}[kind]
                    return SimpleNamespace(st_mode=mode | 0o600)

                launcher = mock.Mock()
                with mock.patch.object(remote, "validate_ssh_executable", VALIDATE_SSH_EXECUTABLE), \
                        mock.patch.object(Path, "lstat", lstat), \
                        mock.patch.object(remote.os, "access", return_value=False), \
                        mock.patch.object(remote.subprocess, "Popen") as popen, \
                        mock.patch.object(remote.subprocess, "run") as run:
                    with self.assertRaisesRegex(remote.SafetyError, "regular executable"):
                        self.store.probe("approval", "run", popen=launcher)
                launcher.assert_not_called()
                popen.assert_not_called()
                run.assert_not_called()
                self.assertEqual(self.read_ledger(), before)
                self.assertEqual(list((self.store.root / "executions").iterdir()), [])

    def test_remote_nonzero_with_proof_is_confirmed_failure(self):
        result = self.store.probe("approval", "run", popen=self.launch(code=67))
        self.assertEqual((result["status"], result["remote_exit"]), ("REMOTE_FAILED", 67))

    def test_missing_proof_is_unknown_even_with_zero_exit(self):
        result = self.store.probe("approval", "run", popen=self.launch(proof=False))
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIsNone(result["remote_exit"])

    def test_wrong_and_duplicate_completion_proofs_are_unknown(self):
        for mode in ("wrong", "duplicate"):
            self.store.approve(mode, "approved-alias", "/approved/repository", self.digest)

        def fake(*args, **kwargs):
            token = shlex.split(args[0][-1])[-1]
            extra = "GPU_SUITE_PROBE_END {0} 0\n".format(token)
            return FakeChild(*args, **kwargs, error=extra.encode())

        result = self.store.probe("duplicate", "duplicate", popen=fake)
        self.assertEqual(result["status"], "UNKNOWN")
        # A separate isolated root is necessary because UNKNOWN blocks every approval.
        other = remote.Store(self.base / "second-state", self.base / "second-coordination")
        other.initialize()
        other.approve("wrong", "approved-alias", "/approved/repository", self.digest)
        result = other.probe("wrong", "wrong", popen=self.launch(proof=False,
                             error=b"GPU_SUITE_PROBE_END different-token 0\n"))
        self.assertEqual(result["status"], "UNKNOWN")

    def test_authentication_diagnostic_takes_precedence_over_proof(self):
        result = self.store.probe("approval", "run", popen=self.launch(error=b"Permission denied\n"))
        self.assertEqual(result["status"], "CONNECTION_STOP")
        self.assertFalse(result["completion_proof"])

    def test_disconnect_is_unknown_even_with_completion_line(self):
        result = self.store.probe("approval", "run", popen=self.launch(code=255))
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertFalse(result["completion_proof"])

    def test_timeout_records_unknown_and_never_retries(self):
        sha256 = b"SHA256:" + b"C" * 43
        result = self.store.probe("approval", "run", popen=self.launch(timeout=True,
                                 error=b"debug1: Offering public key: RSA " + sha256 + b" agent\n"))
        self.assertEqual(result["status"], "UNKNOWN")
        stderr = (self.store.root / "executions/run/stderr.bin").read_bytes()
        self.assertNotIn(sha256, stderr)
        self.assertIn(b"Offering public key: RSA [REDACTED_FINGERPRINT] agent", stderr)
        self.assertFalse(result["completion_proof"])
        self.assertEqual(result["stderr_sha256"], hashlib.sha256(stderr).hexdigest())
        with self.assertRaisesRegex(remote.SafetyError, "unresolved"):
            self.store.probe("approval", "new-run", popen=self.launch())
        self.assertEqual(len(self.children), 1)

    def test_stderr_capture_errors_remain_unknown_without_replay(self):
        for drain_fails in (False, True):
            with self.subTest(drain_fails=drain_fails):
                suffix = "unavailable" if drain_fails else "recovered"
                store = remote.Store(self.base / (suffix + "-state"), self.base / (suffix + "-coordination"))
                store.initialize()
                store.approve("approval", "approved-alias", "/approved/repository", self.digest)
                sha256 = b"SHA256:" + b"D" * 43

                class InterruptedChild(FakeChild):
                    def communicate(self, input=None, timeout=None):
                        if input is not None:
                            self.stderr.write(b"debug1: Offering public key: RSA " + sha256 + b" agent\n")
                            raise RuntimeError("fake interruption after diagnostic output")
                        if drain_fails:
                            raise RuntimeError("fake stderr drain failure")
                        return super().communicate(input, timeout)

                launches = []

                def fake(*args, **kwargs):
                    child = InterruptedChild(*args, **kwargs)
                    launches.append(child)
                    return child

                result = store.probe("approval", "run", popen=fake)
                self.assertEqual(result["status"], "UNKNOWN")
                self.assertIsNone(result["remote_exit"])
                self.assertFalse(result["completion_proof"])
                self.assertEqual((result["ssh_calls"], result["reserved_connections"]), (1, 1))
                self.assertIn("local error after launch: RuntimeError", result["reason"])
                stderr = (store.root / "executions/run/stderr.bin").read_bytes()
                self.assertNotIn(sha256, stderr)
                if drain_fails:
                    self.assertIn("stderr unavailable: RuntimeError", result["reason"])
                    self.assertEqual(stderr, b"")
                else:
                    self.assertIn(b"Offering public key: RSA [REDACTED_FINGERPRINT] agent", stderr)
                self.assertEqual(result["stderr_sha256"], hashlib.sha256(stderr).hexdigest())
                with self.assertRaisesRegex(remote.SafetyError, "unresolved"):
                    store.probe("approval", "retry", popen=fake)
                self.assertEqual(len(launches), 1)
                self.assertTrue(launches[0].killed)
                self.subprocess_popen.assert_not_called()
                self.subprocess_run.assert_not_called()

    def test_authentication_host_key_and_connection_limit_stop(self):
        for index, message in enumerate((b"Permission denied (publickey).", b"Host key verification failed.",
                                         b"REMOTE HOST IDENTIFICATION HAS CHANGED!", b"MaxStartups limit")):
            approval = "case-{0}".format(index)
            self.store.approve(approval, "approved-alias", "/approved/repository", self.digest)
            result = self.store.probe(approval, approval, popen=self.launch(code=255, proof=False, error=message))
            self.assertEqual(result["status"], "CONNECTION_STOP")
            self.assertIsNone(result["remote_exit"])
        self.assertEqual(len(self.children), 4)

    def test_launch_error_consumes_budget_without_claiming_a_call(self):
        launcher = mock.Mock(side_effect=OSError("fake spawn error"))
        result = self.store.probe("approval", "run", popen=launcher)
        self.assertEqual((result["status"], result["ssh_calls"], result["reserved_connections"]),
                         ("LAUNCH_FAILED", 0, 1))
        with self.assertRaisesRegex(remote.SafetyError, "budget"):
            self.store.probe("approval", "new", popen=launcher)
        self.assertEqual(launcher.call_count, 1)

    def test_budget_survives_new_store_and_execution_id(self):
        self.store.probe("approval", "run", popen=self.launch())
        restarted = remote.Store(self.store.root, self.store.coordination)
        launcher = mock.Mock()
        self.executable_check.reset_mock()
        with self.assertRaisesRegex(remote.SafetyError, "budget"):
            restarted.probe("approval", "other", popen=launcher)
        with self.assertRaisesRegex(remote.SafetyError, "already used"):
            restarted.probe("approval", "run", popen=launcher)
        launcher.assert_not_called()
        self.executable_check.assert_not_called()
        self.subprocess_popen.assert_not_called()
        self.subprocess_run.assert_not_called()

    def test_initialization_approval_and_alternate_root_cannot_reset(self):
        self.store.probe("approval", "run", popen=self.launch())
        with self.assertRaises(remote.SafetyError):
            self.store.initialize()
        with self.assertRaises(remote.SafetyError):
            self.store.approve("approval", "approved-alias", "/approved/repository", self.digest)
        other = remote.Store(self.base / "other", self.store.coordination)
        with self.assertRaises(remote.SafetyError):
            other.initialize()
        self.assertFalse(other.root.exists())

    def test_process_lock_is_nonblocking_and_covers_different_approvals(self):
        self.store.approve("other", "approved-alias", "/approved/repository", self.digest)
        ready, release = self.context.Event(), self.context.Event()
        results = self.context.Queue()
        worker = self.context.Process(target=worker_probe,
                                      args=(self.store.root, self.store.coordination, "wait", ready, release, results))
        worker.start()
        try:
            self.assertTrue(ready.wait(10))
            launcher = mock.Mock()
            with self.assertRaisesRegex(remote.SafetyError, "global lock"):
                self.store.probe("other", "contender", popen=launcher)
            launcher.assert_not_called()
            self.executable_check.assert_not_called()
            self.subprocess_popen.assert_not_called()
            self.subprocess_run.assert_not_called()
            self.assertNotIn("contender", self.read_ledger()["executions"])
        finally:
            release.set()
            worker.join(10)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        self.assertEqual(worker.exitcode, 0)
        self.assertEqual(results.get(timeout=2), "COMPLETE")

    def test_reservation_is_persisted_before_spawn(self):
        def fake(*args, **kwargs):
            record = self.read_ledger()["executions"]["run"]
            self.assertEqual(record["status"], "RESERVED")
            self.assertEqual(record["reserved_connections"], 1)
            self.assertIsNone(record["ssh_calls"])
            return FakeChild(*args, **kwargs)
        self.store.probe("approval", "run", popen=fake)

    def crash_and_recover(self, mode, expected_calls):
        worker = self.context.Process(target=worker_probe, args=(self.store.root, self.store.coordination, mode))
        worker.start()
        worker.join(10)
        self.assertFalse(worker.is_alive())
        self.assertIn(worker.exitcode, (23, 24))
        launcher = mock.Mock()
        with self.assertRaisesRegex(remote.SafetyError, "unresolved"):
            self.store.probe("approval", "new-run", popen=launcher)
        launcher.assert_not_called()
        record = self.read_ledger()["executions"]["worker"]
        self.assertEqual(record["status"], "UNKNOWN")
        self.assertEqual(record["ssh_calls"], expected_calls)
        self.assertIsNone(record["ssh_exit"])
        self.assertEqual(record["reserved_connections"], 1)

    def test_restart_between_reservation_and_observed_spawn_is_unknown(self):
        self.crash_and_recover("crash-before-observed", None)

    def test_restart_after_observed_spawn_is_unknown(self):
        self.crash_and_recover("crash-after-observed", 1)

    def test_missing_state_is_never_reinitialized(self):
        # Rename only our disposable test file; never delete existing user state.
        (self.store.root / "ledger.json").rename(self.store.root / "ledger.saved")
        launcher = mock.Mock()
        with self.assertRaises(OSError):
            self.store.probe("approval", "run", popen=launcher)
        with self.assertRaises(remote.SafetyError):
            self.store.initialize()
        launcher.assert_not_called()

    def test_strict_json_rejects_duplicates_nonfinite_and_invalid_utf8(self):
        path = self.store.root / "ledger.json"
        for value in (b'{"a":1,"a":2}', b'{"x":NaN}', b'\xff'):
            path.write_bytes(value)
            launcher = mock.Mock()
            with self.assertRaises(ValueError):
                self.store.probe("approval", "run", popen=launcher)
            launcher.assert_not_called()

    def test_missing_or_changed_output_is_not_accepted(self):
        self.store.probe("approval", "run", popen=self.launch())
        path = self.store.root / "executions/run/stdout.bin"
        path.write_bytes(b"altered")
        with self.assertRaisesRegex(remote.SafetyError, "output"):
            with self.store.locked():
                pass
        path.rename(path.with_name("stdout.saved"))
        with self.assertRaises(OSError):
            with self.store.locked():
                pass

    def test_inconsistent_exit_record_is_rejected(self):
        self.store.probe("approval", "run", popen=self.launch())
        ledger = self.read_ledger()
        ledger["executions"]["run"]["remote_exit"] = 7
        self.store.save(ledger)
        with self.assertRaisesRegex(remote.SafetyError, "completion"):
            with self.store.locked():
                pass

    def test_ledger_boolean_and_missing_count_exit_fields_are_rejected(self):
        self.store.probe("approval", "run", popen=self.launch())
        original = self.read_ledger()
        for field in ("reserved_connections", "ssh_calls", "ssh_exit", "remote_exit"):
            for missing in (False, True):
                ledger = copy.deepcopy(original)
                if missing:
                    del ledger["executions"]["run"][field]
                else:
                    ledger["executions"]["run"][field] = True
                self.store.save(ledger)
                with self.assertRaises(remote.SafetyError):
                    with self.store.locked():
                        pass
        self.store.save(original)

    def test_status_specific_call_counts_are_required(self):
        self.store.probe("approval", "run", popen=self.launch())
        original = self.read_ledger()
        for value in (None, 0):
            ledger = copy.deepcopy(original)
            ledger["executions"]["run"]["ssh_calls"] = value
            self.store.save(ledger)
            with self.assertRaisesRegex(remote.SafetyError, "call count"):
                with self.store.locked():
                    pass
        self.store.save(original)

    def test_approval_after_crash_records_unknown_then_stops(self):
        worker = self.context.Process(target=worker_probe,
                                      args=(self.store.root, self.store.coordination, "crash-after-observed"))
        worker.start()
        worker.join(10)
        self.assertEqual(worker.exitcode, 24)
        with self.assertRaisesRegex(remote.SafetyError, "unresolved"):
            self.store.approve("next", "approved-alias", "/approved/repository", self.digest)
        self.assertEqual(self.read_ledger()["executions"]["worker"]["status"], "UNKNOWN")

    def test_temporary_coordination_loss_never_recreates_existing_state(self):
        (self.store.coordination / "binding.json").rename(self.store.coordination / "binding.saved")
        launcher = mock.Mock()
        with self.assertRaises(OSError):
            self.store.probe("approval", "run", popen=launcher)
        with self.assertRaises(remote.SafetyError):
            self.store.initialize()
        launcher.assert_not_called()

    def test_symlink_state_is_rejected_without_creating_symlinks(self):
        with mock.patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(remote.SafetyError, "symlink"):
                remote.Store(self.base / "fake-link", self.base / "other-coordination")

    def test_production_temporary_state_roots_are_rejected(self):
        roots = (Path("/tmp"), Path("/private/tmp"),
                 Path("/tmp") / "sirius-state-denied",
                 Path("/private/tmp") / "sirius-state-denied",
                 self.base / "sirius-state-denied")
        with mock.patch.object(remote, "is_temporary_state_root", PRODUCTION_TEMPORARY_ROOT):
            for root in roots:
                with self.subTest(root=str(root)):
                    self.assertTrue(PRODUCTION_TEMPORARY_ROOT(root))
                    with self.assertRaisesRegex(remote.SafetyError, "must not be inside a temporary directory"):
                        remote.Store(root, self.base / "other-coordination")
        with mock.patch.object(remote.tempfile, "gettempdir", return_value=str(self.base / "other-temp")):
            self.assertTrue(PRODUCTION_TEMPORARY_ROOT(Path("/tmp/state")))
            self.assertTrue(PRODUCTION_TEMPORARY_ROOT(Path("/private/tmp/state")))

    def test_temporary_root_realpath_aliases_are_rejected(self):
        # Simulate /tmp resolving elsewhere; do not create an actual symlink.
        backing = self.base / "temporary-backing"

        def resolve(path):
            if path == Path("relative-state-alias"):
                return Path("/private/tmp/aliased-state")
            if path == Path("/tmp"):
                return backing
            if Path("/tmp") in path.parents:
                return backing / path.relative_to("/tmp")
            return path

        with mock.patch.object(Path, "resolve", resolve):
            self.assertTrue(PRODUCTION_TEMPORARY_ROOT(backing))
            self.assertTrue(PRODUCTION_TEMPORARY_ROOT(backing / "state"))
            self.assertTrue(PRODUCTION_TEMPORARY_ROOT(Path("relative-state-alias")))

    def test_durable_local_state_root_equivalent_is_accepted(self):
        # Model a durable Mac-local path without reading or writing the user's
        # home directory. Filesystem-device acceptance is exercised by setUp.
        durable = Path("/Users/example/durable-sirius-state")
        with mock.patch.object(Path, "resolve", lambda path: path), \
                mock.patch.object(Path, "is_symlink", return_value=False), \
                mock.patch.object(Path, "exists", return_value=False), \
                mock.patch.object(remote, "is_temporary_state_root", PRODUCTION_TEMPORARY_ROOT):
            self.assertFalse(PRODUCTION_TEMPORARY_ROOT(durable))
            store = remote.Store(durable, self.base / "other-coordination")
            self.assertEqual(store.root, durable)

    def test_fuse_or_sshfs_device_is_rejected_before_initialization(self):
        store = remote.Store(self.base / "fuse-state", self.base / "fuse-coordination")
        original_stat = Path.stat

        def stat_value(path, *args, **kwargs):
            result = original_stat(path, *args, **kwargs)
            if path == self.base:
                values = list(result)
                values[2] = original_stat(Path("/tmp")).st_dev + 1
                return os.stat_result(values)
            return result

        with mock.patch.object(Path, "stat", stat_value):
            with self.assertRaisesRegex(remote.SafetyError, "same Mac local filesystem.*FUSE/SSHFS"):
                store.initialize()
        self.assertFalse(store.root.exists())
        self.assertFalse(store.coordination.exists())

    def test_cli_cannot_bypass_temporary_state_root_policy(self):
        with mock.patch.object(remote, "Store") as store, \
                mock.patch.object(remote.sys, "stderr", io.StringIO()):
            for option in ("--allow-temporary-state", "--allow-temporary-root", "--coordination"):
                with self.subTest(option=option), self.assertRaises(SystemExit) as error:
                    remote.main(["--state-directory", str(self.base / "cli-state"),
                                 "initialize", option])
                self.assertEqual(error.exception.code, 2)
            store.assert_not_called()
        with mock.patch.object(remote, "is_temporary_state_root", PRODUCTION_TEMPORARY_ROOT), \
                mock.patch.object(remote.Store, "initialize") as initialize, \
                mock.patch.object(remote.sys, "stderr", io.StringIO()) as stderr:
            self.assertEqual(remote.main(["--state-directory", str(self.base / "cli-state"),
                                          "initialize"]), 2)
            initialize.assert_not_called()
            self.assertIn("must not be inside a temporary directory", stderr.getvalue())

    def test_alias_and_root_cannot_inject_commands(self):
        for alias in ("-oProxyCommand=bad", "user@host", "name;bad", "name\n"):
            with self.assertRaises(remote.SafetyError):
                self.store.approve("bad", alias, "/approved/repository", self.digest)
        with self.assertRaises(remote.SafetyError):
            self.store.approve("bad", "alias", "/path/../other", self.digest)

    def test_probe_has_no_gpu_build_or_scheduler_commands(self):
        script = remote.SCRIPT.read_text()
        for prohibited in ("rocminfo", "rocm-smi", "qsub", "qstat", "qdel", "sbatch", "srun", "cmake --build"):
            self.assertNotIn(prohibited, script)
        self.assertIn("GIT_OPTIONAL_LOCKS=0", script)
        self.assertIn('"$git_root" != "$expected_root"', script)
        self.assertIn("status --short --branch --untracked-files=all", script)
        self.assertNotIn("--untracked-files=no", script)


class FakeMasterCheck:
    """A piped local control request, with no real socket or child process."""

    def __init__(self, argv, stdin=None, stdout=None, stderr=None, code=0,
                 response=b"Master running (pid=4242)\r\n", timeout=False, env=None):
        self.argv, self.code, self.response, self.env = argv, code, response, env
        self.returncode = None
        self.timeout, self.killed = timeout, False

    def communicate(self, input=None, timeout=None):
        if self.timeout and not self.killed:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        self.returncode = self.code if not self.killed else -9
        return (b"", self.response)

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def worker_master_probe(state, coordination, ready, release, results):
    """Forked worker inherits only the test's metadata mocks and fake launcher."""
    store = remote.Store(state, coordination)

    def fake(*args, **kwargs):
        if "-O" in args[0]:
            return FakeMasterCheck(*args, **kwargs)
        return FakeChild(*args, **kwargs, ready=ready, release=release)

    try:
        value = store.probe("master", "worker", popen=fake)
        results.put(value["status"])
    except remote.SafetyError as error:
        results.put(str(error))


def worker_master_crash(state, coordination, phase):
    store = remote.Store(state, coordination)
    def fake(*args, **kwargs):
        if "-O" in args[0]:
            if phase == "check-spawn":
                os._exit(31)
            return FakeMasterCheck(*args, **kwargs)
        if phase == "session-spawn":
            os._exit(32)
        return FakeChild(*args, **kwargs, crash=True)
    store.probe("master", "crashed", popen=fake)


class SiriusControlMasterTests(unittest.TestCase):
    """Production policy is inspected with fake metadata; no sockets are made."""

    # Explicit helpers avoid inheriting and duplicating the direct test inventory.
    read_ledger = SiriusRemoteTests.read_ledger
    file_snapshot = SiriusRemoteTests.file_snapshot
    legacy_fixture = SiriusRemoteTests.legacy_fixture

    def setUp(self):
        SiriusRemoteTests.setUp(self)
        self.control_path = "/Users/example/private-control/sirius.sock"
        self.metadata = {"socket": {"device": 7, "inode": 101, "uid": os.getuid(),
                                    "mode": stat.S_IFSOCK | 0o600},
                         "parent": {"device": 7, "inode": 100, "uid": os.getuid(),
                                    "mode": stat.S_IFDIR | 0o700}}
        validator = mock.patch.object(remote, "validate_control_socket", return_value=self.metadata)
        self.socket_check = validator.start()
        self.addCleanup(validator.stop)
        self.check_children, self.session_children = [], []
        self.approve_master()
        self.socket_check.reset_mock()

    def approve_master(self, approval_id="master", store=None, **kwargs):
        store = store or self.store
        store.approve(approval_id, "approved-alias", "/approved/repository", self.digest,
                      transport="control-master", control_path=self.control_path, **kwargs)

    def launcher(self, check_code=0, check_response=b"Master running (pid=4242)\r\n",
                 check_timeout=False, session_class=FakeChild, **options):
        def fake(*args, **kwargs):
            if "-O" in args[0]:
                child = FakeMasterCheck(*args, **kwargs, code=check_code,
                                        response=check_response, timeout=check_timeout)
                self.check_children.append(child)
            else:
                child = session_class(*args, **kwargs, **options)
                self.session_children.append(child)
            return child
        return fake

    def assert_no_real_processes(self):
        self.subprocess_popen.assert_not_called()
        self.subprocess_run.assert_not_called()

    def assert_network_zero(self, result):
        for name in ("underlying_connections_created", "new_tcp_attempts", "new_ssh_auth_attempts"):
            self.assertEqual(result[name], 0, name)

    def legacy_pair(self):
        v1_approval, v1_execution = self.legacy_fixture()
        v2_approval, v2_execution = "sirius-env-probe-publickey-v2", "sirius-env-probe-publickey-run2"
        self.store.approve(v2_approval, "approved-alias", "/approved/repository", self.digest)
        ledger = self.read_ledger()
        record = copy.deepcopy(ledger["executions"][v1_execution])
        record.update(approval_id=v2_approval, execution_id=v2_execution,
                      approval=ledger["approvals"][v2_approval],
                      completion_token=v2_execution + "." + "2" * 32)
        boundary = record["argv"].index("--")
        options = ["PreferredAuthentications=publickey", "PubkeyAuthentication=yes",
                   "KbdInteractiveAuthentication=no", "PasswordAuthentication=no",
                   "GSSAPIAuthentication=no", "HostbasedAuthentication=no", "IdentitiesOnly=yes",
                   "IdentityAgent=SSH_AUTH_SOCK", "LogLevel=DEBUG1"]
        record["argv"][boundary:boundary] = [value for option in options for value in ("-o", option)]
        record["argv"][-1] = "bash -s -- /approved/repository " + record["completion_token"]
        logs = self.store.root / "executions" / v2_execution
        logs.mkdir(mode=0o700)
        stderr = (b'Authenticated using "publickey" with partial success.\n'
                  b"Authentications that can continue: keyboard-interactive\n"
                  b"Permission denied (keyboard-interactive).\n")
        remote.exclusive_bytes(logs / "stdout.bin", b"")
        remote.exclusive_bytes(logs / "stderr.bin", stderr)
        record["stderr_sha256"] = hashlib.sha256(stderr).hexdigest()
        ledger["executions"][v2_execution] = record
        self.store.save(ledger)
        return ((v1_approval, v1_execution), (v2_approval, v2_execution))

    def test_legacy_v1_v2_are_read_without_conversion_or_metadata_changes(self):
        pairs = self.legacy_pair()
        before = self.file_snapshot()
        with self.store.locked() as ledger:
            self.store.recover(ledger)
            for approval_id, execution_id in pairs:
                self.assertNotIn("transport", ledger["approvals"][approval_id])
                self.assertEqual(ledger["executions"][execution_id]["status"], "CONNECTION_STOP")
                self.assertEqual(ledger["executions"][execution_id]["ssh_calls"], 1)
                self.assertIsNone(ledger["executions"][execution_id]["remote_exit"])
                self.assertFalse(ledger["executions"][execution_id]["completion_proof"])
        self.assertEqual(before, self.file_snapshot())
        self.socket_check.assert_not_called()
        self.assert_no_real_processes()

    def test_new_master_approval_preserves_terminal_direct_evidence(self):
        self.legacy_pair()
        before, files = self.read_ledger(), self.file_snapshot()
        self.approve_master("separately-human-approved")
        after = self.read_ledger()
        self.assertEqual(before["executions"], after["executions"])
        for key, value in before["approvals"].items():
            self.assertEqual(value, after["approvals"][key])
        ledger_path = str(self.store.root / "ledger.json")
        after_files = self.file_snapshot()
        self.assertEqual({k: v for k, v in files.items() if k != ledger_path},
                         {k: v for k, v in after_files.items() if k != ledger_path})
        for approval_id, execution_id in (("sirius-env-probe-v1", "sirius-env-probe-run1"),
                                          ("sirius-env-probe-publickey-v2", "sirius-env-probe-publickey-run2")):
            with self.assertRaises(remote.SafetyError):
                self.approve_master(approval_id)
            with self.assertRaises(remote.SafetyError):
                self.store.probe(approval_id, execution_id, popen=mock.Mock())
        self.assertEqual(after, self.read_ledger())

    def test_approval_binds_transport_socket_and_logical_budget(self):
        approval = self.read_ledger()["approvals"]["master"]
        self.assertEqual(approval["transport"], "control-master")
        self.assertEqual(approval["control_path"], self.control_path)
        self.assertEqual(approval["connection_budget"], 0)
        self.assertEqual(approval["logical_session_budget"], 1)
        self.assertEqual(approval["execution_budget"], 1)
        self.assertNotIn("transport", self.read_ledger()["approvals"]["approval"])
        launcher = mock.Mock(side_effect=AssertionError("must not launch"))
        plan = self.store.probe("master", "run", dry_run=True, popen=launcher)
        self.assertIn("ControlPath=" + self.control_path, plan["argv"])
        launcher.assert_not_called()

    def test_dry_run_never_checks_socket_or_starts_any_process(self):
        before = self.file_snapshot()
        launcher = mock.Mock(side_effect=AssertionError("must not launch"))
        plan = self.store.probe("master", "run", dry_run=True, popen=launcher)
        self.assertTrue(plan["dry_run"])
        launcher.assert_not_called()
        self.socket_check.assert_not_called()
        self.executable_check.assert_not_called()
        self.assertEqual(before, self.file_snapshot())
        self.assert_no_real_processes()

    def test_alive_check_failure_cannot_start_a_session(self):
        result = self.store.probe("master", "run", popen=self.launcher(check_code=255,
                                  check_response=b"Control socket connect: Connection refused\n"))
        self.assertEqual(result["status"], "MASTER_UNAVAILABLE")
        self.assertEqual(len(self.check_children), 1)
        self.assertEqual(self.session_children, [])
        self.assertEqual(result["local_check_calls"], 1)
        self.assertEqual(result["logical_session_calls"], 0)
        self.assertEqual(result["remote_command_attempts"], 0)
        self.assertEqual(result["remote_commands_confirmed"], 0)
        self.assertEqual(result["logical_sessions_confirmed"], 0)
        self.assert_network_zero(result)
        self.assert_no_real_processes()

    def test_alive_check_requires_master_running_response(self):
        result = self.store.probe("master", "run", popen=self.launcher(check_response=b""))
        self.assertEqual(result["status"], "MASTER_UNAVAILABLE")
        self.assertEqual(self.session_children, [])
        self.assertEqual(len(self.check_children), 1)
        self.assert_network_zero(result)

    def test_alive_check_timeout_is_unavailable_without_session_or_retry(self):
        result = self.store.probe("master", "run", popen=self.launcher(check_timeout=True))
        self.assertEqual(result["status"], "MASTER_UNAVAILABLE")
        self.assertEqual(len(self.check_children), 1)
        self.assertEqual(self.session_children, [])
        self.assert_network_zero(result)

    def test_check_and_session_use_exact_approved_socket_and_no_fallback(self):
        result = self.store.probe("master", "run", popen=self.launcher())
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual((len(self.check_children), len(self.session_children)), (1, 1))
        check, session = self.check_children[0].argv, self.session_children[0].argv
        for argv in (check, session):
            self.assertEqual(argv[0], "/usr/bin/ssh")
            self.assertEqual(argv[argv.index("-S") + 1], self.control_path)
            self.assertIn("ControlPath=" + self.control_path, argv)
            self.assertIn("ControlMaster=no", argv)
            self.assertEqual(argv[argv.index("--") + 1], "approved-alias")
        self.assertEqual(check[check.index("-O") + 1], "check")
        for option in ("ControlPersist=no", "BatchMode=yes", "ClearAllForwardings=yes",
                       "ForwardAgent=no", "ForwardX11=no", "PermitLocalCommand=no",
                       "AddKeysToAgent=no", "UpdateHostKeys=no", "CheckHostIP=no",
                       "ProxyCommand=/usr/bin/false"):
            self.assertIn(option, session)
        self.assertIn("-T", session)
        self.assertEqual(session[session.index("-F") + 1], "/dev/null")
        for option in ("CanonicalizeHostname=no", "IdentityAgent=none", "IdentityFile=none", "PubkeyAuthentication=no",
                       "KbdInteractiveAuthentication=no", "PasswordAuthentication=no",
                       "GSSAPIAuthentication=no", "HostbasedAuthentication=no"):
            self.assertIn(option, session)
        self.assertNotIn("-O", session)
        forbidden = ("ProxyJump=", "IdentityAgent=SSH_AUTH_SOCK",
                     "PreferredAuthentications=publickey", "PubkeyAuthentication=yes",
                     "KbdInteractiveAuthentication=yes", "PasswordAuthentication=yes",
                     "GSSAPIAuthentication=yes", "HostbasedAuthentication=yes")
        for value in session:
            self.assertFalse(value.startswith(forbidden), value)
        for child in self.check_children + self.session_children:
            self.assertEqual(child.env["LC_ALL"], "C")
            self.assertEqual(child.env["LANG"], "C")
            self.assertNotIn("SSH_ASKPASS", child.env)
            self.assertNotIn("SSH_ASKPASS_REQUIRE", child.env)
        self.assertEqual(result["argv"], session)
        self.assertEqual(result["local_check_calls"], 1)
        self.assertEqual(result["logical_session_calls"], 1)
        self.assertEqual(result["ssh_calls"], 2)
        self.assertEqual(result["reserved_connections"], 0)
        self.assertEqual(result["reserved_logical_sessions"], 1)
        self.assertEqual(result["remote_command_attempts"], 1)
        self.assertEqual(result["remote_commands_confirmed"], 1)
        self.assertEqual(result["logical_sessions_confirmed"], 1)
        self.assert_network_zero(result)
        self.assert_no_real_processes()

    def test_missing_socket_stops_before_any_process(self):
        self.socket_check.side_effect = remote.SocketUnavailable("fake absent socket")
        launcher = mock.Mock()
        result = self.store.probe("master", "run", popen=launcher)
        self.assertEqual(result["status"], "MASTER_UNAVAILABLE")
        launcher.assert_not_called()
        self.assert_network_zero(result)

    def test_socket_policy_failure_is_local_stop_without_process(self):
        self.socket_check.side_effect = remote.SafetyError("fake wrong socket owner")
        launcher = mock.Mock()
        result = self.store.probe("master", "run", popen=launcher)
        self.assertEqual(result["status"], "LOCAL_STOP")
        launcher.assert_not_called()
        self.assert_network_zero(result)

    def test_socket_disappears_after_check_is_master_lost_without_session(self):
        self.socket_check.side_effect = [self.metadata, remote.SocketUnavailable("fake vanished socket")]
        result = self.store.probe("master", "run", popen=self.launcher())
        self.assertEqual(result["status"], "MASTER_LOST")
        self.assertEqual(len(self.check_children), 1)
        self.assertEqual(self.session_children, [])
        self.assert_network_zero(result)

    def test_socket_identity_or_permissions_change_after_check_stops(self):
        for field in ("inode", "uid", "mode"):
            with self.subTest(field=field):
                store = remote.Store(self.base / (field + "-state"), self.base / (field + "-coordination"))
                store.initialize()
                self.approve_master(store=store)
                changed = copy.deepcopy(self.metadata)
                changed["socket"][field] += 1
                self.socket_check.side_effect = [self.metadata, changed]
                result = store.probe("master", "run", popen=self.launcher())
                self.assertIn(result["status"], ("LOCAL_STOP", "MASTER_LOST"))
                self.assertEqual(self.session_children, [])
                self.socket_check.side_effect = None

    def test_master_loss_during_session_has_guard_and_no_automatic_reconnection(self):
        result = self.store.probe("master", "run", popen=self.launcher(code=255, proof=False,
                                  error=b"mux_client_request_session: read from master failed: Broken pipe\n"))
        self.assertEqual(result["status"], "MASTER_LOST")
        self.assertIn("ProxyCommand=/usr/bin/false", self.session_children[0].argv)
        for approval, execution in (("master", "retry"), ("approval", "direct-fallback")):
            with self.assertRaisesRegex(remote.SafetyError, "unresolved"):
                self.store.probe(approval, execution, popen=self.launcher())
        with self.assertRaisesRegex(remote.SafetyError, "unresolved"):
            self.approve_master("replacement")
        self.assertEqual((len(self.check_children), len(self.session_children)), (1, 1))
        self.assert_network_zero(result)

    def test_missing_marker_is_unknown_and_cannot_replay(self):
        result = self.store.probe("master", "run", popen=self.launcher(proof=False))
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertFalse(result["completion_proof"])
        with self.assertRaisesRegex(remote.SafetyError, "unresolved"):
            self.store.probe("master", "retry", popen=self.launcher())
        self.assertEqual((len(self.check_children), len(self.session_children)), (1, 1))

    def test_timeout_is_unknown_and_only_owned_session_child_is_killed(self):
        with mock.patch.object(remote.os, "kill") as kill:
            result = self.store.probe("master", "run", popen=self.launcher(timeout=True))
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertFalse(result["completion_proof"])
        self.assertTrue(self.session_children[0].killed)
        self.assertEqual(result["remote_command_attempts"], 1)
        self.assertIsNone(result["remote_commands_confirmed"])
        self.assertIsNone(result["logical_sessions_confirmed"])
        self.assertFalse(self.check_children[0].killed)
        kill.assert_not_called()
        self.assertEqual((len(self.check_children), len(self.session_children)), (1, 1))

    def test_completion_requires_exactly_one_matching_marker_and_exit(self):
        modes = ("success", "nonzero", "mismatch", "duplicate", "wrong-token", "mixed-token")
        for mode in modes:
            with self.subTest(mode=mode):
                store = remote.Store(self.base / (mode + "-state"), self.base / (mode + "-coordination"))
                store.initialize()
                self.approve_master(store=store)

                class ProofChild(FakeChild):
                    def communicate(child, input=None, timeout=None):
                        if input is not None and mode in ("mismatch", "duplicate", "wrong-token", "mixed-token"):
                            token = shlex.split(child.argv[-1])[-1]
                            if mode in ("wrong-token", "mixed-token"):
                                token = "wrong-token"
                            code = 7 if mode == "mismatch" else 0
                            child.error += "GPU_SUITE_PROBE_END {0} {1}\n".format(token, code).encode()
                            child.proof = mode in ("duplicate", "mixed-token")
                        return super().communicate(input, timeout)

                result = store.probe("master", "run", popen=self.launcher(session_class=ProofChild,
                                     code=67 if mode == "nonzero" else 0))
                expected = {"success": "COMPLETE", "nonzero": "REMOTE_FAILED"}.get(mode, "UNKNOWN")
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["completion_proof"], mode in ("success", "nonzero"))

    def test_logical_budget_survives_restart_and_new_execution_id(self):
        self.store.probe("master", "run", popen=self.launcher())
        restarted = remote.Store(self.store.root, self.store.coordination)
        launcher = mock.Mock()
        self.executable_check.reset_mock()
        self.socket_check.reset_mock()
        with self.assertRaisesRegex(remote.SafetyError, "budget"):
            restarted.probe("master", "second", popen=launcher)
        with self.assertRaisesRegex(remote.SafetyError, "already used"):
            restarted.probe("master", "run", popen=launcher)
        launcher.assert_not_called()
        self.executable_check.assert_not_called()
        self.socket_check.assert_not_called()

    def test_global_lock_serializes_master_sessions_across_processes(self):
        self.approve_master("other")
        ready, release, results = self.context.Event(), self.context.Event(), self.context.Queue()
        worker = self.context.Process(target=worker_master_probe,
                                      args=(self.store.root, self.store.coordination, ready, release, results))
        worker.start()
        try:
            self.assertTrue(ready.wait(10))
            launcher = mock.Mock()
            with self.assertRaisesRegex(remote.SafetyError, "global lock"):
                self.store.probe("other", "contender", popen=launcher)
            launcher.assert_not_called()
            self.assertNotIn("contender", self.read_ledger()["executions"])
        finally:
            release.set()
            worker.join(10)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        self.assertEqual(worker.exitcode, 0)
        self.assertEqual(results.get(timeout=2), "COMPLETE")
        self.assert_no_real_processes()

    def test_hash_mismatch_and_unresolved_state_do_not_check_master(self):
        self.store.approve("bad-hash", "approved-alias", "/approved/repository", "0" * 64,
                           transport="control-master", control_path=self.control_path)
        self.socket_check.reset_mock()
        launcher = mock.Mock()
        with self.assertRaisesRegex(remote.SafetyError, "SHA-256"):
            self.store.probe("bad-hash", "bad-run", popen=launcher)
        launcher.assert_not_called()
        self.socket_check.assert_not_called()
        self.executable_check.assert_not_called()
        self.legacy_fixture(status="UNKNOWN")
        self.socket_check.reset_mock()
        with self.assertRaisesRegex(remote.SafetyError, "unresolved"):
            self.store.probe("master", "blocked", popen=launcher)
        launcher.assert_not_called()
        self.socket_check.assert_not_called()

    def test_control_commands_never_exit_stop_or_terminate_master(self):
        with mock.patch.object(remote.os, "kill") as kill, \
                mock.patch.object(remote.os, "killpg") as killpg:
            self.store.probe("master", "run", popen=self.launcher())
        all_argv = [child.argv for child in self.check_children + self.session_children]
        self.assertEqual(len(all_argv), 2)
        for argv in all_argv:
            if "-O" in argv:
                self.assertEqual(argv[argv.index("-O") + 1], "check")
            self.assertNotIn("exit", argv)
            self.assertNotIn("stop", argv)
            self.assertNotIn("-M", argv)
        self.assertFalse(any(child.killed for child in self.check_children + self.session_children))
        kill.assert_not_called()
        killpg.assert_not_called()

    def test_output_files_are_private_separate_and_match_recorded_hashes(self):
        result = self.store.probe("master", "run", popen=self.launcher(error=b"fake session diagnostic\n"))
        logs = self.store.root / "executions/run"
        stdout, stderr = (logs / "stdout.bin").read_bytes(), (logs / "stderr.bin").read_bytes()
        self.assertEqual(stdout, b"probe output\n")
        self.assertNotIn(stdout, stderr)
        self.assertIn(b"fake session diagnostic", stderr)
        self.assertIn(b"GPU_SUITE_PROBE_END", stderr)
        self.assertNotIn(b"Master running", stdout)
        self.assertEqual(stat.S_IMODE(logs.stat().st_mode), 0o700)
        for path, data, field in ((logs / "stdout.bin", stdout, "stdout_sha256"),
                                  (logs / "stderr.bin", stderr, "stderr_sha256")):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(hashlib.sha256(data).hexdigest(), result[field])
        self.assertEqual(self.session_children[0].sent, remote.SCRIPT.read_bytes())
        self.assert_no_real_processes()


    def fake_metadata_validation(self, path=None, case=None):
        socket_path = Path(path or self.control_path)
        def lstat(candidate):
            mode, owner, device, inode = stat.S_IFDIR | 0o700, os.getuid(), 7, 100
            if candidate == socket_path:
                mode, inode = stat.S_IFSOCK | 0o600, 101
                if case == "absent":
                    raise FileNotFoundError("fake absent socket")
                if case == "symlink":
                    mode = stat.S_IFLNK | 0o700
                if case == "regular":
                    mode = stat.S_IFREG | 0o600
                if case == "other-owner":
                    owner += 1
                if case == "socket-permissions":
                    mode |= 0o040
                if case == "fuse":
                    device += 1
            if candidate == socket_path.parent:
                if case == "parent-symlink":
                    mode = stat.S_IFLNK | 0o700
                if case == "parent-owner":
                    owner += 1
                if case == "parent-permissions":
                    mode |= 0o010
                if case == "parent-fuse":
                    device += 1
            return SimpleNamespace(st_mode=mode, st_uid=owner, st_dev=device, st_ino=inode)
        def exists(candidate):
            return case == "git-worktree" and candidate == socket_path.parent / ".git"
        with mock.patch.object(Path, "lstat", lstat), \
                mock.patch.object(Path, "exists", exists), \
                mock.patch.object(remote, "_local_filesystem_device", return_value=7):
            return VALIDATE_CONTROL_SOCKET(str(socket_path), self.store.root)

    def test_control_socket_policy_uses_fake_metadata_only(self):
        accepted = self.fake_metadata_validation()
        self.assertEqual(accepted, self.metadata)
        for case in ("symlink", "regular", "other-owner", "socket-permissions", "fuse",
                     "parent-symlink", "parent-owner", "parent-permissions", "parent-fuse", "git-worktree"):
            with self.subTest(case=case), self.assertRaises(remote.SafetyError):
                self.fake_metadata_validation(case=case)
        with self.assertRaises(remote.SocketUnavailable):
            self.fake_metadata_validation(case="absent")
        self.assert_no_real_processes()

    def test_control_path_rejects_unapproved_spelling_expansions_and_state_overlap(self):
        denied = ("relative.sock", "/Users/example/private-control/../other.sock",
                  "/Users/example//socket", "/Users/example/socket/", "/Users/example/%C",
                  "/Users/example/${SOCKET}", "/Users/example/socket\n", "/",
                  str(self.store.root), str(self.store.root / "master.sock"), str(self.store.root.parent))
        for path in denied:
            with self.subTest(path=path), self.assertRaises(remote.SafetyError):
                remote.validate_control_path(path, self.store.root)
        with mock.patch.object(remote.sys, "stderr", io.StringIO()), \
                mock.patch.object(remote, "Store") as store:
            with self.assertRaises(SystemExit) as failure:
                remote.main(["--state-directory", str(self.store.root), "probe", "--approval-id", "master",
                             "--execution-id", "run", "--control-path", "/Users/example/other.sock"])
            self.assertEqual(failure.exception.code, 2)
            store.assert_not_called()
        self.assert_no_real_processes()

    def test_control_socket_temporary_tree_rejected_by_production_predicate(self):
        roots = ("/tmp/private/socket", "/private/tmp/private/socket", str(self.base / "private/socket"))
        with mock.patch.object(remote, "is_temporary_state_root", PRODUCTION_TEMPORARY_ROOT):
            for path in roots:
                with self.subTest(path=path), self.assertRaisesRegex(remote.SafetyError, "temporary"):
                    self.fake_metadata_validation(path)
        self.assert_no_real_processes()

    def test_socket_replaced_since_approval_stops_before_alive_check(self):
        changed = copy.deepcopy(self.metadata)
        changed["socket"]["inode"] += 1
        self.socket_check.return_value = changed
        launcher = mock.Mock()
        result = self.store.probe("master", "run", popen=launcher)
        self.assertEqual(result["status"], "LOCAL_STOP")
        launcher.assert_not_called()
        self.assert_network_zero(result)


    def test_crash_windows_preserve_unknown_observations_and_prevent_replay(self):
        phases = (("check-spawn", 31, (None, 0)), ("session-spawn", 32, (1, None)),
                  ("session-observed", 24, (1, 1)))
        for phase, exitcode, counts in phases:
            with self.subTest(phase=phase):
                store = remote.Store(self.base / (phase + "-state"), self.base / (phase + "-coordination"))
                store.initialize()
                self.approve_master(store=store)
                worker = self.context.Process(target=worker_master_crash,
                                              args=(store.root, store.coordination, phase))
                worker.start()
                worker.join(10)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
                    self.fail("fake crash worker did not exit")
                self.assertEqual(worker.exitcode, exitcode)
                launcher = mock.Mock()
                with self.assertRaisesRegex(remote.SafetyError, "unresolved"):
                    store.probe("master", "replacement", popen=launcher)
                launcher.assert_not_called()
                ledger = remote.read_json(store.root / "ledger.json")
                record = ledger["executions"]["crashed"]
                self.assertEqual(record["status"], "UNKNOWN")
                self.assertEqual((record["local_check_calls"], record["logical_session_calls"]), counts)
                self.assertEqual(record["reserved_logical_sessions"], 1)
                self.assertEqual(record["reserved_connections"], 0)
                self.assertIsNone(record["remote_exit"])
                self.assertFalse(record["completion_proof"])
                self.assert_network_zero(record)
                with self.assertRaisesRegex(remote.SafetyError, "unresolved"):
                    self.approve_master("new-approval", store=store)
                with store.locked():
                    pass

    def test_ambiguous_fallback_guard_failure_is_unknown_without_retry(self):
        result = self.store.probe("master", "run", popen=self.launcher(code=255, proof=False,
                                  error=b"kex_exchange_identification: Connection closed by remote host\n"))
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual((len(self.check_children), len(self.session_children)), (1, 1))
        self.assertIn("ProxyCommand=/usr/bin/false", self.session_children[0].argv)
        self.assert_network_zero(result)
        with self.assertRaisesRegex(remote.SafetyError, "unresolved"):
            self.store.probe("master", "retry", popen=self.launcher())
        self.assertEqual((len(self.check_children), len(self.session_children)), (1, 1))


    def test_master_counter_and_alive_evidence_tampering_is_rejected(self):
        self.store.probe("master", "run", popen=self.launcher())
        original = self.read_ledger()
        alterations = (("underlying_connections_created", 1), ("new_tcp_attempts", 1),
                       ("new_ssh_auth_attempts", 1), ("ssh_calls", 1), ("local_check_calls", 0),
                       ("logical_session_calls", 0), ("logical_sessions_confirmed", 0),
                       ("remote_commands_confirmed", 0), ("remote_command_attempts", 0),
                       ("check_exit", 255), ("master_alive", False), ("reserved_logical_sessions", True))
        try:
            for field, value in alterations:
                with self.subTest(field=field):
                    ledger = copy.deepcopy(original)
                    ledger["executions"]["run"][field] = value
                    self.store.save(ledger)
                    with self.assertRaises(remote.SafetyError):
                        with self.store.locked():
                            pass
        finally:
            self.store.save(original)
        with self.store.locked():
            pass


    def test_failed_session_with_missing_socket_is_master_lost_without_fallback(self):
        self.socket_check.side_effect = [self.metadata, self.metadata, self.metadata,
                                        remote.SocketUnavailable("fake socket vanished during session")]
        result = self.store.probe("master", "run", popen=self.launcher(code=255, proof=False,
                                  error=b"kex_exchange_identification: Connection closed\n"))
        self.assertEqual(result["status"], "MASTER_LOST")
        self.assertEqual((len(self.check_children), len(self.session_children)), (1, 1))
        self.assert_network_zero(result)
        self.assertIsNone(result["remote_exit"])
        self.assertFalse(result["completion_proof"])
        launcher = mock.Mock()
        with self.assertRaisesRegex(remote.SafetyError, "unresolved"):
            self.store.probe("approval", "direct-fallback", popen=launcher)
        launcher.assert_not_called()


    def test_post_session_recording_failure_reports_unknown_and_preserves_durable_evidence(self):
        for failure in ("capture", "save-before-write", "save-after-write"):
            with self.subTest(failure=failure):
                store = remote.Store(self.base / (failure + "-state"), self.base / (failure + "-coordination"))
                store.initialize()
                self.approve_master(store=store)
                original_capture, original_save = store.capture, store.save
                def capture(execution):
                    if failure == "capture":
                        raise OSError("fake output fsync failure after session")
                    return original_capture(execution)
                def save(ledger):
                    terminal = ledger["executions"].get("run", {}).get("status") in (
                        "COMPLETE", "REMOTE_FAILED", "UNKNOWN", "MASTER_LOST", "LOCAL_STOP")
                    if terminal and failure == "save-before-write":
                        raise OSError("fake ledger failure before write")
                    original_save(ledger)
                    if terminal and failure == "save-after-write":
                        raise OSError("fake directory fsync failure after write")
                self.check_children, self.session_children = [], []
                with mock.patch.object(store, "capture", side_effect=capture), \
                        mock.patch.object(store, "save", side_effect=save), \
                        mock.patch.object(remote, "Store", return_value=store), \
                        mock.patch.object(remote.subprocess, "Popen", side_effect=self.launcher()), \
                        mock.patch.object(remote.sys, "stdout", io.StringIO()) as stdout, \
                        mock.patch.object(remote.sys, "stderr", io.StringIO()):
                    code = remote.main(["--state-directory", str(store.root), "probe",
                                        "--approval-id", "master", "--execution-id", "run"])
                result = json.loads(stdout.getvalue())
                self.assertEqual(result["status"], "UNKNOWN")
                self.assertNotEqual(code, 0)
                self.assertEqual(result["approval_id"], "master")
                self.assertEqual(result["execution_id"], "run")
                self.assertEqual((len(self.check_children), len(self.session_children)), (1, 1))
                ledger = remote.read_json(store.root / "ledger.json")
                record = ledger["executions"]["run"]
                self.assertEqual(record["reserved_logical_sessions"], 1)
                self.assertEqual(record["logical_session_calls"], 1)
                if failure == "save-after-write":
                    self.assertEqual(record["status"], "COMPLETE")
                    self.assertTrue(record["completion_proof"])
                else:
                    self.assertEqual(record["status"], "RUNNING")
                self.assert_network_zero(record)


if __name__ == "__main__":
    unittest.main()

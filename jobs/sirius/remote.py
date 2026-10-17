#!/usr/bin/env python3
"""Approved, serialized Sirius probe transport; no scheduler or arbitrary command API."""

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import os
from pathlib import Path
import posixpath
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import time
import uuid

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))
from gpu_suite import strict_json  # noqa: E402

COORDINATION = Path("/tmp/gpu-library-suite-local-build") / (
    "sirius-control-{0}".format(os.getuid())
)
SCRIPT = Path(__file__).with_name("probe.sh")
SSH_EXECUTABLE = Path("/usr/bin/ssh")
STATE_LOCATION_REQUIREMENT = (
    "state must be on the same Mac local filesystem as /tmp, never FUSE/SSHFS, "
    "but must not be inside a temporary directory"
)
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
FATAL = re.compile(r"permission denied|authentication failed|host key verification failed|"
                   r"remote host identification has changed|too many authentication|"
                   r"too many connections|maxstartups|rate.limit|connection.*limit", re.I)
FINGERPRINT = re.compile(rb"\b(?:SHA256:[A-Za-z0-9+/]+={0,2}|"
                         rb"MD5:(?:[0-9a-f]{2}:){15}[0-9a-f]{2})", re.I)
MASTER_RUNNING = re.compile(rb"^Master running \(pid=[1-9][0-9]*\)\r?$", re.M)
MASTER_LOSS = re.compile(r"control socket connect|mux_client.*(?:failed|broken|read|closed)|"
                         r"master.*(?:lost|closed|dead)|broken pipe", re.I)
UNRESOLVED = ("RESERVED", "RUNNING", "UNKNOWN", "MASTER_LOST")


class SafetyError(RuntimeError):
    """Stop before further remote work; never repair state implicitly."""


class SocketUnavailable(SafetyError):
    """The approved master socket is absent; never create or replace it."""


class OutcomeRecordingError(SafetyError):
    """Report uncertainty without retrying or rewriting a failed final record."""

    def __init__(self, result):
        self.result = result
        super().__init__(result["reason"])


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise SafetyError(message)


def is_temporary_state_root(path):
    """Reject temporary roots and aliases after resolving existing ancestors.

    Tests may mock this predicate for disposable state. The production CLI has
    no override; /tmp and /private/tmp are always included, regardless of TMPDIR.
    """
    root = Path(path).resolve()
    temporary = {Path("/tmp"), Path("/private/tmp"), Path(tempfile.gettempdir())}
    temporary.update(candidate.resolve() for candidate in tuple(temporary))
    return any(root == candidate or candidate in root.parents for candidate in temporary)


def validate_ssh_executable():
    """Inspect the fixed executable without spawning ssh or another process."""
    try:
        mode = SSH_EXECUTABLE.lstat().st_mode
    except OSError as error:
        raise SafetyError("/usr/bin/ssh must be a regular executable file") from error
    require(stat.S_ISREG(mode) and os.access(str(SSH_EXECUTABLE), os.X_OK),
            "/usr/bin/ssh must be a regular executable file")


def validate_control_path(path, state_root):
    """Validate literal syntax without opening or contacting a control socket."""
    require(isinstance(path, str) and path.startswith("/") and path != "/"
            and path == os.path.normpath(path) and not path.startswith("//")
            and not any(c.isspace() or ord(c) < 32 or ord(c) == 127 or c in "%$\\\"'" for c in path),
            "control path must be an approved normalized absolute literal without SSH expansions")
    socket_path, state = Path(path), Path(state_root)
    require(socket_path != state and state not in socket_path.parents
            and socket_path not in state.parents, "control socket must be separate from persistent state")
    return socket_path


def _local_filesystem_device():
    # The same filesystem as the Mac's /tmp excludes separately mounted FUSE,
    # SSHFS and network volumes without invoking a tool or making a connection.
    return Path("/tmp").stat().st_dev


def validate_control_socket(path, state_root):
    """Read only local metadata; tests inject this boundary, never real sockets.

    Approval and each execution bind the socket and its private parent by
    device/inode/uid/mode. Reject every symlink component and SSH expansion so
    the literal path in -S cannot silently name another socket.
    """
    socket_path = validate_control_path(path, state_root)
    require(not is_temporary_state_root(socket_path), "control socket must be outside temporary trees")
    try:
        metadata = {}
        for ancestor in reversed(socket_path.parents):
            value = ancestor.lstat()
            require(stat.S_ISDIR(value.st_mode), "control socket ancestor must not be a symlink")
            require(not (ancestor / ".git").exists(), "control socket must be outside any Git worktree")
            if ancestor == socket_path.parent:
                metadata["parent"] = value
        metadata["socket"] = socket_path.lstat()
    except FileNotFoundError as error:
        raise SocketUnavailable("approved control socket or parent is absent") from error
    require(stat.S_ISSOCK(metadata["socket"].st_mode), "control path must be a Unix-domain socket")
    device = _local_filesystem_device()
    for name, value in metadata.items():
        require(value.st_uid == os.getuid(), "control {0} must belong to the current user".format(name))
        require(value.st_mode & 0o077 == 0, "control {0} permissions must be private".format(name))
        require(value.st_dev == device, "control socket must be on the Mac local filesystem, never FUSE/SSHFS")
    return {name: dict(device=value.st_dev, inode=value.st_ino, uid=value.st_uid, mode=value.st_mode)
            for name, value in metadata.items()}


def validate_socket_snapshot(value):
    require(isinstance(value, dict) and set(value) == {"socket", "parent"}, "invalid socket snapshot")
    for name, metadata in value.items():
        require(isinstance(metadata, dict) and set(metadata) == {"device", "inode", "uid", "mode"}
                and all(type(item) is int and item >= 0 for item in metadata.values()),
                "invalid socket metadata")
        require(metadata["uid"] == os.getuid() and metadata["mode"] & 0o077 == 0,
                "socket snapshot ownership or permission mismatch")
        require((stat.S_ISSOCK if name == "socket" else stat.S_ISDIR)(metadata["mode"]),
                "socket snapshot type mismatch")


def control_options(approval):
    # A multiplex client does not need alias settings: -F /dev/null prevents
    # Match exec, ProxyJump and canonicalization side effects from user config.
    # ProxyCommand makes any failed multiplex fallback terminate locally.
    settings = ["ControlPath=" + approval["control_path"], "ControlMaster=no", "ControlPersist=no",
                "ProxyCommand=/usr/bin/false", "CanonicalizeHostname=no", "BatchMode=yes",
                "StrictHostKeyChecking=yes", "ConnectionAttempts=1",
                "ConnectTimeout={0}".format(approval["connect_timeout"]),
                "ClearAllForwardings=yes", "ForwardAgent=no", "ForwardX11=no",
                "PermitLocalCommand=no", "ForkAfterAuthentication=no", "StdinNull=no",
                "AddKeysToAgent=no", "UpdateHostKeys=no", "CheckHostIP=no", "IdentityAgent=none", "IdentityFile=none",
                "PubkeyAuthentication=no", "KbdInteractiveAuthentication=no", "PasswordAuthentication=no",
                "GSSAPIAuthentication=no", "HostbasedAuthentication=no", "LogLevel=DEBUG1"]
    return [item for setting in settings for item in ("-o", setting)]


def control_check_argv(approval):
    return ([str(SSH_EXECUTABLE), "-F", "/dev/null", "-S", approval["control_path"], "-O", "check"]
            + control_options(approval) + ["--", approval["alias"]])


def control_session_argv(approval, token):
    return ([str(SSH_EXECUTABLE), "-F", "/dev/null", "-T", "-S", approval["control_path"]]
            + control_options(approval)
            + ["--", approval["alias"], "bash -s -- " + shlex.quote(approval["remote_root"])
               + " " + shlex.quote(token)])


def completion_codes(stderr, token):
    return re.findall(r"^GPU_SUITE_PROBE_END " + re.escape(token) + r" ([0-9]+)$",
                      stderr.decode("utf-8", errors="replace"), re.M)


def regular(path):
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and value.st_uid == os.getuid(),
            "state must be an owned regular file: {0}".format(path))
    require(value.st_mode & 0o077 == 0, "state file permissions must be private")


def directory(path):
    value = path.lstat()
    require(stat.S_ISDIR(value.st_mode) and value.st_uid == os.getuid(),
            "state must be an owned directory: {0}".format(path))
    require(value.st_mode & 0o077 == 0, "state directory permissions must be private")


def fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def exclusive_bytes(path, data):
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)


def read_json(path):
    regular(path)
    return strict_json.load(path)


def save_json(path, data):
    """Atomic durable update; the global lock covers this entire transaction."""
    regular(path)
    fd, name = tempfile.mkstemp(prefix="ledger-update-", dir=str(path.parent))
    with os.fdopen(fd, "wb") as stream:
        stream.write(strict_json.dump_bytes(data))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(name, str(path))
    fsync_directory(path.parent)


def validate_approval(value, state_directory):
    require(isinstance(value, dict), "invalid approval")
    fields = {"alias", "remote_root", "script_sha256", "state_directory", "connect_timeout", "timeout",
              "connection_budget", "execution_budget", "approved_at"}
    transport = value.get("transport", "direct")
    require(transport in ("direct", "control-master"), "invalid approval transport")
    if transport == "control-master":
        fields.update(("transport", "control_path", "control_socket", "logical_session_budget"))
        validate_control_path(value.get("control_path"), state_directory)
        validate_socket_snapshot(value.get("control_socket"))
        require(type(value.get("logical_session_budget")) is int and value["logical_session_budget"] == 1,
                "logical session budget must equal one")
    require(set(value) == fields, "invalid approval fields")
    require(isinstance(value["alias"], str) and ID.fullmatch(value["alias"]),
            "alias must be an SSH config alias, without options or user/account syntax")
    root = value["remote_root"]
    require(isinstance(root, str) and root.startswith("/") and root != "/"
            and root == posixpath.normpath(root) and not any(ord(c) < 32 for c in root),
            "remote root must be an explicit normalized absolute path")
    require(isinstance(value["script_sha256"], str) and HASH.fullmatch(value["script_sha256"]),
            "invalid approved SHA-256")
    require(value["state_directory"] == str(state_directory), "approval state root mismatch")
    for key in ("connect_timeout", "timeout"):
        require(type(value[key]) is int and 1 <= value[key] <= 3600, "invalid timeout")
    require(value["connect_timeout"] <= value["timeout"], "connect timeout exceeds total timeout")
    require(type(value["connection_budget"]) is int
            and value["connection_budget"] == (0 if transport == "control-master" else 1)
            and type(value["execution_budget"]) is int and value["execution_budget"] == 1,
            "invalid connection or execution budget")
    require(isinstance(value["approved_at"], str) and value["approved_at"], "missing approval time")


class Store:
    """One persistent approval root, bound to a fixed per-user rendezvous.

    The coordination override is Python-only for isolated fake tests. The CLI
    never accepts another rendezvous, executable, command, or probe path.
    """

    def __init__(self, state_directory, coordination=COORDINATION):
        supplied = Path(state_directory).absolute()
        self.root = supplied.resolve()
        require(not is_temporary_state_root(self.root), STATE_LOCATION_REQUIREMENT)
        require(not supplied.is_symlink(), "state root must not be a symlink")
        self.coordination = Path(coordination).resolve()
        require(self.root != REPOSITORY and REPOSITORY not in self.root.parents,
                "state root must be outside the repository")
        require(self.root != self.coordination and self.coordination not in self.root.parents,
                "persistent state must be separate from temporary coordination")
        for ancestor in (self.root,) + tuple(self.root.parents):
            require(not (ancestor / ".git").exists(), "state root must be outside any Git worktree")

    def initialize(self):
        require(not self.coordination.exists() and not self.root.exists(),
                "initialization is exclusive; existing or lost state needs human reconciliation")
        require(self.coordination.parent.is_dir() and self.root.parent.is_dir(),
                "state and coordination parents must already exist")
        require(self.root.parent.stat().st_dev == Path("/tmp").stat().st_dev,
                STATE_LOCATION_REQUIREMENT)
        # mkdir is exclusive. A crash leaves evidence which is never repaired automatically.
        self.coordination.mkdir(mode=0o700)
        exclusive_bytes(self.coordination / "lock", b"")
        self.root.mkdir(mode=0o700)
        (self.root / "executions").mkdir(mode=0o700)
        identity = {"format_version": 1, "state_directory": str(self.root), "token": uuid.uuid4().hex}
        exclusive_bytes(self.coordination / "binding.json", strict_json.dump_bytes(identity))
        exclusive_bytes(self.root / "identity.json", strict_json.dump_bytes(identity))
        ledger = {"format_version": 1, "token": identity["token"], "approvals": {}, "executions": {}}
        exclusive_bytes(self.root / "ledger.json", strict_json.dump_bytes(ledger))
        fsync_directory(self.root.parent)
        fsync_directory(self.coordination.parent)

    @contextlib.contextmanager
    def locked(self):
        directory(self.coordination)
        regular(self.coordination / "lock")
        fd = os.open(str(self.coordination / "lock"), os.O_RDONLY | os.O_NOFOLLOW)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise SafetyError("another invocation holds the global lock; stopped") from error
            directory(self.root)
            directory(self.root / "executions")
            binding = read_json(self.coordination / "binding.json")
            identity = read_json(self.root / "identity.json")
            require(isinstance(binding, dict) and set(binding) == {"format_version", "state_directory", "token"}
                    and binding == identity and binding.get("state_directory") == str(self.root)
                    and type(binding.get("format_version")) is int and binding["format_version"] == 1
                    and isinstance(binding["token"], str) and re.fullmatch(r"[0-9a-f]{32}", binding["token"]),
                    "state root binding mismatch")
            ledger = read_json(self.root / "ledger.json")
            self.validate(ledger, binding)
            yield ledger
        finally:
            os.close(fd)

    def validate(self, ledger, binding):
        require(isinstance(ledger, dict) and set(ledger) ==
                {"format_version", "token", "approvals", "executions"}, "invalid ledger fields")
        require(type(ledger["format_version"]) is int and ledger["format_version"] == 1
                and ledger["token"] == binding["token"],
                "ledger identity mismatch")
        approvals, executions = ledger["approvals"], ledger["executions"]
        require(isinstance(approvals, dict) and isinstance(executions, dict), "invalid ledger maps")
        for key, approval in approvals.items():
            require(ID.fullmatch(key), "invalid approval ID")
            validate_approval(approval, self.root)
        require(set(p.name for p in (self.root / "executions").iterdir()) == set(executions),
                "execution log inventory mismatch")
        for key, execution in executions.items():
            require(ID.fullmatch(key) and isinstance(execution, dict), "invalid execution")
            require({"approval_id", "execution_id", "approval", "argv", "completion_token", "status",
                     "started_at", "ended_at", "reserved_connections", "ssh_calls", "ssh_exit",
                     "remote_exit", "completion_proof"}.issubset(execution), "missing execution fields")
            require(execution.get("execution_id") == key and execution.get("approval_id") in approvals,
                    "execution identity mismatch")
            approval = approvals[execution["approval_id"]]
            master = approval.get("transport") == "control-master"
            require(execution.get("approval") == approval and type(execution.get("reserved_connections")) is int
                    and execution["reserved_connections"] == (0 if master else 1),
                    "execution approval or budget mismatch")
            states = {"RESERVED", "RUNNING", "UNKNOWN", "COMPLETE", "REMOTE_FAILED"}
            states.update(("MASTER_UNAVAILABLE", "MASTER_LOST", "LOCAL_STOP") if master else
                          ("CONNECTION_STOP", "LAUNCH_FAILED"))
            require(execution.get("status") in states,
                    "invalid execution state")
            require("ssh_calls" in execution and (execution["ssh_calls"] is None or
                    (type(execution["ssh_calls"]) is int and
                     execution["ssh_calls"] in ((0, 1, 2) if master else (0, 1)))),
                    "invalid observed call count")
            expected_calls = {"RESERVED": (None,), "RUNNING": (1,), "UNKNOWN": (None, 1),
                              "COMPLETE": (1,), "REMOTE_FAILED": (1,), "CONNECTION_STOP": (1,),
                              "LAUNCH_FAILED": (0,)}
            if master:
                self.validate_master_execution(execution, approval)
            else:
                require(execution["ssh_calls"] in expected_calls[execution["status"]],
                        "call count disagrees with execution status")
            for name in ("ssh_exit", "remote_exit"):
                require(name in execution and (execution[name] is None or type(execution[name]) is int),
                        "invalid exit type")
            require(type(execution.get("completion_proof")) is bool, "missing completion proof flag")
            token = execution.get("completion_token")
            require(isinstance(token, str) and re.fullmatch(re.escape(key) + r"\.[0-9a-f]{32}", token),
                    "invalid completion token")
            require(isinstance(execution.get("started_at"), str), "missing execution start")
            directory(self.root / "executions" / key)
            for name in ("stdout", "stderr"):
                path = self.root / "executions" / key / (name + ".bin")
                regular(path)
                if execution["status"] not in ("RESERVED", "RUNNING"):
                    require(execution.get(name + "_sha256") == hashlib.sha256(path.read_bytes()).hexdigest(),
                            "recorded output is missing or changed")
            if execution["status"] not in ("RESERVED", "RUNNING"):
                require(isinstance(execution.get("ended_at"), str), "missing execution end")
                require(isinstance(execution.get("reason"), str), "missing outcome reason")
            if execution["status"] not in ("COMPLETE", "REMOTE_FAILED"):
                require(execution["remote_exit"] is None and execution["completion_proof"] is False,
                        "unconfirmed remote exit must be null")
            if execution["status"] in ("COMPLETE", "REMOTE_FAILED"):
                rc = execution.get("remote_exit")
                require(type(rc) is int and 0 <= rc < 255 and rc == execution.get("ssh_exit")
                        and execution.get("completion_proof") is True,
                        "invalid remote completion record")
                require((rc == 0) == (execution["status"] == "COMPLETE"), "completion status mismatch")
                stderr = (self.root / "executions" / key / "stderr.bin").read_bytes()
                require(completion_codes(stderr, token) == [str(rc)] and not FATAL.search(
                    stderr.decode("utf-8", errors="replace")), "completion evidence mismatch")
                if master:
                    require(len(re.findall(rb"^GPU_SUITE_PROBE_END\b", stderr, re.M)) == 1,
                            "multiplex completion has additional or mismatched markers")
        for approval_id in approvals:
            require(sum(e["reserved_connections"] for e in executions.values()
                        if e["approval_id"] == approval_id) <= 1, "approval budget exceeded")
            require(sum(e.get("reserved_logical_sessions", 0) for e in executions.values()
                        if e["approval_id"] == approval_id) <= 1, "logical session budget exceeded")

    def validate_master_execution(self, execution, approval):
        require(execution.get("transport") == "control-master"
                and type(execution.get("reserved_logical_sessions")) is int
                and execution["reserved_logical_sessions"] == 1, "invalid logical session reservation")
        require(execution.get("argv") == control_session_argv(approval, execution["completion_token"])
                and execution.get("check_argv") == control_check_argv(approval),
                "recorded master argv differs from guarded approved transport")
        for field in ("underlying_connections_created", "new_tcp_attempts", "new_ssh_auth_attempts"):
            require(type(execution.get(field)) is int and execution[field] == 0,
                    "master transport cannot create a network connection or authenticate")
        for field in ("local_check_calls", "logical_session_calls", "logical_sessions_confirmed",
                      "remote_commands_confirmed"):
            require(field in execution and (execution[field] is None or
                    (type(execution[field]) is int and execution[field] in (0, 1))),
                    "invalid master observation count")
        require(type(execution.get("remote_command_attempts")) is int
                and execution["remote_command_attempts"] in (0, 1), "invalid remote command attempt count")
        observed = (execution["local_check_calls"], execution["logical_session_calls"])
        require(execution["ssh_calls"] == (None if None in observed else sum(observed)),
                "total SSH process count disagrees with master observations")
        require(execution.get("check_exit") is None or type(execution["check_exit"]) is int,
                "invalid alive check exit")
        require(type(execution.get("master_alive")) is bool, "missing alive-check evidence")
        require(not execution["master_alive"] or execution["local_check_calls"] == 1,
                "alive-check proof lacks an observed local check")
        require(not execution["remote_command_attempts"] or execution["master_alive"],
                "session attempted without successful master check")
        require(execution["logical_session_calls"] == 0 or execution["remote_command_attempts"] == 1,
                "session process lacks durable launch intent")
        confirmed = execution["status"] in ("COMPLETE", "REMOTE_FAILED")
        if confirmed:
            require(execution["local_check_calls"] == execution["logical_session_calls"] == 1
                    and execution["remote_commands_confirmed"] == execution["logical_sessions_confirmed"] == 1,
                    "completion counters lack observed session")
        else:
            expected = 0 if execution["logical_session_calls"] == 0 else None
            require(execution["remote_commands_confirmed"] == expected
                    and execution["logical_sessions_confirmed"] == expected, "unproven remote command count")
        if execution["status"] in ("MASTER_UNAVAILABLE", "LOCAL_STOP"):
            require(execution["logical_session_calls"] == 0, "local stop cannot conceal a possible session")
        if execution["status"] == "MASTER_UNAVAILABLE":
            require(not execution["master_alive"], "unavailable master cannot have successful alive check")
        if execution["status"] == "MASTER_LOST":
            require(execution["master_alive"] and execution["local_check_calls"] == 1,
                    "master loss requires prior successful alive check")
        if execution["local_check_calls"] is None:
            require(execution["status"] in ("RESERVED", "UNKNOWN"), "unresolved check launch status mismatch")
        if execution["logical_session_calls"] is None:
            require(execution["status"] in ("RUNNING", "UNKNOWN"), "unresolved session launch status mismatch")
        logs = self.root / "executions" / execution["execution_id"]
        for name in ("check_stdout", "check_stderr"):
            regular(logs / (name + ".bin"))
            if execution["status"] not in ("RESERVED", "RUNNING"):
                require(execution.get(name + "_sha256") == hashlib.sha256(
                    (logs / (name + ".bin")).read_bytes()).hexdigest(), "alive-check evidence changed")
        if execution["master_alive"]:
            check_output = (logs / "check_stdout.bin").read_bytes() + b"\n" + (logs / "check_stderr.bin").read_bytes()
            require(execution["check_exit"] == 0 and len(MASTER_RUNNING.findall(check_output)) == 1,
                    "missing successful alive-check response")

    def save(self, ledger):
        save_json(self.root / "ledger.json", ledger)

    def approve(self, approval_id, alias, remote_root, script_sha256, connect_timeout=20, timeout=120,
                transport="direct", control_path=None):
        require(ID.fullmatch(approval_id), "invalid approval ID")
        require(transport in ("direct", "control-master"), "invalid transport")
        require(transport == "control-master" or control_path is None, "direct transport has no control path")
        approval = dict(alias=alias, remote_root=remote_root, script_sha256=script_sha256,
                        state_directory=str(self.root), connect_timeout=connect_timeout, timeout=timeout,
                        connection_budget=1, execution_budget=1, approved_at=now())
        if transport == "control-master":
            snapshot = validate_control_socket(control_path, self.root)
            approval.update(transport=transport, control_path=control_path, control_socket=snapshot,
                            connection_budget=0, logical_session_budget=1)
        validate_approval(approval, self.root)
        with self.locked() as ledger:
            self.recover(ledger)
            require(approval_id not in ledger["approvals"], "approval already exists; never overwrite")
            require(not any(e["status"] in UNRESOLVED
                            for e in ledger["executions"].values()), "unresolved execution blocks approval")
            ledger["approvals"][approval_id] = approval
            self.save(ledger)

    def recover(self, ledger):
        changed = False
        for execution in ledger["executions"].values():
            if execution["status"] in ("RESERVED", "RUNNING"):
                execution.update(status="UNKNOWN", ended_at=now(), ssh_exit=None, remote_exit=None,
                                 completion_proof=False, reason="interrupted invocation; remote outcome unknown")
                # Between spawn and its durable observation, zero is not proof of no launch.
                if execution["ssh_calls"] != 1:
                    if execution.get("transport") != "control-master":
                        execution["ssh_calls"] = None
                self.capture(execution)
                changed = True
        if changed:
            self.save(ledger)

    def capture(self, execution):
        names = ("stdout", "stderr", "check_stdout", "check_stderr") if execution.get("transport") == "control-master" else ("stdout", "stderr")
        for name in names:
            path = self.root / "executions" / execution["execution_id"] / (name + ".bin")
            with path.open("rb") as stream:
                data = stream.read()
                os.fsync(stream.fileno())
            execution[name + "_sha256"] = hashlib.sha256(data).hexdigest()

    def save_stderr(self, logs, data):
        """Persist DEBUG1 evidence without public/host key fingerprints.

        Raw stderr stays in memory. Authentication stages and the completion
        marker are retained; the recorded hash covers these sanitized bytes.
        Existing execution logs are never passed through this function.
        """
        require(isinstance(data, bytes), "SSH stderr was not captured as bytes")
        with (logs / "stderr.bin").open("wb") as stream:
            stream.write(FINGERPRINT.sub(b"[REDACTED_FINGERPRINT]", data))
            stream.flush()
            os.fsync(stream.fileno())

    def _master_output(self, logs, name, data):
        require(isinstance(data, bytes), "master output was not captured as bytes")
        if "stderr" in name:
            data = FINGERPRINT.sub(b"[REDACTED_FINGERPRINT]", data)
        with (logs / (name + ".bin")).open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    def _probe_master(self, ledger, approval_id, execution_id, token, script, popen):
        """One check and at most one multiplex client, while holding the lock.

        Counts refer to observed local processes, not network connections. A
        durable launch intent plus null observation survives a process crash
        without inventing either a launch or proof that nothing ran remotely.
        """
        approval = ledger["approvals"][approval_id]
        logs = self.root / "executions" / execution_id
        logs.mkdir(mode=0o700)
        for name in ("stdout", "stderr", "check_stdout", "check_stderr"):
            exclusive_bytes(logs / (name + ".bin"), b"")
        execution = dict(approval_id=approval_id, execution_id=execution_id, approval=approval,
                         transport="control-master", argv=control_session_argv(approval, token),
                         check_argv=control_check_argv(approval), completion_token=token,
                         status="RESERVED", started_at=now(), ended_at=None,
                         reserved_connections=0, reserved_logical_sessions=1,
                         ssh_calls=0, local_check_calls=0, logical_session_calls=0,
                         underlying_connections_created=0, new_tcp_attempts=0, new_ssh_auth_attempts=0,
                         remote_command_attempts=0, remote_commands_confirmed=0,
                         logical_sessions_confirmed=0, check_exit=None, master_alive=False,
                         ssh_exit=None, remote_exit=None, completion_proof=False,
                         phase="socket-validation")
        ledger["executions"][execution_id] = execution
        self.save(ledger)  # Reserve the one logical budget before either local client.
        deadline = time.monotonic() + approval["timeout"]
        environment = dict(os.environ, LC_ALL="C", LANG="C")
        environment.pop("SSH_ASKPASS", None)
        environment.pop("SSH_ASKPASS_REQUIRE", None)
        child = None

        def socket_unchanged():
            snapshot = validate_control_socket(approval["control_path"], self.root)
            require(snapshot == approval["control_socket"], "approved control socket attributes changed")
            return snapshot

        def persist_count():
            observed = (execution["local_check_calls"], execution["logical_session_calls"])
            execution["ssh_calls"] = None if None in observed else sum(observed)
            self.save(ledger)

        def finish(status, reason):
            execution.update(status=status, reason=reason, ended_at=now())
            try:
                self.capture(execution)
                self.save(ledger)
            except Exception as error:
                # The write may have succeeded before fsync failed. Never
                # attempt another final save or overwrite a durable completion.
                uncertain = execution["logical_session_calls"] != 0
                result = dict(execution)
                result.update(status="UNKNOWN" if uncertain else "LOCAL_STOP",
                              reason="outcome recording failed: " + type(error).__name__,
                              remote_exit=None, completion_proof=False,
                              remote_commands_confirmed=None if uncertain else 0,
                              logical_sessions_confirmed=None if uncertain else 0)
                raise OutcomeRecordingError(result) from error
            return execution

        try:
            socket_unchanged()
        except SocketUnavailable:
            return finish("MASTER_UNAVAILABLE", "approved master socket is absent; no session attempted")
        except (SafetyError, OSError) as error:
            return finish("LOCAL_STOP", "socket validation stopped: " + str(error))

        try:
            execution.update(phase="check-launch", local_check_calls=None)
            persist_count()
            child = popen(execution["check_argv"], stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
            execution.update(status="RUNNING", phase="check", local_check_calls=1)
            persist_count()
            try:
                stdout_bytes, stderr_bytes = child.communicate(
                    timeout=max(0.001, min(approval["connect_timeout"], deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                child.kill()  # This local check client only; never signal the master.
                stdout_bytes, stderr_bytes = child.communicate()
                execution["check_exit"] = child.returncode
                self._master_output(logs, "check_stdout", stdout_bytes)
                self._master_output(logs, "check_stderr", stderr_bytes)
                return finish("MASTER_UNAVAILABLE", "alive check timed out; no session attempted")
            execution["check_exit"] = child.returncode
            self._master_output(logs, "check_stdout", stdout_bytes)
            self._master_output(logs, "check_stderr", stderr_bytes)
            if child.returncode != 0 or len(MASTER_RUNNING.findall(stdout_bytes + b"\n" + stderr_bytes)) != 1:
                return finish("MASTER_UNAVAILABLE", "alive check did not prove a running master")
            execution["master_alive"] = True
            self.save(ledger)
            socket_unchanged()  # Recheck inode, ownership and permissions after check.
            socket_unchanged()  # Check again at the session launch boundary.
        except SocketUnavailable:
            return finish("MASTER_LOST", "master socket disappeared after alive check; no session attempted")
        except OutcomeRecordingError:
            raise
        except Exception as error:
            if child is None:
                execution["local_check_calls"] = 0
                execution["ssh_calls"] = 0
            else:
                try:
                    if child.poll() is None:
                        child.kill()
                    stdout_bytes, stderr_bytes = child.communicate()
                    self._master_output(logs, "check_stdout", stdout_bytes)
                    self._master_output(logs, "check_stderr", stderr_bytes)
                    execution["check_exit"] = child.returncode
                except Exception:
                    pass  # Missing evidence remains missing, never a synthetic success.
            return finish("LOCAL_STOP", "local alive-check failure: " + type(error).__name__)

        child = None
        try:
            if time.monotonic() >= deadline:
                return finish("LOCAL_STOP", "overall timeout elapsed before session launch")
            execution.update(phase="session-launch", logical_session_calls=None,
                             remote_command_attempts=1, remote_commands_confirmed=None,
                             logical_sessions_confirmed=None)
            persist_count()
            with (logs / "stdout.bin").open("wb") as stdout:
                child = popen(execution["argv"], stdin=subprocess.PIPE, stdout=stdout,
                              stderr=subprocess.PIPE, env=environment)
                execution.update(phase="session", logical_session_calls=1)
                persist_count()
                try:
                    _, stderr_bytes = child.communicate(input=script,
                                                       timeout=max(0.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    child.kill()  # Only our multiplex client, never the authenticated master.
                    _, stderr_bytes = child.communicate()
                    self.save_stderr(logs, stderr_bytes)
                    execution["ssh_exit"] = child.returncode
                    return finish("UNKNOWN", "session timeout; remote outcome unknown")
                execution["ssh_exit"] = child.returncode
                self.save_stderr(logs, stderr_bytes)
            stderr_bytes = (logs / "stderr.bin").read_bytes()
            text = stderr_bytes.decode("utf-8", errors="replace")
            proofs = completion_codes(stderr_bytes, token)
            marker_count = len(re.findall(rb"^GPU_SUITE_PROBE_END\b", stderr_bytes, re.M))
            rc = child.returncode
            if (len(proofs) == marker_count == 1 and rc is not None and 0 <= rc < 255
                    and int(proofs[0]) == rc and not FATAL.search(text)):
                execution.update(remote_exit=rc, completion_proof=True,
                                 remote_commands_confirmed=1, logical_sessions_confirmed=1)
                return finish("COMPLETE" if rc == 0 else "REMOTE_FAILED", "authenticated multiplex completion marker")
            if MASTER_LOSS.search(text):
                return finish("MASTER_LOST", "master lost after check; remote outcome unconfirmed")
            try:
                socket_unchanged()
            except SocketUnavailable:
                return finish("MASTER_LOST", "master socket disappeared during session; remote outcome unknown")
            except (SafetyError, OSError):
                return finish("UNKNOWN", "socket changed during session; remote outcome unknown")
            return finish("UNKNOWN", "missing or inconsistent completion; remote outcome unknown")
        except OutcomeRecordingError:
            raise
        except Exception as error:
            if child is None:
                execution.update(logical_session_calls=0, ssh_calls=1, remote_commands_confirmed=0,
                                 logical_sessions_confirmed=0)
                return finish("LOCAL_STOP", "multiplex client launch failed: " + type(error).__name__)
            try:
                if child.poll() is None:
                    child.kill()
                _, stderr_bytes = child.communicate()
                self.save_stderr(logs, stderr_bytes)
            except Exception:
                pass
            execution["ssh_exit"] = child.returncode
            return finish("UNKNOWN", "local failure after session launch: " + type(error).__name__)

    def probe(self, approval_id, execution_id, dry_run=False, popen=None):
        popen = subprocess.Popen if popen is None else popen
        require(ID.fullmatch(execution_id), "invalid execution ID")
        script = SCRIPT.read_bytes()  # Exactly one read; send these exact approved bytes.
        digest = hashlib.sha256(script).hexdigest()
        with self.locked() as ledger:
            if not dry_run:
                self.recover(ledger)
            require(not any(e["status"] in UNRESOLVED
                            for e in ledger["executions"].values()), "unresolved execution; no automatic replay")
            require(approval_id in ledger["approvals"], "unknown approval ID")
            approval = ledger["approvals"][approval_id]
            require(digest == approval["script_sha256"], "probe SHA-256 differs from approval; stopped")
            require(execution_id not in ledger["executions"], "execution ID already used; no replay")
            require(not any(e["approval_id"] == approval_id for e in ledger["executions"].values()),
                    "approval connection budget exhausted")
            token = execution_id + "." + uuid.uuid4().hex
            command = [str(SSH_EXECUTABLE), "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                       "-o", "ConnectionAttempts=1", "-o", "ConnectTimeout={0}".format(approval["connect_timeout"]),
                       "-o", "ControlMaster=no", "-o", "ControlPath=none", "-o", "ControlPersist=no",
                       "-o", "ClearAllForwardings=yes", "-o", "ForwardAgent=no", "-o", "ForwardX11=no",
                       "-o", "PermitLocalCommand=no", "-o", "ForkAfterAuthentication=no", "-o", "StdinNull=no",
                       "-o", "AddKeysToAgent=no", "-o", "UpdateHostKeys=no", "-o", "CheckHostIP=no",
                       "-o", "PreferredAuthentications=publickey", "-o", "PubkeyAuthentication=yes",
                       "-o", "KbdInteractiveAuthentication=no", "-o", "PasswordAuthentication=no",
                       "-o", "GSSAPIAuthentication=no", "-o", "HostbasedAuthentication=no",
                       "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=SSH_AUTH_SOCK",
                       "-o", "LogLevel=DEBUG1",
                       "--", approval["alias"], "bash -s -- " + shlex.quote(approval["remote_root"])
                       + " " + shlex.quote(token)]
            plan = {"approval_id": approval_id, "execution_id": execution_id, "argv": command,
                    "script_sha256": digest, "timeout": approval["timeout"], "dry_run": dry_run}
            master = approval.get("transport") == "control-master"
            if master:
                plan.update(transport="control-master", argv=control_session_argv(approval, token),
                            check_argv=control_check_argv(approval), logical_session_budget=1)
            if dry_run:
                return plan
            validate_ssh_executable()  # No reservation or execution log until this succeeds.
            if master:
                return self._probe_master(ledger, approval_id, execution_id, token, script, popen)
            logs = self.root / "executions" / execution_id
            logs.mkdir(mode=0o700)
            exclusive_bytes(logs / "stdout.bin", b"")
            exclusive_bytes(logs / "stderr.bin", b"")
            execution = dict(approval_id=approval_id, execution_id=execution_id, approval=approval,
                             argv=command, completion_token=token, status="RESERVED", started_at=now(),
                             ended_at=None, reserved_connections=1, ssh_calls=None,
                             ssh_exit=None, remote_exit=None, completion_proof=False)
            ledger["executions"][execution_id] = execution
            self.save(ledger)  # Durable reservation BEFORE launching even one process.
            child = None
            try:
                with (logs / "stdout.bin").open("wb") as stdout:
                    child = popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=subprocess.PIPE)
                    execution.update(status="RUNNING", ssh_calls=1)
                    self.save(ledger)
                    try:
                        _, stderr_bytes = child.communicate(input=script, timeout=approval["timeout"])
                    except subprocess.TimeoutExpired:
                        child.kill()  # Only our own child, never a master or other SSH process.
                        _, stderr_bytes = child.communicate()
                        execution.update(status="UNKNOWN", reason="timeout; remote outcome unknown",
                                         ssh_exit=child.returncode)
                    else:
                        execution["ssh_exit"] = child.returncode
                    self.save_stderr(logs, stderr_bytes)
                if execution["status"] == "RUNNING":
                    stderr_bytes = (logs / "stderr.bin").read_bytes()
                    text = stderr_bytes.decode("utf-8", errors="replace")
                    proofs = completion_codes(stderr_bytes, token)
                    rc = child.returncode
                    if FATAL.search(text):
                        execution.update(status="CONNECTION_STOP", reason="authentication/host key/connection limit")
                    elif len(proofs) == 1 and rc is not None and 0 <= rc < 255 and int(proofs[0]) == rc:
                        execution.update(status="COMPLETE" if rc == 0 else "REMOTE_FAILED",
                                         remote_exit=rc, completion_proof=True, reason="authenticated completion marker")
                    else:
                        execution.update(status="UNKNOWN", reason="missing or inconsistent completion; remote outcome unknown")
            except Exception as error:
                if child is not None:
                    execution.update(status="UNKNOWN", reason="local error after launch: " + type(error).__name__,
                                     ssh_exit=child.returncode)
                    try:
                        if child.poll() is None:
                            child.kill()
                        _, stderr_bytes = child.communicate()
                        self.save_stderr(logs, stderr_bytes)
                    except Exception as capture_error:
                        execution["reason"] += "; stderr unavailable: " + type(capture_error).__name__
                    execution["ssh_exit"] = child.returncode
                else:
                    execution.update(status="LAUNCH_FAILED", ssh_calls=0,
                                     reason="SSH process creation failed: " + type(error).__name__)
            execution["ended_at"] = now()
            self.capture(execution)
            self.save(ledger)
            return execution


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-directory", required=True, type=Path)
    sub = parser.add_subparsers(dest="operation", required=True)
    sub.add_parser("initialize", help="exclusive local setup; requires separately approved local path")
    approve = sub.add_parser("approve", help="register a human-approved probe; never replace an approval")
    approve.add_argument("--approval-id", required=True)
    approve.add_argument("--alias", required=True)
    approve.add_argument("--remote-root", required=True)
    approve.add_argument("--script-sha256", required=True)
    approve.add_argument("--connect-timeout", type=int, default=20)
    approve.add_argument("--timeout", type=int, default=120)
    approve.add_argument("--transport", choices=("direct", "control-master"), default="direct")
    approve.add_argument("--control-path", help="owner-approved absolute private existing master socket")
    probe = sub.add_parser("probe")
    probe.add_argument("--approval-id", required=True)
    probe.add_argument("--execution-id", required=True)
    probe.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        store = Store(args.state_directory)
        if args.operation == "initialize":
            store.initialize()
        elif args.operation == "approve":
            store.approve(args.approval_id, args.alias, args.remote_root, args.script_sha256,
                          args.connect_timeout, args.timeout, args.transport, args.control_path)
        else:
            result = store.probe(args.approval_id, args.execution_id, args.dry_run)
            print(strict_json.dumps(result))
            return 0 if args.dry_run or result["status"] == "COMPLETE" else 1
        return 0
    except OutcomeRecordingError as error:
        print(strict_json.dumps(error.result))
        print("Sirius probe stopped: {0}".format(error), file=sys.stderr)
        return 1
    except (SafetyError, OSError, ValueError, KeyError, TypeError) as error:
        if args.operation == "probe":
            # Do not repair or mutate a ledger which failed validation. The CLI
            # still gives local policy/ledger/hash failures an explicit status.
            print(strict_json.dumps({"status": "LOCAL_STOP", "reason": str(error)}))
        print("Sirius probe stopped: {0}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

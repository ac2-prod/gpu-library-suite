"""Read-only proof for reusing an approved AMD build in a new smoke campaign.

This module never imports code from the supplied snapshot, runs a process, or
writes to a build.  The caller must separately validate the live dependencies
and loaded libraries before execution.  Artifact source and launcher identity
are deliberately separate: a new launcher cannot relabel an old binary.
"""

import hashlib
import functools
import json
import math
import os
from os import walk as directory_walk
from pathlib import Path, PurePosixPath
import re
import shlex
import stat


FAMILIES = ("rocblas", "rocfft", "rocsparse", "rocsolver", "rocrand", "rocthrust")
PROFILES = ("cpu", "hip", "openmp")
ROLES = ("example", "benchmark")
SOURCE_FILES = tuple("amd-build/" + name + ".json" for name in (
    "source-before", "source-before-cpu", "source-before-hip",
    "source-before-openmp", "source-after"))
EVIDENCE_FILES = (
    "dispatch-result.json", "manifest.json", "prepared-files-sha256.json",
    "source-preflight.json", "submitted-job.json", "submission-exit.json",
) + SOURCE_FILES
PROFILE_FILES = (
    "partial-manifest-input.json", "built-manifest-input.json",
    "partial-manifest.json", "build-metadata.json", "CMakeCache.txt",
    "compile_commands.json",
)
IDENTITY_FIELDS = ("library", "implementation", "executable_role", "target_name",
                   "build_profile", "backend_variant", "binary_sha256")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
             "Invalid approved SHA-256")
    return value


def _relative(value):
    _require(isinstance(value, str) and value and "\0" not in value,
             "Invalid relative evidence path")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and ".." not in path.parts
             and path.as_posix() == value and value != ".",
             "Noncanonical or escaping evidence path")
    value.encode("utf-8", errors="strict")
    return value


def _absolute(value):
    _require(isinstance(value, str) and value.startswith("/") and "\0" not in value,
             "Expected an absolute recorded path")
    path = PurePosixPath(value)
    _require(path.as_posix() == value and ".." not in path.parts
             and not value.startswith("//"), "Noncanonical recorded path")
    return path


def _plain_path(path, directory=False):
    path = Path(path)
    _require(path.is_absolute() and path == Path(os.path.abspath(str(path))),
             "An absolute canonical local path is required")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        _require(not current.is_symlink(), "Symbolic link in evidence path")
    mode = path.stat().st_mode
    _require(stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode),
             "Evidence is not a regular directory/file")
    return path


def _read(path):
    path = _plain_path(path)
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    _require(all(getattr(before, key) == getattr(after, key) for key in fields)
             and len(data) == after.st_size, "Evidence changed while reading")
    return data, bool(before.st_mode & 0o111)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Non-standard JSON constant: " + value)
    def finite(value):
        result = float(value)
        _require(math.isfinite(result), "Nonfinite JSON number")
        return result
    return json.loads(data.decode("utf-8", errors="strict"), object_pairs_hook=pairs,
                      parse_constant=invalid, parse_float=finite)


def snapshot_identity(root, paths):
    """Check the complete saved tree and canonical v1 path/bytes/mode hash."""
    root = _plain_path(root, directory=True)
    _require(isinstance(paths, list) and paths and len(paths) == len(set(paths)),
             "Missing or duplicate source inventory")
    expected = sorted(_relative(value) for value in paths)
    actual = []
    for directory, directories, files in directory_walk(str(root), followlinks=False):
        for name in directories:
            _plain_path(Path(directory) / name, directory=True)
        for name in files:
            actual.append((Path(directory) / name).relative_to(root).as_posix())
    _require(sorted(actual) == expected, "Saved source has extra or missing files")
    digest = hashlib.sha256(b"gpu-library-suite-source-snapshot-v1\0")
    digest.update(len(expected).to_bytes(8, "big"))
    inventory = []
    for relative in expected:
        data, executable = _read(root / relative)
        path_bytes = relative.encode("utf-8")
        content_hash = _digest(data)
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(bytes((int(executable),)))
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(bytes.fromhex(content_hash))
        inventory.append({"path": relative, "sha256": content_hash,
                          "executable": executable, "size": len(data)})
    return {"sha256": digest.hexdigest(), "files": inventory}


def _target(library, profile, role):
    stem = {"rocblas": "blas", "rocfft": "fft", "rocsparse": "sparse",
            "rocsolver": "solver", "rocrand": "rand", "rocthrust": "reduce"}[library]
    name = ("openmp_" + library if profile == "openmp"
            else stem + ("_cpu" if profile == "cpu" else "_gpu"))
    return name + ("_bench" if role == "benchmark" else "")


def _hip_link_binding(link_data, stdout_data, entry, compiler, cmake):
    """Bind a generated link command to its already-approved successful log."""
    lines = [line.strip() for line in link_data.decode("utf-8", errors="strict").splitlines()
             if line.strip()]
    _require(len(lines) == 1, "Native HIP link evidence must contain exactly one command")
    line = lines[0]
    argv = shlex.split(line, posix=True)
    binary = _absolute(entry["executable_path"])
    compiler_path = str(_absolute(compiler["compiler_path"]))
    _require(argv and argv[0] == compiler_path and "--hip-link" in argv
             and argv.count("-o") == 1 and argv.index("-o") + 1 < len(argv),
             "Native HIP link compiler or output differs")
    output = PurePosixPath(argv[argv.index("-o") + 1])
    _require((binary.parent / output) == binary and ".." not in output.parts,
             "Native HIP link output differs from the approved target")
    stdout_lines = [value.strip() for value in stdout_data.decode("utf-8", errors="strict").splitlines()]
    positions = [index for index, value in enumerate(stdout_lines) if value == line]
    _require(len(positions) == 1 and positions[0] > 0,
             "Native HIP link command is absent or ambiguous in successful target stdout")
    position = positions[0]
    invocation = ["cd", str(binary.parent), "&&", cmake, "-E", "cmake_link_script",
                  "CMakeFiles/" + entry["target_name"] + ".dir/link.txt", "--verbose=1"]
    _require(shlex.split(stdout_lines[position - 1], posix=True) == invocation,
             "Native HIP link command directory or target binding differs")
    completion = re.compile(r"\[\s*100%\] Built target " + re.escape(entry["target_name"]))
    completed = [index for index, value in enumerate(stdout_lines) if completion.fullmatch(value)]
    _require(len(completed) == 1 and completed[0] > position,
             "Native HIP link command lacks one subsequent successful target completion")


def _reject_malformed(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (KeyError, TypeError, IndexError, AttributeError) as error:
            raise ValueError("Malformed retained build evidence: " + str(error)) from error
    return wrapped


@_reject_malformed
def verify_prebuilt_source(snapshot_root, build_run_root, *, expected_source_sha256,
                           expected_summary_sha256, expected_evidence_sha256,
                           expected_build_run_id, expected_recorded_source_root,
                           harness_path, expected_harness_sha256,
                           _allow_path_mapping=False):
    """Return immutable-proof content; raise ValueError/OSError on any mismatch.

    ``build_run_root`` is the old top-level runs/<RUN_ID>, not amd-build/.
    Production requires real paths to equal recorded absolute paths.  The
    private mapping switch supports read-only Mac audits and synthetic fixtures;
    it is never exposed as a production CLI escape hatch.
    """
    source_hash = _sha(expected_source_sha256)
    _sha(expected_summary_sha256)
    _sha(expected_harness_sha256)
    _require(isinstance(expected_build_run_id, str)
             and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", expected_build_run_id)
             and ".." not in expected_build_run_id, "Invalid approved build run ID")
    root = _plain_path(build_run_root, directory=True)
    _require(type(_allow_path_mapping) is bool, "Invalid internal path-mapping setting")
    source = _absolute(expected_recorded_source_root)
    recorded_root = source / "runs" / expected_build_run_id
    _require(root.name == expected_build_run_id, "Build run directory identity mismatch")
    _require(_allow_path_mapping or str(root) == str(recorded_root),
             "Production build root differs from its recorded path")
    _require(set(expected_evidence_sha256) == set(EVIDENCE_FILES),
             "Approved evidence hash inventory must be complete and exact")
    for value in expected_evidence_sha256.values():
        _sha(value)
    checked = {}
    def read(relative, expected=None):
        relative = _relative(relative)
        data, _ = _read(root / relative)
        digest = _digest(data)
        if expected is not None:
            _require(digest == _sha(expected), "Retained evidence hash mismatch: " + relative)
        _require(relative not in checked or checked[relative] == digest,
                 "Retained evidence changed during inspection: " + relative)
        checked[relative] = digest
        return data
    def document(relative, expected=None):
        value = _json(read(relative, expected))
        _require(isinstance(value, dict), "Expected JSON evidence object")
        return value
    def relative_recorded(path):
        absolute = _absolute(path)
        try:
            return _relative(absolute.relative_to(recorded_root).as_posix())
        except ValueError as error:
            raise ValueError("Recorded artifact/log escapes approved build run") from error
    def identity(value, relative):
        _require(isinstance(value, dict)
                 and value.get("path") == str(recorded_root / relative)
                 and value.get("resolved_path") == value.get("path"),
                 "Recorded file path or resolved identity mismatch")
        return read(relative, value["sha256"])
    def command_logs(command):
        _require(isinstance(command, dict), "Invalid command evidence")
        for stream in ("stdout", "stderr"):
            read(relative_recorded(command[stream]), command[stream + "_sha256"])

    evidence = {name: document(name, expected_evidence_sha256[name]) for name in EVIDENCE_FILES}
    summary = document("amd-build/summary.json", expected_summary_sha256)
    _require(summary.get("run_id") == expected_build_run_id and summary.get("status") == "BUILT"
             and summary.get("counts") == {"BUILT": 36}, "Build is not a complete 36-artifact success")
    for key in ("artifact_execution_count", "gpu_query_count", "scheduler_call_count", "ssh_call_count"):
        _require(type(summary.get(key)) is int and summary[key] == 0,
                 "Build-only execution counters are invalid: " + key)
    first = evidence[SOURCE_FILES[0]]
    head = first.get("git_head")
    _require(isinstance(head, str) and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", head), "Invalid source Git HEAD")
    _require(first.get("source_snapshot_sha256") == source_hash, "Build source differs from approval")
    for name in SOURCE_FILES:
        item = evidence[name]
        _require(all(item.get(key) == first.get(key) for key in
                     ("git_head", "paths", "source_snapshot_sha256")),
                 "Build source changed before/during/after a profile")
        _require(item.get("exclusions") == ["runs/", ".fuse_hidden* in any path component"],
                 "Unexpected historical source exclusions")
    snapshot = snapshot_identity(snapshot_root, first["paths"])
    _require(snapshot["sha256"] == source_hash, "Saved source bytes or executable bits differ from build")
    preflight = evidence["source-preflight.json"]
    _require(preflight.get("expected_source_snapshot_sha256") == source_hash
             and preflight.get("source_snapshot_sha256") == source_hash
             and preflight.get("git_head") == head
             and preflight.get("paths") == first["paths"], "Build dispatch source preflight differs")

    packet = evidence["manifest.json"]
    _require(packet.get("run_id") == expected_build_run_id
             and packet.get("remote_source_root") == str(source)
             and packet.get("remote_run_root") == str(recorded_root)
             and packet.get("source_snapshot_sha256") == source_hash
             and packet.get("requested_artifact_slots") == 36
             and set(packet.get("libraries", [])) == set(FAMILIES)
             and packet.get("profiles") == list(PROFILES), "Prepared build packet identity differs")
    prepared = evidence["prepared-files-sha256.json"]
    _require(prepared.get("manifest.json") == expected_evidence_sha256["manifest.json"],
             "Prepared manifest hash differs")
    _require(set(prepared) == set(packet.get("helper_sha256", {})) | {"manifest.json"},
             "Prepared helper inventory differs")
    for name, digest in prepared.items():
        read(name, digest)
        if name != "manifest.json":
            _require(packet["helper_sha256"][name] == digest, "Helper hash binding differs")
    source_files = {item["path"]: item for item in snapshot["files"]}
    for name in ("amd_build.py", "amd_build_job.sh", "rocfft_compute_modules.sh"):
        _require(prepared.get(name) == source_files.get("jobs/sirius/" + name, {}).get("sha256")
                 and name in prepared, "Retained build helper differs from its source snapshot")

    dispatch = evidence["dispatch-result.json"]
    job_id = summary.get("job_id")
    _require(isinstance(job_id, str) and re.fullmatch(r"[0-9]+\.[A-Za-z0-9_.-]+", job_id),
             "Invalid retained PBS job ID")
    _require(all(dispatch.get(k) == v for k, v in {
        "status": "BUILD_COMPLETE", "build_status": "BUILT", "pbs_job_state": "F",
        "pbs_exit_status": 0, "qsub_exit": 0, "job_submissions": 1,
        "artifact_counts": {"BUILT": 36}, "requested_artifact_slots": 36,
        "run_id": expected_build_run_id, "job_id": job_id, "purpose": "build-only",
        "run_root": str(recorded_root), "source_root": str(source),
    }.items()), "Build dispatch completion proof is not successful")
    proof = dispatch.get("proof_sha256", {})
    _require(set(proof) == {"amd-build-job-exit.txt", "amd-build/summary.json", "job.stdout", "job.stderr"},
             "Incomplete dispatch proof inventory")
    _require(proof["amd-build/summary.json"] == expected_summary_sha256, "Dispatch summary binding differs")
    proof_bytes = {name: read(name, digest) for name, digest in proof.items()}
    _require(proof_bytes["amd-build-job-exit.txt"] in (b"0", b"0\n"), "Build body exit is not zero")
    marker = "GPU_SUITE_SIRIUS_AMD_BUILD_END " + expected_build_run_id + " 0"
    stdout = proof_bytes["job.stdout"].decode("utf-8", errors="strict")
    markers = [line for line in stdout.splitlines() if line.startswith("GPU_SUITE_SIRIUS_AMD_BUILD_END")]
    _require(markers == [marker] and dispatch.get("completion_marker") == marker,
             "Missing, duplicate or mismatched build completion marker")
    terminal = dispatch.get("terminal_observation", {})
    _require(terminal.get("argv") == ["/opt/pbs/bin/qstat", "-xf", job_id]
             and terminal.get("exit") == 0 and terminal.get("error") is None,
             "Terminal PBS observation is not successful")
    command_logs(terminal)
    pbs = read(relative_recorded(terminal["stdout"])).decode("utf-8", errors="strict")
    _require(re.findall(r"(?m)^Job Id:\s*(\S+)\s*$", pbs) == [job_id]
             and re.findall(r"(?m)^\s*job_state\s*=\s*(\S+)\s*$", pbs) == ["F"]
             and re.findall(r"(?m)^\s*Exit_status\s*=\s*(-?[0-9]+)\s*$", pbs) == ["0"]
             and re.findall(r"(?m)^\s*run_count\s*=\s*([0-9]+)\s*$", pbs) == ["1"]
             and [value.lower() for value in re.findall(r"(?m)^\s*Rerunable\s*=\s*(\S+)\s*$", pbs)] == ["false"],
             "PBS terminal output disagrees with dispatch")
    submitted, submission = evidence["submitted-job.json"], evidence["submission-exit.json"]
    argv = ["/opt/pbs/bin/qsub", str(recorded_root / "job.sh")]
    _require(submitted.get("argv") == argv and submitted.get("job_id") == job_id
             and submitted.get("run_id") == expected_build_run_id and submitted.get("qsub_exit") == 0
             and submission.get("argv") == argv and submission.get("exit") == 0
             and submission.get("error") is None, "Submission evidence differs")
    command_logs(submission)
    _require(read(relative_recorded(submission["stdout"])).decode("utf-8").strip() == job_id,
             "Submission stdout job ID differs")

    expected_slots = {(library, profile, role): _target(library, profile, role)
                      for library in FAMILIES for profile in PROFILES for role in ROLES}
    slots = summary.get("slots", [])
    _require(len(slots) == 36, "Missing or duplicate build slots")
    observed = {}
    for slot in slots:
        key = (slot.get("library"), slot.get("profile"), slot.get("role"))
        _require(key in expected_slots and key not in observed and slot.get("target") == expected_slots[key]
                 and slot.get("status") == "BUILT" and slot.get("compile_status") == "SUCCEEDED"
                 and slot.get("failure_stage") is None, "Invalid, duplicate or unsuccessful artifact slot")
        observed[key] = slot
    commands = summary.get("commands", [])
    _require(isinstance(commands, list) and commands, "Missing build command evidence")
    for command in commands:
        _require(command.get("exit_status") == 0 and command.get("process_started") is True
                 and command.get("timed_out") is False and "error" not in command,
                 "Build command did not finish successfully")
        command_logs(command)

    profiles = summary.get("profiles", {})
    _require(set(profiles) == set(PROFILES), "Missing or extra build profile")
    entries, metadata_by_hash, manifests, link_evidence = [], {}, [], {}
    for profile in PROFILES:
        saved = profiles[profile]
        base = "amd-build/build-" + profile + "/"
        _require(set(saved.get("files", {})) == set(PROFILE_FILES), "Incomplete profile file proofs")
        files = {name: identity(saved["files"][name], base + name) for name in PROFILE_FILES}
        _require(set(saved.get("configured_file_identities", {})) == {"build-metadata.json", "partial-manifest-input.json"},
                 "Missing configure-time identity")
        for name, value in saved["configured_file_identities"].items():
            _require(identity(value, base + name) == files[name], "Profile metadata changed after configure")
        metadata = _json(files["build-metadata.json"])
        metadata_hash = _digest(files["build-metadata.json"])
        cmake_commands = re.findall(r"(?m)^CMAKE_COMMAND:INTERNAL=(.+)$",
                                    files["CMakeCache.txt"].decode("utf-8", errors="strict")) if profile == "hip" else []
        if profile == "hip":
            _require(len(cmake_commands) == 1, "Missing or ambiguous configured CMake command")
            cmake_command = str(_absolute(cmake_commands[0]))
        _require(metadata == saved.get("build_metadata") and metadata.get("vendor") == "amd"
                 and metadata.get("build_profile") == profile and metadata.get("build_type") == "Release"
                 and metadata.get("git_metadata_available") is True and metadata.get("git_commit") == head
                 and metadata.get("git_dirty") is True, "Build metadata/source identity differs")
        metadata_by_hash[metadata_hash] = metadata
        dependencies = metadata.get("dependencies", {})
        _require(isinstance(dependencies, dict) and set(saved.get("dependencies", {})) == set(dependencies),
                 "Saved dependency evidence differs from build metadata")
        for name, value in dependencies.items():
            _absolute(value["path"])
            _sha(value["sha256"])
            _require(saved["dependencies"][name] == dict(value, resolved_path=value["path"]),
                     "Saved dependency identity differs from build metadata")
        manifest = _json(files["partial-manifest.json"])
        _require(manifest.get("manifest_schema_version") == 1 and manifest.get("build_metadata_sha256") == metadata_hash,
                 "Partial manifest metadata binding differs")
        manifests.append({"path": str(root / (base + "partial-manifest.json")), "sha256": _digest(files["partial-manifest.json"])})
        descriptors = []
        for name in ("partial-manifest-input.json", "built-manifest-input.json"):
            descriptor = _json(files[name])
            _require(descriptor.get("build_metadata_path") == str(recorded_root / (base + "build-metadata.json")),
                     "Descriptor metadata path escapes or differs")
            descriptors.append(descriptor.get("entries"))
        _require(descriptors[0] == descriptors[1], "Complete build has unequal configured/built descriptors")
        rows = manifest.get("entries", [])
        names = {value for (family, prof, role), value in expected_slots.items() if prof == profile}
        _require(isinstance(rows, list) and len(rows) == 12 and {row.get("target_name") for row in rows} == names
                 and set(saved.get("artifacts", {})) == names and set(saved.get("target_builds", {})) == names
                 and set(saved.get("configured_targets", [])) == names, "Profile artifact inventory differs")
        _require(isinstance(descriptors[0], list) and len(descriptors[0]) == 12
                 and {item.get("target_name") for item in descriptors[0]} == names, "Descriptor target inventory differs")
        descriptor_by_name = {item["target_name"]: item for item in descriptors[0]}
        for entry in rows:
            name = entry["target_name"]
            key = (entry.get("library"), profile, entry.get("executable_role"))
            _require(key in expected_slots and expected_slots[key] == name, "Artifact semantics differ")
            relative = base + "amd/c-cpp/" + key[0] + "/" + name
            recorded_path = str(recorded_root / relative)
            _require(entry.get("executable_path") == recorded_path
                     and entry.get("implementation") == profile and entry.get("build_profile") == profile
                     and entry.get("build_metadata_sha256") == metadata_hash, "Artifact path/profile differs")
            language = "hip" if profile == "hip" else "cxx" if profile == "openmp" or key[0] in {"rocrand", "rocthrust"} else "c"
            compiler = metadata.get(language, {})
            _require(entry.get("compiler_language") == language
                     and all(entry.get(k) == metadata.get(k) for k in ("git_metadata_available", "git_commit", "git_dirty", "build_type"))
                     and all(entry.get(k) == compiler.get(k) for k in ("compiler", "compiler_version", "global_configure_flags")),
                     "Artifact compiler/source metadata differs")
            descriptor = descriptor_by_name[name]
            _require(all(entry.get(k) == descriptor.get(k) for k in (
                "library", "implementation", "executable_role", "target_name", "build_profile",
                "backend_variant", "executable_path", "compiler_language"))
                and entry.get("supported_cpu_backends") == descriptor.get("cpu_backends"),
                "Artifact differs from configured target descriptor")
            artifact = saved["artifacts"][name]
            _require(artifact.get("manifest_entry") == entry, "Summary artifact differs from partial manifest")
            binary = identity(artifact["identity"], relative)
            _require(_digest(binary) == entry.get("binary_sha256"), "Artifact binary hash differs")
            content = json.dumps({k: entry[k] for k in IDENTITY_FIELDS}, ensure_ascii=False,
                                 sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            _require(entry.get("artifact_id") == _digest(content), "Artifact ID differs")
            index = observed[key].get("command_index")
            target = saved["target_builds"][name]
            _require(type(index) is int and 1 <= index <= len(commands)
                     and target.get("command_index") == index and target.get("compile_status") == "SUCCEEDED",
                     "Target build command binding differs")
            argv = commands[index - 1].get("argv", [])
            _require(len(argv) >= 6 and "--build" in argv and "--target" in argv
                     and argv[argv.index("--build") + 1] == str(recorded_root / base.rstrip("/"))
                     and argv[argv.index("--target") + 1:] == [name], "Target compile command differs")
            link_base = str(PurePosixPath(relative).parent / "CMakeFiles" / (name + ".dir"))
            if profile == "hip":
                link_path = link_base + "/link.txt"
                link_data = read(link_path)
                command = commands[index - 1]
                stdout_data = read(relative_recorded(command["stdout"]), command["stdout_sha256"])
                _hip_link_binding(link_data, stdout_data, entry, compiler, cmake_command)
                link_evidence[entry["artifact_id"]] = {
                    "mode": "native-hip-link-command",
                    "link_command": {"path": str(recorded_root / link_path), "sha256": _digest(link_data)},
                    "build_stdout": {"path": command["stdout"], "sha256": command["stdout_sha256"]},
                    "build_directory": str(PurePosixPath(recorded_path).parent),
                }
            else:
                link_path = link_base + "/link.d"
                _require(bool(read(link_path).strip()), "Empty linker dependency evidence")
                link_evidence[entry["artifact_id"]] = str(recorded_root / link_path)
            entries.append(entry)
    _require(len({entry["artifact_id"] for entry in entries}) == 36, "Ambiguous artifact IDs")
    harness_data, _ = _read(harness_path)
    _require(_digest(harness_data) == expected_harness_sha256, "New smoke harness hash differs from approval")
    # A second pass detects changes during this multi-file inspection, including
    # changes to source executable bits.  No file is created or repaired here.
    for relative, digest in list(checked.items()):
        read(relative, digest)
    _require(snapshot_identity(snapshot_root, first["paths"]) == snapshot,
             "Saved source changed during proof inspection")
    _require(_read(harness_path)[0] == harness_data, "New smoke harness changed during inspection")
    return {"schema_version": 1, "status": "VERIFIED", "build_run_id": expected_build_run_id,
            "build_run_root": str(root), "recorded_build_run_root": str(recorded_root),
            "path_mapping_used": str(root) != str(recorded_root),
            "artifact_source": {"source_snapshot_sha256": source_hash, "git_commit": head,
                                "git_dirty": True, "snapshot_root": str(snapshot_root),
                                "files": snapshot["files"]},
            "harness": {"path": str(harness_path), "sha256": expected_harness_sha256},
            "evidence_sha256": checked, "entries": entries, "metadata_by_hash": metadata_by_hash,
            "partial_manifests": manifests, "link_evidence": link_evidence,
            "runtime_dependencies_verified": False, "loaded_libraries_verified": False}

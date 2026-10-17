#!/usr/bin/env python3
"""Audited, CPU-only local validation of a small Git-less source snapshot.

Only a fresh directory below /tmp/gpu-library-suite-local-build is writable.
CTest/unittest remain the test frameworks. No remote command is executed.
"""

import argparse
import ctypes
import datetime
import hashlib
import importlib.util
import json
import os
from os import walk as directory_walk
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

# Loading this entry must never create a cache next to repository sources.
sys.dont_write_bytecode = True
from run_profiled_unittests import (  # noqa: E402
    audit_inventory, CPU_FIXTURE_MODULE_PREFIXES, CPU_FIXTURE_TEST_IDS, POLICY_EXCLUSIONS,
)


LOCAL_BASE = Path("/tmp/gpu-library-suite-local-build").resolve()
CTEST_EXCLUSIONS = {
    "test_result": "uses fixed /tmp paths and tmpfile() outside dedicated work directory",
    "python_unittest": "replaced by audited safe-independent Python profile",
    "non_git_source_build": "nested full CTest does not inherit policy exclusions; safe isolation test retained",
    "python39_compat": "shared fixed cache; existing check_file functions run with dedicated cache",
}
CTEST_FIXED = frozenset("""
    test_checked test_clock test_json test_cli test_result test_cpp_compat
    test_verification_nonfinite test_rocfft_verification cmake_compile_link_probe
    cmake_fftw_positive_configure cmake_fftw_positive_build
    fftw_fixture_partial_manifest fftw_fixture_serial_compute
    fftw_fixture_threaded_end_to_end cmake_fftw_header_only_negative
    cmake_fftw_header_only_has_no_target cmake_cpu_provider_positive_configure
    cmake_cpu_provider_positive_build cpu_provider_fixture_partial_manifest
    cblas_fixture_compute mkl_sparse_fixture_compute lapacke_fixture_end_to_end
    cmake_cpu_provider_symbol_negative cpu_provider_negative_no_blas_cpu
    cpu_provider_negative_no_sparse_cpu cpu_provider_negative_no_solver_cpu
    cmake_cpu_provider_mixed_rejected cmake_cpu_provider_mixed_no_blas_target
    cmake_cpu_provider_mixed_no_solver_target cmake_cpu_provider_switch_onemkl
    cmake_cpu_provider_switch_openblas cmake_cpu_provider_switch_build
    rand_cpu_fixture_compute reduce_cpu_fixture_end_to_end
    cmake_dependency_root_discovery python_cpu_fixture_unittest
    cmake_openacc_only_tests_generate cmake_openacc_only_test_inventory
    python_unittest python39_compat non_git_source_build
""".split())
CTEST_SYNTAX = frozenset(
    ["cufft_syntax_" + stem for stem in
     ("fft_gpu", "openacc_cufft", "fft_gpu_bench", "openacc_cufft_bench")]
    + ["phase3_syntax_" + library + "_" + filename
       for library, stem in (("cublas", "blas"), ("cusparse", "sparse"),
                             ("cusolver", "solver"), ("curand", "rand"),
                             ("thrust", "reduce"))
       for filename in (stem + "_gpu", "openacc_" + library,
                        stem + "_gpu_bench", "openacc_" + library + "_bench")]
)
CTEST_METADATA = frozenset("test_cuda_metadata_" + name for name in
                          ("cufft", "cublas", "cusparse", "cusolver", "curand", "thrust"))
CTEST_SHELL = frozenset("pegasus_bash_syntax_" + name for name in
                       ("build_cpu_cuda_sh", "build_openacc_sh", "run_node_sh", "run_benchmarks_pbs_in"))
CTEST_SHELLCHECK = frozenset("pegasus_shellcheck_" + name for name in
                            ("build_cpu_cuda_sh", "build_openacc_sh", "run_node_sh"))
CTEST_ALL = CTEST_FIXED | CTEST_SYNTAX | CTEST_METADATA | CTEST_SHELL | CTEST_SHELLCHECK
# Audited registration source: detects removed and newly conditional tests even
# when this machine does not register them. Update only after policy review.
CTEST_REGISTRATION_SHA256 = "1b71a3a6d1a7b417420128c8f2b0a63441ede856515acf80dc66a383f14775bd"
STATUSES = {"PASS", "FAIL", "SKIP", "POLICY_EXCLUDED", "NOT_RUN_DEPENDENCY"}


class ValidationError(RuntimeError):
    pass


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, document):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValidationError("duplicate JSON key: " + key)
            result[key] = value
        return result

    def constant(value):
        raise ValidationError("non-standard JSON number: " + value)

    return json.loads(Path(path).read_text(encoding="utf-8", errors="strict"),
                      object_pairs_hook=pairs, parse_constant=constant)


def local_filesystem(path):
    """Fail closed on remote/FUSE/unknown filesystems; inspect only metadata."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        class DarwinStatFS(ctypes.Structure):
            _fields_ = [
                ("bsize", ctypes.c_uint32), ("iosize", ctypes.c_int32),
                ("blocks", ctypes.c_uint64), ("bfree", ctypes.c_uint64),
                ("bavail", ctypes.c_uint64), ("files", ctypes.c_uint64),
                ("ffree", ctypes.c_uint64), ("fsid", ctypes.c_int32 * 2),
                ("owner", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("flags", ctypes.c_uint32), ("subtype", ctypes.c_uint32),
                ("fstypename", ctypes.c_char * 16), ("reserved", ctypes.c_byte * 4096),
            ]
        info = DarwinStatFS()
        if libc.statfs(os.fsencode(path), ctypes.byref(info)):
            raise ValidationError("statfs failed for " + str(path))
        kind = info.fstypename.decode("ascii", errors="strict")
        if kind not in {"apfs", "hfs"}:
            raise ValidationError("not an approved local filesystem: " + kind)
        return kind
    if sys.platform.startswith("linux"):
        # The Linux statfs ABI begins with a native long f_type; reserve more
        # than the complete structure, without assuming the remaining layout.
        info = (ctypes.c_long * 128)()
        if libc.statfs(os.fsencode(path), ctypes.byref(info)):
            raise ValidationError("statfs failed for " + str(path))
        known = {0xEF53: "ext", 0x01021994: "tmpfs", 0x58465342: "xfs",
                 0x9123683E: "btrfs", 0x794C7630: "overlay"}
        kind = known.get(info[0] & 0xffffffff)
        if kind is None:
            raise ValidationError("remote/FUSE/unknown filesystem is prohibited")
        return kind
    raise ValidationError("local filesystem validation is unsupported on this OS")


def validated_path(path):
    path = Path(os.path.abspath(str(path)))
    # /tmp is the OS-provided alias on macOS, not a user-created source link.
    if sys.platform == "darwin" and str(path).startswith("/tmp/"):
        path = Path("/private") / path.relative_to("/")
    try:
        relative = path.relative_to(LOCAL_BASE)
    except ValueError:
        raise ValidationError("path must be below " + str(LOCAL_BASE))
    if not relative.parts:
        raise ValidationError("the shared local validation base is not a work directory")
    current = LOCAL_BASE
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValidationError("symbolic link is prohibited: " + str(current))
    return path


def snapshot_files(source):
    files = {}
    total = 0
    for directory, children, filenames in directory_walk(str(source)):
        for name in children + filenames:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValidationError("symbolic link in snapshot: " + str(path))
            if name in {".git", "__pycache__", "CMakeCache.txt", "CMakeFiles"}:
                raise ValidationError("snapshot contains Git/build/cache data: " + str(path))
        for name in filenames:
            path = Path(directory) / name
            if not path.is_file():
                raise ValidationError("snapshot contains a non-regular file")
            total += path.stat().st_size
            if total > 32 * 1024 * 1024 or len(files) >= 2000:
                raise ValidationError("snapshot exceeds small local validation limits")
            files[path.relative_to(source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def validate_paths(source, work):
    source, work = validated_path(source), validated_path(work)
    if not source.is_dir() or not (source / "CMakeLists.txt").is_file():
        raise ValidationError("source-copy is not a repository source snapshot")
    if source == work or source in work.parents or work in source.parents:
        raise ValidationError("source-copy and work-dir must not overlap")
    if work.exists() or not work.parent.is_dir():
        raise ValidationError("work-dir must be new and its parent must already exist")
    local_filesystem(source)
    local_filesystem(work.parent)
    source_files = snapshot_files(source)
    if source_files.get("tests/CMakeLists.txt") != CTEST_REGISTRATION_SHA256:
        raise ValidationError("CTest registration changed; audit the fixed local profile first")
    audit_inventory(source / "tests" / "python")
    return source, work, source_files


def make_environment(source, work):
    environment = dict(os.environ)
    # Do not inherit unrelated profile inputs, Git context, Python overrides,
    # provider paths, or compiler launcher/caching settings.
    for name in list(environment):
        if name.startswith(("GPU_SUITE_", "GIT_", "PYTHON", "CMAKE_", "CCACHE_", "SCCACHE_")):
            environment.pop(name)
    for name in ("CC", "CXX", "FC", "CFLAGS", "CXXFLAGS", "LDFLAGS", "CPATH",
                 "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "MKLROOT",
                 "ONEAPI_ROOT", "FFTW_ROOT", "FFTW3_ROOT", "FFTW3F_ROOT", "FFTW3f_ROOT",
                 "PKG_CONFIG_PATH", "PKG_CONFIG_LIBDIR"):
        environment.pop(name, None)
    for name, directory in (("TMPDIR", "tmp"), ("TMP", "tmp"), ("TEMP", "tmp"),
                            ("PYTHONPYCACHEPREFIX", "cache/python"),
                            ("XDG_CACHE_HOME", "cache"), ("MPLCONFIGDIR", "cache/matplotlib")):
        destination = work / directory
        destination.mkdir(parents=True, exist_ok=True)
        environment[name] = str(destination)
    environment.update({
        "PYTHONPATH": str(source / "tools"), "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1", "GIT_OPTIONAL_LOCKS": "0",
        "GIT_CEILING_DIRECTORIES": str(source.parent), "GPU_SUITE_SAFE_LOCAL": "1",
    })
    return environment


def ctest_selection(inventory, metadata, shellcheck_available):
    tests = inventory.get("tests", [])
    names = [test["name"] for test in tests]
    if len(names) != len(set(names)) or set(names) - CTEST_ALL:
        raise ValidationError("duplicate or unaudited CTest IDs")
    unavailable = {}
    if not shellcheck_available:
        unavailable.update({name: "ShellCheck is not installed" for name in CTEST_SHELLCHECK})
    if metadata["cxx"]["compiler"] not in {"GNU", "Clang", "AppleClang"}:
        unavailable.update({name: "GNU-like syntax-only compiler is unavailable" for name in CTEST_SYNTAX})
    missing = CTEST_ALL - set(names)
    if missing - set(unavailable):
        raise ValidationError("audited CTest IDs disappeared: " + ", ".join(sorted(missing - set(unavailable))))
    rows = [{"id": "ctest:" + name, "status": "NOT_RUN_DEPENDENCY", "reason": unavailable[name]}
            for name in sorted(missing)]
    rows.extend({"id": "ctest:" + name, "status": "POLICY_EXCLUDED", "reason": reason}
                for name, reason in CTEST_EXCLUSIONS.items())
    return sorted(set(names) - set(CTEST_EXCLUSIONS)), rows


def junit_results(path, selected):
    records = []
    seen = set()
    for case in ET.parse(path).getroot().iter("testcase"):
        name = case.get("name")
        if name not in selected or name in seen:
            raise ValidationError("unexpected or duplicate JUnit test: " + str(name))
        seen.add(name)
        if case.find("failure") is not None or case.find("error") is not None:
            status, reason = "FAIL", "CTest reports failure; see CTest stdout/stderr and JUnit"
        elif case.find("skipped") is not None:
            status, reason = "SKIP", "CTest reports skipped: " + case.find("skipped").get("message", "")
        elif case.get("status") not in {"run", "passed"}:
            status, reason = "FAIL", "CTest did not confirm execution: " + str(case.get("status"))
        else:
            status, reason = "PASS", "CTest JUnit confirms completion"
        records.append({"id": "ctest:" + name, "status": status, "reason": reason})
    records.extend({"id": "ctest:" + name, "status": "FAIL", "reason": "selected test has no JUnit completion evidence"}
                   for name in sorted(set(selected) - seen))
    return records


def validated_python_report(document, audited_ids):
    inventory, selected = document["inventory"], document["selected"]
    if len(inventory) != len(set(inventory)) or set(inventory) != set(audited_ids):
        raise ValidationError("Python report inventory differs from audited IDs")
    cpu = {name for name in audited_ids if name in CPU_FIXTURE_TEST_IDS
           or name.startswith(CPU_FIXTURE_MODULE_PREFIXES)}
    if document["profile"] == "cpu-fixture":
        expected, excluded = cpu, set()
    elif document["profile"] == "safe-independent":
        excluded = set(POLICY_EXCLUSIONS)
        expected = set(audited_ids) - cpu - excluded
    else:
        raise ValidationError("unexpected Python validation profile")
    if len(selected) != len(set(selected)) or set(selected) != expected:
        raise ValidationError("Python report selected IDs differ from fixed profile")
    rows, seen = document["results"], set()
    for row in rows:
        name, status = row["id"], row["status"]
        if name in seen or status not in STATUSES or not isinstance(row.get("reason"), str):
            raise ValidationError("duplicate or invalid Python result")
        seen.add(name)
        if name in excluded:
            if status != "POLICY_EXCLUDED":
                raise ValidationError("policy-excluded case was executed or misclassified")
        elif name in expected:
            if status not in {"PASS", "FAIL", "SKIP"}:
                raise ValidationError("selected Python case lacks an execution outcome")
        else:
            fixture = re.fullmatch(r"(?:setUpClass|tearDownClass|setUpModule|tearDownModule) \((.*?)\)", name)
            known_fixture = fixture and any(case.startswith(fixture.group(1) + ".") for case in expected)
            coverage = name in {"test_sirius_remote.required-coverage", "test_safe_local_validation.required-coverage"}
            if not ((known_fixture and status in {"SKIP", "FAIL"}) or (coverage and status == "FAIL")):
                raise ValidationError("unknown Python result ID: " + name)
    if not (expected | excluded).issubset(seen):
        raise ValidationError("Python report lacks selected or excluded outcome evidence")
    return rows


class ValidationRun:
    def __init__(self, source, work, source_files):
        self.source, self.work, self.source_files = source, work, source_files
        self.results, self.commands = [], []
        self.python_ids = audit_inventory(source / "tests/python")
        self.document = {"format_version": 1, "started_at": utc_now(),
                         "source_copy": str(source), "work_dir": str(work),
                         "python": {"executable": sys.executable, "version": sys.version},
                         "ssh_calls": 0, "scheduler_operations": 0,
                         "results": self.results, "commands": self.commands,
                         "planned_inventory": sorted({"command:probe-bash-syntax", "command:probe-shellcheck"}
                                                     | {"ctest:" + name for name in CTEST_ALL}
                                                     | {"python:" + name for name in self.python_ids})}
        work.mkdir()
        (work / "logs").mkdir()
        write_json(work / "source-inventory.json", source_files)
        self.runtime_source = work / "source-runtime"
        shutil.copytree(str(source), str(self.runtime_source))
        self.environment = make_environment(self.runtime_source, work)

    def record(self, test_id, status, reason):
        if status not in STATUSES:
            raise ValidationError("invalid result classification")
        self.results.append({"id": test_id, "status": status, "reason": reason})

    def command(self, name, argv):
        stdout = self.work / "logs" / (name + ".stdout.log")
        stderr = self.work / "logs" / (name + ".stderr.log")
        entry = {"id": name, "argv": [str(item) for item in argv], "cwd": str(self.work),
                 "started_at": utc_now(), "stdout": str(stdout), "stderr": str(stderr)}
        self.commands.append(entry)
        try:
            with stdout.open("xb") as out, stderr.open("xb") as err:
                completed = subprocess.run(entry["argv"], cwd=str(self.work),
                                           env=self.environment, stdout=out, stderr=err, check=False)
            entry["exit_code"] = completed.returncode
        except OSError as error:
            entry["exit_code"] = None
            entry["error"] = str(error)
        entry["ended_at"] = utc_now()
        status = "PASS" if entry["exit_code"] == 0 else "FAIL"
        self.record("command:" + name, status, "see preserved argv, exit code and logs")
        # Append after every command so interruption retains command evidence.
        with (self.work / "command-transcript.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
        return entry["exit_code"] == 0

    def python_audit(self):
        path = self.runtime_source / "tools" / "check_python39.py"
        spec = importlib.util.spec_from_file_location("safe_local_python39", str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cache = self.work / "cache" / "python39"
        cache.mkdir()
        for source_file in module.python_files(self.runtime_source):
            errors = module.check_file(source_file, cache)
            self.record("python39:" + source_file.relative_to(self.runtime_source).as_posix(),
                        "FAIL" if errors else "PASS", "\n".join(errors) if errors else "Python 3.9 syntax/API and byte compilation")

    def import_report(self, path):
        document = strict_json(path)
        rows = validated_python_report(document, self.python_ids)
        for row in rows:
            self.record("python:" + row["id"], row["status"], row["reason"])
        return document

    def check_probe_syntax(self):
        probe = self.runtime_source / "jobs/sirius/probe.sh"
        for tool, name, flags in (("bash", "probe-bash-syntax", ["-n"]),
                                  ("shellcheck", "probe-shellcheck", [])):
            executable = shutil.which(tool)
            if executable is None:
                self.record("command:" + name, "NOT_RUN_DEPENDENCY", tool + " is not installed")
            else:
                self.command(name, [executable] + flags + [probe])

    def execute(self, parallel):
        self.python_audit()
        self.check_probe_syntax()
        cmake, ctest = shutil.which("cmake"), shutil.which("ctest")
        build = self.work / "build"
        configured = built = False
        if cmake:
            self.environment["GPU_SUITE_CMAKE_COMMAND"] = cmake
        if cmake and ctest:
            self.command("cmake-version", [cmake, "--version"])
            self.command("ctest-version", [ctest, "--version"])
            self.command("ctest-help", [ctest, "--help"])
            if "--output-junit" not in (self.work / "logs/ctest-help.stdout.log").read_text():
                raise ValidationError("CTest lacks --output-junit; update CMake externally before validation")
            configured = self.command("configure", [
                cmake, "-S", self.runtime_source, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                "-DGPU_SUITE_BUILD_CPU=ON", "-DGPU_SUITE_BUILD_CUDA=OFF",
                "-DGPU_SUITE_BUILD_OPENACC=OFF", "-DGPU_SUITE_BUILD_TESTS=ON",
                "-DGPU_SUITE_SOURCE_LANGUAGE=c-cpp", "-DPython3_EXECUTABLE=" + sys.executable,
            ])
            if configured:
                self.environment.update({
                    "GPU_SUITE_BUILD_METADATA_FILE": str(build / "build-metadata.json"),
                    "GPU_SUITE_BUILD_METADATA_HEADER": str(build / "generated/include/gpu_suite/build_metadata.h"),
                    "GPU_SUITE_CMAKE_COMMAND": cmake,
                    "GPU_SUITE_SAFE_LOCAL_CPU_REPORT": str(self.work / "python-cpu-fixture.json"),
                })
                metadata = strict_json(build / "build-metadata.json")
                self.document["build_metadata"] = metadata
                if (metadata["git_metadata_available"] is not False or
                        metadata["git_commit"] is not None or metadata["git_dirty"] is not None):
                    raise ValidationError("Git-less snapshot unexpectedly has Git provenance")
                self.record("non-git-build-metadata", "PASS", "metadata unavailable; commit/dirty null")
                configured_text = (self.work / "logs/configure.stdout.log").read_text()
                for line in configured_text.splitlines():
                    match = re.match(r"-- (.*?): DISABLED \((.*?)\)$", line)
                    if match:
                        label, reason = match.groups()
                        policy = any(term in reason.lower() for term in ("not requested", "controlled by", "disabled"))
                        self.record("configure-capability:" + label,
                                    "POLICY_EXCLUDED" if policy else "NOT_RUN_DEPENDENCY", reason)
                if self.command("ctest-inventory", [ctest, "--test-dir", build, "--show-only=json-v1"]):
                    inventory = strict_json(self.work / "logs/ctest-inventory.stdout.log")
                    selected, exclusions = ctest_selection(inventory, metadata, bool(shutil.which("shellcheck")))
                    self.results.extend(exclusions)
                    write_json(self.work / "ctest-selection.json", {"selected": selected, "inventory": inventory})
                    built = self.command("build", [cmake, "--build", build, "--parallel", str(parallel)])
                    if built:
                        self.command("partial-manifest", [cmake, "--build", build, "--target", "gpu_suite_partial_manifest", "--parallel", str(parallel)])
                        expression = "^(" + "|".join(re.escape(name) for name in selected) + ")$"
                        self.command("ctest", [ctest, "--test-dir", build, "--output-on-failure",
                                               "--parallel", str(parallel), "-R", expression,
                                               "--output-junit", self.work / "ctest.xml"])
                        try:
                            self.results.extend(junit_results(self.work / "ctest.xml", selected))
                            self.import_report(self.work / "python-cpu-fixture.json")
                        except Exception as error:
                            self.record("ctest-completion-evidence", "FAIL", str(error))
                    else:
                        for test_id in selected:
                            self.record("ctest:" + test_id, "NOT_RUN_DEPENDENCY", "CPU build failed; see build command FAIL")
        else:
            self.record("cmake-ctest", "NOT_RUN_DEPENDENCY", "CMake and CTest must already be installed")
            for test_id in sorted(CTEST_ALL):
                self.record("ctest:" + test_id,
                            "POLICY_EXCLUDED" if test_id in CTEST_EXCLUSIONS else "NOT_RUN_DEPENDENCY",
                            CTEST_EXCLUSIONS.get(test_id, "CMake/CTest unavailable"))
        if not built:
            for test_id in self.python_ids:
                if test_id in CPU_FIXTURE_TEST_IDS or test_id.startswith(CPU_FIXTURE_MODULE_PREFIXES):
                    self.record("python:" + test_id, "NOT_RUN_DEPENDENCY", "CPU fixture build unavailable")
        report = self.work / "python-independent.json"
        self.command("python-independent", [sys.executable,
                     self.runtime_source / "tests/python/run_profiled_unittests.py",
                     "--start-directory", self.runtime_source / "tests/python",
                     "--profile", "safe-independent", "--report", report])
        self.import_report(report)
        if built and (build / "partial-manifest.json").is_file():
            output = self.work / "dry-run-must-not-exist"
            digest = hashlib.sha256(json.dumps(self.source_files, sort_keys=True).encode()).hexdigest()
            synthetic = hashlib.sha256(b"safe-local dry-run synthetic runtime identity; not measurement evidence").hexdigest()
            self.document["dry_run_runtime_hash_kind"] = "synthetic; not measurement evidence"
            self.command("runner-dry-run", [sys.executable, self.runtime_source / "tools/run_suite.py",
                         "--config", self.runtime_source / "configs/pilot.json",
                         "--manifest", build / "partial-manifest.json", "--build-metadata", build / "build-metadata.json",
                         "--output", output, "--run-id", "safe-local-dry-run", "--system-label", "local-validation",
                         "--wave", "0", "--node-index", "0", "--hostname", "local-validation",
                         "--runtime-environment-sha256", synthetic, "--source-snapshot-sha256", digest, "--dry-run"])
            self.record("dry-run-output-absent", "FAIL" if output.exists() else "PASS", "dry-run must not create output directory")
        else:
            self.record("runner-dry-run", "NOT_RUN_DEPENDENCY", "built partial manifest unavailable")
        self.record("actual-gpu-execution", "POLICY_EXCLUDED", "local-only profile; no CUDA/ROCm/OpenMP offload execution")
        self.record("actual-remote-execution", "POLICY_EXCLUDED", "no SSH, scheduler, remote build or machine probes")
        if importlib.util.find_spec("matplotlib") is None:
            self.record("real-matplotlib-rendering", "NOT_RUN_DEPENDENCY", "Matplotlib is absent from the selected Python environment; fake renderer tests only")
        else:
            self.record("real-matplotlib-rendering", "POLICY_EXCLUDED", "this profile runs fake renderer tests; real rendering is outside this profile")

    def finish(self):
        observed = {row["id"] for row in self.results}
        for test_id in self.document.get("planned_inventory", []):
            if test_id not in observed:
                self.record(test_id, "FAIL", "planned check has no completion or exclusion evidence")
        try:
            if snapshot_files(self.source) != self.source_files:
                raise ValidationError("supplied source snapshot changed during validation")
            self.record("source-copy-unchanged", "PASS", "input file inventory and content hashes unchanged")
        except Exception as error:
            self.record("source-copy-unchanged", "FAIL", str(error))
        self.document["ended_at"] = utc_now()
        self.document["counts"] = {status: sum(row["status"] == status for row in self.results) for status in sorted(STATUSES)}
        write_json(self.work / "validation-summary.json", self.document)
        print(json.dumps({"summary": str(self.work / "validation-summary.json"), "counts": self.document["counts"]}, indent=2))
        return 1 if self.document["counts"]["FAIL"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-copy", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--parallel", type=int, default=2, choices=(1, 2))
    arguments = parser.parse_args(argv)
    try:
        source, work, files = validate_paths(arguments.source_copy, arguments.work_dir)
        run = ValidationRun(source, work, files)
    except Exception as error:
        print("safe local validation refused: " + str(error), file=sys.stderr)
        return 1
    try:
        run.execute(arguments.parallel)
    except Exception as error:
        run.record("validation-entry", "FAIL", str(error))
    return run.finish()


if __name__ == "__main__":
    sys.exit(main())

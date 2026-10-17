"""Allocated AMD build controller tests: synthetic files/processes, no SDK or PBS."""
from contextlib import redirect_stdout
import builtins
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gpu_suite.hashing import source_snapshot_sha256
from gpu_suite.manifest import artifact_id

SPEC = importlib.util.spec_from_file_location("sirius_amd_build", ROOT / "jobs/sirius/amd_build.py")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)
GENERATOR_SPEC = importlib.util.spec_from_file_location("sirius_test_manifest_generator", ROOT / "tools/generate_partial_manifest.py")
generator = importlib.util.module_from_spec(GENERATOR_SPEC)
GENERATOR_SPEC.loader.exec_module(generator)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class FakeProcesses:
    def __init__(self, test, missing_cpu=(), fail_profile=None, corrupt=None, fail_targets=(), fail_manifest=None,
                 durations=None, clock=None):
        self.test, self.missing_cpu = test, set(missing_cpu)
        self.fail_profile, self.corrupt = fail_profile, corrupt
        self.fail_targets, self.fail_manifest = set(fail_targets), fail_manifest
        self.durations, self.clock = durations or {}, clock
        self.build_attempts = []
        self.calls = []
        self.pid = 424242

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        code, output, error = self.dispatch(argv)
        kwargs["stdout"].write(output.encode())
        kwargs["stderr"].write(error.encode())
        duration_key = argv[-1] if "--build" in argv else ""
        if argv[:2] == [sys.executable, "-B"]:
            duration_key = "manifest-" + Path(argv[-1]).parent.name.removeprefix("build-")
        duration = self.durations.get(duration_key, 0)
        waited = False
        def wait(timeout=None):
            nonlocal waited
            if waited:
                return 0
            waited = True
            if self.clock:
                self.clock.value += min(duration, timeout) if timeout is not None else duration
            if timeout is not None and duration > timeout:
                raise subprocess.TimeoutExpired(argv, timeout)
            return code
        return mock.Mock(pid=self.pid, wait=wait)

    def dispatch(self, argv):
        t = self.test
        if argv[0] == str(t.tools["git"]):
            if argv[-2:] == ["rev-parse", "--show-toplevel"]:
                return 0, str(t.source) + "\n", ""
            if argv[-2:] == ["rev-parse", "HEAD"]:
                return 0, "a" * 40 + "\n", ""
            if "status" in argv:
                return 0, "## fixture\n M source.txt\n", ""
            if "diff" in argv:
                return 0, "", ""
            if "ls-files" in argv:
                return 0, "\0".join(t.source_paths) + "\0", ""
        if argv[1:] == ["--version"]:
            return 0, "fake compiler version\n", "version diagnostic\n"
        if argv[0] == str(t.tools["cmake"]):
            if "-S" in argv:
                directory = Path(argv[argv.index("-B") + 1])
                profile = directory.name.removeprefix("build-")
                if profile == self.fail_profile:
                    return 1, "configure failed\n", "fixture compile/link failure\n"
                flags = dict(value[2:].split("=", 1) for value in argv if value.startswith("-D"))
                if profile == "cpu":
                    for family, prefix in (("rocblas", "GPU_SUITE_GENERIC_CBLAS_"),
                                           ("rocsolver", "GPU_SUITE_GENERIC_LAPACKE_"),
                                           ("rocsparse", "GPU_SUITE_AOCL_SPARSE_")):
                        include = flags.get(prefix + "INCLUDE_DIR")
                        if include and not Path(include).is_dir():
                            self.missing_cpu.add(family)
                self.configure(directory, profile, flags)
                output = ""
                if profile == "cpu":
                    for family in self.missing_cpu:
                        reason = "explicit GENERIC provider not found" if flags.get("GPU_SUITE_CPU_BLAS_BACKEND") == "GENERIC_CBLAS" else "ONEMKL provider not found"
                        output += "-- " + family + " CPU targets: DISABLED (" + reason + ")\n"
                return 0, output, ""
            if "--build" in argv:
                directory = Path(argv[argv.index("--build") + 1])
                targets = argv[argv.index("--target") + 1:]
                if len(targets) != 1 or targets[0] == "gpu_suite_partial_manifest":
                    raise AssertionError("Build must not retry targets through the whole manifest dependency graph")
                target = targets[0]
                key = (directory.name, target)
                if key in self.build_attempts:
                    raise AssertionError("Repeated target build: " + repr(key))
                self.build_attempts.append(key)
                descriptor = build.load_json(directory / "partial-manifest-input.json")
                if target in self.fail_targets:
                    # A failed link can leave an output; mere existence must
                    # never put that target into the successful manifest.
                    for entry in descriptor["entries"]:
                        if entry["target_name"] == target:
                            Path(entry["executable_path"]).write_text("Incomplete output from failed build")
                    return 1, "failed target " + target + "\n", "fixture compile failure\n"
                for entry in descriptor["entries"]:
                    if entry["target_name"] == target:
                        Path(entry["executable_path"]).write_text("Non-executable synthetic artifact: " + target)
                if self.corrupt == "metadata-during-build" and directory.name == "build-cpu":
                    metadata = build.load_json(directory / "build-metadata.json")
                    metadata["c"]["global_configure_flags"] += " -DUNEXPECTED"
                    save(directory / "build-metadata.json", metadata)
                return 0, "fake compile/link only\n", ""
        if argv[:2] == [sys.executable, "-B"] and argv[2] == str(t.source / "tools/generate_partial_manifest.py"):
            directory = Path(argv[3]).parent
            if directory.name == "build-" + str(self.fail_manifest):
                return 1, "", "fixture manifest generation failure\n"
            if self.corrupt == "missing-manifest" and directory.name == "build-cpu":
                return 0, "fixture generator returned zero without output\n", ""
            try:
                self.manifest(directory, Path(argv[3]))
            except (OSError, ValueError) as error:
                return 1, "", str(error) + "\n"
            return 0, "generated with actual repository manifest generator\n", ""
        raise AssertionError("Unapproved subprocess boundary: " + repr(argv))

    def configure(self, directory, profile, flags):
        t = self.test
        directory.mkdir()
        rows = [row for row in build.slots() if row["profile"] == profile
                and not (profile == "cpu" and row["library"] in self.missing_cpu)]
        entries = []
        for row in rows:
            target = directory / "amd/c-cpp" / row["library"] / row["target"]
            target.parent.mkdir(parents=True, exist_ok=True)
            language = "hip" if profile == "hip" else "cxx" if profile == "openmp" or row["library"] in {"rocrand", "rocthrust"} else "c"
            backends = []
            if profile == "cpu":
                backends = ["cpu-fftw-serial", "cpu-fftw-threaded"] if row["library"] == "rocfft" else ["cpu-onemkl"]
                if t.args.aocl_cpu_candidates and row["library"] in {"rocblas", "rocsolver"}:
                    backends = ["cpu-generic-cblas" if row["library"] == "rocblas" else "cpu-generic-lapacke"]
                if row["library"] == "rocsparse" and t.args.cpu_sparse_backend == "AOCLSPARSE":
                    backends = ["cpu-aocl-sparse"]
            entries.append({"target_name": row["target"], "library": row["library"],
                            "executable_path": str(target), "implementation": profile,
                            "build_profile": profile, "executable_role": row["role"],
                            "compiler_language": language,
                            "backend_variant": "aocl-sparse" if backends == ["cpu-aocl-sparse"] else "fixture",
                            "cpu_backends": backends})
        save(directory / "partial-manifest-input.json", {"build_metadata_path": str(directory / "build-metadata.json"), "entries": entries})
        cxx = t.args.amd_cxx_compiler if profile == "openmp" else t.args.cpu_cxx_compiler
        metadata = {"vendor": "amd", "build_profile": profile, "build_type": "Release",
                    "git_commit": "a" * 40, "git_metadata_available": True, "git_dirty": True,
                    "amd_build_inputs": dict(build.cpu_options(t.args), GPU_SUITE_AMD_LIBRARIES=";".join(build.FAMILIES)),
                    "dependencies": {}, "fftw_threading": {"requested": "OPENMP", "backend": "openmp", "compile_options": ["-fopenmp"], "link_options": [], "dependency_names": ["fftw3f", "fftw3f_omp"]},
                    "rocm": {"root": str(t.args.rocm_root)},
                    "openmp": {"gpu_target": t.args.gpu_arch, "compile_flags": "-fopenmp -target", "link_flags": "-fopenmp -target"}}
        for language, compiler in (("c", t.args.cpu_c_compiler), ("cxx", cxx), ("hip", t.args.amd_cxx_compiler)):
            metadata[language] = {"compiler": "Fake", "compiler_version": "1.0", "compiler_path": str(compiler), "global_configure_flags": "-O2"}
        metadata["hip"]["architectures"] = t.args.gpu_arch
        names = {"cblas", "lapacke", "mkl_sparse", "fftw3f", "fftw3f_omp"} if profile == "cpu" else set(build.FAMILIES) | {"hip", "rocprim"}
        if profile == "cpu" and t.args.cpu_sparse_backend == "AOCLSPARSE":
            names.discard("mkl_sparse")
            names.add("aocl_sparse")
        for name in names:
            path = t.root / "dependencies" / (name + ".fixture")
            path.parent.mkdir(exist_ok=True)
            path.write_text(name)
            metadata["dependencies"][name] = {"path": str(path), "sha256": build.file_identity(path)["sha256"]}
        if profile == "cpu":
            for family, provider, prefix, dependency in (
                    ("rocblas", "CBLAS", "GPU_SUITE_GENERIC_CBLAS_", "cblas"),
                    ("rocsolver", "LAPACKE", "GPU_SUITE_GENERIC_LAPACKE_", "lapacke"),
                    ("rocsparse", "AOCLSparse", "GPU_SUITE_AOCL_SPARSE_", "aocl_sparse")):
                if family not in self.missing_cpu and flags.get(prefix + "LIBRARY"):
                    libraries = [flags[prefix + "LIBRARY"]] + [p for p in flags.get(prefix + "EXTRA_LIBRARIES", "").split(";") if p]
                    metadata["amd_build_inputs"][provider + "_INTERFACE_INCLUDE_DIRECTORIES"] = flags[prefix + "INCLUDE_DIR"]
                    metadata["amd_build_inputs"][provider + "_INTERFACE_LINK_LIBRARIES"] = ";".join(libraries + ["Threads::Threads"])
                    for index, path in enumerate(libraries):
                        key = dependency + ("_link_" + str(index) if index else "")
                        metadata["dependencies"][key] = {"path": path, "sha256": build.file_identity(path)["sha256"]}
                    if provider == "AOCLSparse":
                        identity = build.file_identity(Path(flags[prefix + "INCLUDE_DIR"]) / "aoclsparse.h")
                        metadata["dependencies"]["aocl_sparse_header"] = {
                            "path": identity["resolved_path"], "sha256": identity["sha256"]}
            if self.corrupt == "aocl-order":
                key = "LAPACKE_INTERFACE_LINK_LIBRARIES"
                metadata["amd_build_inputs"][key] = ";".join(reversed(metadata["amd_build_inputs"][key].split(";")))
            if self.corrupt == "aocl-sparse-order":
                key = "AOCLSparse_INTERFACE_LINK_LIBRARIES"
                metadata["amd_build_inputs"][key] = ";".join(reversed(metadata["amd_build_inputs"][key].split(";")))
            if self.corrupt == "aocl-sparse-dependency":
                metadata["dependencies"].pop("aocl_sparse", None)
            if self.corrupt == "aocl-sparse-header":
                metadata["dependencies"].pop("aocl_sparse_header", None)
            if self.corrupt == "aocl-sparse-extra-order":
                key = "AOCLSparse_INTERFACE_LINK_LIBRARIES"
                parts = metadata["amd_build_inputs"][key].split(";")
                parts[1:3] = reversed(parts[1:3])
                metadata["amd_build_inputs"][key] = ";".join(parts)
        if profile == "openmp":
            wrapper = "amd/c-cpp/rocthrust/examples/rocthrust_wrapper.hip"
            metadata["rocthrust_interop"] = {"source": wrapper, "source_sha256": build.file_identity(t.source / wrapper)["sha256"],
                "compiler_language": "hip", "compiler_path": str(t.args.amd_cxx_compiler), "compile_flags": "-O2",
                "caller_language": "cxx", "caller_compile_flags": "-fopenmp -target", "link_flags": "-fopenmp -target",
                "linker_language": "cxx", "linker_path": str(t.args.amd_cxx_compiler)}
            if self.corrupt == "wrapper":
                metadata["rocthrust_interop"]["source_sha256"] = "0" * 64
        if self.corrupt == "dependency" and profile == "hip":
            del metadata["dependencies"]["hip"]
        save(directory / "build-metadata.json", metadata)
        if self.corrupt == "existing-manifest" and profile == "cpu":
            (directory / "partial-manifest.json").write_bytes(b"prior evidence must not be overwritten\n")
        (directory / "CMakeCache.txt").write_text("fixture:BOOL=ON\n")
        save(directory / "compile_commands.json", [])

    def manifest(self, directory, descriptor):
        manifest = generator.build_manifest(descriptor)
        entries = manifest["entries"]
        if self.corrupt == "binary" and directory.name == "build-hip":
            entries[0]["binary_sha256"] = "0" * 64
            entries[0]["artifact_id"] = artifact_id(entries[0])
        if self.corrupt == "role" and directory.name == "build-hip":
            entries[0]["executable_role"] = "benchmark"
        save(directory / "partial-manifest.json", manifest)
        if self.corrupt == "source" and directory.name == "build-cpu":
            (self.test.source / "source.txt").write_text("Changed during CPU build")


class SiriusAmdBuildTests(unittest.TestCase):
    def setUp(self):
        base = Path("/private/tmp/gpu-library-suite-local-build")
        if not base.parent.exists():
            base = Path("/tmp/gpu-library-suite-local-build")
        base.mkdir(exist_ok=True)
        requested = Path(os.environ.get("TMPDIR", str(base))).resolve()
        if requested == base.resolve() or base.resolve() in requested.parents:
            base = requested
        temporary = tempfile.TemporaryDirectory(prefix="amd-allocated-build-", dir=base)
        self.addCleanup(temporary.cleanup)
        old_sys_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), old_sys_path))
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "source"
        self.run = self.source / "runs/fake-run"
        self.run.mkdir(parents=True)
        self.source_paths = ["source.txt", "amd/c-cpp/rocthrust/examples/rocthrust_wrapper.hip"]
        for relative in self.source_paths:
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic source " + relative)
        self.tools = {}
        for name in ("cc", "c++", "amdclang++", "git", "cmake"):
            path = self.root / "tools" / name
            path.parent.mkdir(exist_ok=True)
            path.write_text("Fake executable; never run")
            path.chmod(0o700)
            self.tools[name] = path
        self.rocm = self.root / "rocm"
        self.rocm.mkdir()
        self.argv = ["--source-root", str(self.source), "--run-root", str(self.run), "--run-id", "fake-run",
                     "--expected-source-sha256", source_snapshot_sha256(self.source, self.source_paths),
                     "--gpu-arch", "gfx942:sramecc+:xnack-", "--cpu-c-compiler", str(self.tools["cc"]),
                     "--cpu-cxx-compiler", str(self.tools["c++"]), "--amd-cxx-compiler", str(self.tools["amdclang++"]),
                     "--rocm-root", str(self.rocm)]
        self.args = build.parser().parse_args(self.argv)
        self.env = {"GPU_SUITE_SIRIUS_AMD_BUILD_APPROVED": "yes", "PBS_JOBID": "123.pbs", "PBS_ENVIRONMENT": "PBS_BATCH", "PATH": str(self.root / "tools")}

    def which(self, name, **kwargs):
        return str(self.tools[name]) if name in self.tools else None

    def execute(self, **kwargs):
        fake = FakeProcesses(self, **kwargs)
        job = build.Build(self.args, env=self.env, popen=fake)
        with mock.patch.object(build.shutil, "which", side_effect=self.which):
            build.execute(job)
        return job, fake

    def artifact_builds(self, fake):
        names = {row["target"] for row in build.slots()}
        return [argv for argv in fake.calls if "--build" in argv and argv[-1] in names]

    def test_missing_allocation_approval_and_unsafe_paths_start_no_process(self):
        cases = [({}, "approval"), ({"GPU_SUITE_SIRIUS_AMD_BUILD_APPROVED": "yes"}, "PBS_JOBID"),
                 (dict(self.env, PBS_ENVIRONMENT="PBS_INTERACTIVE"), "batch")]
        with mock.patch.object(build.subprocess, "Popen") as popen:
            for env, text in cases:
                with self.subTest(text=text), self.assertRaisesRegex(ValueError, text):
                    build.Build(self.args, env=env)
            wrong = build.parser().parse_args(self.argv)
            wrong.run_root = self.root
            with self.assertRaisesRegex(ValueError, "Run root"):
                build.Build(wrong, env=self.env)
            popen.assert_not_called()
        self.assertFalse((self.run / "amd-build").exists())

    def test_existing_attempt_and_symlink_guard_preserve_protected_files(self):
        protected = self.run / "amd-build"
        protected.mkdir()
        marker = protected / "summary.json"
        marker.write_bytes(b"previous evidence")
        with mock.patch.object(build.subprocess, "Popen") as popen:
            with self.assertRaises(FileExistsError):
                build.Build(self.args, env=self.env)
            self.assertEqual(marker.read_bytes(), b"previous evidence")
            with mock.patch.object(Path, "is_symlink", return_value=True):
                with self.assertRaisesRegex(ValueError, "canonical"):
                    build.validate(self.args, self.env)
            popen.assert_not_called()

    def test_bound_snapshot_mismatch_stops_before_configure(self):
        self.args.expected_source_sha256 = "0" * 64
        fake = FakeProcesses(self)
        job = build.Build(self.args, env=self.env, popen=fake)
        with mock.patch.object(build.shutil, "which", side_effect=self.which), self.assertRaisesRegex(ValueError, "approved snapshot"):
            build.execute(job)
        self.assertTrue(fake.calls)
        self.assertTrue(all(argv[0] == str(self.tools["git"]) for argv in fake.calls))
        self.assertTrue(all(row["status"] == "NOT_ATTEMPTED" for row in job.slots))

    def test_snapshot_bootstrap_matches_common_without_importing_unapproved_source(self):
        path = self.source / "executable-入力.txt"
        path.write_bytes(b"content\x00with exact bytes\n")
        path.chmod(0o700)
        paths = self.source_paths + [path.name]
        self.assertEqual(build.source_snapshot_sha256(self.source, reversed(paths)), source_snapshot_sha256(self.source, paths))
        unverified = self.source / "tools/gpu_suite/hashing.py"
        unverified.parent.mkdir(parents=True)
        unverified.write_text("raise AssertionError('unapproved module must never execute')\n")
        self.source_paths.append("tools/gpu_suite/hashing.py")
        original_import = builtins.__import__
        imports = []
        def guarded_import(name, *args, **kwargs):
            if name.startswith("gpu_suite"):
                imports.append(name)
                raise AssertionError("Repository import before source hash approval")
            return original_import(name, *args, **kwargs)
        fake = FakeProcesses(self)
        job = build.Build(self.args, env=self.env, popen=fake)
        with mock.patch.object(build.shutil, "which", side_effect=self.which), mock.patch.object(builtins, "__import__", side_effect=guarded_import), self.assertRaisesRegex(ValueError, "approved snapshot"):
            build.execute(job)
        self.assertEqual(imports, [])
        self.assertTrue(all(argv[0] == str(self.tools["git"]) for argv in fake.calls))

    def test_all_36_builds_record_artifacts_without_executing_them(self):
        job, fake = self.execute()
        self.assertEqual(len(job.slots), 36)
        self.assertTrue(all(row["status"] == "BUILT" for row in job.slots))
        commands = self.artifact_builds(fake)
        self.assertEqual(len(commands), 36)
        for argv in commands:
            self.assertEqual(argv[argv.index("--parallel") + 1], "1")
            self.assertEqual(len(argv[argv.index("--target") + 1:]), 1)
            self.assertNotIn("gpu_suite_partial_manifest", argv)
        self.assertEqual(sum(target == "gpu_suite_common" for _, target in fake.build_attempts), 3)
        self.assertEqual(sum(target == "rocthrust_openmp_wrapper" for _, target in fake.build_attempts), 1)
        self.assertTrue(all("--version" in argv or argv[0] in {str(self.tools["git"]), str(self.tools["cmake"]), sys.executable} for argv in fake.calls))
        self.assertTrue(all(not os.access(item["identity"]["path"], os.X_OK) for profile in job.profiles.values() for item in profile["artifacts"].values()))
        for item in job.commands:
            self.assertEqual(item["exit_status"], 0)
            self.assertEqual(build.file_identity(item["stdout"])["sha256"], item["stdout_sha256"])
            self.assertEqual(build.file_identity(item["stderr"])["sha256"], item["stderr_sha256"])
        self.assertEqual(build.load_json(job.root / "source-before.json")["source_snapshot_sha256"],
                         build.load_json(job.root / "source-after.json")["source_snapshot_sha256"])

    def test_missing_cpu_providers_are_explicit_without_reference_fallback(self):
        job, fake = self.execute(missing_cpu=("rocblas", "rocsolver", "rocsparse"))
        counts = {status: sum(row["status"] == status for row in job.slots) for status in ("BUILT", "NOT_BUILT_DEPENDENCY")}
        self.assertEqual(counts, {"BUILT": 30, "NOT_BUILT_DEPENDENCY": 6})
        self.assertTrue(all("ONEMKL provider" in row["reason"] for row in job.slots if row["status"] == "NOT_BUILT_DEPENDENCY"))
        for argv in fake.calls:
            self.assertFalse(any("REFERENCE" in value for value in argv))
        cpu_command = next(argv for argv in fake.calls if "--build" in argv and argv[2].endswith("build-cpu"))
        self.assertNotIn("solver_cpu", cpu_command)

    def test_configure_failure_keeps_other_profiles_and_stderr(self):
        job, fake = self.execute(fail_profile="cpu")
        self.assertEqual(sum(row["status"] == "BUILT" for row in job.slots), 24)
        self.assertEqual(sum(row["status"] == "FAILED" for row in job.slots), 12)
        failed = next(item for item in job.commands if item["stage"] == "configure-cpu")
        self.assertEqual(failed["exit_status"], 1)
        self.assertIn("compile/link failure", Path(failed["stderr"]).read_text())
        self.assertEqual(len(self.artifact_builds(fake)), 24)

    def test_binary_hash_mismatch_rejects_profile_without_retry(self):
        job, fake = self.execute(corrupt="binary")
        self.assertEqual(sum(row["status"] == "FAILED" for row in job.slots), 12)
        self.assertIn("binary hash mismatch", job.profiles["hip"]["error"])
        self.assertEqual(len(self.artifact_builds(fake)), 36)

    def test_failed_early_target_preserves_verified_successes_and_continues_once(self):
        job, fake = self.execute(fail_targets=("blas_cpu_bench",))
        self.assertEqual(len(job.slots), 36)
        self.assertEqual(sum(row["status"] == "BUILT" for row in job.slots), 35)
        failed = next(row for row in job.slots if row["target"] == "blas_cpu_bench")
        self.assertEqual((failed["status"], failed["compile_status"], failed["failure_stage"]), ("FAILED", "FAILED", "COMPILE_LINK"))
        command = job.commands[failed["command_index"] - 1]
        self.assertEqual(command["exit_status"], 1)
        self.assertIn("fixture compile failure", Path(command["stderr"]).read_text())
        cpu = job.root / "build-cpu"
        self.assertTrue((cpu / "amd/c-cpp/rocblas/blas_cpu_bench").is_file())
        original = build.load_json(cpu / "partial-manifest-input.json")
        subset = build.load_json(cpu / "built-manifest-input.json")
        manifest = build.load_json(cpu / "partial-manifest.json")
        self.assertEqual(len(original["entries"]), 12)
        self.assertEqual(len(subset["entries"]), 11)
        self.assertEqual({entry["target_name"] for entry in subset["entries"]}, {entry["target_name"] for entry in manifest["entries"]})
        self.assertNotIn("blas_cpu_bench", job.profiles["cpu"]["artifacts"])
        self.assertIn("solver_cpu_bench", job.profiles["cpu"]["artifacts"])
        self.assertEqual(fake.build_attempts.count(("build-cpu", "blas_cpu_bench")), 1)
        self.assertEqual(len(self.artifact_builds(fake)), 36)

    def test_common_dependency_failure_is_not_retried_and_examples_continue(self):
        job, fake = self.execute(fail_targets=("gpu_suite_common",))
        self.assertEqual(sum(row["status"] == "BUILT" for row in job.slots), 18)
        for row in job.slots:
            if row["role"] == "benchmark":
                self.assertEqual((row["status"], row["compile_status"], row["failure_stage"]), ("NOT_ATTEMPTED", "NOT_ATTEMPTED", "SHARED_DEPENDENCY"))
                self.assertNotIn("command_index", row)
        for profile in build.PROFILES:
            self.assertEqual(fake.build_attempts.count(("build-" + profile, "gpu_suite_common")), 1)
            self.assertEqual(len(job.profiles[profile]["artifacts"]), 6)
        self.assertEqual(len(self.artifact_builds(fake)), 18)

    def test_wrapper_dependency_failure_blocks_both_consumers_without_retry(self):
        job, fake = self.execute(fail_targets=("rocthrust_openmp_wrapper",))
        self.assertEqual(sum(row["status"] == "BUILT" for row in job.slots), 34)
        self.assertEqual(fake.build_attempts.count(("build-openmp", "rocthrust_openmp_wrapper")), 1)
        for name in ("openmp_rocthrust", "openmp_rocthrust_bench"):
            self.assertNotIn(("build-openmp", name), fake.build_attempts)
            row = next(row for row in job.slots if row["target"] == name)
            self.assertEqual((row["status"], row["failure_stage"]), ("NOT_ATTEMPTED", "SHARED_DEPENDENCY"))

    def test_manifest_generation_failure_never_certifies_compiled_targets(self):
        job, fake = self.execute(fail_targets=("blas_cpu_bench",), fail_manifest="cpu")
        self.assertEqual(sum(row["status"] == "BUILT" for row in job.slots), 24)
        for row in job.slots:
            if row["profile"] == "cpu" and row["target"] != "blas_cpu_bench":
                self.assertEqual((row["status"], row["compile_status"], row["failure_stage"]), ("FAILED", "SUCCEEDED", "MANIFEST_GENERATION"))
        self.assertEqual(len(self.artifact_builds(fake)), 36)
        self.assertNotIn("artifacts", job.profiles["cpu"])

    def test_manifest_zero_exit_without_output_does_not_certify_compiled_targets(self):
        job, _ = self.execute(corrupt="missing-manifest")
        self.assertEqual(sum(row["status"] == "BUILT" for row in job.slots), 24)
        for row in job.slots:
            if row["profile"] == "cpu":
                self.assertEqual((row["status"], row["compile_status"], row["failure_stage"]), ("FAILED", "SUCCEEDED", "ARTIFACT_VERIFICATION"))
        self.assertNotIn("artifacts", job.profiles["cpu"])

    def test_existing_manifest_is_preserved_without_generator_overwrite(self):
        job, fake = self.execute(corrupt="existing-manifest")
        self.assertEqual(sum(row["status"] == "BUILT" for row in job.slots), 24)
        self.assertEqual((job.root / "build-cpu/partial-manifest.json").read_bytes(), b"prior evidence must not be overwritten\n")
        self.assertIn("Refusing existing subset manifest", job.profiles["cpu"]["error"])
        self.assertFalse(any(argv[:2] == [sys.executable, "-B"] and "build-cpu" in argv[-1] for argv in fake.calls))

    def test_changed_configured_metadata_stops_profile_without_rebinding_binaries(self):
        job, fake = self.execute(corrupt="metadata-during-build")
        cpu = [row for row in job.slots if row["profile"] == "cpu"]
        self.assertEqual(sum(row["status"] == "BUILT" for row in job.slots), 24)
        self.assertEqual(sum(row["status"] == "NOT_ATTEMPTED" for row in cpu), 11)
        self.assertEqual(cpu[0]["failure_stage"], "PROFILE_PROVENANCE")
        self.assertIn("Configured provenance changed", job.profiles["cpu"]["error"])
        self.assertEqual(sum(directory == "build-cpu" for directory, _ in fake.build_attempts), 1)
        self.assertFalse((job.root / "build-cpu/partial-manifest.json").exists())

    def test_benchmark_only_success_still_requires_rocthrust_interop_proof(self):
        job, _ = self.execute(fail_targets=("openmp_rocthrust",), corrupt="wrapper")
        self.assertIn("translation-unit provenance", job.profiles["openmp"]["error"])
        self.assertTrue(all(row["status"] == "FAILED" for row in job.slots if row["profile"] == "openmp"))

    def test_benchmark_only_success_still_requires_aocl_ordered_dependency_proof(self):
        self.opt_in_aocl(self.aocl_fixture())
        job, _ = self.execute(fail_targets=("blas_cpu", "solver_cpu"), corrupt="aocl-order")
        self.assertIn("selected link order", job.profiles["cpu"]["error"])
        self.assertTrue(all(row["status"] == "FAILED" for row in job.slots if row["profile"] == "cpu"))

    def test_profile_build_timeout_is_shared_and_unstarted_targets_are_not_failed_compiles(self):
        clock = mock.Mock(value=0.0)
        clock.side_effect = lambda: clock.value
        self.args.build_timeout = 3
        fake = FakeProcesses(self, durations={"blas_cpu": 1, "gpu_suite_common": 1, "blas_cpu_bench": 20}, clock=clock)
        job = build.Build(self.args, env=self.env, popen=fake, clock=clock)
        with mock.patch.object(build.shutil, "which", side_effect=self.which), mock.patch.object(build.os, "killpg") as kill:
            build.execute(job)
        kill.assert_called_once_with(fake.pid, signal.SIGTERM)
        cpu = [row for row in job.slots if row["profile"] == "cpu"]
        timed_out = next(row for row in cpu if row["target"] == "blas_cpu_bench")
        self.assertEqual(timed_out["compile_status"], "TIMED_OUT")
        self.assertEqual(job.commands[timed_out["command_index"] - 1]["effective_timeout_seconds"], 1)
        self.assertEqual(sum(row["status"] == "NOT_ATTEMPTED" for row in cpu), 10)
        self.assertEqual(sum(row["status"] == "BUILT" for row in job.slots), 24)
        self.assertEqual(sum(directory == "build-cpu" for directory, _ in fake.build_attempts), 3)
        self.assertFalse((job.root / "build-cpu/partial-manifest.json").exists())

    def test_global_deadline_launches_no_further_process_and_requires_final_source_proof(self):
        clock = mock.Mock(value=0.0)
        clock.side_effect = lambda: clock.value
        fake = FakeProcesses(self, durations={"blas_cpu": 2, "gpu_suite_common": 2, "blas_cpu_bench": 99}, clock=clock)
        original_build = build.Build
        def create(args):
            return original_build(args, clock=clock)
        argv = self.argv + ["--timeout-seconds", "30", "--configure-timeout", "20", "--build-timeout", "30"]
        with mock.patch.dict(os.environ, self.env, clear=True), mock.patch.object(build, "Build", side_effect=create), mock.patch.object(build.shutil, "which", side_effect=self.which), mock.patch.object(build.subprocess, "Popen", side_effect=fake), mock.patch.object(build.os, "killpg"), redirect_stdout(io.StringIO()):
            code = build.main(argv)
        summary = build.load_json(self.run / "amd-build/summary.json")
        self.assertEqual(code, 1)
        self.assertEqual(clock.value, 30)
        self.assertIn("deadline exhausted", summary["error"])
        self.assertEqual(len(summary["slots"]), 36)
        self.assertFalse(any(row["status"] == "BUILT" for row in summary["slots"]))
        self.assertEqual(fake.calls[-1][-1], "blas_cpu_bench")
        self.assertEqual(len([argv for argv in fake.calls if "-S" in argv]), 1)

    def test_manifest_completion_at_profile_deadline_cannot_certify_artifacts(self):
        clock = mock.Mock(value=0.0)
        clock.side_effect = lambda: clock.value
        self.args.build_timeout = 3
        fake = FakeProcesses(self, durations={"manifest-cpu": 3}, clock=clock)
        job = build.Build(self.args, env=self.env, popen=fake, clock=clock)
        with mock.patch.object(build.shutil, "which", side_effect=self.which):
            build.execute(job)
        self.assertTrue((job.root / "build-cpu/partial-manifest.json").is_file())
        manifest = next(item for item in job.commands if item["stage"] == "manifest-cpu")
        self.assertEqual((manifest["exit_status"], manifest["timed_out"]), (0, False))
        self.assertEqual(sum(row["status"] == "BUILT" for row in job.slots), 24)
        for row in job.slots:
            if row["profile"] == "cpu":
                self.assertEqual((row["status"], row["compile_status"], row["failure_stage"]), ("FAILED", "SUCCEEDED", "ARTIFACT_VERIFICATION"))
        self.assertIn("deadline exhausted before artifact verification", job.profiles["cpu"]["error"])
        self.assertNotIn("artifacts", job.profiles["cpu"])

    def test_rocthrust_two_translation_unit_hash_mismatch_is_rejected(self):
        job, _ = self.execute(corrupt="wrapper")
        self.assertIn("translation-unit provenance", job.profiles["openmp"]["error"])
        self.assertTrue(all(row["status"] == "FAILED" for row in job.slots if row["profile"] == "openmp"))

    def test_missing_dependency_identity_rejects_manifest_even_with_matching_hash(self):
        job, _ = self.execute(corrupt="dependency")
        self.assertIn("dependency identities are missing", job.profiles["hip"]["error"])
        self.assertEqual(sum(row["status"] == "FAILED" for row in job.slots), 12)

    def test_source_change_invalidates_built_artifacts_and_preserves_unattempted_slots(self):
        fake = FakeProcesses(self, corrupt="source")
        with mock.patch.dict(os.environ, self.env, clear=True), mock.patch.object(build.shutil, "which", side_effect=self.which), mock.patch.object(build.subprocess, "Popen", side_effect=fake), redirect_stdout(io.StringIO()):
            code = build.main(self.argv)
        summary = build.load_json(self.run / "amd-build/summary.json")
        self.assertEqual(code, 1)
        self.assertEqual(summary["counts"], {"FAILED": 12, "NOT_ATTEMPTED": 24})
        self.assertIn("Source changed", summary["error"])
        self.assertEqual(len(self.artifact_builds(fake)), 12)

    def test_timeout_stops_only_owned_child_and_exhausted_deadline_does_not_launch(self):
        process = mock.Mock(pid=424242)
        process.wait.side_effect = [subprocess.TimeoutExpired("fake", 1), 0]
        popen = mock.Mock(return_value=process)
        job = build.Build(self.args, env=self.env, popen=popen)
        with mock.patch.object(build.os, "killpg") as kill:
            with self.assertRaisesRegex(build.CommandFailure, "timeout"):
                job.run("fake-timeout", ["fake-owned-compiler"], timeout=1)
            kill.assert_called_once_with(424242, signal.SIGTERM)
        self.assertIsNone(job.commands[0]["exit_status"])
        self.assertIn("no retry", job.commands[0]["error"])
        job.deadline = job.clock() - 1
        with self.assertRaisesRegex(build.CommandFailure, "deadline"):
            job.run("not-started", ["fake-owned-compiler"])
        self.assertEqual(popen.call_count, 1)

    def test_provider_inventory_is_bounded_nonrecursive_and_never_selects(self):
        provider = self.root / "aocl"
        (provider / "include/nested").mkdir(parents=True)
        (provider / "lib_LP64").mkdir()
        (provider / "include/cblas.h").write_text("cblas")
        (provider / "lib_LP64/libblis.so").write_text("blis")
        (provider / "include/nested/lapacke.h").write_text("must not recurse")
        (provider / "include/unrelated.h").write_text("unrelated")
        with mock.patch.object(build.subprocess, "Popen") as popen:
            result = build.provider_inventory({"AOCL_ROOT": str(provider)}, float("inf"))
            popen.assert_not_called()
        self.assertFalse(result["automatic_provider_selection"])
        self.assertFalse(result["recursive_search"])
        self.assertEqual({Path(row["path"]).name for row in result["candidates"]}, {"cblas.h", "libblis.so"})
        self.assertTrue(all(row["hash_status"] == "HASHED" for row in result["candidates"]))

    def test_explicit_provider_options_allow_only_corresponding_existing_inputs(self):
        library = self.root / "libblis.so"
        library.write_text("fake")
        extra = self.root / "libomp.so"
        extra.write_text("fake extra")
        self.args.cpu_cmake_option = ["GPU_SUITE_GENERIC_CBLAS_LIBRARY=" + str(library)]
        with self.assertRaisesRegex(ValueError, "corresponding"):
            build.cpu_options(self.args)
        self.args.cpu_blas_backend = "GENERIC_CBLAS"
        self.args.cpu_cmake_option.append("GPU_SUITE_GENERIC_CBLAS_EXTRA_LIBRARIES=" + str(extra) + ";" + str(library))
        self.assertEqual(build.cpu_options(self.args)["GPU_SUITE_GENERIC_CBLAS_EXTRA_LIBRARIES"], str(extra) + ";" + str(library))
        self.args.cpu_cmake_option.append("CMAKE_PROJECT_INCLUDE=/unapproved/script.cmake")
        with self.assertRaisesRegex(ValueError, "Invalid"):
            build.cpu_options(self.args)

    def aocl_fixture(self, name="aocl"):
        root = self.root / name
        for relative in ("include_LP64/cblas.h", "include_LP64/lapacke.h", "lib_LP64/libblis.so", "lib_LP64/libflame.so"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Synthetic AOCL candidate " + relative)
        return root

    def opt_in_aocl(self, root):
        self.args.aocl_cpu_candidates = True
        self.args.cpu_blas_backend = "GENERIC_CBLAS"
        self.args.cpu_lapack_backend = "GENERIC_LAPACKE"
        self.env["AOCL_ROOT"] = str(root)

    def select_aocl(self, root, sparse_backend="ONEMKL"):
        return build.select_aocl_candidates({"AOCL_ROOT": str(root)}, self.run / "unavailable", float("inf"),
                                            sparse_backend=sparse_backend)

    def sparse_fixture(self, name="aocl-sparse", missing=()):
        root = self.aocl_fixture(name)
        for relative in ("include_LP64/aoclsparse.h", "lib_LP64/libaoclsparse.so"):
            if relative not in missing:
                (root / relative).write_text("Synthetic AOCL Sparse candidate " + relative)
        return root

    def test_aocl_sparse_requires_explicit_backend_and_preserves_default(self):
        self.assertEqual(self.args.cpu_sparse_backend, "ONEMKL")
        root = self.sparse_fixture()
        self.assertNotIn("AOCLSPARSE", self.select_aocl(root)["providers"])
        args = build.parser().parse_args(self.argv + ["--cpu-sparse-backend", "AOCLSPARSE"])
        self.assertEqual(build.cpu_options(args)["GPU_SUITE_CPU_SPARSE_BACKEND"], "AOCLSPARSE")
        with redirect_stdout(io.StringIO()), mock.patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            build.parser().parse_args(self.argv + ["--cpu-sparse-backend", "REFERENCE"])

    def test_aocl_sparse_explicit_inputs_preserve_order_and_reject_wrong_backend(self):
        root = self.sparse_fixture()
        extra = root / "lib_LP64/libextra.so"
        extra.write_text("Synthetic explicit link input")
        options = ["GPU_SUITE_AOCL_SPARSE_INCLUDE_DIR=" + str(root / "include_LP64"),
                   "GPU_SUITE_AOCL_SPARSE_LIBRARY=" + str(root / "lib_LP64/libaoclsparse.so"),
                   "GPU_SUITE_AOCL_SPARSE_EXTRA_LIBRARIES=" + str(extra)]
        self.args.cpu_cmake_option = options
        with self.assertRaisesRegex(ValueError, "explicit AOCLSPARSE"):
            build.cpu_options(self.args)
        self.args.cpu_sparse_backend = "AOCLSPARSE"
        result = build.cpu_options(self.args)
        self.assertEqual(result["GPU_SUITE_AOCL_SPARSE_EXTRA_LIBRARIES"], str(extra))
        for value in ("-lblis", "relative.so", str(root / "missing.so"), str(root / "lib_LP64"), str(extra) + ";"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.args.cpu_cmake_option = options[:2] + ["GPU_SUITE_AOCL_SPARSE_EXTRA_LIBRARIES=" + value]
                build.cpu_options(self.args)

    def test_aocl_sparse_candidates_pin_header_library_and_no_implicit_extras(self):
        root = self.sparse_fixture()
        with mock.patch.object(build.subprocess, "Popen") as process:
            selection = self.select_aocl(root, "AOCLSPARSE")
            process.assert_not_called()
        record = selection["providers"]["AOCLSPARSE"]
        self.assertEqual(record["status"], "CANDIDATE_SELECTED")
        self.assertEqual(record["header"]["sha256"], build.file_identity(root / "include_LP64/aoclsparse.h")["sha256"])
        self.assertEqual([item["resolved_path"] for item in record["ordered_libraries"]], [str(root / "lib_LP64/libaoclsparse.so")])
        self.assertEqual(record["cmake_options"]["GPU_SUITE_AOCL_SPARSE_EXTRA_LIBRARIES"], "")
        self.assertEqual(build.validate_aocl_selection(selection)["AOCLSPARSE"]["status"], "NOT_ENABLED_BY_CMAKE")

    def test_aocl_sparse_manual_extra_link_order_is_verified(self):
        root = self.sparse_fixture()
        extras = [root / "lib_LP64/libextra1.so", root / "lib_LP64/libextra2.so"]
        for extra in extras:
            extra.write_text(extra.name)
        self.args.cpu_sparse_backend = "AOCLSPARSE"
        self.args.cpu_cmake_option = [
            "GPU_SUITE_AOCL_SPARSE_INCLUDE_DIR=" + str(root / "include_LP64"),
            "GPU_SUITE_AOCL_SPARSE_LIBRARY=" + str(root / "lib_LP64/libaoclsparse.so"),
            "GPU_SUITE_AOCL_SPARSE_EXTRA_LIBRARIES=" + ";".join(map(str, extras))]
        for corruption in (None, "aocl-sparse-extra-order"):
            with self.subTest(corruption=corruption):
                self.args.run_id = str(corruption or "manual-good")
                self.args.run_root = self.source / "runs" / self.args.run_id
                self.args.run_root.mkdir()
                job, _ = self.execute(corrupt=corruption)
                if corruption:
                    self.assertEqual(sum(row["status"] == "BUILT" for row in job.slots), 24)
                    self.assertIn("extra link order", job.profiles["cpu"]["error"])
                else:
                    self.assertEqual(sum(row["status"] == "BUILT" for row in job.slots), 36)
                    dependencies = job.profiles["cpu"]["dependencies"]
                    self.assertEqual(dependencies["aocl_sparse_link_1"]["resolved_path"], str(extras[0]))
                    self.assertEqual(dependencies["aocl_sparse_link_2"]["resolved_path"], str(extras[1]))

    def test_aocl_sparse_missing_ambiguous_escape_and_ilp64_reject_without_fallback(self):
        original_resolve = Path.resolve
        for case in ("missing-header", "missing-library", "ambiguous-header", "ambiguous-library", "escape", "ilp64"):
            with self.subTest(case=case):
                missing = {"include_LP64/aoclsparse.h"} if case == "missing-header" else {"lib_LP64/libaoclsparse.so"} if case == "missing-library" else set()
                root = self.sparse_fixture(case, missing=missing)
                aliases = {}
                if case in {"ambiguous-header", "ambiguous-library"}:
                    relative = "include/aoclsparse.h" if case == "ambiguous-header" else "lib/libaoclsparse.so"
                    (root / relative).parent.mkdir()
                    (root / relative).write_text("distinct candidate")
                elif case in {"escape", "ilp64"}:
                    destination = self.root / "outside-sparse.so" if case == "escape" else root / "lib_ILP64/libaoclsparse.so"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text("invalid candidate")
                    aliases[root / "lib_LP64/libaoclsparse.so"] = destination
                def resolve(path, *args, **kwargs):
                    return original_resolve(aliases.get(path, path), *args, **kwargs)
                with mock.patch.object(Path, "resolve", resolve):
                    record = self.select_aocl(root, "AOCLSPARSE")["providers"]["AOCLSPARSE"]
                self.assertEqual(record["status"], "REJECTED")
                self.assertTrue(record["rejection_reasons"])
                self.assertEqual(record["ordered_libraries"], [])
                self.assertTrue(all(not Path(value).exists() for value in record["cmake_options"].values() if value))

    def test_aocl_sparse_builds_all_36_and_binds_selected_provenance(self):
        root = self.sparse_fixture()
        self.opt_in_aocl(root)
        self.args.cpu_sparse_backend = "AOCLSPARSE"
        job, fake = self.execute()
        self.assertEqual(sum(row["status"] == "BUILT" for row in job.slots), 36)
        self.assertEqual(len(self.artifact_builds(fake)), 36)
        self.assertEqual(job.profiles["cpu"]["aocl_cpu_provider_validation"]["AOCLSPARSE"]["status"], "API_COMPILE_LINK_PASSED")
        for name in ("sparse_cpu", "sparse_cpu_bench"):
            self.assertEqual(job.profiles["cpu"]["artifacts"][name]["manifest_entry"]["supported_cpu_backends"], ["cpu-aocl-sparse"])
        self.assertIn("aocl_sparse", job.profiles["cpu"]["dependencies"])
        self.assertEqual(job.profiles["cpu"]["dependencies"]["aocl_sparse_header"]["resolved_path"], str(root / "include_LP64/aoclsparse.h"))
        self.assertNotIn("mkl_sparse", job.profiles["cpu"]["dependencies"])
        selection = build.load_json(job.root / "cpu-provider-selection.json")
        self.assertEqual(selection, job.aocl_selection)
        for argv in fake.calls:
            if "-S" in argv:
                self.assertIn("-DGPU_SUITE_CPU_SPARSE_BACKEND=AOCLSPARSE", argv)

    def test_aocl_sparse_missing_candidate_records_34_without_reference_substitution(self):
        self.opt_in_aocl(self.sparse_fixture(missing={"lib_LP64/libaoclsparse.so"}))
        self.args.cpu_sparse_backend = "AOCLSPARSE"
        job, fake = self.execute()
        self.assertEqual(sum(row["status"] == "BUILT" for row in job.slots), 34)
        self.assertEqual(sum(row["status"] == "NOT_BUILT_DEPENDENCY" for row in job.slots), 2)
        for row in job.slots:
            if row["profile"] == "cpu" and row["library"] == "rocsparse":
                self.assertEqual(row["compile_status"], "NOT_ATTEMPTED")
                self.assertIn("AOCL selection", row["reason"])
        self.assertFalse(any("REFERENCE" in arg for argv in fake.calls for arg in argv))

    def test_aocl_sparse_changed_library_rejects_identity(self):
        root = self.sparse_fixture()
        selection = self.select_aocl(root, "AOCLSPARSE")
        (root / "lib_LP64/libaoclsparse.so").write_text("changed library")
        with self.assertRaisesRegex(ValueError, "Selected AOCL input changed"):
            build.validate_aocl_selection(selection)

    def test_aocl_sparse_link_order_and_missing_identity_invalidate_cpu_artifacts(self):
        for corruption in ("aocl-sparse-order", "aocl-sparse-dependency", "aocl-sparse-header"):
            with self.subTest(corruption=corruption):
                root = self.sparse_fixture(corruption)
                self.opt_in_aocl(root)
                self.args.cpu_sparse_backend = "AOCLSPARSE"
                self.args.run_id = corruption
                self.args.run_root = self.source / "runs" / corruption
                self.args.run_root.mkdir()
                job, _ = self.execute(corrupt=corruption)
                self.assertEqual(sum(row["status"] == "BUILT" for row in job.slots), 24)
                self.assertTrue(all(row["status"] == "FAILED" for row in job.slots if row["profile"] == "cpu"))
                self.assertIn("link order" if corruption.endswith("order") else "dependency identities", job.profiles["cpu"]["error"])

    def test_aocl_candidate_mode_requires_explicit_generics_without_manual_mix(self):
        self.args.aocl_cpu_candidates = True
        with self.assertRaisesRegex(ValueError, "both explicit GENERIC"):
            build.cpu_options(self.args)
        self.opt_in_aocl(self.aocl_fixture())
        self.assertEqual(build.cpu_options(self.args)["GPU_SUITE_CPU_SPARSE_BACKEND"], "ONEMKL")
        self.args.cpu_cmake_option = ["GPU_SUITE_GENERIC_CBLAS_LIBRARY=" + str(self.root / "anything")]
        with self.assertRaisesRegex(ValueError, "no manual"):
            build.cpu_options(self.args)

    def test_aocl_unique_candidates_deduplicate_aliases_and_record_ordered_hashes(self):
        root = self.aocl_fixture()
        aliases = {}
        original_resolve = Path.resolve
        for directory, real_directory, names in (("include", "include_LP64", ("cblas.h", "lapacke.h")),
                                                 ("lib", "lib_LP64", ("libblis.so", "libflame.so"))):
            (root / directory).mkdir()
            aliases[root / directory] = root / real_directory
            for name in names:
                alias, real = root / directory / name, root / real_directory / name
                alias.write_bytes(real.read_bytes())
                aliases[alias] = real
        def resolve(path, *args, **kwargs):
            return aliases.get(path, original_resolve(path, *args, **kwargs))
        with mock.patch.object(Path, "resolve", resolve), mock.patch.object(build.subprocess, "Popen") as process:
            result = self.select_aocl(root)
            process.assert_not_called()
        self.assertTrue(all(row["status"] == "CANDIDATE_SELECTED" for row in result["providers"].values()))
        self.assertEqual(len(result["candidates"]["blis"]["files"]), 1)
        self.assertEqual(len(result["candidates"]["blis"]["files"][0]["aliases"]), 2)
        libraries = result["providers"]["LAPACKE"]["ordered_libraries"]
        self.assertEqual([Path(item["resolved_path"]).name for item in libraries], ["libflame.so", "libblis.so"])
        self.assertTrue(all(item["sha256"] == build.file_identity(item["resolved_path"])["sha256"] for item in libraries))
        self.assertEqual(result["cmake_options"]["GPU_SUITE_GENERIC_LAPACKE_EXTRA_LIBRARIES"], str(root / "lib_LP64/libblis.so"))
        self.assertEqual(result["api_validation"], "DEFERRED_TO_CMAKE_COMPILE_LINK")

    def test_aocl_missing_ambiguous_escape_and_ilp64_inputs_reject(self):
        original_resolve = Path.resolve
        for case in ("missing-root", "missing-header", "ambiguous-header", "ambiguous-library", "escape", "ilp64"):
            with self.subTest(case=case):
                root = self.aocl_fixture(case)
                aliases = {}
                if case == "missing-root":
                    root = root / "absent"
                elif case == "missing-header":
                    aliases[root / "include_LP64/cblas.h"] = root / "include_LP64/not-present.h"
                elif case == "ambiguous-header":
                    (root / "include").mkdir()
                    (root / "include/cblas.h").write_text("distinct header")
                    # An alternate regular file within the LP64 directory is
                    # still ambiguous, even if it has identical declarations.
                    extra = root / "include_LP64/other-cblas.h"
                    extra.write_text("distinct header")
                    aliases[root / "include/cblas.h"] = extra
                elif case == "ambiguous-library":
                    (root / "lib_LP64/liblapacke.so").write_text("second LAPACKE provider")
                else:
                    destination = (self.root / "outside.so" if case == "escape" else
                                   root / "lib_ILP64/provider.so")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text("invalid candidate")
                    aliases[root / "lib_LP64/libblis.so"] = destination
                def resolve(path, *args, **kwargs):
                    replacement = aliases.get(path)
                    return original_resolve(replacement if replacement is not None else path, *args, **kwargs)
                with mock.patch.object(Path, "resolve", resolve):
                    result = self.select_aocl(root)
                rejected = [row for row in result["providers"].values() if row["status"] == "REJECTED"]
                self.assertTrue(rejected)
                for row in rejected:
                    self.assertTrue(row["rejection_reasons"])
                    self.assertEqual(row["ordered_libraries"], [])
                    for key, path in row["cmake_options"].items():
                        if path:
                            self.assertTrue(Path(path).is_absolute())
                            self.assertFalse(Path(path).exists())
                            self.assertNotIn("NOTFOUND", path)

    def test_aocl_unmarked_layout_is_recorded_without_claiming_verified_abi(self):
        root = self.root / "unmarked-aocl"
        for relative in ("include/cblas.h", "include/lapacke.h", "lib/libblis.so", "lib/liblapacke.so"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Unmarked synthetic candidate")
        result = self.select_aocl(root)
        self.assertEqual(result["mode"], "AOCL_SHARED_CANDIDATES")
        self.assertEqual(result["abi_validation"], "NOT_VERIFIED_BUILD_ONLY")
        self.assertTrue(all(row["status"] == "CANDIDATE_SELECTED" for row in result["providers"].values()))
        self.assertEqual(result["providers"]["CBLAS"]["ordered_libraries"][0]["observed_layout"], "lib")

    def test_aocl_rejected_candidates_preserve_all_independent_profiles(self):
        self.opt_in_aocl(self.root / "missing-aocl")
        job, fake = self.execute(missing_cpu=("rocsparse",))
        self.assertEqual(sum(row["status"] == "BUILT" for row in job.slots), 30)
        self.assertEqual(sum(row["status"] == "NOT_BUILT_DEPENDENCY" for row in job.slots), 6)
        self.assertTrue(all(row["status"] == "BUILT" for row in job.slots if row["profile"] != "cpu" or row["library"] in {"rocfft", "rocrand", "rocthrust"}))
        for row in job.slots:
            if row["profile"] == "cpu" and row["library"] in {"rocblas", "rocsolver"}:
                self.assertIn("AOCL selection", row["reason"])
        self.assertEqual(len(self.artifact_builds(fake)), 30)
        self.assertFalse(any("REFERENCE" in arg for argv in fake.calls for arg in argv))

    def test_aocl_selected_paths_and_api_validation_are_bound_to_provenance(self):
        root = self.aocl_fixture()
        self.opt_in_aocl(root)
        job, fake = self.execute()
        selection = build.load_json(job.root / "cpu-provider-selection.json")
        self.assertEqual(selection, job.aocl_selection)
        toolchain = build.load_json(job.root / "toolchain.json")
        for key, value in selection["cmake_options"].items():
            self.assertEqual(toolchain["cpu_options"][key], value)
            self.assertTrue(all("-D" + key + "=" + value in argv for argv in fake.calls if "-S" in argv))
        validation = job.profiles["cpu"]["aocl_cpu_provider_validation"]
        self.assertEqual(validation, {"CBLAS": {"status": "API_COMPILE_LINK_PASSED"}, "LAPACKE": {"status": "API_COMPILE_LINK_PASSED"}})
        self.assertEqual(sum(row["status"] == "BUILT" for row in job.slots), 36)

    def test_aocl_api_rejection_does_not_enable_or_substitute_provider(self):
        self.opt_in_aocl(self.aocl_fixture())
        job, _ = self.execute(missing_cpu=("rocsolver", "rocsparse"))
        self.assertEqual(job.aocl_selection["providers"]["LAPACKE"]["status"], "CANDIDATE_SELECTED")
        self.assertEqual(job.profiles["cpu"]["aocl_cpu_provider_validation"]["LAPACKE"]["status"], "NOT_ENABLED_BY_CMAKE")
        self.assertEqual(sum(row["status"] == "BUILT" for row in job.slots), 32)
        self.assertTrue(all(row["status"] == "NOT_BUILT_DEPENDENCY" for row in job.slots if row["profile"] == "cpu" and row["library"] == "rocsolver"))

    def test_aocl_changed_files_and_link_order_are_rejected(self):
        root = self.aocl_fixture()
        selection = self.select_aocl(root)
        library = root / "lib_LP64/libblis.so"
        library.write_text("changed after selection")
        with self.assertRaisesRegex(ValueError, "Selected AOCL input changed"):
            build.validate_aocl_selection(selection)
        self.opt_in_aocl(root)
        job, _ = self.execute(corrupt="aocl-order")
        self.assertIn("selected link order", job.profiles["cpu"]["error"])
        self.assertEqual(sum(row["status"] == "BUILT" for row in job.slots), 24)

    def test_aocl_consumed_header_alias_retarget_is_rejected(self):
        root = self.aocl_fixture()
        header = root / "include_LP64/cblas.h"
        original = root / "include_LP64/cblas-original.h"
        replacement = root / "include_LP64/cblas-replacement.h"
        original.write_bytes(header.read_bytes())
        replacement.write_bytes(header.read_bytes())
        resolve = Path.resolve
        with mock.patch.object(Path, "resolve", lambda path, *a, **kw: original if path == header else resolve(path, *a, **kw)):
            selection = self.select_aocl(root)
        with mock.patch.object(Path, "resolve", lambda path, *a, **kw: replacement if path == header else resolve(path, *a, **kw)):
            with self.assertRaisesRegex(ValueError, "consumed header changed"):
                build.validate_aocl_selection(selection)

    def test_aocl_rejection_sentinels_disable_real_cmake_search_without_compilers(self):
        cmake = shutil.which("cmake")
        if not cmake:
            self.skipTest("CMake unavailable for local sentinel configure")
        project = self.root / "sentinel-project"
        project.mkdir()
        alternate = self.root / "unrelated-provider"
        for name in ("include/cblas.h", "include/lapacke.h", "lib/libcblas.a", "lib/liblapacke.a",
                     "include/aoclsparse.h", "lib/libaoclsparse.so"):
            path = alternate / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Would be an unapproved fallback")
        selection = self.select_aocl(self.root / "no-aocl", "AOCLSPARSE")
        (project / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.21)
project(AoclSentinel NONE)
include(GpuSuiteCpuDependencies)
gpu_suite_find_cblas()
gpu_suite_find_lapacke()
gpu_suite_find_aocl_sparse()
if(GPU_SUITE_CBLAS_FOUND OR GPU_SUITE_LAPACKE_FOUND OR GPU_SUITE_AOCL_SPARSE_FOUND)
  message(FATAL_ERROR "Unexpected provider fallback")
endif()
if(TARGET GpuSuite::CBLAS OR TARGET GpuSuite::LAPACKE OR TARGET GpuSuite::AOCLSparse)
  message(FATAL_ERROR "Unexpected provider target")
endif()
message(STATUS "REJECTED: ${GPU_SUITE_CBLAS_REASON}; ${GPU_SUITE_LAPACKE_REASON}; ${GPU_SUITE_AOCL_SPARSE_REASON}")
''')
        command = [cmake, "-S", str(project), "-B", str(self.root / "sentinel-build"),
                   "-DCMAKE_MODULE_PATH=" + str(ROOT / "cmake"), "-DCMAKE_PREFIX_PATH=" + str(alternate),
                   "-DGPU_SUITE_CPU_BLAS_BACKEND=GENERIC_CBLAS", "-DGPU_SUITE_CPU_LAPACK_BACKEND=GENERIC_LAPACKE",
                   "-DGPU_SUITE_VENDOR=amd", "-DGPU_SUITE_CPU_SPARSE_BACKEND=AOCLSPARSE"]
        command += ["-D" + key + "=" + value for key, value in selection["cmake_options"].items()]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        (self.root / "sentinel-configure.stdout").write_text(result.stdout)
        (self.root / "sentinel-configure.stderr").write_text(result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("CBLAS header or provider library was not found", result.stdout)
        self.assertIn("LAPACKE header or provider library was not found", result.stdout)
        self.assertIn("AOCL Sparse aoclsparse.h or aoclsparse library was not found", result.stdout)
        self.assertNotIn("CMAKE_C_COMPILER:", (self.root / "sentinel-build/CMakeCache.txt").read_text())

    def test_main_records_incomplete_summary_and_all_slots(self):
        fake = FakeProcesses(self, missing_cpu=("rocblas",))
        with mock.patch.dict(os.environ, self.env, clear=True), mock.patch.object(build.shutil, "which", side_effect=self.which), mock.patch.object(build.subprocess, "Popen", side_effect=fake), redirect_stdout(io.StringIO()):
            code = build.main(self.argv)
        summary = build.load_json(self.run / "amd-build/summary.json")
        self.assertEqual(code, 1)
        self.assertEqual(summary["status"], "INCOMPLETE")
        self.assertEqual(summary["counts"], {"BUILT": 34, "NOT_BUILT_DEPENDENCY": 2})
        self.assertEqual(len(summary["slots"]), 36)
        for key in ("artifact_execution_count", "gpu_query_count", "scheduler_call_count", "ssh_call_count"):
            self.assertEqual(summary[key], 0)

    def test_job_body_reuses_only_module_helper_and_gates_before_commands(self):
        text = (ROOT / "jobs/sirius/amd_build_job.sh").read_text()
        self.assertLess(text.index("approved PBS allocation required"), text.index("date -u"))
        self.assertIn("rocfft_compute_modules.sh", text)
        self.assertIn("STAGE31_SOURCE_ROOT=$AMD_BUILD_SOURCE_ROOT", text)
        self.assertIn("AMD_BUILD_CONTROLLER_SHA256", text)
        self.assertIn('AMD_BUILD_CPU_SPARSE_BACKEND=${AMD_BUILD_CPU_SPARSE_BACKEND:-ONEMKL}', text)
        self.assertIn('--cpu-sparse-backend "$AMD_BUILD_CPU_SPARSE_BACKEND"', text)
        self.assertLess(text.index("invalid explicit CPU Sparse provider"), text.index("date -u"))
        self.assertIn("set -C", text)
        for forbidden in ("compute_smoke.py", "run_suite.py", "rocminfo", "rocm-smi", "ldd ", "qsub", "qstat", "ssh ", "ctest", "hipcc"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()

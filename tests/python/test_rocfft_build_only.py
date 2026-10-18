"""Build-only orchestration tests: all command processes are fake."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("rocfft_build_only", ROOT / "jobs/sirius/rocfft_build_only.py")
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class RocfftBuildOnlyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name).resolve() / "source"
        self.run = self.source / "runs" / "fixture-build"
        self.run.mkdir(parents=True)
        self.cc, self.cxx = self.source / "gcc", self.source / "g++"
        for compiler in (self.cc, self.cxx):
            compiler.write_text("fixture compiler; must never be executed\n")
            compiler.chmod(0o700)
        self.args = SimpleNamespace(source_root=self.source, run_root=self.run, run_id=self.run.name,
                                    profile="cpu", c_compiler=self.cc, cxx_compiler=self.cxx,
                                    gpu_arch=None, rocm_root=None)
        self.popen = mock.patch.object(BUILD.subprocess, "Popen", side_effect=self.fake_process).start()
        self.addCleanup(mock.patch.stopall)
        self.env = mock.patch.dict(os.environ, {"PATH": "/fixture/bin", "CMAKE_PREFIX_PATH": "",
            "CMAKE_INCLUDE_PATH": "", "CMAKE_LIBRARY_PATH": "", "CPATH": "", "C_INCLUDE_PATH": "",
            "CPLUS_INCLUDE_PATH": "", "LIBRARY_PATH": "", "LD_LIBRARY_PATH": ""}, clear=True).start()
        self.configure_code = 0
        self.build_code = 0
        self.missing_target = False
        self.threading_backend = None

    def job(self, profile="cpu"):
        self.args.profile = profile
        if profile != "cpu":
            self.args.gpu_arch, self.args.rocm_root = "gfx942:sramecc+:xnack-", self.source
        return BUILD.Build(self.args)

    def fake_process(self, argv, **kwargs):
        kwargs["stdout"].write(b"fake command output\n")
        kwargs["stderr"].write(b"fake command diagnostics\n")
        code = 0
        if "-S" in argv:
            build = Path(argv[argv.index("-B") + 1])
            build.mkdir()
            stem = BUILD.STEMS[self.args.profile]
            entries = [] if self.missing_target else [{"target_name": name} for name in (stem, stem + "_bench")]
            (build / "partial-manifest-input.json").write_text(json.dumps({"entries": entries}))
            code = self.configure_code
        elif "--build" in argv:
            code = self.build_code
            if not code:
                self.artifacts(Path(argv[argv.index("--build") + 1]))
        process = mock.Mock(pid=123456)
        process.wait.return_value = code
        return process

    def artifacts(self, build):
        metadata = {"vendor": "amd", "build_profile": self.args.profile,
                    "c": {"compiler_path": str(self.cc)}, "cxx": {"compiler_path": str(self.cxx)}}
        if self.threading_backend:
            names = {"pthread": ["fftw3f_threads"],
                     "openmp": ["fftw3f_omp", "fftw_openmp_runtime_0"], "combined": []}[self.threading_backend]
            names = ["fftw3f"] + names
            dependencies = {}
            for name in names:
                library = build / (name + ".so")
                library.write_bytes(("fixture " + name).encode())
                identity = BUILD.file_identity(library)
                dependencies[name] = {key: identity[key] for key in ("path", "sha256")}
            metadata.update(dependencies=dependencies, fftw_threading={
                "requested": self.threading_backend.upper(), "backend": self.threading_backend,
                "compile_options": [], "link_options": [], "dependency_names": names})
        (build / "build-metadata.json").write_text(json.dumps(metadata))
        meta_hash = BUILD.file_identity(build / "build-metadata.json")["sha256"]
        directory = build / "amd/c-cpp/rocfft"
        directory.mkdir(parents=True)
        stem = BUILD.STEMS[self.args.profile]
        entries = []
        for name in (stem, stem + "_bench"):
            path = directory / name
            path.write_bytes(b"fixture artifact; must never be executed\n")
            entries.append({"target_name": name, "executable_path": str(path),
                            "binary_sha256": BUILD.file_identity(path)["sha256"],
                            "build_metadata_sha256": meta_hash, "implementation": self.args.profile,
                            "library": "rocfft"})
        (build / "partial-manifest.json").write_text(json.dumps({"entries": entries, "build_metadata_sha256": meta_hash}))
        (build / "compile_commands.json").write_text("[]\n")
        include, library = build / "fixture-include", build / "libfftw3f.so"
        include.mkdir()
        (include / "fftw3.h").write_text("fixture header\n")
        library.write_text("fixture library\n")
        (build / "CMakeCache.txt").write_text("FFTW3f_INCLUDE_DIR:PATH=" + str(include) +
            "\nFFTW3f_LIBRARY:FILEPATH=" + str(library) + "\n" +
            ("FFTW3f_THREADING:STRING=" + self.threading_backend.upper() + "\n"
             if self.threading_backend else ""))

    def execute(self, job):
        with mock.patch.object(BUILD, "source_evidence", return_value={"source_snapshot_sha256": "a" * 64}), \
                mock.patch.object(BUILD.shutil, "which", return_value=sys.executable):
            return BUILD.execute(job)

    def test_explicit_compilers_and_cpu_only_configuration(self):
        BUILD.validate(self.args)
        argv = BUILD.configure_argv(self.args, "/fixture/cmake", self.run / "build-cpu", {})
        self.assertIn("-DCMAKE_C_COMPILER=" + str(self.cc), argv)
        self.assertIn("-DCMAKE_CXX_COMPILER=" + str(self.cxx), argv)
        for flag in ("CPU=ON", "HIP=OFF", "OPENMP=OFF", "TESTS=OFF"):
            self.assertIn("-DGPU_SUITE_BUILD_" + flag, argv)
        self.popen.assert_not_called()

    def test_loaded_aocl_environment_translates_to_bounded_cmake_hints(self):
        env = {"CPATH": "/aocl/include:/extra/include", "C_INCLUDE_PATH": "/aocl/include",
               "LIBRARY_PATH": "/aocl/lib", "LD_LIBRARY_PATH": "/compiler/lib64",
               "AOCL_ROOT": "/aocl", "FFTW3f_ROOT": "/aocl/fftw"}
        hints = BUILD.cpu_hints(env)
        self.assertEqual(hints["CMAKE_INCLUDE_PATH"], "/aocl/include;/extra/include")
        self.assertEqual(hints["CMAKE_LIBRARY_PATH"], "/aocl/lib;/compiler/lib64")
        self.assertEqual(hints["CMAKE_PREFIX_PATH"], "/aocl")
        self.assertEqual(hints["FFTW3f_ROOT"], "/aocl/fftw")
        with self.assertRaises(ValueError):
            BUILD.cpu_hints({"CPATH": "relative/include"})

    def test_cpu_threading_choice_is_forwarded_only_to_cpu_configuration(self):
        for selection in ("AUTO", "PTHREAD", "OPENMP", "COMBINED", "NONE"):
            env = {"FFTW3f_THREADING": selection}
            self.args.profile = "cpu"
            argv = BUILD.configure_argv(self.args, "/fixture/cmake", self.run / "cpu", env)
            self.assertIn("-DFFTW3f_THREADING=" + selection, argv)
            self.args.profile, self.args.rocm_root, self.args.gpu_arch = "hip", self.source, "gfx942"
            argv = BUILD.configure_argv(self.args, "/fixture/cmake", self.run / "hip", env)
            self.assertFalse(any("FFTW3f_THREADING" in flag for flag in argv))
        with self.assertRaisesRegex(ValueError, "Invalid FFTW3f_THREADING"):
            BUILD.cpu_hints({"FFTW3f_THREADING": "invalid"})
        self.popen.assert_not_called()

    def test_threading_selection_and_dependency_hashes_survive_build_only_report(self):
        self.threading_backend = "openmp"
        job = self.job()
        result = self.execute(job)
        self.assertEqual(result["fftw_cache"]["FFTW3f_THREADING"], "OPENMP")
        self.assertEqual(set(result["selected_fftw_dependencies"]),
                         {"fftw3f", "fftw3f_omp", "fftw_openmp_runtime_0"})
        for name, identity in result["selected_fftw_dependencies"].items():
            self.assertEqual(identity["sha256"], result["build_metadata"]["dependencies"][name]["sha256"])
        selected = Path(result["selected_fftw_dependencies"]["fftw_openmp_runtime_0"]["path"])
        selected.write_bytes(b"changed runtime after build")
        with self.assertRaisesRegex(ValueError, "threading dependency hash mismatch"):
            BUILD.verify_artifacts(job)

    def test_combined_threads_require_no_invented_separate_library(self):
        self.threading_backend = "combined"
        result = self.execute(self.job())
        self.assertEqual(set(result["selected_fftw_dependencies"]), {"fftw3f"})
        self.assertEqual(result["build_metadata"]["fftw_threading"]["backend"], "combined")

    def test_gpu_flags_use_explicit_architecture_and_native_compiler(self):
        for profile in ("hip", "openmp"):
            self.args.profile = profile
            self.args.rocm_root, self.args.gpu_arch = self.source, "gfx942:sramecc+:xnack-"
            argv = BUILD.configure_argv(self.args, "/fixture/cmake", self.run / profile, {"AOCL_ROOT": "/unused"})
            self.assertNotIn("-DCMAKE_PREFIX_PATH=/unused", argv)
            flag = "CMAKE_HIP_ARCHITECTURES" if profile == "hip" else "GPU_SUITE_AMD_GPU_TARGET"
            self.assertIn("-D" + flag + "=gfx942:sramecc+:xnack-", argv)
            if profile == "hip":
                self.assertIn("-DCMAKE_HIP_COMPILER=" + str(self.cxx), argv)
        self.popen.assert_not_called()

    def test_validation_rejects_missing_architecture_hipcc_and_wrong_run(self):
        self.args.profile, self.args.rocm_root = "hip", self.source
        with self.assertRaisesRegex(ValueError, "architecture"):
            BUILD.validate(self.args)
        self.args.gpu_arch = "gfx942"
        hipcc = self.source / "hipcc"
        hipcc.write_text("fake wrapper\n")
        hipcc.chmod(0o700)
        self.args.cxx_compiler = hipcc
        with self.assertRaisesRegex(ValueError, "hipcc"):
            BUILD.validate(self.args)
        self.args.cxx_compiler = self.cxx
        self.args.run_id = "different"
        with self.assertRaisesRegex(ValueError, "RUN_ID"):
            BUILD.validate(self.args)
        self.popen.assert_not_called()

    def test_build_success_never_executes_artifacts_or_gpu_scheduler_commands(self):
        job = self.job()
        result = self.execute(job)
        self.assertEqual(set(result["binaries"]), {"fft_cpu", "fft_cpu_bench"})
        calls = [call.args[0] for call in self.popen.call_args_list]
        self.assertEqual(len(calls), 6)  # Four tool versions, configure, build.
        self.assertTrue(all(argv[0] in (str(self.cc), str(self.cxx), sys.executable) for argv in calls))
        build = calls[-1]
        self.assertEqual(build[build.index("--parallel") + 1], "1")
        self.assertEqual(build[build.index("--target") + 1:], ["fft_cpu", "fft_cpu_bench", "gpu_suite_partial_manifest"])
        for entry in job.commands:
            self.assertEqual(Path(entry["stdout"]).read_bytes(), b"fake command output\n")
            self.assertEqual(Path(entry["stderr"]).read_bytes(), b"fake command diagnostics\n")

    def test_configure_success_with_missing_targets_stops_before_build(self):
        self.missing_target = True
        with self.assertRaisesRegex(ValueError, "disabled required"):
            self.execute(self.job())
        self.assertFalse(any("--build" in call.args[0] for call in self.popen.call_args_list))

    def test_failed_configure_and_build_are_not_retried(self):
        self.configure_code = 7
        job = self.job()
        with self.assertRaisesRegex(ValueError, "configure"):
            self.execute(job)
        self.assertEqual(sum("-S" in call.args[0] for call in self.popen.call_args_list), 1)
        self.assertFalse(any("--build" in call.args[0] for call in self.popen.call_args_list))
        self.assertEqual(job.commands[-1]["exit_status"], 7)
        self.assertTrue((job.root / "source-after.json").is_file())

    def test_failed_link_stops_and_preserves_error_status(self):
        self.build_code = 9
        job = self.job()
        with self.assertRaisesRegex(ValueError, "build"):
            self.execute(job)
        self.assertEqual(job.commands[-1]["exit_status"], 9)
        self.assertEqual(sum("--build" in call.args[0] for call in self.popen.call_args_list), 1)

    def test_source_mutation_rejects_otherwise_successful_artifacts(self):
        job = self.job()
        with mock.patch.object(BUILD, "source_evidence", side_effect=[
                {"source_snapshot_sha256": "before"}, {"source_snapshot_sha256": "after"}]), \
                mock.patch.object(BUILD.shutil, "which", return_value=sys.executable), \
                self.assertRaisesRegex(ValueError, "Source changed"):
            BUILD.execute(job)

    def test_existing_build_or_attempt_is_never_overwritten(self):
        (self.run / "build-cpu").mkdir()
        with self.assertRaisesRegex(ValueError, "existing build"):
            self.job()
        self.args.profile = "hip"
        (self.run / "build-only-hip").mkdir()
        with self.assertRaises(FileExistsError):
            BUILD.Build(self.args)
        self.popen.assert_not_called()

    def test_artifact_hash_or_compiler_mismatch_is_rejected(self):
        job = self.job()
        self.execute(job)
        binary = job.build / "amd/c-cpp/rocfft/fft_cpu"
        binary.write_bytes(b"changed bytes\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            BUILD.verify_artifacts(job)
        metadata_path = job.build / "build-metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["c"]["compiler_path"] = "/unexpected/compiler"
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "different compiler"):
            BUILD.verify_artifacts(job)

    def test_timeout_terminates_only_owned_process_group_without_retry(self):
        job = self.job()
        process = mock.Mock(pid=123456)
        process.wait.side_effect = [subprocess.TimeoutExpired(["fixture"], 1), -15]
        self.popen.side_effect = None
        self.popen.return_value = process
        with mock.patch.object(BUILD.os, "killpg") as kill, self.assertRaisesRegex(ValueError, "failed"):
            job.run("fixture-timeout", [str(self.cc)], timeout=1)
        self.popen.assert_called_once()
        kill.assert_called_once_with(process.pid, BUILD.signal.SIGTERM)
        self.assertIsNone(job.commands[0]["exit_status"])
        self.assertIn("not retried", job.commands[0]["failure"])

    def test_subprocess_launch_failure_records_missing_exit(self):
        job = self.job()
        self.popen.side_effect = OSError("fixture process launch denied")
        with self.assertRaisesRegex(ValueError, "failed"):
            job.run("fixture-launch", [str(self.cc)])
        self.assertIsNone(job.commands[0]["exit_status"])
        self.assertIn("launch denied", job.commands[0]["failure"])


if __name__ == "__main__":
    unittest.main()

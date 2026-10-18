"""Exercise fail-before-build smoke gates; never compile, SSH or touch a GPU."""

import json
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "jobs/sirius/rocfft_build_smoke.sh"
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "rocfft_smoke_evidence", ROOT / "jobs/sirius/rocfft_smoke_evidence.py")
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


class RocfftSmokeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rocfft-smoke-gate-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.output = self.base / "new-output"
        self.environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                            "PYTHONDONTWRITEBYTECODE": "1"}

    def run_script(self):
        result = subprocess.run(["/bin/bash", str(SCRIPT)], env=self.environment,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists(), "rejected input must not start a build")
        return result

    def set_inputs(self):
        # Python fills executable-path slots only; rejected preflight must never
        # invoke them as compiler or CMake commands.
        self.environment.update({
            "GPU_SUITE_ALLOCATED_COMPUTE_NODE": "yes",
            "SOURCE_ROOT": str(ROOT), "RUN_ROOT": str(self.output),
            "CMAKE": sys.executable, "PYTHON": sys.executable,
            "CPU_CC": sys.executable, "CPU_CXX": sys.executable,
            "HIP_CLANG": sys.executable, "OPENMP_CXX": sys.executable,
            "ROCM_ROOT": str(self.base), "FFTW_ROOT": str(self.base),
            "GPU_ARCH": "test-explicit-target", "BUILD_JOBS": "1",
            "HIP_DEVICE": "0", "OMP_DEVICE": "1",
        })

    def test_allocation_confirmation_required_before_any_work(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 2)
        self.assertIn("allocated-compute-node confirmation", result.stderr)

    def test_native_architecture_rejected_before_toolchain_commands(self):
        self.set_inputs()
        self.environment["GPU_ARCH"] = "native"
        result = self.run_script()
        self.assertIn("never native", result.stderr)

    def test_large_workload_rejected_before_build_directory_creation(self):
        self.set_inputs()
        config = json.loads((ROOT / "jobs/sirius/rocfft-smoke.json").read_text())
        config["benchmarks"]["rocfft"]["cases"][0]["parameters"]["nfft"] = 512
        path = self.base / "oversize.json"
        path.write_text(json.dumps(config))
        self.environment["SMOKE_CONFIG"] = str(path)
        result = self.run_script()
        self.assertIn("reviewed small bounds", result.stderr)

    def test_single_batch_rejected_before_build_directory_creation(self):
        self.set_inputs()
        config = json.loads((ROOT / "jobs/sirius/rocfft-smoke.json").read_text())
        config["benchmarks"]["rocfft"]["cases"][0]["parameters"]["batch"] = 1
        path = self.base / "single-batch.json"
        path.write_text(json.dumps(config))
        self.environment["SMOKE_CONFIG"] = str(path)
        result = self.run_script()
        self.assertIn("multiple batches", result.stderr)

    def test_existing_output_is_preserved_before_any_build(self):
        self.set_inputs()
        existing = self.base / "existing-output"
        existing.mkdir()
        evidence = existing / "evidence.txt"
        evidence.write_bytes(b"preserve prior evidence\n")
        self.environment["RUN_ROOT"] = str(existing)
        result = self.run_script()
        self.assertIn("must not already exist", result.stderr)
        self.assertEqual(evidence.read_bytes(), b"preserve prior evidence\n")

    def test_output_outside_runs_is_rejected_before_any_build(self):
        self.set_inputs()
        result = self.run_script()
        self.assertIn("direct child of SOURCE_ROOT/runs", result.stderr)

    def test_smoke_shell_syntax(self):
        result = subprocess.run(["/bin/bash", "-n", str(SCRIPT)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("shellcheck"), "ShellCheck dependency unavailable")
    def test_smoke_shellcheck(self):
        result = subprocess.run([shutil.which("shellcheck"), str(SCRIPT)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def library(self, directory, name, content=b"synthetic library evidence"):
        directory.mkdir(exist_ok=True)
        path = directory / name
        path.write_bytes(content)
        return path.resolve(), hashlib.sha256(content).hexdigest()

    def test_same_name_alternate_loaded_library_is_rejected(self):
        configured, digest = self.library(self.base / "configured", "librocfft.so.0")
        loaded, _ = self.library(self.base / "loaded", "librocfft.so.0")
        metadata = {"build_profile": "cpu", "dependencies": {
            "rocfft": {"path": str(configured), "sha256": digest}}}
        with self.assertRaisesRegex(ValueError, "loaded dependency differs"):
            evidence.check_configured_dependencies(metadata, {str(loaded): digest})

    def test_changed_configured_library_is_rejected(self):
        path, digest = self.library(self.base, "librocfft.so.0")
        metadata = {"build_profile": "cpu", "dependencies": {
            "rocfft": {"path": str(path), "sha256": digest}}}
        path.write_bytes(b"changed since configure")
        with self.assertRaisesRegex(ValueError, "changed since build"):
            evidence.check_configured_dependencies(metadata, {str(path): digest})

    def test_static_cpu_fftw_does_not_require_dynamic_resolution(self):
        path, digest = self.library(self.base, "libfftw3f.a")
        metadata = {"build_profile": "cpu", "dependencies": {
            "fftw3f": {"path": str(path), "sha256": digest}}}
        evidence.check_configured_dependencies(metadata, {})

    def test_gpu_profile_requires_both_hip_and_rocfft_evidence(self):
        with self.assertRaisesRegex(ValueError, "lacks HIP/rocFFT"):
            evidence.check_configured_dependencies(
                {"build_profile": "hip", "dependencies": {}}, {})

    def openmp_script_fixture(self, template="INPUT ( {runtime} )", archive=False):
        base, base_hash = self.library(self.base, "libfftw3f.so", b"\x7fELFfake fftw")
        omp, omp_hash = self.library(self.base, "libfftw3f_omp.so", b"\x7fELFfake omp fftw")
        runtime, runtime_hash = self.library(self.base / "system", "libgomp.so.1", b"\x7fELFfake runtime")
        script, script_hash = self.library(self.base / "toolset", "libgomp.so",
                                           template.format(runtime=runtime).encode("ascii"))
        dependencies = {"fftw3f": {"path": str(base), "sha256": base_hash},
                        "fftw3f_omp": {"path": str(omp), "sha256": omp_hash},
                        "fftw_openmp_runtime_0": {"path": str(script), "sha256": script_hash}}
        metadata = {"build_profile": "cpu", "dependencies": dependencies,
                    "fftw_threading": {"requested": "OPENMP", "backend": "openmp",
                                       "compile_options": ["-fopenmp"], "link_options": [],
                                       "dependency_names": list(dependencies)}}
        resolved = {str(base): base_hash, str(omp): omp_hash, str(runtime): runtime_hash}
        depfile = self.base / "link.d"
        paths = [script, runtime]
        if archive:
            static, _ = self.library(self.base / "toolset", "libgomp_nonshared.a", b"!<arch>\nfake static evidence")
            paths.append(static)
        depfile.write_text("fft_cpu_bench: \\\n  " + " \\\n  ".join(map(str, paths)) + "\n\n" + str(script) + ":\n")
        return metadata, resolved, script, runtime, depfile

    def test_openmp_linker_script_binds_exact_loaded_runtime_and_script_hash(self):
        metadata, resolved, script, runtime, _ = self.openmp_script_fixture()
        bindings = evidence.check_configured_dependencies(metadata, resolved)
        binding = bindings["fftw_openmp_runtime_0"]
        self.assertEqual(binding["script_path"], str(script))
        self.assertEqual(binding["script_sha256"], hashlib.sha256(script.read_bytes()).hexdigest())
        self.assertEqual(binding["parsed_inputs"], [str(runtime)])
        self.assertEqual(binding["inputs"], [{"reference": str(runtime), "path": str(runtime),
                                              "sha256": resolved[str(runtime)], "kind": "loaded-elf"}])

    def test_openmp_group_as_needed_comments_and_output_format_are_read_only(self):
        metadata, resolved, _, _, _ = self.openmp_script_fixture(
            "/* GNU ld script */\nOUTPUT_FORMAT(elf64-x86-64)\nGROUP(AS_NEEDED({runtime})); # comment\n")
        with mock.patch.object(subprocess, "run", side_effect=AssertionError("must never execute")), \
             mock.patch.object(subprocess, "Popen", side_effect=AssertionError("must never execute")):
            self.assertIn("fftw_openmp_runtime_0", evidence.check_configured_dependencies(metadata, resolved))

    def test_openmp_linker_script_different_same_named_runtime_is_rejected(self):
        metadata, resolved, _, runtime, _ = self.openmp_script_fixture()
        other, digest = self.library(self.base / "alternate", "libgomp.so.1", runtime.read_bytes())
        del resolved[str(runtime)]
        resolved[str(other)] = digest
        with self.assertRaisesRegex(ValueError, "referenced path/hash"):
            evidence.check_configured_dependencies(metadata, resolved)

    def test_openmp_linker_script_loaded_hash_mismatch_is_rejected(self):
        metadata, resolved, _, runtime, _ = self.openmp_script_fixture()
        resolved[str(runtime)] = "0" * 64
        with self.assertRaisesRegex(ValueError, "referenced path/hash"):
            evidence.check_configured_dependencies(metadata, resolved)

    def test_changed_openmp_linker_script_is_rejected_before_parsing(self):
        metadata, resolved, script, _, _ = self.openmp_script_fixture()
        script.write_text("INPUT(/changed/runtime.so)\n")
        with self.assertRaisesRegex(ValueError, "changed since build"):
            evidence.check_configured_dependencies(metadata, resolved)

    def test_unrelated_so_does_not_gain_linker_script_exemption(self):
        metadata, resolved, script, _, _ = self.openmp_script_fixture()
        unrelated = {"build_profile": "cpu", "dependencies": {
            "rocfft": metadata["dependencies"]["fftw_openmp_runtime_0"]}}
        with self.assertRaisesRegex(ValueError, "loaded dependency differs"):
            evidence.check_configured_dependencies(unrelated, resolved)
        self.assertTrue(script.is_file())

    def test_openmp_linker_script_named_archive_needs_link_dependency_evidence(self):
        metadata, resolved, _, _, _ = self.openmp_script_fixture("INPUT({runtime} -lgomp_nonshared)", True)
        with self.assertRaisesRegex(ValueError, "requires recorded link"):
            evidence.check_configured_dependencies(metadata, resolved)

    def test_openmp_linker_script_static_archive_bound_to_recorded_link_inputs(self):
        metadata, resolved, script, _, depfile = self.openmp_script_fixture("INPUT({runtime} -lgomp_nonshared)", True)
        binding = evidence.check_configured_dependencies(metadata, resolved, depfile)["fftw_openmp_runtime_0"]
        self.assertEqual(binding["link_dependency_file"]["sha256"], hashlib.sha256(depfile.read_bytes()).hexdigest())
        self.assertEqual(binding["inputs"][1]["kind"], "static-archive")
        self.assertEqual(binding["inputs"][1]["path"], str(script.parent / "libgomp_nonshared.a"))

    def test_openmp_linker_script_missing_or_unbound_archive_is_rejected(self):
        metadata, resolved, script, runtime, depfile = self.openmp_script_fixture("INPUT({runtime} -lgomp_nonshared)")
        with self.assertRaisesRegex(ValueError, "unambiguous recorded static archive"):
            evidence.check_configured_dependencies(metadata, resolved, depfile)
        depfile.write_text("fft_cpu_bench: " + str(runtime) + "\n")
        with self.assertRaisesRegex(ValueError, "does not bind configured script"):
            evidence.check_configured_dependencies(metadata, resolved, depfile)
        self.assertTrue(script.is_file())

    def test_openmp_linker_script_ambiguous_or_dynamic_l_target_is_rejected(self):
        metadata, resolved, script, runtime, depfile = self.openmp_script_fixture("INPUT({runtime} -lgomp_nonshared)", True)
        other, _ = self.library(self.base / "alternate", "libgomp_nonshared.a", b"!<arch>\nother")
        for paths in ([script, runtime, script.parent / "libgomp_nonshared.a", other],
                      [script, runtime, self.library(self.base / "alternate", "libgomp_nonshared.so", b"\x7fELFdynamic")[0]]):
            with self.subTest(paths=paths):
                depfile.write_text("fft_cpu_bench: " + " ".join(map(str, paths)) + "\n")
                with self.assertRaisesRegex(ValueError, "unambiguous recorded static archive"):
                    evidence.check_configured_dependencies(metadata, resolved, depfile)

    def test_openmp_linker_script_rejects_unsupported_syntax_and_relative_inputs(self):
        for text in ("INPUT(libgomp.so.1)", "INCLUDE /etc/passwd", "SEARCH_DIR(/anything) INPUT(/a)",
                     "INPUT(/a); $(touch /tmp/not-run)", "INPUT(AS_NEEDED())", "INPUT(/a", "/* unfinished"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                evidence._linker_script_inputs(text.encode())

    def test_openmp_linker_script_limits_size_and_nesting(self):
        with self.assertRaisesRegex(ValueError, "size limit"):
            evidence._linker_script_inputs(b" " * 65537)
        with self.assertRaisesRegex(ValueError, "nesting limit"):
            evidence._linker_script_inputs(("INPUT(" + "AS_NEEDED(" * 10 + "/a" + ")" * 11).encode())

    def test_openmp_linker_script_fake_referenced_so_is_rejected(self):
        metadata, resolved, _, runtime, _ = self.openmp_script_fixture()
        runtime.write_bytes(b"not an ELF library")
        resolved[str(runtime)] = hashlib.sha256(runtime.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "unsupported linker script input content"):
            evidence.check_configured_dependencies(metadata, resolved)

    def test_aslr_addresses_do_not_change_runtime_library_identity(self):
        path, digest = self.library(self.base, "librocfft.so.0")
        first = "librocfft.so.0 => {0} (0x00001234)\n".format(path)
        second = "librocfft.so.0 => {0} (0x43210000)\n".format(path)
        normalized, libraries = evidence.resolved_libraries(first)
        self.assertEqual((normalized, libraries), evidence.resolved_libraries(second))
        self.assertEqual(libraries, {str(path): digest})
        self.assertNotIn("0x", normalized)
        with self.assertRaisesRegex(ValueError, "unresolved runtime"):
            evidence.resolved_libraries("librocfft.so => not found\n")


    def test_gpu_visibility_environment_changes_runtime_identity(self):
        fftw, fftw_hash = self.library(self.base, "libfftw3f.a")
        hip, hip_hash = self.library(self.base, "libamdhip64.so.7")
        rocfft, rocfft_hash = self.library(self.base, "librocfft.so.0")
        config = ROOT / "jobs/sirius/rocfft-smoke.json"
        variables = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                     "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")

        def prepare_fixture(name):
            output = self.base / name
            output.mkdir()
            (output / "logs").mkdir()
            for profile in ("cpu", "hip", "openmp"):
                (output / profile).mkdir()
                dependencies = ({"fftw3f": {"path": str(fftw), "sha256": fftw_hash}}
                                if profile == "cpu" else {
                                    "hip": {"path": str(hip), "sha256": hip_hash},
                                    "rocfft": {"path": str(rocfft), "sha256": rocfft_hash}})
                (output / profile / "build-metadata.json").write_text(json.dumps({
                    "build_profile": profile, "dependencies": dependencies}))
                resolution = ("" if profile == "cpu" else
                              "libamdhip64.so.7 => {0} (0x1234)\n"
                              "librocfft.so.0 => {1} (0x5678)\n".format(hip, rocfft))
                (output / "logs" / ("001-resolved-" + profile + ".stdout")).write_text(resolution)
            digest = evidence.prepare(output, config, 0, 1)
            document = json.loads((output / "runtime-environment.json").read_text())
            return digest, document

        with mock.patch.dict(os.environ, {name: "0" for name in variables}):
            baseline_hash, baseline = prepare_fixture("baseline")
            for variable in variables:
                with self.subTest(variable=variable), mock.patch.dict(os.environ, {variable: "1"}):
                    changed_hash, changed = prepare_fixture(variable)
                    self.assertNotEqual(changed_hash, baseline_hash)
                    self.assertEqual(changed["environment"][variable], "1")
                    changed["environment"][variable] = "0"
                    self.assertEqual(changed, baseline)


if __name__ == "__main__":
    unittest.main()

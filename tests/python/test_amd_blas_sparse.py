"""Host numerical/API fixtures, never evidence of real HIP or target execution."""
import csv
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/amd_blas_sparse"
AMD = ROOT / "amd/c-cpp"


@unittest.skipUnless(shutil.which("cc") and shutil.which("c++"),
                     "host C and C++ compilers required for API fixtures")
class AmdBlasSparseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path("/tmp/gpu-library-suite-local-build")
        base.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(tempfile.gettempdir()).resolve()
        if temporary_root == base.resolve() or base.resolve() in temporary_root.parents:
            base = temporary_root  # Honor the fixed validation runner's private TMPDIR.
        temporary = tempfile.TemporaryDirectory(prefix="amd-blas-sparse-", dir=base)
        cls.addClassCleanup(temporary.cleanup)
        cls.work = Path(temporary.name)
        cls.environment = {k: v for k, v in os.environ.items()
                           if not k.startswith(("GPU_SUITE_", "FAKE_"))}
        cls.environment["TMPDIR"] = str(cls.work)
        generated = cls.work / "include/gpu_suite"
        generated.mkdir(parents=True)
        template = (ROOT / "cmake/gpu_suite_build_metadata.h.in").read_text()
        template = re.sub(r"@([A-Z0-9_]+)@", lambda match:
                          "a" * 64 if match[1] == "GPU_SUITE_BUILD_METADATA_SHA256"
                          else "0" if match[1] in {
                              "GPU_SUITE_META_GIT_METADATA_AVAILABLE", "GPU_SUITE_META_GIT_DIRTY_C"}
                          else "", template)
        (generated / "build_metadata.h").write_text(template)
        includes = ["-I", ROOT / "common/c-cpp/include", "-I", cls.work / "include"]
        cls.objects = []
        for source in sorted((ROOT / "common/c-cpp/src").glob("*.c")):
            obj = cls.work / (source.stem + ".o")
            cls.checked(["cc", "-std=c11", "-D_POSIX_C_SOURCE=200809L", *includes,
                         "-c", source, "-o", obj])
            cls.objects.append(obj)
        provider = cls.work / "cpu-provider.o"
        cls.checked(["cc", "-std=c11", "-I", ROOT / "tests/fixtures/cpu_libs/include",
                     "-c", ROOT / "tests/fixtures/cpu_libs/fake_cpu_providers.c", "-o", provider])
        fixture_includes = ["-I", FIXTURE / "include", "-I", ROOT / "tests/fixtures/rocm/include"]
        fake = cls.work / "fake-amd.o"
        cls.checked(["c++", "-std=c++17", *fixture_includes,
                     "-c", FIXTURE / "fake_amd.cpp", "-o", fake])
        cls.binaries = {}
        for family, stem in (("rocblas", "blas"), ("rocsparse", "sparse")):
            for implementation in ("cpu", "hip", "openmp"):
                name = stem + "_cpu_bench.c" if implementation == "cpu" else (
                    stem + "_gpu_bench.cpp" if implementation == "hip" else "openmp_" + family + "_bench.cpp")
                source = AMD / family / "benchmarks" / name
                exe = cls.work / (family + "-" + implementation)
                if implementation == "cpu":
                    backend = "cpu-generic-cblas" if family == "rocblas" else "cpu-reference-csr"
                    command = ["cc", "-std=c11", *includes,
                               "-I", ROOT / "tests/fixtures/cpu_libs/include",
                               '-DGPU_SUITE_CBLAS_HEADER="cblas.h"',
                               '-DGPU_SUITE_COMPILED_CPU_BACKEND="' + backend + '"',
                               source, provider, *cls.objects, "-lm", "-o", exe]
                else:
                    # Pragmas are intentionally ignored in this HOST simulation.
                    command = ["c++", "-std=c++17", "-Wno-unknown-pragmas", "-D_OPENMP=201511",
                               *includes, *fixture_includes, source, fake, *cls.objects, "-o", exe]
                cls.checked(command)
                cls.binaries[family, implementation] = exe

    @classmethod
    def checked(cls, argv):
        result = subprocess.run([str(a) for a in argv], capture_output=True, text=True,
                                env=cls.environment, check=False)
        if result.returncode:
            raise AssertionError("command failed: {}\n{}\n{}".format(argv, result.stdout, result.stderr))
        return result

    def invoke(self, family, implementation, scope="compute", output_format="jsonl", extra=(), environment=None):
        argv = [str(self.binaries[family, implementation]), "--warmup", "2", "--repeat", "3",
                "--trials", "2", "--scope", scope, "--verify", "true", "--output", "-",
                "--format", output_format, "--alpha", "1.25", "--beta", "0.5"]
        argv += ["--m", "3", "--n", "2", "--k", "4"] if family == "rocblas" else ["--nx", "3", "--ny", "4"]
        if implementation == "cpu":
            backend = "cpu-generic-cblas" if family == "rocblas" else "cpu-reference-csr"
            argv += ["--cpu-backend", backend, "--cpu-threads", "1", "--cpu-threads-effective", "1",
                     "--cpu-backend-role", "reference", "--series-role", "auxiliary", "--cpu-parallelism", "serial"]
        for key, value in zip(extra[::2], extra[1::2]):
            if key in argv:
                argv[argv.index(key) + 1] = value
            else:
                argv += [key, value]
        env = dict(self.environment)
        env.update(environment or {})
        return subprocess.run(argv, capture_output=True, text=True, env=env, check=False)

    def test_all_host_fixture_variants_verify_both_scopes_and_formats(self):
        for family, implementation in self.binaries:
            for scope in ("compute", "end-to-end"):
                for fmt in ("jsonl", "csv"):
                    with self.subTest(family=family, implementation=implementation, scope=scope, format=fmt):
                        result = self.invoke(family, implementation, scope, fmt)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        rows = ([json.loads(line) for line in result.stdout.splitlines()] if fmt == "jsonl"
                                else list(csv.DictReader(io.StringIO(result.stdout))))
                        self.assertEqual(len(rows), 2)
                        for row in rows:
                            self.assertEqual(row["benchmark"], family)
                            self.assertEqual(row["implementation"], implementation)
                            self.assertEqual(row["status"], "success")
                            self.assertEqual(row["verification_status"], "pass")
                            self.assertAlmostEqual(float(row["elapsed_sec"]) * 3,
                                                   float(row["elapsed_total_sec"]), places=12)

    def test_nonfinite_gpu_fixture_output_fails_verification(self):
        for family in ("rocblas", "rocsparse"):
            for implementation in ("hip", "openmp"):
                for scope in ("compute", "end-to-end"):
                    for value in ("nan", "inf"):
                        with self.subTest(family=family, implementation=implementation, scope=scope, value=value):
                            result = self.invoke(family, implementation, scope,
                                                 environment={"GPU_SUITE_TEST_NONFINITE": value})
                            self.assertNotEqual(result.returncode, 0)
                            for row in map(json.loads, result.stdout.splitlines()):
                                self.assertEqual(row["verification_status"], "nonfinite")
                                self.assertEqual(row["failure_origin"], "verification")

    def test_gpu_fixture_library_failure_has_no_success_records(self):
        for family in ("rocblas", "rocsparse"):
            for implementation in ("hip", "openmp"):
                for scope in ("compute", "end-to-end"):
                    with self.subTest(family=family, implementation=implementation, scope=scope):
                        result = self.invoke(family, implementation, scope,
                                             environment={"FAKE_LIBRARY_FAIL": "1"})
                        self.assertNotEqual(result.returncode, 0)
                        rows = [json.loads(line) for line in result.stdout.splitlines()]
                        self.assertEqual(len(rows), 2)
                        self.assertTrue(all(row["status"] != "success" for row in rows))

    def test_openmp_host_fallback_and_mapped_device_mismatch_reject(self):
        for family in ("rocblas", "rocsparse"):
            for scope in ("compute", "end-to-end"):
                for failure in ("FAKE_HOST_FALLBACK", "FAKE_DEVICE_MISMATCH"):
                    with self.subTest(family=family, scope=scope, failure=failure):
                        result = self.invoke(family, "openmp", scope, environment={failure: "1"})
                        self.assertNotEqual(result.returncode, 0)
                        rows = [json.loads(line) for line in result.stdout.splitlines()]
                        self.assertEqual(len(rows), 2)
                        self.assertTrue(all(row["status"] != "success" for row in rows))

    def test_measured_failure_marks_remaining_trials_unattempted(self):
        for family in ("rocblas", "rocsparse"):
            for implementation in ("hip", "openmp"):
                for scope in ("compute", "end-to-end"):
                    with self.subTest(family=family, implementation=implementation, scope=scope):
                        result = self.invoke(family, implementation, scope,
                                             environment={"FAKE_FAIL_AFTER": "2"})
                        self.assertNotEqual(result.returncode, 0)
                        rows = [json.loads(line) for line in result.stdout.splitlines()]
                        self.assertEqual(len(rows), 2)
                        self.assertEqual(rows[0]["status"], "failure")
                        self.assertTrue(rows[0]["attempted"])
                        self.assertEqual(rows[1]["status"], "skipped")
                        self.assertFalse(rows[1]["attempted"])
                        self.assertEqual(rows[1]["failure_origin"], "prior-failure")

    def test_zero_warmup_still_checks_openmp_device_mapping(self):
        for family in ("rocblas", "rocsparse"):
            result = self.invoke(family, "openmp", extra=("--warmup", "0"),
                                 environment={"FAKE_DEVICE_MISMATCH": "1"})
            self.assertNotEqual(result.returncode, 0)
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["status"] != "success" for row in rows))

    def test_overflow_rejected_before_allocation(self):
        for family, implementation in self.binaries:
            with self.subTest(family=family, implementation=implementation):
                dimension = "--m" if family == "rocblas" else "--nx"
                result = self.invoke(family, implementation, extra=(dimension, "2147483648"))
                self.assertNotEqual(result.returncode, 0)
                rows = [json.loads(line) for line in result.stdout.splitlines()]
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(not row["attempted"] and row["elapsed_sec"] is None for row in rows))

    def test_hip_allocation_failure_is_recorded(self):
        for family in ("rocblas", "rocsparse"):
            result = self.invoke(family, "hip", environment={"FAKE_ALLOC_FAIL": "1"})
            self.assertNotEqual(result.returncode, 0)
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["status"], "failure")

    def test_explicit_cpu_backend_does_not_fallback(self):
        for family in ("rocblas", "rocsparse"):
            result = self.invoke(family, "cpu", extra=("--cpu-backend", "cpu-onemkl"))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("backend", result.stderr.lower())

    def test_actual_openmp_frontend_parses_examples_and_benchmarks(self):
        compiler = shutil.which("clang++")
        if not compiler:
            self.skipTest("Clang OpenMP syntax frontend unavailable")
        for family, stem in (("rocblas", "blas"), ("rocsparse", "sparse")):
            sources = [AMD / family / "examples" / (stem + "_gpu.cpp"),
                       AMD / family / "examples" / ("openmp_" + family + ".cpp"),
                       AMD / family / "benchmarks" / ("openmp_" + family + "_bench.cpp")]
            self.checked([compiler, "-std=c++17", "-Xclang", "-fopenmp", "-fsyntax-only",
                          "-I", FIXTURE / "include", "-I", ROOT / "tests/fixtures/rocm/include",
                          "-I", ROOT / "common/c-cpp/include", *sources])

    def test_standalone_cpu_teaching_sources_compile_for_explicit_backends(self):
        for family, stem in (("rocblas", "blas"), ("rocsparse", "sparse")):
            source = AMD / family / "examples" / (stem + "_cpu.c")
            self.checked(["cc", "-std=c11", "-fsyntax-only", "-I",
                          ROOT / "tests/fixtures/cpu_libs/include", source])
            if family == "rocsparse":
                self.checked(["cc", "-std=c11", "-fsyntax-only", "-DGPU_SUITE_USE_MKL_SPARSE=1",
                              "-I", ROOT / "tests/fixtures/cpu_libs/include", source])

    def test_sources_do_not_depend_on_nvidia_or_hidden_teaching_helpers(self):
        for family in ("rocblas", "rocsparse"):
            for source in (AMD / family).rglob("*"):
                if source.suffix not in (".c", ".cpp", ".hpp"):
                    continue
                text = source.read_text()
                self.assertNotIn("nvidia/", text)
                self.assertNotIn("#pragma acc", text)
                if source.parent.name == "examples":
                    self.assertNotIn('#include "', text)
                    self.assertNotIn("gpu_suite_options", text)


if __name__ == "__main__":
    unittest.main()

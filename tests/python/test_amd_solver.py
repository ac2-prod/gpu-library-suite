"""Host-only rocSOLVER orchestration tests; synthetic HIP is not GPU validation."""
import csv
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "amd/c-cpp/rocsolver"
FIXTURE = ROOT / "tests/fixtures/amd_solver"
sys.path.insert(0, str(ROOT / "tools"))
from gpu_suite.schema import validate_raw_result


@unittest.skipUnless(shutil.which("cc") and shutil.which("c++"),
                     "host C and C++ compilers are required")
class AmdSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path("/private/tmp/gpu-library-suite-local-build")
        if not base.parent.exists():
            base = Path("/tmp/gpu-library-suite-local-build")
        base.mkdir(exist_ok=True)
        requested_tmp = Path(os.environ.get("TMPDIR", str(base))).resolve()
        if base.resolve() in requested_tmp.parents or requested_tmp == base.resolve():
            base = requested_tmp
        cls.temporary = tempfile.TemporaryDirectory(prefix="amd-solver-", dir=base)
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        cls.environment = os.environ.copy()
        cls.environment["TMPDIR"] = str(cls.work)
        include = cls.work / "gpu_suite"
        include.mkdir()
        template = (ROOT / "cmake/gpu_suite_build_metadata.h.in").read_text()
        values = {"GPU_SUITE_META_GIT_METADATA_AVAILABLE": "0",
                  "GPU_SUITE_META_GIT_DIRTY_C": "0",
                  "GPU_SUITE_BUILD_METADATA_SHA256": "a" * 64}
        template = re.sub(r"@([A-Z0-9_]+)@", lambda m: values.get(m[1], "fixture"), template)
        (include / "build_metadata.h").write_text(template)
        cls.includes = ["-I" + str(ROOT / "common/c-cpp/include"), "-I" + str(cls.work)]
        cls.objects = []
        for source in sorted((ROOT / "common/c-cpp/src").glob("*.c")):
            target = cls.work / (source.stem + ".o")
            cls.execute(["cc", "-std=c11", "-D_POSIX_C_SOURCE=200809L", *cls.includes,
                         "-c", str(source), "-o", str(target)])
            cls.objects.append(str(target))
        cpu_include = ROOT / "tests/fixtures/cpu_libs/include"
        cpu_object = cls.work / "cpu_provider.o"
        cls.execute(["cc", "-std=c11", "-I" + str(cpu_include), "-c",
                     str(ROOT / "tests/fixtures/cpu_libs/fake_cpu_providers.c"),
                     "-o", str(cpu_object)])
        cls.binaries = {}
        cls.execute(["cc", "-std=c11", *cls.includes, "-I" + str(cpu_include),
                     '-DGPU_SUITE_LAPACKE_HEADER="lapacke.h"',
                     '-DGPU_SUITE_COMPILED_CPU_BACKEND="cpu-generic-lapacke"',
                     str(FAMILY / "benchmarks/solver_cpu_bench.c"), str(cpu_object),
                     *cls.objects, "-lm", "-o", str(cls.work / "cpu")])
        cls.binaries["cpu"] = cls.work / "cpu"
        cls.execute(["cc", "-std=c11", "-I" + str(cpu_include),
                     str(FAMILY / "examples/solver_cpu.c"), str(cpu_object),
                     "-lm", "-o", str(cls.work / "cpu_example")])
        for kind, stem in (("hip", "solver_gpu"), ("openmp", "openmp_rocsolver")):
            flags = ["-D_OPENMP=202011"] if kind == "openmp" else []
            cls.execute(["c++", "-std=c++17", "-Wno-unknown-pragmas", *flags,
                         "-DGPU_SUITE_HAVE_HIP_RUNTIME_METADATA=1",
                         "-DGPU_SUITE_HIP_LIBRARY_ROCSOLVER=1", *cls.includes,
                         "-I" + str(FIXTURE), str(FAMILY / "benchmarks" / (stem + "_bench.cpp")),
                         *cls.objects, "-o", str(cls.work / kind)])
            cls.binaries[kind] = cls.work / kind
            cls.execute(["c++", "-std=c++17", "-Wno-unknown-pragmas", *flags,
                         "-I" + str(FIXTURE), str(FAMILY / "examples" / (stem + ".cpp")),
                         "-o", str(cls.work / (kind + "_example"))])
        # Real OpenMP parsing, syntax-only: no runtime link or offload execution.
        compiler_version = cls.execute(["c++", "--version"]).stdout
        omp_flags = ["-Xclang", "-fopenmp"] if "Apple clang" in compiler_version else ["-fopenmp"]
        cls.openmp_syntax = subprocess.run(
            ["c++", "-std=c++17", *omp_flags, "-fsyntax-only", *cls.includes,
             "-I" + str(FIXTURE), str(FAMILY / "benchmarks/openmp_rocsolver_bench.cpp"),
             str(FAMILY / "examples/openmp_rocsolver.cpp")],
            text=True, capture_output=True, check=False)

    @classmethod
    def execute(cls, command):
        result = subprocess.run(command, env=cls.environment, text=True, capture_output=True, check=False)
        if result.returncode:
            raise AssertionError("command failed: " + repr(command) + "\n" + result.stdout + result.stderr)
        return result

    def run_bench(self, kind="hip", scope="compute", fault=None, warmup=2, output_format="jsonl", extra=None, omp_device=0):
        env = self.environment.copy()
        env.pop("GPU_SUITE_SOLVER_FAULT", None)
        env.pop("GPU_SUITE_TEST_NONFINITE", None)
        env["GPU_SUITE_SOLVER_TRACE"] = "1"
        env["OMP_DEFAULT_DEVICE"] = str(omp_device)
        if fault:
            env["GPU_SUITE_SOLVER_FAULT"] = fault
            if kind == "cpu" and fault == "nonfinite":
                env["GPU_SUITE_TEST_NONFINITE"] = "nan"
        command = [str(self.binaries[kind]), "--size", "8", "--nrhs", "3",
                   "--warmup", str(warmup), "--repeat", "1", "--trials", "2",
                   "--scope", scope, "--verify", "true", "--format", output_format,
                   "--output", "-"]
        if kind == "cpu":
            command += ["--cpu-backend", "cpu-generic-lapacke", "--cpu-threads", "1",
                        "--cpu-threads-effective", "1"]
        if extra:
            for option, value in extra.items():
                command[command.index(option) + 1] = value
        result = subprocess.run(command, text=True, capture_output=True, env=env, check=False)
        records = [json.loads(line) for line in result.stdout.splitlines()] if output_format == "jsonl" else list(csv.DictReader(io.StringIO(result.stdout)))
        if output_format == "jsonl":
            for row in records:
                validate_raw_result(row)
        return result, records

    def test_all_variants_both_scopes_restore_destructive_inputs(self):
        for kind in self.binaries:
            for scope in ("compute", "end-to-end"):
                with self.subTest(kind=kind, scope=scope):
                    proc, rows = self.run_bench(kind, scope)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    self.assertEqual(len(rows), 2)
                    for row in rows:
                        self.assertEqual(row["benchmark"], "rocsolver")
                        self.assertEqual(row["status"], "success")
                        self.assertEqual(row["verification_status"], "pass")
                        self.assertEqual(row["getrf_info"], 0)
                        self.assertEqual(row["getrs_info"], 0)
                        self.assertEqual(row["elapsed_sec"], row["elapsed_total_sec"])
                        self.assertEqual(row["parameters"]["nrhs"], 3)
                    if kind != "cpu":
                        self.assertEqual(proc.stderr.count("fixture:getrf\n"), 4)
                        self.assertEqual(proc.stderr.count("fixture:getrs\n"), 4)
                        self.assertEqual(proc.stderr.count("fixture:allocate\n"), proc.stderr.count("fixture:free\n"))
                        self.assertEqual(proc.stderr.count("fixture:create\n"), 1 if scope == "compute" else 4)
                        self.assertEqual(proc.stderr.count("fixture:create\n"), proc.stderr.count("fixture:destroy\n"))

    def test_nonfinite_solution_is_not_a_pass(self):
        for kind in self.binaries:
            for scope in ("compute", "end-to-end"):
                with self.subTest(kind=kind, scope=scope):
                    proc, rows = self.run_bench(kind, scope, "nonfinite", warmup=0)
                    self.assertNotEqual(proc.returncode, 0)
                    self.assertEqual(len(rows), 2)
                    self.assertTrue(all(r["verification_status"] == "nonfinite" for r in rows))
                    self.assertTrue(all(r["failure_origin"] == "verification" for r in rows))

    def test_last_rhs_component_is_verified(self):
        for kind in ("hip", "openmp"):
            proc, rows = self.run_bench(kind, fault="wrong-solution", warmup=0)
            self.assertNotEqual(proc.returncode, 0)
            self.assertTrue(all(r["verification_status"] == "failure" for r in rows))

    def test_singular_factorization_never_calls_getrs(self):
        for kind in ("hip", "openmp"):
            proc, rows = self.run_bench(kind, fault="singular", warmup=0)
            self.assertNotEqual(proc.returncode, 0)
            self.assertNotIn("fixture:getrs\n", proc.stderr)
            self.assertEqual(rows[0]["getrf_info"], 2)
            self.assertIsNone(rows[0]["getrs_info"])
            self.assertIsNone(rows[0]["elapsed_sec"])
            self.assertEqual(rows[1]["failure_origin"], "prior-failure")

    def test_api_failures_keep_unknown_status_null(self):
        for fault in ("getrf", "getrs"):
            for kind in ("hip", "openmp"):
                with self.subTest(fault=fault, kind=kind):
                    proc, rows = self.run_bench(kind, fault=fault, warmup=0)
                    self.assertNotEqual(proc.returncode, 0)
                    self.assertIsNone(rows[0]["getrs_info"])
                    self.assertEqual(rows[0]["getrf_info"], 0 if fault == "getrs" else None)
                    self.assertIsNone(rows[0]["elapsed_sec"])
                    self.assertEqual(proc.stderr.count("fixture:allocate\n"), proc.stderr.count("fixture:free\n"))

    def test_query_copy_allocation_and_completion_errors_fail(self):
        for fault in ("query", "allocation", "copy", "synchronize"):
            proc, rows = self.run_bench(fault=fault, warmup=0)
            with self.subTest(fault=fault):
                self.assertNotEqual(proc.returncode, 0)
                self.assertEqual(len(rows), 2)
                self.assertFalse(any(r["status"] == "success" for r in rows))
                self.assertEqual(proc.stderr.count("fixture:allocate\n"), proc.stderr.count("fixture:free\n"))

    def test_openmp_rejects_host_fallback_and_wrong_mapping(self):
        for fault in ("host-fallback", "wrong-device"):
            proc, rows = self.run_bench("openmp", fault=fault)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(len(rows), 2)
            self.assertNotIn("fixture:getrf\n", proc.stderr)
            self.assertFalse(any(r["status"] == "success" for r in rows))

    def test_distinct_openmp_and_hip_ordinals_remain_valid(self):
        for scope in ("compute", "end-to-end"):
            proc, rows = self.run_bench("openmp", scope, omp_device=1)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("OpenMP offload ordinal=1", proc.stderr)
            self.assertTrue(all(row["device_id"] == 0 for row in rows))
            self.assertTrue(all(row["verification_status"] == "pass" for row in rows))
        proc, rows = self.run_bench("openmp", omp_device=2)
        self.assertNotEqual(proc.returncode, 0)
        self.assertNotIn("fixture:getrf\n", proc.stderr)
        self.assertFalse(any(row["status"] == "success" for row in rows))

    def test_repeat_greater_than_one_is_rejected(self):
        for kind in self.binaries:
            proc, rows = self.run_bench(kind, extra={"--repeat": "2"})
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(rows, [])

    def test_dimensions_overflow_before_device_allocations(self):
        for kind in self.binaries:
            proc, rows = self.run_bench(kind, extra={"--size": "2147483648"})
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(not r["attempted"] for r in rows))
            self.assertNotIn("fixture:allocate\n", proc.stderr)

    def test_csv_records_and_gpu_metadata_are_distinct(self):
        for kind in ("hip", "openmp"):
            proc, rows = self.run_bench(kind, output_format="csv")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(len(rows), 2)
            self.assertEqual(proc.stdout.count("result_schema_version"), 1)
            proc, rows = self.run_bench(kind)
            for row in rows:
                self.assertEqual(row["library_version"], "synthetic-rocsolver")
                self.assertIsNone(row["cuda_runtime_version"])
                self.assertEqual(row["parameters"]["runtime"]["hip_runtime_version"], 1)

    def test_teaching_gpu_sources_run_only_with_fake_apis(self):
        for kind in ("hip", "openmp"):
            env = self.environment.copy()
            env.pop("GPU_SUITE_SOLVER_FAULT", None)
            proc = subprocess.run([str(self.work / (kind + "_example"))], env=env,
                                  capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("max solution error=0", proc.stdout)
            env["GPU_SUITE_SOLVER_FAULT"] = "nonfinite"
            proc = subprocess.run([str(self.work / (kind + "_example"))], env=env,
                                  capture_output=True, text=True, check=False)
            self.assertNotEqual(proc.returncode, 0)

    def test_openmp_structured_blocks_parse(self):
        self.assertEqual(self.openmp_syntax.returncode, 0, self.openmp_syntax.stderr)


if __name__ == "__main__":
    unittest.main()

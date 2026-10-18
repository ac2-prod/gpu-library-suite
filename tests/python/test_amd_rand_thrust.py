"""Host executables and API doubles only; these tests do not prove GPU support."""
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
AMD = ROOT / "amd/c-cpp"
FIXTURE = ROOT / "tests/fixtures/amd_rand_thrust/include"
sys.path.insert(0, str(ROOT / "tools"))
from gpu_suite.results_io import parse_csv_stdout
from gpu_suite.schema import validate_raw_result


@unittest.skipUnless(shutil.which("cc") and shutil.which("c++"), "C and C++ compilers required")
class AmdRandThrustTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path("/tmp/gpu-library-suite-local-build")
        base.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="amd-rand-thrust-", dir=base)
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        include = cls.work / "include/gpu_suite"
        include.mkdir(parents=True)
        template = (ROOT / "cmake/gpu_suite_build_metadata.h.in").read_text()
        substitutions = {
            "GPU_SUITE_META_GIT_METADATA_AVAILABLE": "0", "GPU_SUITE_META_GIT_DIRTY_C": "0",
            "GPU_SUITE_META_BUILD_PROFILE": "cpu", "GPU_SUITE_META_BUILD_TYPE": "Debug",
            "GPU_SUITE_META_C_COMPILER_ID": "host-fixture", "GPU_SUITE_META_CXX_COMPILER_ID": "host-fixture",
            "GPU_SUITE_META_HIP_COMPILER_ID": "host-fixture", "GPU_SUITE_META_C_COMPILER_VERSION": "1",
            "GPU_SUITE_META_CXX_COMPILER_VERSION": "1", "GPU_SUITE_META_HIP_COMPILER_VERSION": "1",
            "GPU_SUITE_BUILD_METADATA_SHA256": "0" * 64,
        }
        (include / "build_metadata.h").write_text(re.sub(
            r"@([A-Z0-9_]+)@", lambda match: substitutions.get(match[1], ""), template))
        cls.common_include = ROOT / "common/c-cpp/include"
        cls.objects = []
        for source in (ROOT / "common/c-cpp/src").glob("*.c"):
            target = cls.work / (source.stem + ".o")
            cls.command(["cc", "-std=c17", "-D_POSIX_C_SOURCE=200809L", "-I", cls.common_include,
                         "-I", cls.work / "include", "-c", source, "-o", target])
            cls.objects.append(target)
        cls.executables = {}
        for family, stem in (("rocrand", "rand"), ("rocthrust", "reduce")):
            for implementation in ("cpu", "hip", "openmp"):
                source = ("openmp_" + family if implementation == "openmp" else
                          stem + ("_gpu" if implementation == "hip" else "_cpu")) + "_bench.cpp"
                output = cls.work / (family + "-" + implementation)
                args = ["c++", "-std=c++17", "-O0", "-Wno-unknown-pragmas", "-I", FIXTURE,
                        "-I", cls.common_include, "-I", cls.work / "include"]
                if implementation != "cpu":
                    args += ["-DGPU_SUITE_HAVE_HIP_RUNTIME_METADATA=1",
                             "-DGPU_SUITE_HIP_LIBRARY_" + family.upper() + "=1"]
                if implementation == "openmp":
                    args += ["-D_OPENMP=201811"]
                if family == "rocthrust" and implementation == "cpu":
                    args += ['-DGPU_SUITE_COMPILED_CPU_BACKEND="cpu-stl-serial"']
                args += [AMD / family / "benchmarks" / source]
                if family == "rocthrust" and implementation == "openmp":
                    args += ["-x", "c++", AMD / family / "examples/rocthrust_wrapper.hip", "-x", "none"]
                args += cls.objects + ["-o", output]
                cls.command(args)
                cls.executables[family, implementation] = output

    @classmethod
    def command(cls, args, environment=None):
        result = subprocess.run(list(map(str, args)), text=True, capture_output=True,
                                env=environment, check=False)
        if result.returncode != 0:
            raise AssertionError(" ".join(map(str, args)) + "\n" + result.stdout + result.stderr)
        return result

    def execute(self, family, implementation="hip", scope="compute", extra=(), **environment):
        args = [str(self.executables[family, implementation]), "--size", "65536", "--warmup", "2",
                "--repeat", "3", "--trials", "2", "--scope", scope, "--verify", "true",
                "--output", "-", "--format", "jsonl"]
        if implementation == "cpu":
            backend = "cpu-std-random-serial" if family == "rocrand" else "cpu-stl-serial"
            args += ["--cpu-backend", backend, "--cpu-threads", "1", "--cpu-threads-effective", "1",
                     "--cpu-backend-role", "production", "--series-role", "primary", "--cpu-parallelism", "serial"]
        # Override values in-place; duplicate CLI arguments remain forbidden.
        for key, value in extra:
            args[args.index(key) + 1] = value
        env = os.environ.copy()
        for key in tuple(env):
            if key.startswith("GPU_SUITE_FIXTURE_"): del env[key]
        env.update(environment)
        result = subprocess.run(args, text=True, capture_output=True, env=env, check=False)
        rows = (parse_csv_stdout(result.stdout) if args[args.index("--format") + 1] == "csv"
                else [json.loads(line) for line in result.stdout.splitlines()])
        return result, rows

    def test_cpu_benchmarks_both_scopes_and_trial_identity(self):
        for family in ("rocrand", "rocthrust"):
            for scope in ("compute", "end-to-end"):
                with self.subTest(family=family, scope=scope):
                    result, rows = self.execute(family, "cpu", scope)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual([row["trial"] for row in rows], [0, 1])
                    for row in rows:
                        validate_raw_result(row)
                        self.assertEqual(row["benchmark"], family)
                        self.assertEqual(row["verification_status"], "pass")
                        self.assertEqual(row["elapsed_sec"], row["elapsed_total_sec"] / 3)

    def test_fake_hip_both_scopes_record_real_family_and_runtime_fields(self):
        for family in ("rocrand", "rocthrust"):
            for scope in ("compute", "end-to-end"):
                with self.subTest(family=family, scope=scope):
                    result, rows = self.execute(family, scope=scope)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(len(rows), 2)
                    for row in rows:
                        validate_raw_result(row)
                        self.assertEqual(row["benchmark"], family)
                        self.assertEqual(row["implementation"], "hip")
                        self.assertEqual(row["gpu_name"], "HOST FIXTURE, NOT A GPU")
                        self.assertEqual(row["verification_status"], "pass")
                        self.assertEqual(row["parameters"]["runtime"]["hip_runtime_version"], 70200000)
                        self.assertIsNone(row["cuda_runtime_version"])

    def test_csv_single_header_and_both_family_schemas(self):
        for family in ("rocrand", "rocthrust"):
            for implementation in ("cpu", "hip"):
                result, rows = self.execute(family, implementation, "end-to-end", extra=(("--format", "csv"),))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.count("result_schema_version"), 1)
                self.assertEqual(len(rows), 2)
                for row in rows:
                    validate_raw_result(row)
                    self.assertEqual(row["benchmark"], family)

    def test_serial_cpu_rejects_false_effective_parallelism(self):
        for family in ("rocrand", "rocthrust"):
            result, rows = self.execute(family, "cpu", extra=(("--cpu-threads-effective", "8"),))
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(rows, [])

    def test_rng_warmup_and_prior_trial_do_not_change_canonical_sequence(self):
        for implementation in ("cpu", "hip"):
            first, rows = self.execute("rocrand", implementation)
            second, without = self.execute("rocrand", implementation, extra=(("--warmup", "0"),))
            self.assertEqual((first.returncode, second.returncode), (0, 0))
            self.assertEqual(rows[0]["verification_metrics"], rows[1]["verification_metrics"])
            self.assertEqual(rows[0]["verification_metrics"], without[0]["verification_metrics"])

    def test_rng_compute_explicit_init_and_boundary_sync_counts(self):
        result, rows = self.execute("rocrand", GPU_SUITE_FIXTURE_EVENTS="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        events = result.stderr.splitlines()
        self.assertNotIn("fixture:lazy_initialize", events)
        self.assertEqual(events.count("fixture:create_generator"), 1)
        self.assertEqual(events.count("fixture:generate"), 8)  # 2 warm-up + 2 * 3 repeats.
        self.assertEqual(events.count("fixture:initialize"), 4)
        self.assertEqual(events.count("fixture:sync"), 7)  # Initializations + warm-up/trial boundaries.
        self.assertEqual(events.count("fixture:destroy_generator"), 1)

    def test_end_to_end_recreates_context_per_repeat(self):
        for family in ("rocrand", "rocthrust"):
            result, _ = self.execute(family, scope="end-to-end", GPU_SUITE_FIXTURE_EVENTS="1")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr.splitlines().count("fixture:allocate"), 8)
            self.assertEqual(result.stderr.splitlines().count("fixture:free"), 8)

    def test_nonfinite_and_bad_range_are_verification_failures(self):
        for family in ("rocrand", "rocthrust"):
            for scope in ("compute", "end-to-end"):
                result, rows = self.execute(family, scope=scope, GPU_SUITE_FIXTURE_NONFINITE="1")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(rows), 2)
                for row in rows:
                    validate_raw_result(row)
                    self.assertEqual(row["failure_origin"], "verification")
                    self.assertEqual(row["verification_status"], "nonfinite")
                    self.assertIsNone(row["verification_metrics"][row["verification_primary_metric"]])
        result, rows = self.execute("rocrand", GPU_SUITE_FIXTURE_BAD_RANGE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(all(row["verification_status"] == "failure" for row in rows))

    def test_generator_failure_releases_owned_context_and_stops(self):
        result, rows = self.execute("rocrand", GPU_SUITE_FIXTURE_GENERATION_FAIL="1", GPU_SUITE_FIXTURE_EVENTS="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([row["attempted"] for row in rows], [True, False])
        self.assertEqual(result.stderr.splitlines().count("fixture:destroy_generator"), 1)
        self.assertEqual(result.stderr.splitlines().count("fixture:free"), 1)

    def test_openmp_host_fallback_stops_before_allocation(self):
        for family in ("rocrand", "rocthrust"):
            result, rows = self.execute(family, "openmp", GPU_SUITE_FIXTURE_EVENTS="1")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("host fallback rejected", result.stderr)
            self.assertNotIn("fixture:allocate", result.stderr)
            self.assertNotIn("fixture:create_generator", result.stderr)
            self.assertTrue(all(not row["attempted"] for row in rows))

    def test_size_overflow_stops_before_gpu_allocation(self):
        for family in ("rocrand", "rocthrust"):
            result, rows = self.execute(family, extra=(("--size", "18446744073709551615"),),
                                        GPU_SUITE_FIXTURE_EVENTS="1")
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("fixture:allocate", result.stderr)
            self.assertTrue(all(not row["attempted"] for row in rows))

    def test_wrapper_catches_standard_and_unknown_exceptions_without_freeing_input(self):
        source = self.work / "wrapper-harness.cpp"
        source.write_text('''#include <cstddef>
extern "C" int rocthrust_square_sum(const double *, std::size_t, double *) noexcept;
int main() { double values[3]={1,2,3}; double result=-1;
  int status=rocthrust_square_sum(values,3,&result);
  return status ? 7 : (result==14 ? 0 : 8); }
''')
        output = self.work / "wrapper-harness"
        self.command(["c++", "-std=c++17", "-I", FIXTURE, source, "-x", "c++",
                      AMD / "rocthrust/examples/rocthrust_wrapper.hip", "-o", output])
        self.command([output])
        for mode in ("GPU_SUITE_FIXTURE_THROW", "GPU_SUITE_FIXTURE_THROW_UNKNOWN"):
            env = os.environ.copy()
            env.update({mode: "1", "GPU_SUITE_FIXTURE_EVENTS": "1"})
            result = subprocess.run([output], env=env, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertNotIn("fixture:free", result.stderr)

    def test_openmp_context_rejects_mismatched_device_without_freeing_mapping(self):
        for family, stem, macro in (("rocrand", "rand", "RAND"), ("rocthrust", "reduce", "REDUCE")):
            source = self.work / (family + "-mapping-harness.cpp")
            source.write_text('''#define GPU_SUITE_''' + macro + '''_OPENMP 1
#include "''' + str(AMD / family / "benchmarks" / (stem + "_gpu_impl.hpp")) + '''"
int main() { gpu_suite_options options{}; options.device=0;
  std::vector<double> values(8,1.0); gpu_suite_''' + family + '''::Context context;
  bool ok=context.create(values,options); context.destroy(); return ok ? 0 : 7; }
''')
            output = self.work / (family + "-mapping-harness")
            args = ["c++", "-std=c++17", "-D_OPENMP=201811", "-Wno-unknown-pragmas", "-I", FIXTURE,
                    "-I", self.common_include, "-I", self.work / "include", source]
            if family == "rocthrust":
                args += ["-x", "c++", AMD / family / "examples/rocthrust_wrapper.hip", "-x", "none"]
            self.command(args + self.objects + ["-o", output])
            env = os.environ.copy()
            env.update(GPU_SUITE_FIXTURE_WRONG_DEVICE="1", GPU_SUITE_FIXTURE_EVENTS="1")
            result = subprocess.run([output], env=env, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertIn("not on requested HIP device", result.stderr)
            self.assertNotIn("fixture:free", result.stderr)

    def test_openmp_sources_parse_with_real_frontend_and_fake_declarations(self):
        clang = shutil.which("clang++")
        if not clang: self.skipTest("Clang frontend unavailable")
        for family in ("rocrand", "rocthrust"):
            self.command([clang, "-std=c++17", "-Xclang", "-fopenmp", "-fsyntax-only",
                          "-I", FIXTURE, "-I", self.common_include,
                          AMD / family / ("examples/openmp_" + family + ".cpp"),
                          AMD / family / ("benchmarks/openmp_" + family + "_bench.cpp")])


if __name__ == "__main__":
    unittest.main()

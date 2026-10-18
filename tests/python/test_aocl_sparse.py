"""AOCL Sparse 5.1 API/numerical host fixtures; no installed AOCL or GPU proof."""
import csv
import hashlib
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
CMAKE = r'''
cmake_minimum_required(VERSION 3.21)
project(aocl_sparse_fixture C)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(GpuSuiteCpuDependencies)
include(GpuSuiteTargets)
include(GpuSuiteStatus)
include(GpuSuiteBuildMetadata)
include(GpuSuiteAmdMetadata)
set(GPU_SUITE_BUILD_CPU ON)
set(GPU_SUITE_BUILD_EXAMPLES ON)
set(GPU_SUITE_BUILD_BENCHMARKS ON)
set(GPU_SUITE_GENERATED_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/generated")
gpu_suite_find_onemkl_sparse()
gpu_suite_find_aocl_sparse()
_gpu_suite_escape_json_string("${GPU_SUITE_AOCL_SPARSE_REASON}" reason)
file(WRITE "${CMAKE_BINARY_DIR}/provider.json"
  "{\"found\":\"${GPU_SUITE_AOCL_SPARSE_FOUND}\",\"reason\":\"${reason}\"}")
add_subdirectory(common/c-cpp)
add_subdirectory(amd/c-cpp/rocsparse)
gpu_suite_prepare_amd_metadata()
file(WRITE "${CMAKE_BINARY_DIR}/metadata.json"
  "{\"dependencies\":${GPU_SUITE_META_DEPENDENCIES_JSON},\"inputs\":${GPU_SUITE_META_AMD_INPUTS_JSON}}")
'''


@unittest.skipUnless(all(shutil.which(tool) for tool in ("cc", "ar", "cmake")),
                     "Host C compiler, ar and CMake are required")
class AoclSparseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        allowed = Path("/tmp/gpu-library-suite-local-build")
        allowed.mkdir(parents=True, exist_ok=True)
        requested = Path(tempfile.gettempdir()).resolve()
        base = requested if requested == allowed.resolve() or allowed.resolve() in requested.parents else allowed
        temporary = tempfile.TemporaryDirectory(prefix="aocl-sparse-fixture-", dir=base)
        cls.addClassCleanup(temporary.cleanup)
        cls.work = Path(temporary.name).resolve()
        cls.source = cls.work / "source"
        cls.source.mkdir()
        cls.environment = {k: v for k, v in os.environ.items() if not k.startswith(("GPU_SUITE_", "FAKE_"))}
        for name in ("CC", "CFLAGS", "CPPFLAGS", "LDFLAGS", "CPATH", "C_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH", "MKLROOT", "AOCL_ROOT", "CMAKE_PREFIX_PATH", "CMAKE_INCLUDE_PATH", "CMAKE_LIBRARY_PATH", "AOCLSPARSE_NUM_THREADS", "AOCL_ENABLE_INSTRUCTIONS", "OMP_NUM_THREADS", "OMP_DYNAMIC"):
            cls.environment.pop(name, None)
        cls.environment["TMPDIR"] = str(cls.work)
        for relative in ("cmake", "common/c-cpp", "amd/c-cpp/rocsparse", "tests/fixtures/aocl_sparse", "tests/fixtures/cpu_libs"):
            shutil.copytree(ROOT / relative, cls.source / relative)
        (cls.source / "CMakeLists.txt").write_text(CMAKE)
        generated = cls.source / "generated/gpu_suite"
        generated.mkdir(parents=True)
        template = (ROOT / "cmake/gpu_suite_build_metadata.h.in").read_text()
        template = re.sub(r"@([A-Z0-9_]+)@", lambda m: "a"*64 if m[1] == "GPU_SUITE_BUILD_METADATA_SHA256" else "0" if m[1] in {"GPU_SUITE_META_GIT_METADATA_AVAILABLE", "GPU_SUITE_META_GIT_DIRTY_C"} else "", template)
        (generated / "build_metadata.h").write_text(template)
        cls.headers = cls.source / "tests/fixtures/aocl_sparse"
        cls.library = cls.work / "libaoclsparse.a"
        for name, flags in (("aoclsparse", []), ("dependent", ["-DFIXTURE_EXTERNAL_DEPENDENCY=1"])):
            cls.checked(["cc", "-std=c17", *flags, "-I", cls.headers, "-c", cls.headers / "fake_aocl_sparse.c", "-o", cls.work / (name + ".o")])
            cls.checked(["ar", "rcs", cls.work / ("lib" + name + ".a"), cls.work / (name + ".o")])
        (cls.work / "dependency.c").write_text("int fixture_dependency(void) { return 7; }\n")
        (cls.work / "empty.c").write_text("int fixture_unrelated(void) { return 0; }\n")
        for name in ("dependency", "empty"):
            cls.checked(["cc", "-std=c17", "-c", cls.work / (name + ".c"), "-o", cls.work / (name + ".o")])
            cls.checked(["ar", "rcs", cls.work / ("lib" + name + ".a"), cls.work / (name + ".o")])
        cls.build, status = cls.configure("built")
        if status["found"] != "TRUE":
            raise AssertionError(status)
        cls.checked(["cmake", "--build", cls.build, "--parallel", "1", "--target", "sparse_cpu", "sparse_cpu_bench"])
        cls.example = cls.build / "amd/c-cpp/rocsparse/sparse_cpu"
        cls.benchmark = cls.build / "amd/c-cpp/rocsparse/sparse_cpu_bench"

    @classmethod
    def command(cls, argv, env=None):
        return subprocess.run(list(map(str, argv)), env=dict(cls.environment, **(env or {})),
                              capture_output=True, text=True, check=False, timeout=90)

    @classmethod
    def checked(cls, argv):
        result = cls.command(argv)
        if result.returncode:
            raise AssertionError(str(argv) + "\n" + result.stdout + result.stderr)
        return result

    @classmethod
    def configure(cls, name, extra=()):
        build = cls.work / name
        result = cls.checked(["cmake", "-S", cls.source, "-B", build,
            "-DGPU_SUITE_VENDOR=amd", "-DGPU_SUITE_CPU_SPARSE_BACKEND=AOCLSPARSE",
            "-DGPU_SUITE_AOCL_SPARSE_INCLUDE_DIR=" + str(cls.headers),
            "-DGPU_SUITE_AOCL_SPARSE_LIBRARY=" + str(cls.library), *extra])
        (cls.work / (name + ".stdout")).write_text(result.stdout)
        (cls.work / (name + ".stderr")).write_text(result.stderr)
        return build, json.loads((build / "provider.json").read_text())

    def invoke(self, scope="compute", fmt="jsonl", extra=(), env=None):
        argv = [self.benchmark, "--nx", "3", "--ny", "4", "--warmup", "2", "--repeat", "3", "--trials", "2", "--scope", scope, "--verify", "true", "--output", "-", "--format", fmt, "--alpha", "1.25", "--beta", "0.5", "--cpu-backend", "cpu-aocl-sparse", "--cpu-backend-role", "production", "--series-role", "primary", "--cpu-threads", "2", "--cpu-parallelism", "threaded"]
        for key, value in zip(extra[::2], extra[1::2]):
            if key in argv:
                argv[argv.index(key) + 1] = value
            else:
                argv.extend((key, value))
        return self.command(argv, env)

    def test_explicit_provider_builds_teaching_and_benchmark_with_strict_c17(self):
        result = self.command([self.example])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("AOCL Sparse CPU", result.stdout)
        self.assertIn("max error = 0", result.stdout)
        commands = json.loads((self.build / "compile_commands.json").read_text())
        for stem in ("sparse_cpu", "sparse_cpu_bench"):
            command = next(row["command"] for row in commands if Path(row["file"]).stem == stem)
            self.assertIn("-std=c17", command)
            self.assertIn("GPU_SUITE_USE_AOCL_SPARSE", command)
            self.assertNotIn("GPU_SUITE_USE_MKL_SPARSE", command)

    def test_numerics_restoration_threads_version_and_formats(self):
        for scope in ("compute", "end-to-end"):
            for fmt in ("jsonl", "csv"):
                with self.subTest(scope=scope, fmt=fmt):
                    trace = self.work / (scope + "-" + fmt + ".trace")
                    result = self.invoke(scope, fmt, env={"GPU_SUITE_AOCL_FIXTURE_TRACE": str(trace), "GPU_SUITE_AOCL_FIXTURE_EXPECT_THREADS": "2", "AOCLSPARSE_NUM_THREADS": "99", "OMP_NUM_THREADS": "88", "OMP_DYNAMIC": "TRUE"})
                    self.assertEqual(result.returncode, 0, result.stderr)
                    rows = [json.loads(line) for line in result.stdout.splitlines()] if fmt == "jsonl" else list(csv.DictReader(io.StringIO(result.stdout)))
                    self.assertEqual(len(rows), 2)
                    for row in rows:
                        self.assertEqual(row["status"], "success")
                        self.assertEqual(row["verification_status"], "pass")
                        self.assertEqual(row["library_version"], "AOCL-Sparse 5.1-fixture")
                        self.assertIn(row["cpu_threads_effective"], (None, ""))
                        self.assertEqual(int(row["cpu_threads_requested"]), 2)
                        parameters = row["parameters"] if fmt == "jsonl" else json.loads(row["parameters"])
                        self.assertEqual(parameters["runtime"]["aocl_sparse_environment"], {
                            "AOCLSPARSE_NUM_THREADS": "2", "OMP_NUM_THREADS": "2",
                            "OMP_DYNAMIC": "FALSE", "AOCL_ENABLE_INSTRUCTIONS": None})
                    events = trace.read_text().splitlines()
                    before = [float(line.split()[1]) for line in events if line.startswith("mv ")]
                    expected = [1, 3, 1, 3, 4, 1, 3, 4] if scope == "compute" else [1] * 8
                    self.assertEqual(before, expected)
                    for created, destroyed in (("create", "destroy"), ("create", "hint"), ("create", "optimize"), ("descriptor", "destroy_descriptor")):
                        self.assertEqual(sum(line.startswith(created + " ") for line in events), sum(line.startswith(destroyed + " ") for line in events))

    def test_provider_errors_fail_and_cleanup_partial_descriptors(self):
        for scope in ("compute", "end-to-end"):
            for failure in ("descriptor", "base", "type", "diag", "create", "hint", "optimize", "mv", "destroy", "destroy_descriptor"):
                with self.subTest(scope=scope, failure=failure):
                    trace = self.work / ("failure-" + scope + "-" + failure + ".trace")
                    result = self.invoke(scope, extra=("--trials", "1"), env={"GPU_SUITE_AOCL_FIXTURE_FAIL": failure, "GPU_SUITE_AOCL_FIXTURE_TRACE": str(trace)})
                    self.assertNotEqual(result.returncode, 0)
                    rows = list(map(json.loads, result.stdout.splitlines()))
                    self.assertEqual(len(rows), 1)
                    self.assertTrue(all(row["status"] != "success" for row in rows))
                    self.assertTrue(all(row["library_version"] == "AOCL-Sparse 5.1-fixture" for row in rows))
                    events = [line.split()[0] for line in trace.read_text().splitlines()]
                    self.assertEqual(events.count("destroy_descriptor"), 0 if failure == "descriptor" else 1)
                    self.assertEqual(events.count("destroy"), 1 if failure in {"hint", "optimize", "mv", "destroy", "destroy_descriptor"} else 0)

    def test_nonfinite_and_invalid_dimensions_never_pass(self):
        for scope in ("compute", "end-to-end"):
            result = self.invoke(scope, env={"GPU_SUITE_TEST_NONFINITE": "nan"})
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(all(row["verification_status"] == "nonfinite" for row in map(json.loads, result.stdout.splitlines())))
        result = self.invoke(extra=("--nx", "2147483648"))
        self.assertNotEqual(result.returncode, 0)
        rows = list(map(json.loads, result.stdout.splitlines()))
        self.assertEqual(len(rows), 2, result.stderr)
        self.assertTrue(all(not row["attempted"] and row["elapsed_sec"] is None for row in rows))
        self.assertTrue(all(row["library_version"] == "AOCL-Sparse 5.1-fixture" for row in rows))

    def test_empty_version_and_backend_mismatch_stop(self):
        result = self.invoke(env={"GPU_SUITE_AOCL_FIXTURE_FAIL": "version"})
        self.assertNotEqual(result.returncode, 0)
        rows = list(map(json.loads, result.stdout.splitlines()))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["status"] != "success" for row in rows))
        result = self.invoke(extra=("--cpu-backend", "cpu-onemkl"))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("backend mismatch", result.stderr)

    def test_zero_warmup_preserves_first_trial_and_repeat_restoration(self):
        for scope in ("compute", "end-to-end"):
            trace = self.work / ("zero-" + scope + ".trace")
            result = self.invoke(scope, extra=("--warmup", "0"), env={"GPU_SUITE_AOCL_FIXTURE_TRACE": str(trace)})
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = list(map(json.loads, result.stdout.splitlines()))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["verification_status"] == "pass" for row in rows))
            before = [float(line.split()[1]) for line in trace.read_text().splitlines() if line.startswith("mv ")]
            self.assertEqual(before, [1, 3, 4, 1, 3, 4] if scope == "compute" else [1] * 6)

    def test_onemkl_and_explicit_reference_remain_independent(self):
        headers = self.source / "tests/fixtures/cpu_libs/include"
        obj = self.work / "mkl.o"
        library = self.work / "libmkl_rt.a"
        self.checked(["cc", "-std=c17", "-I", headers, "-c", self.source / "tests/fixtures/cpu_libs/fake_cpu_providers.c", "-o", obj])
        self.checked(["ar", "rcs", library, obj])
        for backend, label in (("ONEMKL", "oneMKL"), ("REFERENCE", "Reference")):
            build, status = self.configure("retained-" + backend, (
                "-DGPU_SUITE_CPU_SPARSE_BACKEND=" + backend,
                "-DGPU_SUITE_MKL_SPARSE_INCLUDE_DIR=" + str(headers),
                "-DGPU_SUITE_MKL_SPARSE_LIBRARY=" + str(library)))
            self.assertEqual(status["found"], "FALSE")
            self.checked(["cmake", "--build", build, "--target", "sparse_cpu", "sparse_cpu_bench"])
            result = self.command([build / "amd/c-cpp/rocsparse/sparse_cpu"])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(label, result.stdout)
            self.assertNotIn("aocl_sparse", json.loads((build / "metadata.json").read_text())["dependencies"])

    def test_missing_header_or_api_disables_only_sparse_without_fallback(self):
        for name, option in (("missing-header", "-DGPU_SUITE_AOCL_SPARSE_INCLUDE_DIR=" + str(self.work / "missing")), ("missing-api", "-DGPU_SUITE_AOCL_SPARSE_LIBRARY=" + str(self.work / "libempty.a"))):
            build, status = self.configure(name, (option,))
            self.assertEqual(status["found"], "FALSE")
            targets = self.checked(["cmake", "--build", build, "--target", "help"]).stdout
            self.assertNotIn("sparse_cpu", targets)
            self.assertIn("gpu_suite_common", targets)
            self.assertEqual(json.loads((build / "metadata.json").read_text())["dependencies"], {})

    def test_ilp64_is_rejected_at_compile_probe_and_manual_source(self):
        _, status = self.configure("ilp64", ("-DCMAKE_C_FLAGS=-DFIXTURE_ILP64=1",))
        self.assertEqual(status["found"], "FALSE")
        self.assertIn("LP64", status["reason"])
        result = self.command(["cc", "-std=c17", "-DFIXTURE_ILP64=1", "-DGPU_SUITE_USE_AOCL_SPARSE=1", "-I", self.headers, "-fsyntax-only", self.source / "amd/c-cpp/rocsparse/examples/sparse_cpu.c"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LP64", result.stderr)
        ilp64 = self.work / "lib_ILP64"
        ilp64.mkdir()
        shutil.copyfile(self.library, ilp64 / "libaoclsparse.a")
        _, status = self.configure("ilp64-library", (
            "-DGPU_SUITE_AOCL_SPARSE_LIBRARY=" + str(ilp64 / "libaoclsparse.a"),))
        self.assertEqual(status["found"], "FALSE")
        self.assertIn("ILP64 paths", status["reason"])

    def test_explicit_ordered_extra_libraries_are_required_and_hashed(self):
        primary = self.work / "libdependent.a"
        extra = self.work / "libdependency.a"
        options = ("-DGPU_SUITE_AOCL_SPARSE_LIBRARY=" + str(primary),)
        _, status = self.configure("unresolved-extra", options)
        self.assertEqual(status["found"], "FALSE")
        build, status = self.configure("resolved-extra", (*options, "-DGPU_SUITE_AOCL_SPARSE_EXTRA_LIBRARIES=" + str(extra)))
        self.assertEqual(status["found"], "TRUE")
        self.checked(["cmake", "--build", build, "--target", "sparse_cpu"])
        metadata = json.loads((build / "metadata.json").read_text())
        self.assertEqual(metadata["inputs"]["AOCLSparse_INTERFACE_LINK_LIBRARIES"].split(";")[:2], [str(primary), str(extra)])
        for key, path in (("aocl_sparse", primary), ("aocl_sparse_link_1", extra), ("aocl_sparse_header", self.headers / "aoclsparse.h")):
            self.assertEqual(metadata["dependencies"][key], {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    def test_invalid_explicit_extra_inputs_cannot_become_flags_or_searches(self):
        for i, value in enumerate(("-lgomp", "relative.a", str(self.work), str(self.work / "absent.a"))):
            _, status = self.configure("invalid-extra-" + str(i), ("-DGPU_SUITE_AOCL_SPARSE_EXTRA_LIBRARIES=" + value,))
            self.assertEqual(status["found"], "FALSE")
            self.assertIn("existing absolute files", status["reason"])

    def test_nonselected_provider_does_not_enable_aocl(self):
        for backend in ("REFERENCE", "INVALID"):
            _, status = self.configure("selection-" + backend, ("-DGPU_SUITE_CPU_SPARSE_BACKEND=" + backend,))
            self.assertEqual(status["found"], "FALSE")
            self.assertIn("explicit", status["reason"])
        _, status = self.configure("nvidia", ("-DGPU_SUITE_VENDOR=nvidia",))
        self.assertEqual(status["found"], "FALSE")


if __name__ == "__main__":
    unittest.main()

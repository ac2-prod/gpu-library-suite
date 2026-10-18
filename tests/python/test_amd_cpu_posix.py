"""Strict-C17 feature-selection fixtures, not verification of installed AOCL."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
TARGETS = {
    "blas_cpu": "rocblas/examples/blas_cpu.c",
    "blas_cpu_bench": "rocblas/benchmarks/blas_cpu_bench.c",
    "solver_cpu": "rocsolver/examples/solver_cpu.c",
    "solver_cpu_bench": "rocsolver/benchmarks/solver_cpu_bench.c",
}
FEATURES = r'''
#ifndef FIXTURE_FEATURES_FROZEN
#define FIXTURE_FEATURES_FROZEN
/* Model libc's one-time feature selection, even on hosts without barriers. */
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200809L
typedef struct { int value; } fixture_pthread_barrier_t;
typedef struct { int value; } fixture_pthread_barrierattr_t;
#endif
#endif
'''
PROVIDER_PREFIX = r'''
/* Some providers select features themselves, too late after the suite header. */
#if !defined(FIXTURE_PROVIDER_DOES_NOT_SELECT_FEATURES) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include "fixture_features.h"
#if defined(FIXTURE_EXPECT_POSIX) && _POSIX_C_SOURCE != FIXTURE_EXPECT_POSIX
#error "Caller POSIX feature level was changed"
#endif
typedef struct {
  fixture_pthread_barrier_t barrier;
  fixture_pthread_barrierattr_t attributes;
} fixture_provider_barriers;
'''
CMAKE = r'''
cmake_minimum_required(VERSION 3.21)
project(amd_cpu_posix_fixture C)
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
set(GPU_SUITE_CPU_BLAS_BACKEND GENERIC_CBLAS)
set(GPU_SUITE_CPU_LAPACK_BACKEND GENERIC_LAPACKE)
set(GPU_SUITE_GENERIC_CBLAS_INCLUDE_DIR "${FIXTURE_HEADERS}")
set(GPU_SUITE_GENERIC_LAPACKE_INCLUDE_DIR "${FIXTURE_HEADERS}")
set(GPU_SUITE_GENERIC_CBLAS_LIBRARY "${FIXTURE_LIBRARY}")
set(GPU_SUITE_GENERIC_LAPACKE_LIBRARY "${FIXTURE_LIBRARY}")
gpu_suite_find_cblas()
gpu_suite_find_lapacke()
file(WRITE "${CMAKE_BINARY_DIR}/providers.txt"
  "${GPU_SUITE_CBLAS_FOUND}\n${GPU_SUITE_LAPACKE_FOUND}\n${GPU_SUITE_CBLAS_REASON}\n${GPU_SUITE_LAPACKE_REASON}\n")
if(GPU_SUITE_VENDOR STREQUAL "amd")
  set(GPU_SUITE_GENERATED_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/generated")
  add_subdirectory(common/c-cpp)
  # Freeze feature selection when the real benchmark's umbrella is included.
  target_include_directories(gpu_suite_common BEFORE PUBLIC "${FIXTURE_HEADERS}")
  add_subdirectory(amd/c-cpp/rocblas)
  add_subdirectory(amd/c-cpp/rocsolver)
endif()
gpu_suite_prepare_amd_metadata()
file(WRITE "${CMAKE_BINARY_DIR}/metadata.json"
  "{\"inputs\":${GPU_SUITE_META_AMD_INPUTS_JSON},\"dependencies\":${GPU_SUITE_META_DEPENDENCIES_JSON}}")
'''


@unittest.skipUnless(shutil.which("cc") and shutil.which("ar") and shutil.which("cmake"),
                     "Host C compiler, archive tool and CMake are required")
class AmdCpuPosixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        allowed = Path("/tmp/gpu-library-suite-local-build")
        allowed.mkdir(parents=True, exist_ok=True)
        requested = Path(tempfile.gettempdir()).resolve()
        base = requested if requested == allowed.resolve() or allowed.resolve() in requested.parents else allowed
        temporary = tempfile.TemporaryDirectory(prefix="amd-cpu-posix-", dir=base)
        cls.addClassCleanup(temporary.cleanup)
        cls.work = Path(temporary.name).resolve()
        cls.source = cls.work / "source"
        cls.source.mkdir()
        cls.environment = dict(os.environ, TMPDIR=str(cls.work))
        for name in ("CC", "CFLAGS", "CPPFLAGS", "LDFLAGS", "CPATH", "C_INCLUDE_PATH",
                     "LIBRARY_PATH", "LD_LIBRARY_PATH", "CMAKE_PREFIX_PATH", "CMAKE_INCLUDE_PATH",
                     "CMAKE_LIBRARY_PATH", "GPU_SUITE_TEST_NONFINITE"):
            cls.environment.pop(name, None)
        for relative in ("cmake", "common/c-cpp"):
            shutil.copytree(ROOT / relative, cls.source / relative)
        for family in ("rocblas", "rocsolver"):
            destination = cls.source / "amd/c-cpp" / family
            destination.mkdir(parents=True)
            shutil.copyfile(ROOT / "amd/c-cpp" / family / "CMakeLists.txt", destination / "CMakeLists.txt")
        for relative in TARGETS.values():
            destination = cls.source / "amd/c-cpp" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "amd/c-cpp" / relative, destination)
        generated = cls.source / "generated/gpu_suite"
        generated.mkdir(parents=True)
        metadata = (ROOT / "cmake/gpu_suite_build_metadata.h.in").read_text()
        metadata = re.sub(r"@([A-Z0-9_]+)@", lambda match:
                          "a" * 64 if match[1] == "GPU_SUITE_BUILD_METADATA_SHA256"
                          else "0" if match[1] in {"GPU_SUITE_META_GIT_METADATA_AVAILABLE", "GPU_SUITE_META_GIT_DIRTY_C"}
                          else "", metadata)
        (generated / "build_metadata.h").write_text(metadata)
        cls.headers = cls.work / "provider/include"
        cls.headers.mkdir(parents=True)
        (cls.headers / "fixture_features.h").write_text(FEATURES)
        base_headers = cls.work / "provider/base"
        shutil.copytree(ROOT / "tests/fixtures/cpu_libs/include", base_headers)
        for name in ("cblas.h", "lapacke.h"):
            guard = "FIXTURE_" + name.replace(".", "_").upper()
            (cls.headers / name).write_text("#ifndef " + guard + "\n#define " + guard + "\n" +
                PROVIDER_PREFIX.replace("fixture_provider_barriers", "fixture_" + guard + "_barriers") +
                '\n#include "' + str(base_headers / name) + '"\n#endif\n')
        umbrella = cls.headers / "gpu_suite/gpu_suite.h"
        umbrella.parent.mkdir()
        umbrella.write_text('#include "fixture_features.h"\n#include "' +
                            str(cls.source / "common/c-cpp/include/gpu_suite/gpu_suite.h") + '"\n')
        provider = cls.work / "provider.c"
        shutil.copyfile(ROOT / "tests/fixtures/cpu_libs/fake_cpu_providers.c", provider)
        cls.library = cls.work / "libfixture_provider.a"
        for command in (["cc", "-std=c17", "-I", base_headers, "-c", provider, "-o", cls.work / "provider.o"],
                        ["ar", "rcs", cls.library, cls.work / "provider.o"]):
            result = cls.command(command)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
        (cls.source / "CMakeLists.txt").write_text(CMAKE)

    @classmethod
    def command(cls, argv):
        return subprocess.run(list(map(str, argv)), env=cls.environment,
                              capture_output=True, text=True, check=False, timeout=90)

    def syntax(self, source, extra=()):
        provider = "CBLAS" if "blas_cpu" in source.name else "LAPACKE"
        header = "cblas.h" if provider == "CBLAS" else "lapacke.h"
        backend = "cpu-generic-cblas" if provider == "CBLAS" else "cpu-generic-lapacke"
        return self.command(["cc", "-std=c17", "-Werror", "-fsyntax-only", *extra,
                             "-I", self.headers, "-I", self.source / "common/c-cpp/include",
                             "-I", self.source / "generated", '-DGPU_SUITE_' + provider + '_HEADER="' + header + '"',
                             '-DGPU_SUITE_COMPILED_CPU_BACKEND="' + backend + '"',
                             source])

    def configure(self, name, vendor="amd", extra_flags=""):
        directory = self.work / name
        flags = "-DFIXTURE_PROVIDER_DOES_NOT_SELECT_FEATURES=1 " + extra_flags
        result = self.command(["cmake", "-S", self.source, "-B", directory,
                               "-DGPU_SUITE_VENDOR=" + vendor,
                               "-DFIXTURE_HEADERS=" + str(self.headers), "-DFIXTURE_LIBRARY=" + str(self.library),
                               "-DCMAKE_C_FLAGS=" + flags])
        (self.work / (name + ".stdout")).write_text(result.stdout)
        (self.work / (name + ".stderr")).write_text(result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return directory, (directory / "providers.txt").read_text().splitlines()

    def test_benchmark_late_feature_selection_fails_and_early_selection_compiles(self):
        for target in ("blas_cpu_bench", "solver_cpu_bench"):
            with self.subTest(target=target):
                source = self.source / "amd/c-cpp" / TARGETS[target]
                text = source.read_text()
                legacy = self.work / (target + "-before.c")
                legacy.write_text(text[text.index('#include "gpu_suite/gpu_suite.h"'):])
                before = self.syntax(legacy)
                self.assertNotEqual(before.returncode, 0)
                self.assertIn("fixture_pthread_barrier", before.stderr)
                after = self.syntax(source)
                self.assertEqual(after.returncode, 0, after.stdout + after.stderr)
                (self.work / (target + ".before.stderr")).write_text(before.stderr)
                (self.work / (target + ".after.stderr")).write_text(after.stderr)

    def test_manual_cpu_sources_preserve_higher_and_reject_lower_feature_levels(self):
        for target, relative in TARGETS.items():
            with self.subTest(target=target):
                source = self.source / "amd/c-cpp" / relative
                for level in (200809, 202405):
                    result = self.syntax(source, ("-D_POSIX_C_SOURCE=" + str(level) + "L",
                                                  "-DFIXTURE_EXPECT_POSIX=" + str(level) + "L"))
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                result = self.syntax(source, ("-D_POSIX_C_SOURCE=200112L",))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("AMD CPU library headers require _POSIX_C_SOURCE >= 200809L", result.stderr)

    def test_configured_amd_cpu_targets_compile_and_link_with_sensitive_headers(self):
        directory, status = self.configure("configured-amd")
        self.assertEqual(status[:2], ["TRUE", "TRUE"])
        result = self.command(["cmake", "--build", directory, "--parallel", "1", "--target", *TARGETS])
        (self.work / "configured-amd-build.stdout").write_text(result.stdout)
        (self.work / "configured-amd-build.stderr").write_text(result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for target, relative in TARGETS.items():
            self.assertTrue((directory / "amd/c-cpp" / Path(relative).parts[0] / target).is_file())
        commands = json.loads((directory / "compile_commands.json").read_text())
        for target in TARGETS:
            command = next(row["command"] for row in commands if Path(row["file"]).stem == target)
            self.assertIn("-std=c17", command)
            self.assertNotIn("-D_POSIX_C_SOURCE", command)

    def test_probe_feature_selection_is_amd_only_and_preserves_caller_level(self):
        _, status = self.configure("probe-nvidia", vendor="nvidia")
        self.assertEqual(status[:2], ["FALSE", "FALSE"])
        self.assertTrue(all("did not compile and link" in reason for reason in status[2:4]))
        _, status = self.configure("probe-higher", extra_flags="-D_POSIX_C_SOURCE=202405L -DFIXTURE_EXPECT_POSIX=202405L")
        self.assertEqual(status[:2], ["TRUE", "TRUE"])
        _, status = self.configure("probe-lower", extra_flags="-D_POSIX_C_SOURCE=200112L")
        self.assertEqual(status[:2], ["FALSE", "FALSE"])
        self.assertTrue(all("did not compile and link" in reason for reason in status[2:4]))


if __name__ == "__main__":
    unittest.main()

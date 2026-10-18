"""Compile the actual CMake probes on the host with synthetic API declarations.

These tests verify C++ source composition and link requirements only. OpenMP
pragmas are ignored deliberately; no probe executable is run and no GPU or
OpenMP offload support is claimed.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

HIP_HEADER = r"""
#include <stddef.h>
typedef struct { int type, device; } hipPointerAttribute_t;
typedef struct { int unused; } hipDeviceProp_t;
enum hipMemcpyKind { hipMemcpyHostToDevice, hipMemcpyDeviceToHost };
extern "C" {
int hipGetDeviceProperties(hipDeviceProp_t *, int);
int hipRuntimeGetVersion(int *);
int hipDriverGetVersion(int *);
int hipPointerGetAttributes(hipPointerAttribute_t *, const void *);
int hipMalloc(void **, size_t);
int hipMemcpy(void *, const void *, size_t, hipMemcpyKind);
int hipDeviceSynchronize(void);
int hipFree(void *);
}
"""

ROCFFT_HEADER = r"""
#include <stddef.h>
typedef void *rocfft_plan;
typedef void *rocfft_execution_info;
enum rocfft_result_placement { rocfft_placement_notinplace };
enum rocfft_transform_type { rocfft_transform_type_complex_forward };
enum rocfft_precision { rocfft_precision_single };
extern "C" {
int rocfft_get_version_string(char *, size_t);
int rocfft_setup(void);
int rocfft_plan_create(rocfft_plan *, rocfft_result_placement,
                      rocfft_transform_type, rocfft_precision, size_t,
                      const size_t *, size_t, const void *);
int rocfft_execution_info_create(rocfft_execution_info *);
int rocfft_plan_get_work_buffer_size(rocfft_plan, size_t *);
int rocfft_execution_info_set_work_buffer(rocfft_execution_info, void *, size_t);
int rocfft_execute(rocfft_plan, void **, void **, rocfft_execution_info);
int rocfft_execution_info_destroy(rocfft_execution_info);
int rocfft_plan_destroy(rocfft_plan);
int rocfft_cleanup(void);
}
"""

API_STUB = r"""
#include <hip/hip_runtime_api.h>
#include <rocfft/rocfft.h>
int hipGetDeviceProperties(hipDeviceProp_t *, int) { return 0; }
int hipRuntimeGetVersion(int *) { return 0; }
int hipDriverGetVersion(int *) { return 0; }
int hipPointerGetAttributes(hipPointerAttribute_t *, const void *) { return 0; }
int hipMalloc(void **, size_t) { return 0; }
int hipMemcpy(void *, const void *, size_t, hipMemcpyKind) { return 0; }
int hipDeviceSynchronize(void) { return 0; }
int hipFree(void *) { return 0; }
int rocfft_get_version_string(char *, size_t) { return 0; }
int rocfft_setup(void) { return 0; }
int rocfft_plan_create(rocfft_plan *, rocfft_result_placement,
                      rocfft_transform_type, rocfft_precision, size_t,
                      const size_t *, size_t, const void *) { return 0; }
int rocfft_execution_info_create(rocfft_execution_info *) { return 0; }
int rocfft_plan_get_work_buffer_size(rocfft_plan, size_t *) { return 0; }
int rocfft_execution_info_set_work_buffer(rocfft_execution_info, void *, size_t) { return 0; }
int rocfft_execute(rocfft_plan, void **, void **, rocfft_execution_info) { return 0; }
int rocfft_execution_info_destroy(rocfft_execution_info) { return 0; }
int rocfft_plan_destroy(rocfft_plan) { return 0; }
int rocfft_cleanup(void) { return 0; }
"""


@unittest.skipUnless(shutil.which("cmake") and shutil.which("cc") and shutil.which("c++"),
                     "CMake and host C/C++ compilers are required")
class RocfftProbeCompileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rocfft-probe-compile-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = os.environ.copy()
        for name in ("ROCM_PATH", "HIP_PATH", "CMAKE_PREFIX_PATH", "CXXFLAGS", "LDFLAGS"):
            self.environment.pop(name, None)
        include = self.root / "include"
        (include / "hip").mkdir(parents=True)
        (include / "rocfft").mkdir()
        (include / "hip/hip_runtime_api.h").write_text(HIP_HEADER)
        (include / "rocfft/rocfft.h").write_text(ROCFFT_HEADER)
        (include / "omp.h").write_text("inline int omp_is_initial_device() { return 1; }\n")
        # FindRocFFT resolves concrete files here, but no synthetic library is
        # consumed by CMake. The host compiler below links the API stub object.
        library = self.root / "unlinked-fixture.a"
        library.write_bytes(b"")
        project = self.root / "project"
        project.mkdir()
        (project / "CMakeLists.txt").write_text("\n".join((
            "cmake_minimum_required(VERSION 3.21)",
            "project(rocfft_probe_sources NONE)",
            'list(APPEND CMAKE_MODULE_PATH "' + str(ROOT / "cmake") + '")',
            "set(GPU_SUITE_BUILD_CPU OFF)",
            "set(GPU_SUITE_BUILD_HIP OFF)",
            "set(GPU_SUITE_BUILD_OPENMP ON)",
            "set(GPU_SUITE_AMD_LIBRARIES rocfft)",
            'set(RocFFT_INCLUDE_DIR "' + str(include) + '")',
            'set(RocFFT_HIP_INCLUDE_DIR "' + str(include) + '")',
            'set(RocFFT_LIBRARY "' + str(library) + '")',
            'set(RocFFT_HIP_LIBRARY "' + str(library) + '")',
            "include(GpuSuiteAmd)",
            'file(WRITE "${CMAKE_BINARY_DIR}/hip.cpp" "${GPU_SUITE_ROCFFT_API_PROBE}")',
            'file(WRITE "${CMAKE_BINARY_DIR}/openmp.cpp" "${GPU_SUITE_ROCFFT_OPENMP_API_PROBE}")',
            "")))
        self.build = self.root / "build"
        self.command(["cmake", "-S", project, "-B", self.build])
        stub = self.root / "api.cpp"
        stub.write_text(API_STUB)
        self.api_object = self.root / "api.o"
        self.compile(stub, self.api_object)

    def command(self, args, success=True):
        result = subprocess.run([str(item) for item in args], capture_output=True,
                                text=True, env=self.environment, check=False)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def compile(self, source, output):
        self.command(["c++", "-std=c++17", "-O2", "-Wno-unknown-pragmas",
                      "-I", self.root / "include", "-c", source, "-o", output])

    def probe_object(self, profile):
        source = self.build / (profile + ".cpp")
        self.assertTrue(source.read_text().strip(), "CMake did not export the probe source")
        output = self.root / (profile + ".o")
        self.compile(source, output)
        return output

    def test_actual_composed_openmp_probe_compiles_and_links_on_host(self):
        output = self.probe_object("openmp")
        self.command(["c++", output, self.api_object, "-o", self.root / "openmp-probe"])

    def test_actual_hip_api_probe_compiles_and_links_with_cpp_runtime(self):
        output = self.probe_object("hip")
        self.command(["c++", output, self.api_object, "-o", self.root / "hip-probe"])

    def test_optimized_probes_reject_link_without_cpp_runtime(self):
        for profile in ("hip", "openmp"):
            with self.subTest(profile=profile):
                output = self.probe_object(profile)
                result = self.command(["cc", output, self.api_object,
                                       "-o", self.root / (profile + "-without-cpp")],
                                      success=False)
                self.assertIn("nothrow", result.stderr)


if __name__ == "__main__":
    unittest.main()

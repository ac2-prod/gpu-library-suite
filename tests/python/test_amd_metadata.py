"""Synthetic CMake metadata checks: no SDK, compiler, GPU or remote execution."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = "Ninja" if shutil.which("ninja") else "Unix Makefiles"

CMAKE_SOURCE = r'''
cmake_minimum_required(VERSION 3.21)
project(amd_metadata_fixture NONE)
list(APPEND CMAKE_MODULE_PATH "@MODULES@")
include(GpuSuiteBuildMetadata)
include(GpuSuiteAmdMetadata)
set(GPU_SUITE_META_GIT_COMMIT_JSON_VALUE null)
set(GPU_SUITE_META_GIT_DIRTY_JSON null)
set(GPU_SUITE_META_GIT_METADATA_AVAILABLE_JSON false)
set(GPU_SUITE_META_BUILD_PROFILE_JSON openmp)
set(GPU_SUITE_META_BUILD_TYPE_JSON Fixture)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_COMPILER_ID FixtureClang)
set(CMAKE_HIP_COMPILER_VERSION fixture)
set(CMAKE_HIP_COMPILER "${CMAKE_SOURCE_DIR}/nonexecuted-hip-compiler")
set(CMAKE_CXX_COMPILER "${CMAKE_SOURCE_DIR}/nonexecuted-openmp-linker")
set(CMAKE_HIP_FLAGS "-DFIXTURE_HIP")
set(CMAKE_HIP_ARCHITECTURES gfx900)
set(GPU_SUITE_OPENMP_COMPILE_OPTIONS "-fopenmp;--offload-arch=gfx900")
set(GPU_SUITE_OPENMP_LINK_OPTIONS "-fopenmp;--offload-arch=gfx900")
set(_cxx_flags "-DFIXTURE_CALLER")
set(_link_flags "-Wl,fixture")
set(GPU_SUITE_AMD_LIBRARIES "${TEST_LIBRARIES}")
set(GPU_SUITE_CPU_THRUST_BACKEND "${TEST_CPU_THRUST}")
set(GPU_SUITE_CPU_LAPACK_BACKEND "${TEST_CPU_LAPACK}")
set(OpenMP_CXX_FLAGS "${TEST_CPU_OMP_FLAGS}")
set(RocFFT_FOUND FALSE)
set(RocBLAS_FOUND TRUE)
set(RocBLAS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(RocBLAS_LIBRARY "${CMAKE_SOURCE_DIR}/lib/librocblas.so")
set(RocBLAS_HIP_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(RocBLAS_HIP_LIBRARY "${CMAKE_SOURCE_DIR}/lib/libamdhip64.so")
if(TEST_THRUST)
  set(RocThrust_FOUND TRUE)
  set(RocThrust_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
  set(RocThrust_ROCPRIM_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
  set(RocThrust_HIP_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
  set(RocThrust_HIP_LIBRARY "${CMAKE_SOURCE_DIR}/lib/libamdhip64.so")
endif()
if(TEST_MISMATCH)
  set(RocSOLVER_FOUND TRUE)
  set(RocSOLVER_HIP_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/other-include")
endif()
if(TEST_PROVIDER)
  add_library(GpuSuite::LAPACKE INTERFACE IMPORTED)
  set_target_properties(GpuSuite::LAPACKE PROPERTIES
    INTERFACE_LINK_LIBRARIES "${CMAKE_SOURCE_DIR}/lib/liblapacke.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/cpu-include"
    INTERFACE_COMPILE_DEFINITIONS "GPU_SUITE_LAPACKE_HEADER=\"lapacke.h\""
    INTERFACE_COMPILE_OPTIONS "${TEST_PROVIDER_FLAGS}"
    INTERFACE_LINK_OPTIONS "-Wl,provider")
endif()
set(GPU_SUITE_ROCTHRUST_OPENMP_ENABLED "${TEST_WRAPPER}")
gpu_suite_prepare_amd_metadata()
configure_file("@TEMPLATE@" "${CMAKE_BINARY_DIR}/metadata.json" @ONLY)
add_custom_target(metadata_fixture ALL COMMAND "${CMAKE_COMMAND}" -E true)
'''


@unittest.skipUnless(shutil.which("cmake") and (shutil.which("ninja") or shutil.which("make")),
                     "CMake and a build-file executor are required")
class AmdMetadataTests(unittest.TestCase):
    def setUp(self):
        base = Path("/private/tmp/gpu-library-suite-local-build")
        if not base.parent.exists():
            base = Path("/tmp/gpu-library-suite-local-build")
        base.mkdir(exist_ok=True)
        requested = Path(os.environ.get("TMPDIR", str(base))).resolve()
        if requested == base.resolve() or base.resolve() in requested.parents:
            base = requested
        temporary = tempfile.TemporaryDirectory(prefix="amd-metadata-", dir=base)
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.source = self.work / "source"
        self.build = self.work / "build"
        self.source.mkdir()
        self.environment = os.environ.copy()
        self.environment["TMPDIR"] = str(self.work)
        fixture_files = {
            "include/hip/hip_runtime.h": "// synthetic HIP header\n",
            "include/thrust/version.h": "#define THRUST_VERSION 100000\n",
            "include/rocprim/rocprim.hpp": "// synthetic rocPRIM identity\n",
            "other-include/hip/hip_runtime.h": "// different HIP headers\n",
            "cpu-include/lapacke.h": "// synthetic LAPACKE header\n",
            "lib/librocblas.so": "synthetic rocBLAS library identity\n",
            "lib/libamdhip64.so": "synthetic HIP library identity\n",
            "lib/liblapacke.so": "synthetic LAPACKE library identity\n",
            "amd/c-cpp/rocthrust/examples/rocthrust_wrapper.hip": "// wrapper input one\n",
        }
        for name, content in fixture_files.items():
            target = self.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        self.wrapper = self.source / "amd/c-cpp/rocthrust/examples/rocthrust_wrapper.hip"
        script = CMAKE_SOURCE.replace("@MODULES@", str(ROOT / "cmake")).replace(
            "@TEMPLATE@", str(ROOT / "cmake/gpu_suite_amd_build_metadata.json.in"))
        (self.source / "CMakeLists.txt").write_text(script)

    def command(self, arguments, success=True):
        result = subprocess.run(arguments, env=self.environment, text=True,
                                capture_output=True, check=False)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def configure(self, success=True, **settings):
        return self.command(["cmake", "-S", str(self.source), "-B", str(self.build),
                             "-G", GENERATOR,
                             *["-D" + key + "=" + value for key, value in settings.items()]],
                            success=success)

    def metadata(self):
        content = (self.build / "metadata.json").read_bytes()
        return json.loads(content), hashlib.sha256(content).hexdigest()

    def test_wrapper_content_change_reconfigures_on_build(self):
        self.configure(TEST_THRUST="ON", TEST_WRAPPER="ON")
        before, before_hash = self.metadata()
        self.assertEqual(before["rocthrust_interop"]["source_sha256"],
                         hashlib.sha256(self.wrapper.read_bytes()).hexdigest())
        self.assertEqual(before["rocthrust_interop"]["compiler_language"], "hip")
        self.assertIn("-DFIXTURE_HIP", before["rocthrust_interop"]["compile_flags"])
        self.assertIn("-DFIXTURE_CALLER", before["rocthrust_interop"]["caller_compile_flags"])
        self.assertIn("-fopenmp", before["rocthrust_interop"]["link_flags"])
        self.assertEqual(before["rocthrust_interop"]["linker_language"], "cxx")
        self.assertEqual(before["rocthrust_interop"]["linker_path"],
                         str(self.source / "nonexecuted-openmp-linker"))
        # Support generators/filesystems whose timestamp resolution is one second.
        time.sleep(1.05)
        self.wrapper.write_text("// wrapper input two, requiring a new content identity\n")
        self.command(["cmake", "--build", str(self.build)])
        after, after_hash = self.metadata()
        self.assertNotEqual(after_hash, before_hash)
        self.assertEqual(after["rocthrust_interop"]["source_sha256"],
                         hashlib.sha256(self.wrapper.read_bytes()).hexdigest())
        self.assertNotEqual(after["rocthrust_interop"]["source_sha256"],
                            before["rocthrust_interop"]["source_sha256"])

    def test_hip_header_identity_does_not_require_rocfft(self):
        self.configure(TEST_LIBRARIES="rocblas")
        metadata, _ = self.metadata()
        self.assertEqual(metadata["rocm"]["rocfft_include"], "")
        self.assertEqual(metadata["rocm"]["hip_include"], str((self.source / "include").resolve()))
        self.assertEqual(metadata["dependencies"]["hip"]["sha256"],
                         hashlib.sha256((self.source / "lib/libamdhip64.so").read_bytes()).hexdigest())
        self.assertIn("rocblas", metadata["dependencies"])
        self.assertNotIn("rocfft", metadata["dependencies"])

    def test_deselected_rocfft_cached_libraries_do_not_pollute_dependencies(self):
        old_hip = self.source / "lib/liboldhip.so"
        old_fft = self.source / "lib/librocfft.so"
        old_hip.write_text("previously selected HIP identity\n")
        old_fft.write_text("previously selected rocFFT identity\n")
        cmake_file = self.source / "CMakeLists.txt"
        cmake_file.write_text(cmake_file.read_text().replace(
            'if(TEST_THRUST)',
            r'''if(TEST_ROCFFT)
  set(RocFFT_FOUND TRUE)
  set(RocFFT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
  set(RocFFT_HIP_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
  set(RocFFT_LIBRARY "${CMAKE_SOURCE_DIR}/lib/librocfft.so" CACHE FILEPATH "retained probe result")
  set(RocFFT_HIP_LIBRARY "${CMAKE_SOURCE_DIR}/lib/liboldhip.so" CACHE FILEPATH "retained probe result")
  set(RocBLAS_HIP_LIBRARY "${RocFFT_HIP_LIBRARY}")
endif()
if(TEST_THRUST)''', 1))
        self.configure(TEST_ROCFFT="ON", TEST_LIBRARIES="rocfft;rocblas")
        before, _ = self.metadata()
        self.assertEqual(before["dependencies"]["hip"]["path"], str(old_hip.resolve()))
        self.assertEqual(before["dependencies"]["rocfft"]["path"], str(old_fft.resolve()))
        self.configure(TEST_ROCFFT="OFF", TEST_LIBRARIES="rocblas")
        cache = (self.build / "CMakeCache.txt").read_text()
        self.assertIn("RocFFT_HIP_LIBRARY:FILEPATH=" + str(old_hip), cache)
        self.assertIn("RocFFT_LIBRARY:FILEPATH=" + str(old_fft), cache)
        after, _ = self.metadata()
        self.assertNotIn("rocfft", after["dependencies"])
        self.assertEqual(set(after["dependencies"]), {"hip", "rocblas"})
        current_hip = self.source / "lib/libamdhip64.so"
        self.assertEqual(after["dependencies"]["hip"]["path"], str(current_hip.resolve()))
        self.assertEqual(after["dependencies"]["hip"]["sha256"],
                         hashlib.sha256(current_hip.read_bytes()).hexdigest())

    def test_inconsistent_hip_header_directories_are_rejected(self):
        result = self.configure(success=False, TEST_MISMATCH="ON")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("different HIP header directories", result.stdout + result.stderr)
        self.assertFalse((self.build / "metadata.json").exists())

    def test_cpu_openmp_flags_and_library_selection_change_identity(self):
        self.configure(TEST_LIBRARIES="rocblas;rocthrust", TEST_CPU_THRUST="OPENMP",
                       TEST_CPU_OMP_FLAGS="-fopenmp")
        first, first_hash = self.metadata()
        self.assertEqual(first["amd_build_inputs"]["GPU_SUITE_AMD_LIBRARIES"], "rocblas;rocthrust")
        self.assertEqual(first["amd_build_inputs"]["GPU_SUITE_CPU_THRUST_BACKEND"], "OPENMP")
        self.configure(TEST_CPU_OMP_FLAGS="-fopenmp=libomp")
        second, second_hash = self.metadata()
        self.assertNotEqual(first_hash, second_hash)
        self.assertEqual(second["amd_build_inputs"]["OpenMP_CXX_FLAGS"], "-fopenmp=libomp")
        self.configure(TEST_LIBRARIES="rocthrust")
        third, third_hash = self.metadata()
        self.assertNotEqual(second_hash, third_hash)
        self.assertEqual(third["amd_build_inputs"]["GPU_SUITE_AMD_LIBRARIES"], "rocthrust")

    def test_provider_options_and_header_only_dependencies_are_bound(self):
        self.configure(TEST_THRUST="ON", TEST_PROVIDER="ON", TEST_CPU_LAPACK="GENERIC_LAPACKE",
                       TEST_PROVIDER_FLAGS="-DPROVIDER_A;-fno-fast-math")
        before, before_hash = self.metadata()
        for name, relative in (("rocthrust", "include/thrust/version.h"),
                               ("rocprim", "include/rocprim/rocprim.hpp")):
            self.assertEqual(before["dependencies"][name]["sha256"],
                             hashlib.sha256((self.source / relative).read_bytes()).hexdigest())
        inputs = before["amd_build_inputs"]
        self.assertEqual(inputs["GPU_SUITE_CPU_LAPACK_BACKEND"], "GENERIC_LAPACKE")
        self.assertEqual(inputs["LAPACKE_INTERFACE_INCLUDE_DIRECTORIES"], str(self.source / "cpu-include"))
        self.assertEqual(inputs["LAPACKE_INTERFACE_COMPILE_DEFINITIONS"], 'GPU_SUITE_LAPACKE_HEADER="lapacke.h"')
        self.assertEqual(inputs["LAPACKE_INTERFACE_COMPILE_OPTIONS"], "-DPROVIDER_A;-fno-fast-math")
        self.assertEqual(inputs["LAPACKE_INTERFACE_LINK_OPTIONS"], "-Wl,provider")
        self.assertEqual(inputs["RocThrust_ROCPRIM_INCLUDE_DIR"], str(self.source / "include"))
        self.configure(TEST_CPU_LAPACK="OPENBLAS", TEST_PROVIDER_FLAGS="-DPROVIDER_B")
        after, after_hash = self.metadata()
        self.assertNotEqual(before_hash, after_hash)
        self.assertEqual(after["amd_build_inputs"]["GPU_SUITE_CPU_LAPACK_BACKEND"], "OPENBLAS")
        self.assertEqual(after["amd_build_inputs"]["LAPACKE_INTERFACE_COMPILE_OPTIONS"], "-DPROVIDER_B")


if __name__ == "__main__":
    unittest.main()

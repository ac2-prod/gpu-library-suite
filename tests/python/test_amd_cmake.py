"""CPU-only integration of all AMD families; no HIP/GPU execution is claimed."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = {"rocfft", "rocblas", "rocsparse", "rocsolver", "rocrand", "rocthrust"}


@unittest.skipUnless(shutil.which("cmake") and shutil.which("cc") and shutil.which("c++"),
                     "CMake and host C/C++ compilers are required")
class AmdCmakeTests(unittest.TestCase):
    def setUp(self):
        allowed = Path("/tmp/gpu-library-suite-local-build")
        allowed.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(tempfile.gettempdir()).resolve()
        if temporary_root != allowed.resolve() and allowed.resolve() not in temporary_root.parents:
            temporary_root = allowed
        temporary = tempfile.TemporaryDirectory(prefix="amd-cmake-", dir=temporary_root)
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.empty_sysroot = self.work / "no-optional-libraries"
        self.empty_sysroot.mkdir()
        self.environment = os.environ.copy()
        for key in ("ROCM_PATH", "HIP_PATH", "MKLROOT", "FFTW_ROOT", "FFTW3_ROOT",
                    "FFTW3F_ROOT", "FFTW3f_ROOT", "CMAKE_PREFIX_PATH", "CMAKE_INCLUDE_PATH",
                    "CMAKE_LIBRARY_PATH", "PKG_CONFIG_PATH", "PKG_CONFIG_LIBDIR",
                    "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH",
                    "CFLAGS", "CXXFLAGS", "LDFLAGS"):
            self.environment.pop(key, None)
        self.environment["TMPDIR"] = str(self.work)

    def command(self, argv, success=True):
        result = subprocess.run([str(arg) for arg in argv], env=self.environment,
                                capture_output=True, text=True, check=False)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def configure(self, *options, success=True, trace=False):
        build = self.work / "build"
        argv = ["cmake", "-S", ROOT, "-B", build,
                "-DGPU_SUITE_VENDOR=amd", "-DGPU_SUITE_SOURCE_LANGUAGE=c-cpp",
                "-DGPU_SUITE_BUILD_CPU=ON", "-DGPU_SUITE_BUILD_HIP=OFF",
                "-DGPU_SUITE_BUILD_OPENMP=OFF", "-DGPU_SUITE_BUILD_CUDA=OFF",
                "-DGPU_SUITE_BUILD_OPENACC=OFF", "-DGPU_SUITE_BUILD_TESTS=OFF",
                "-DGPU_SUITE_CPU_BLAS_BACKEND=ONEMKL", "-DGPU_SUITE_CPU_LAPACK_BACKEND=ONEMKL",
                "-DGPU_SUITE_CPU_SPARSE_BACKEND=ONEMKL", "-DGPU_SUITE_CPU_THRUST_BACKEND=STL",
                "-DCMAKE_DISABLE_FIND_PACKAGE_FFTW3f=ON",
                "-DCMAKE_FIND_ROOT_PATH=" + str(self.empty_sysroot),
                "-DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY",
                "-DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY",
                "-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY",
                "-DPython3_EXECUTABLE=" + sys.executable]
        if trace:
            argv += ["--trace-expand", "--trace-format=json-v1",
                     "--trace-redirect=" + str(self.work / "trace.jsonl")]
            for source in ("CMakeLists.txt", "cmake/GpuSuiteAmd.cmake",
                           "cmake/GpuSuiteAmdFamilies.cmake", "cmake/GpuSuiteCpuDependencies.cmake"):
                argv.append("--trace-source=" + str(ROOT / source))
        result = self.command(argv + list(options), success=success)
        return build, result

    def manifest_entries(self, build):
        document = json.loads((build / "partial-manifest-input.json").read_text())
        return document["entries"]

    def test_all_six_cpu_families_keep_dependency_free_targets_buildable(self):
        build, result = self.configure()
        output = result.stdout.lower()
        for family in FAMILIES:
            self.assertIn(family + " cpu targets", output)
        for reason in ("fftw3f discovery explicitly disabled",
                       "onemkl cblas header or provider library was not found",
                       "onemkl lapacke header or provider library was not found",
                       "onemkl mkl_spblas.h or mkl_rt was not found"):
            self.assertIn(reason, output)
        self.assertNotIn("Looking for a HIP compiler", result.stdout)
        self.command(["cmake", "--build", build, "--target", "gpu_suite_common",
                      "rand_cpu", "rand_cpu_bench", "reduce_cpu", "reduce_cpu_bench", "--parallel", "2"])
        metadata = json.loads((build / "build-metadata.json").read_text())
        self.assertEqual(metadata["vendor"], "amd")
        self.assertEqual(metadata["build_profile"], "cpu")
        self.assertEqual(metadata["dependencies"], {})
        self.assertEqual(metadata["hip"]["compiler"], "")
        self.assertNotIn("cuda", metadata)
        inputs = metadata["amd_build_inputs"]
        self.assertEqual(set(inputs["GPU_SUITE_AMD_LIBRARIES"].split(";")), FAMILIES)
        self.assertEqual(inputs["GPU_SUITE_CPU_THRUST_BACKEND"], "STL")
        self.assertEqual(inputs["GPU_SUITE_CPU_SPARSE_BACKEND"], "ONEMKL")
        entries = self.manifest_entries(build)
        self.assertEqual({entry["target_name"] for entry in entries},
                         {"rand_cpu", "rand_cpu_bench", "reduce_cpu", "reduce_cpu_bench"})
        self.assertEqual({entry["library"] for entry in entries}, {"rocrand", "rocthrust"})
        for entry in entries:
            self.assertEqual(entry["implementation"], "cpu")
            self.assertEqual(entry["build_profile"], "cpu")
            self.assertEqual(entry["compiler_language"], "cxx")
            self.assertTrue(Path(entry["executable_path"]).is_file())
            self.assertIn("/amd/c-cpp/" + entry["library"] + "/", entry["executable_path"])
        # Build-system target inventory must agree with the manifest exclusion.
        help_result = self.command(["cmake", "--build", build, "--target", "help"])
        for absent in ("fft_cpu_bench", "blas_cpu_bench", "sparse_cpu_bench", "solver_cpu_bench",
                       "rand_gpu_bench", "openmp_rocthrust_bench"):
            self.assertNotIn(absent, help_result.stdout)

    def test_unknown_library_selector_is_rejected(self):
        _, result = self.configure("-DGPU_SUITE_AMD_LIBRARIES=rocrand;unrecognized", success=False)
        self.assertIn("GPU_SUITE_AMD_LIBRARIES", result.stderr)
        self.assertIn("unknown", result.stderr.lower())

    def test_duplicate_library_selector_is_rejected(self):
        _, result = self.configure("-DGPU_SUITE_AMD_LIBRARIES=rocblas;rocblas", success=False)
        self.assertIn("GPU_SUITE_AMD_LIBRARIES", result.stderr)
        self.assertIn("duplicate", result.stderr.lower())

    def test_empty_library_selector_is_rejected(self):
        _, result = self.configure("-DGPU_SUITE_AMD_LIBRARIES=", success=False)
        self.assertIn("GPU_SUITE_AMD_LIBRARIES", result.stderr)
        self.assertIn("nonempty", result.stderr.lower())

    def test_rocrand_subset_does_not_discover_unselected_dependencies(self):
        build, result = self.configure("-DGPU_SUITE_AMD_LIBRARIES=rocrand", trace=True)
        self.command(["cmake", "--build", build, "--target", "rand_cpu", "rand_cpu_bench", "--parallel", "2"])
        entries = self.manifest_entries(build)
        self.assertEqual({entry["target_name"] for entry in entries}, {"rand_cpu", "rand_cpu_bench"})
        self.assertEqual({entry["library"] for entry in entries}, {"rocrand"})
        for family in FAMILIES - {"rocrand"}:
            self.assertNotIn(family + " cpu targets", result.stdout.lower())
        trace = [json.loads(line) for line in (self.work / "trace.jsonl").read_text().splitlines()]
        for event in trace:
            self.assertNotIn(event.get("cmd"), {"gpu_suite_find_cblas", "gpu_suite_find_lapacke",
                                               "gpu_suite_find_onemkl_sparse"})
            if event.get("cmd") == "find_package":
                self.assertNotIn(event["args"][0], {"FFTW3f", "RocFFT", "RocBLAS", "RocSPARSE",
                                                    "RocSOLVER", "RocRAND", "RocThrust"})
        metadata = json.loads((build / "build-metadata.json").read_text())
        self.assertEqual(metadata["dependencies"], {})
        self.assertEqual(metadata["amd_build_inputs"]["GPU_SUITE_AMD_LIBRARIES"], "rocrand")

    def test_requested_cpu_openmp_absence_does_not_fall_back_to_stl(self):
        build, result = self.configure("-DGPU_SUITE_AMD_LIBRARIES=rocthrust",
                                       "-DGPU_SUITE_CPU_THRUST_BACKEND=OPENMP",
                                       "-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON")
        self.assertIn("OpenMP CXX runtime unavailable", result.stdout)
        self.assertIn("requested backend is not replaced", result.stdout)
        self.assertFalse((build / "partial-manifest-input.json").exists())
        self.command(["cmake", "--build", build, "--target", "gpu_suite_common", "--parallel", "2"])
        help_result = self.command(["cmake", "--build", build, "--target", "help"])
        self.assertNotIn("reduce_cpu", help_result.stdout)
        metadata = json.loads((build / "build-metadata.json").read_text())
        self.assertEqual(metadata["dependencies"], {})

    def test_rocthrust_openmp_wrapper_architecture_checked_before_compiler_probe(self):
        # Exercise the real selection module without an SDK or enabled language.
        source = self.work / "architecture-fixture"
        modules = source / "modules"
        modules.mkdir(parents=True)
        (modules / "FindRocThrust.cmake").write_text("set(RocThrust_FOUND TRUE)\n")
        (modules / "CheckLanguage.cmake").write_text(
            'file(WRITE "${CMAKE_BINARY_DIR}/unexpected-compiler-probe" "called")\n'
            'message(FATAL_ERROR "compiler discovery must not run in this fixture")\n')
        (source / "CMakeLists.txt").write_text('''
cmake_minimum_required(VERSION 3.21)
project(amd_architecture_fixture NONE)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/modules" "@MODULES@")
set(GPU_SUITE_AMD_LIBRARIES rocthrust)
set(GPU_SUITE_BUILD_CPU FALSE)
set(GPU_SUITE_BUILD_HIP FALSE)
set(GPU_SUITE_BUILD_OPENMP TRUE)
set(GPU_SUITE_AMD_GPU_TARGET gfx942)
# Accepted cases deliberately skip compiler discovery and OpenMP API probes.
# The mismatch case leaves HIP unloaded to prove the guard runs first.
set(CMAKE_HIP_COMPILER_LOADED "${FIXTURE_HIP_LOADED}")
set(CMAKE_HIP_COMPILER_ID Fixture)
set(CMAKE_CXX_COMPILER_ID Fixture)
include(GpuSuiteAmdFamilies)
file(WRITE "${CMAKE_BINARY_DIR}/selection.txt"
  "${CMAKE_HIP_ARCHITECTURES}\\n${GPU_SUITE_ROCTHRUST_OPENMP_ENABLED}\\n")
'''.replace("@MODULES@", str(ROOT / "cmake")))
        for name, architecture, loaded, accepted in (
                ("mismatch", "gfx90a", "FALSE", False),
                ("matching", "gfx942", "TRUE", True),
                ("inherited", None, "TRUE", True)):
            with self.subTest(name=name):
                build = self.work / ("architecture-" + name)
                arguments = ["cmake", "-S", source, "-B", build,
                             "-DFIXTURE_HIP_LOADED=" + loaded]
                if architecture is not None:
                    arguments.append("-DCMAKE_HIP_ARCHITECTURES=" + architecture)
                result = self.command(arguments, success=accepted)
                self.assertFalse((build / "unexpected-compiler-probe").exists())
                if accepted:
                    self.assertEqual((build / "selection.txt").read_text(), "gfx942\nFALSE\n")
                else:
                    diagnostic = " ".join(result.stderr.split())
                    self.assertIn("CMAKE_HIP_ARCHITECTURES to equal GPU_SUITE_AMD_GPU_TARGET",
                                  diagnostic)
                    self.assertFalse((build / "selection.txt").exists())


if __name__ == "__main__":
    unittest.main()

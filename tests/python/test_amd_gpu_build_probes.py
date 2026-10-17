"""Host-only CMake/link regressions; no ROCm compiler or GPU is exercised."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
HEADERS = ROOT / "tests/fixtures/amd_rand_thrust/include"
WRAPPER = ROOT / "amd/c-cpp/rocthrust/examples/rocthrust_wrapper.hip"


@unittest.skipUnless(shutil.which("cmake") and shutil.which("cc") and
                     shutil.which("c++") and shutil.which("ar"),
                     "CMake and host C/C++ archive tools are required")
class AmdGpuBuildProbeTests(unittest.TestCase):
    def setUp(self):
        base = Path("/tmp/gpu-library-suite-local-build")
        base.mkdir(parents=True, exist_ok=True)
        requested = Path(os.environ.get("TMPDIR", str(base))).resolve()
        if requested == base.resolve() or base.resolve() in requested.parents:
            base = requested
        temporary = tempfile.TemporaryDirectory(prefix="amd-probe-host-", dir=base)
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.includes = self.work / "include"
        shutil.copytree(HEADERS, self.includes)
        (self.includes / "rocprim").mkdir()
        (self.includes / "rocprim/rocprim.hpp").write_text("// Host fixture only.\n")
        self.environment = os.environ.copy()
        self.environment["TMPDIR"] = str(self.work)
        for key in ("HIP_PATH", "ROCM_PATH", "CMAKE_PREFIX_PATH", "CPATH",
                    "CPLUS_INCLUDE_PATH", "CFLAGS", "CXXFLAGS", "HIPFLAGS", "LDFLAGS"):
            self.environment.pop(key, None)
        (self.work / "empty.c").write_text("void fixture_archive_marker(void) {}\n")
        self.command(["cc", "-c", self.work / "empty.c", "-o", self.work / "empty.o"])
        self.library = self.work / "libhostfixture.a"
        self.command(["ar", "rcs", self.library, self.work / "empty.o"])

    def command(self, arguments):
        result = subprocess.run(list(map(str, arguments)), capture_output=True,
                                text=True, check=False, env=self.environment)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def project(self, name, body):
        source = self.work / name
        source.mkdir()
        prelude = '''cmake_minimum_required(VERSION 3.21)
project(host_only_fixture LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "@MODULES@")
include(GpuSuiteProbe)
'''.replace("@MODULES@", str(ROOT / "cmake"))
        (source / "CMakeLists.txt").write_text(prelude + body)
        build = self.work / (name + "-build")
        self.command(["cmake", "-S", source, "-B", build])
        return build

    def imported_paths(self, package):
        return '\n'.join('set({0}_{1} "{2}" CACHE {3} "")'.format(
            package, field, path, "FILEPATH" if field.endswith("LIBRARY") else "PATH")
            for field, path in (("INCLUDE_DIR", self.includes),
                                ("ROCPRIM_INCLUDE_DIR", self.includes),
                                ("HIP_INCLUDE_DIR", self.includes),
                                ("HIP_LIBRARY", self.library),
                                ("LIBRARY", self.library))) + '\n'

    def test_actual_rocrand_api_probe_compiles_links_and_rejects_missing_ordering(self):
        body = self.imported_paths("RocRAND") + '''
find_package(RocRAND REQUIRED)
gpu_suite_check_cxx_link(actual_api SOURCE "${GPU_SUITE_ROCRAND_API_PROBE}"
  LIBRARIES RocRAND::rocrand RocRAND::hip)
file(WRITE "${CMAKE_BINARY_DIR}/result.txt" "${actual_api}")
'''
        build = self.project("rand-valid", body)
        self.assertEqual((build / "result.txt").read_text(), "1")
        # The actual probe must use this declaration, not just include a header.
        header = self.includes / "rocrand/rocrand.h"
        header.write_text(header.read_text().replace("rocrand_set_ordering(", "fixture_removed_ordering("))
        build = self.project("rand-missing-api", body)
        self.assertIn((build / "result.txt").read_text(), ("", "0", "FALSE"))

    def mixed_probe(self, broken=False):
        wrapper = self.work / "rocthrust_wrapper.hip"
        original = WRAPPER.read_text()
        wrapper.write_text(original.replace("rocthrust_square_sum(", "missing_wrapper_symbol(")
                           if broken else original)
        # Intercept only native HIP compilation in the test: keep the helper's
        # generated project, real wrapper/caller, imported targets and link.
        # This is explicitly a host simulation, not a HIP language test.
        simulation = r'''
function(try_compile result binary source project target)
  if(NOT project STREQUAL "GpuSuiteRocThrustInterop" OR NOT target STREQUAL "interop_probe")
    message(FATAL_ERROR "expected project-form two-TU try_compile")
  endif()
  file(READ "${source}/CMakeLists.txt" code)
  file(WRITE "${source}/native-project.txt" "${code}")
  string(REPLACE "LANGUAGES CXX HIP" "LANGUAGES CXX" code "${code}")
  string(REPLACE "PROPERTIES LANGUAGE HIP" "PROPERTIES LANGUAGE CXX" code "${code}")
  file(WRITE "${source}/CMakeLists.txt" "${code}")
  execute_process(COMMAND "${CMAKE_COMMAND}" -S "${source}" -B "${binary}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    RESULT_VARIABLE configured OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT configured EQUAL 0)
    message(FATAL_ERROR "host simulation configure failed: ${output}${error}")
  endif()
  execute_process(COMMAND "${CMAKE_COMMAND}" --build "${binary}" --verbose
    RESULT_VARIABLE built OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(built EQUAL 0)
    set(${result} TRUE PARENT_SCOPE)
  else()
    set(${result} FALSE PARENT_SCOPE)
  endif()
  set(_output "${output}${error}" PARENT_SCOPE)
endfunction()
set(CMAKE_HIP_COMPILER_LOADED TRUE)
set(CMAKE_HIP_ARCHITECTURES fixture-explicit-architecture)
set(caller [=[
#include <hip/hip_runtime.h>
#include <omp.h>
#include <cstddef>
#include <cstdio>
#ifndef FIXTURE_CALLER_COMPILE_FLAG
#error caller compile conditions were not forwarded
#endif
extern "C" int rocthrust_square_sum(const double *, std::size_t, double *) noexcept;
int main() {
  std::fclose(std::fopen("@MARKER@", "w"));
  double *p = nullptr, result = 0.0;
  int initial = 1, status = 1;
  #pragma omp target map(from:initial)
  { initial = omp_is_initial_device(); }
  #pragma omp target data map(to:p[0:1]) use_device_ptr(p)
  { status = rocthrust_square_sum(p, 1, &result); }
  return status;
}
]=])
gpu_suite_check_rocthrust_openmp_link(mixed_api SOURCE "${caller}" WRAPPER "@WRAPPER@"
  COMPILE_OPTIONS -DFIXTURE_CALLER_COMPILE_FLAG=1 -Wno-unknown-pragmas)
file(WRITE "${CMAKE_BINARY_DIR}/result.txt" "${mixed_api}")
'''.replace("@MARKER@", str(self.work / "must-not-run")).replace("@WRAPPER@", str(wrapper))
        build = self.project("mixed", self.imported_paths("RocThrust") + simulation)
        probe = build / "CMakeFiles/mixed_api"
        native_project = (probe / "source/native-project.txt").read_text()
        self.assertIn("LANGUAGES CXX HIP", native_project)
        self.assertIn("LINKER_LANGUAGE CXX", native_project)
        self.assertIn(str(wrapper), native_project)
        compile_commands = json.loads((probe / "build/compile_commands.json").read_text())
        self.assertEqual({Path(row["file"]).name for row in compile_commands},
                         {"caller.cpp", "rocthrust_wrapper.hip"})
        for row in compile_commands:
            self.assertIn("__HIP_PLATFORM_AMD__", row["command"])
            self.assertIn(str(self.includes), row["command"])
        self.assertFalse((self.work / "must-not-run").exists())
        return build, probe

    def test_mixed_rocthrust_host_simulation_links_actual_wrapper_without_execution(self):
        build, probe = self.mixed_probe()
        self.assertEqual((build / "result.txt").read_text(), "TRUE")
        log = (probe / "output.log").read_text()
        self.assertIn("Linking CXX executable interop_probe", log)
        self.assertIn("rocthrust_wrapper.hip.o", log)

    def test_mixed_rocthrust_host_simulation_rejects_missing_wrapper_symbol(self):
        build, probe = self.mixed_probe(broken=True)
        self.assertEqual((build / "result.txt").read_text(), "FALSE")
        self.assertIn("rocthrust_square_sum", (probe / "output.log").read_text())

    def test_production_rocthrust_openmp_target_links_both_tus_with_cxx(self):
        body = self.imported_paths("RocThrust") + '''
find_package(RocThrust REQUIRED)
set(GPU_SUITE_BUILD_CPU FALSE)
set(GPU_SUITE_BUILD_EXAMPLES TRUE)
set(GPU_SUITE_BUILD_BENCHMARKS FALSE)
set(GPU_SUITE_ROCTHRUST_HIP_ENABLED FALSE)
set(GPU_SUITE_ROCTHRUST_OPENMP_ENABLED TRUE)
function(gpu_suite_record_status)
endfunction()
function(gpu_suite_configure_executable target source)
  target_compile_definitions(${target} PRIVATE _OPENMP=201811)
  target_compile_options(${target} PRIVATE -Wno-unknown-pragmas)
endfunction()
add_subdirectory("@FAMILY@" family)
# Host simulation of only the HIP language; consume the actual target layout.
set_source_files_properties("@WRAPPER@" DIRECTORY "@FAMILY@" PROPERTIES LANGUAGE CXX)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/linker.txt"
  CONTENT "$<TARGET_PROPERTY:openmp_rocthrust,LINKER_LANGUAGE>")
'''.replace("@FAMILY@", str(ROOT / "amd/c-cpp/rocthrust")).replace("@WRAPPER@", str(WRAPPER))
        build = self.project("production-layout", body)
        result = self.command(["cmake", "--build", build, "--verbose"])
        self.assertEqual((build / "linker.txt").read_text(), "CXX")
        self.assertIn("rocthrust_wrapper.hip.o", result.stdout)
        commands = json.loads((build / "compile_commands.json").read_text())
        self.assertEqual({Path(row["file"]).name for row in commands},
                         {"openmp_rocthrust.cpp", "rocthrust_wrapper.hip"})


if __name__ == "__main__":
    unittest.main()

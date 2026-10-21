"""Small host regressions for saved Sirius build failures; no ROCm/GPU execution."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class AmdFortranBuildFixTests(unittest.TestCase):
    def setUp(self):
        base = Path("/tmp/gpu-library-suite-local-build")
        base.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="fortran-build-fix-", dir=base)
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.environment = dict(os.environ, TMPDIR=str(self.work))

    def command(self, argv):
        result = subprocess.run(list(map(str, argv)), cwd=self.work,
                                env=self.environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    @unittest.skipUnless(shutil.which("cmake"), "CMake unavailable")
    def test_failed_probe_reason_preserves_following_target(self):
        (self.work / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.20)
project(status_fixture NONE)
include("@ROOT@/cmake/GpuSuiteStatus.cmake")
gpu_suite_record_status("failed API" FALSE "compile/link failed; see x|y.log")
gpu_suite_record_status("healthy API" TRUE "")
gpu_suite_record_status("missing dependency" FALSE "not installed")
gpu_suite_print_status()
add_custom_target(healthy COMMAND "${CMAKE_COMMAND}" -E echo "healthy target remains")
'''.replace("@ROOT@", str(ROOT)))
        result = self.command(["cmake", "-S", self.work, "-B", self.work / "build"])
        self.assertIn("failed API: DISABLED (compile/link failed; see x|y.log)", result.stdout)
        self.assertIn("healthy API: ENABLED", result.stdout)
        self.assertIn("missing dependency: DISABLED (not installed)", result.stdout)
        self.command(["cmake", "--build", self.work / "build", "--target", "healthy"])

    @unittest.skipUnless(shutil.which("gfortran") and shutil.which("cc"),
                         "Host Fortran/C compilers unavailable")
    def test_public_c_bindings_preserve_values_pointers_and_query_nulls(self):
        # Exercise declarations extracted from the real programs against C,
        # rather than inventing hipFORT generics that accept the desired calls.
        provider = self.work / "provider.c"
        provider.write_text('''
#include <stddef.h>
int rocsolver_dgetrf(void *h, int m, int n, double *a, int lda, int *piv, int *info) {
    if (m != 3 || n != 3 || lda != 3) return -1;
    if (!h) return (!a && !piv && !info) ? 201 : -2;
    if (h != a || !piv || !info || a[0] != 7.) return -3;
    piv[0] = 2; *info = 0; return 17;
}
int rocrand_set_ordering(void *gen, int ordering) {
    if (!gen || ordering != 101) return -4;
    ++*(int *)gen; return 19;
}
int rocblas_stop_device_memory_size_query(void *handle, size_t *bytes) {
    if (!handle || !bytes) return -5;
    *bytes = 4096; return 0;
}
''')
        self.command(["cc", "-c", provider, "-o", self.work / "provider.o"])
        solver = ROOT / "amd/fortran/rocsolver"
        rand = ROOT / "amd/fortran/rocrand"
        pairs = (("examples/solver_gpu.f90", "examples/rand_gpu.f90"),
                 ("examples/openmp_rocsolver.f90", "examples/openmp_rocrand.f90"),
                 ("benchmarks/solver_workload.F90", "benchmarks/rand_workload.F90"))
        for index, (solver_file, rand_file) in enumerate(pairs):
            with self.subTest(solver=solver_file, rand=rand_file):
                solver_text = (solver / solver_file).read_text()
                rand_text = (rand / rand_file).read_text()
                functions = []
                for text, name in ((solver_text, "rocsolver_dgetrf"), (rand_text, "rocrand_set_ordering")):
                    match = re.search(r"(?ms)^\s*function " + name + r"\(.*?^\s*end function", text)
                    self.assertIsNotNone(match, name)
                    functions.append(match.group())
                constant = re.search(r"(?m)^.*parameter :: rocrand_ordering_pseudo_default=.*$", rand_text)
                self.assertIsNotNone(constant)
                source = self.work / ("caller" + str(index) + ".f90")
                source.write_text('''program caller
use iso_c_binding
implicit none
@CONSTANT@
interface
@FUNCTIONS@
end interface
real(c_double), target :: a(1)=[7.0_c_double]
integer(c_int), target :: piv(1)=[-1_c_int], info=-1, generator=5
if (rocsolver_dgetrf(c_loc(a),3_c_int,3_c_int,c_loc(a),3_c_int,c_loc(piv),c_loc(info)) /= 17) stop 1
if (piv(1) /= 2 .or. info /= 0) stop 2
if (rocsolver_dgetrf(c_null_ptr,3_c_int,3_c_int,c_null_ptr,3_c_int,c_null_ptr,c_null_ptr) /= 201) stop 3
if (rocrand_set_ordering(c_loc(generator),rocrand_ordering_pseudo_default) /= 19) stop 4
if (generator /= 6) stop 5
end program
'''.replace("@FUNCTIONS@", "\n".join(functions)).replace("@CONSTANT@", constant.group()))
                binary = self.work / ("caller" + str(index))
                self.command(["gfortran", source, self.work / "provider.o", "-o", binary])
                self.command([binary])  # Host C ABI fixture only, never a GPU call.

        # The C API returns a host size_t by pointer. hipFORT's corresponding
        # typed Fortran output is passed by reference: c_loc(bytes) is wrong.
        # Check the actual workload expression against that typed interface,
        # including a C write-back and rejection of the previous call form.
        call = re.search(r"call check\((.*),'stop workspace query'\)",
                         (solver / "benchmarks/solver_workload.F90").read_text())
        self.assertIsNotNone(call)
        source = self.work / "workspace_query.f90"
        program = '''module hipfort_rocblas
use iso_c_binding
implicit none
interface rocblas_stop_device_memory_size_query
  function stop_query(handle,bytes) bind(C,name='rocblas_stop_device_memory_size_query') result(status)
    import c_ptr,c_size_t,c_int
    type(c_ptr), value :: handle
    integer(c_size_t) :: bytes
    integer(c_int) :: status
  end function
end interface
end module
program caller
use iso_c_binding
use hipfort_rocblas
implicit none
integer(c_int), target :: token=7
type(c_ptr) :: handle
integer(c_size_t), target :: bytes=-1
integer(c_int) :: status
handle=c_loc(token)
status=@CALL@
if (status /= 0 .or. bytes /= 4096_c_size_t) stop 1
end program
'''
        source.write_text(program.replace("@CALL@", call.group(1)))
        binary = self.work / "workspace_query"
        self.command(["gfortran", source, self.work / "provider.o", "-o", binary])
        self.command([binary])
        source.write_text(program.replace("@CALL@",
                          "rocblas_stop_device_memory_size_query(handle,c_loc(bytes))"))
        rejected = subprocess.run(["gfortran", "-fsyntax-only", str(source)],
                                  cwd=self.work, env=self.environment, capture_output=True, text=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("rocblas_stop_device_memory_size_query", rejected.stderr)

    @unittest.skipUnless(shutil.which("cmake") and shutil.which("gfortran") and shutil.which("c++"),
                         "CMake and host Fortran/C++ compilers unavailable")
    def test_mixed_probe_links_with_fortran_without_running(self):
        version = re.search(r"cmake version (\d+)\.(\d+)", self.command(["cmake", "--version"]).stdout)
        if tuple(map(int, version.groups())) < (3, 29):
            self.skipTest("Explicit try_compile linker selection requires CMake 3.29+")
        # Native HIP is unavailable locally. A real Fortran/C++ pair checks
        # the same helper's explicit linker selection and failure propagation.
        (self.work / "caller.f90").write_text('''program caller
use iso_c_binding
implicit none
interface
  function host_value() bind(C) result(value)
    import c_int
    integer(c_int) :: value
  end function
end interface
integer :: unit
open(newunit=unit,file='must-not-run',status='new')
write(unit,*) host_value()
close(unit)
end program
''')
        (self.work / "wrapper.cpp").write_text(
            '#include <string>\nextern "C" int host_value() { return std::string("fixture").size(); }\n')
        (self.work / "broken.cpp").write_text('extern "C" int wrong_name() { return 7; }\n')
        (self.work / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.29)
project(mixed_host_fixture LANGUAGES Fortran CXX)
include("@ROOT@/cmake/GpuSuiteFortran.cmake")
gpu_suite_fortran_probe(good SOURCES "${CMAKE_SOURCE_DIR}/caller.f90" "${CMAKE_SOURCE_DIR}/wrapper.cpp"
  LINKER_LANGUAGE Fortran)
gpu_suite_fortran_probe(bad SOURCES "${CMAKE_SOURCE_DIR}/caller.f90" "${CMAKE_SOURCE_DIR}/broken.cpp"
  LINKER_LANGUAGE Fortran)
if(NOT good OR bad)
  message(FATAL_ERROR "mixed probe accepted/rejected the wrong link")
endif()
'''.replace("@ROOT@", str(ROOT)))
        self.command(["cmake", "-S", self.work, "-B", self.work / "build",
                      "-DCMAKE_Fortran_COMPILER=" + shutil.which("gfortran")])
        log = (self.work / "build/fortran-probes/good.log").read_text()
        self.assertIn("Linking Fortran executable", log)
        self.assertFalse(list(self.work.rglob("must-not-run")))


if __name__ == "__main__":
    unittest.main()

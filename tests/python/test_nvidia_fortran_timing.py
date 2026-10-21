"""Host-only NVIDIA timing/build-profile checks; never CUDA/NVHPC/GPU execution evidence."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("gfortran") and shutil.which("cc"),
                     "Host Fortran/C compilers unavailable")
class NvidiaFortranTimingTests(unittest.TestCase):
    def setUp(self):
        base = Path("/tmp/gpu-library-suite-local-build")
        base.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="nvidia-timing-", dir=base)
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.environment = dict(os.environ, TMPDIR=str(self.work))

    def command(self, argv, environment=None):
        result = subprocess.run(list(map(str, argv)), cwd=self.work,
                                env=environment or self.environment,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_driver_scope_lifetimes_and_repetition(self):
        # Compile the real driver with observable CLI/clock/workload callbacks.
        (self.work / "bridge.c").write_text(r'''
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
static int library;
void gpu_suite_f_initialize(int lib, int impl) {
    library=lib; (void)impl; setvbuf(stdout,NULL,_IONBF,0);
}
void gpu_suite_f_argument(const char *arg) { (void)arg; }
int gpu_suite_f_parse(void) { return 0; }
int64_t gpu_suite_f_integer(int key) {
    if (key==11) return atoi(getenv("TIMING_TEST_WARMUP"));
    if (key==13) return 2;
    if (key==12) return library==3 ? 1 : 3;
    if (key==14) return atoi(getenv("TIMING_TEST_SCOPE"));
    return 1;
}
double gpu_suite_f_real(int key) { (void)key; return 1.; }
void gpu_suite_f_trial(int index) { printf("trial %d\n",index); }
void gpu_suite_f_start(void) { puts("start"); }
void gpu_suite_f_end(void) { puts("end"); }
int gpu_suite_f_finish(const double *m, double s, int rf, int rs) {
    (void)m; (void)s; (void)rf; (void)rs; puts("finish"); return 0;
}
void gpu_suite_f_close(int status) { exit(status); }
void gpu_suite_f_fail(const char *message) { fprintf(stderr,"%s\n",message); exit(2); }
''')
        (self.work / "trace.f90").write_text('''program trace
use gpu_suite_fortran
use iso_c_binding
implicit none
character(16) :: arg
integer :: lib,impl
call get_command_argument(1,arg); read(arg,*) lib
call get_command_argument(2,arg); read(arg,*) impl
call run_benchmark(lib,impl,host_setup,reset,setup,restore,operation, &
                   synchronize,download,cleanup,verify,host_cleanup)
contains
subroutine host_setup()
 print '(A)', 'host_setup'
end subroutine
subroutine reset()
 print '(A)', 'reset'
end subroutine
subroutine setup()
 print '(A)', 'setup'
end subroutine
subroutine restore()
 print '(A)', 'restore'
end subroutine
subroutine operation(repetitions)
 integer, intent(in) :: repetitions
 integer :: i
 do i=1,repetitions
   print '(A)', 'task'
 end do
end subroutine
subroutine synchronize()
 print '(A)', 'sync'
end subroutine
subroutine download(end_to_end)
 logical, intent(in) :: end_to_end
 print '(A)', 'download'
end subroutine
subroutine cleanup()
 print '(A)', 'cleanup'
end subroutine
subroutine verify(metrics,scale,getrf,getrs)
 real(c_double), intent(out) :: metrics(4),scale
 integer, intent(out) :: getrf,getrs
 metrics=0; scale=1; getrf=0; getrs=0
 print '(A)', 'verify'
end subroutine
subroutine host_cleanup()
 print '(A)', 'host_cleanup'
end subroutine
end program
''')
        self.command(["cc", "-c", "bridge.c", "-o", "bridge.o"])
        self.command(["gfortran", ROOT / "common/fortran/benchmark_support.f90",
                      "trace.f90", "bridge.o", "-o", "trace"])
        for library in range(6):
            repeat = 1 if library == 3 else 3
            for implementation in range(3):
                for scope in (0, 1):
                    for warmup in (0, 2):
                        with self.subTest(library=library, implementation=implementation,
                                          scope=scope, warmup=warmup):
                            env = dict(self.environment, TIMING_TEST_SCOPE=str(scope),
                                       TIMING_TEST_WARMUP=str(warmup), GFORTRAN_UNBUFFERED_ALL="1")
                            lines = self.command([self.work / "trace", library, implementation], env).splitlines()
                            self.assertEqual(lines.count("host_setup"), 1)
                            self.assertEqual(lines.count("host_cleanup"), 1)
                            self.assertEqual(lines.count("finish"), 2)
                            self.assertEqual(lines.count("task"), warmup + 2 * repeat)
                            self.assertEqual(lines.count("start"), 2 * repeat if scope else 2)
                            self.assertEqual(lines.count("end"), lines.count("start"))
                            warm_setups = 0 if library == 1 and implementation == 0 else warmup
                            self.assertEqual(lines.count("setup"), warm_setups + 2 * repeat if scope else 1)
                            self.assertEqual(lines.count("cleanup"), lines.count("setup"))
                            for index, line in enumerate(lines):
                                if line != "start":
                                    continue
                                end = lines.index("end", index)
                                expected = (["setup", "task", "sync", "download"] if scope else
                                            ["task"] * repeat + ["sync"])
                                self.assertEqual(lines[index + 1:end], expected)
                                if scope:
                                    self.assertEqual(lines[index - 1], "reset")
                                    self.assertEqual(lines[end + 1], "cleanup")
                                else:
                                    self.assertEqual(lines[index - 3:index], ["reset", "restore", "sync"])
                                    self.assertEqual(lines[end + 1], "download")
                            if not scope:
                                warm = lines[:lines.index("trial 0")]
                                self.assertEqual(warm.count("download"), 0)
                                syncs = warmup if library in (3, 4) else 1
                                if library == 0 and implementation == 2 and warmup == 0:
                                    syncs = 0
                                self.assertEqual(warm.count("sync"), syncs)
                                restored = warmup if library == 3 or (library == 1 and implementation == 1) else 0
                                self.assertEqual(warm.count("restore"), restored)

    def test_gpu_repeat_regions_setup_and_result_completion(self):
        # Preprocess actual GPU branches; no CUDA modules or runtime are used.
        stems = dict(cufft="fft", cublas="blas", cusparse="sparse",
                     cusolver="solver", curand="rand", thrust="reduce")
        texts = {(lib, impl): self.command(["gfortran", "-cpp", "-E", f"-DGPU_SUITE_IMPL={impl}",
                                            ROOT / f"nvidia/fortran/{lib}/benchmarks/{stem}_workload.F90"]).lower()
                 for lib, stem in stems.items() for impl in (1, 2)}

        def sub(lib, impl, name):
            match = re.search(r"(?ms)^\s*subroutine " + name + r"\(.*?^\s*end subroutine", texts[lib, impl])
            self.assertIsNotNone(match, (lib, impl, name))
            return match.group()

        def ordered(text, tokens):
            offset = 0
            for token in tokens:
                position = text.find(token, offset)
                self.assertGreaterEqual(position, 0, token)
                offset = position + len(token)

        for lib in ("cufft", "cublas"):
            ordered(sub(lib, 2, "operation"), ["!$acc host_data", "do repetition=1,repetitions",
                                               "end do", "!$acc end host_data"])
        for lib in ("curand", "thrust"):
            ordered(sub(lib, 2, "operation"), ["do repetition=1,repetitions", "!$acc host_data",
                                               "!$acc end host_data", "end do"])
        for lib in stems:
            for impl in (1, 2):
                self.assertNotIn("cudadevicesynchronize", sub(lib, impl, "operation"))
                self.assertIn("cudadevicesynchronize", sub(lib, impl, "synchronize"))
        self.assertNotIn("!$acc host_data", sub("cusparse", 2, "operation"))
        for lib, create in (("cufft", "cufftplanmany"), ("cublas", "cublascreate"),
                            ("cusparse", "cusparsecreate"), ("curand", "curandcreategenerator")):
            ordered(sub(lib, 2, "setup"), [create, "!$acc enter data"])
        ordered(sub("cusolver", 2, "download"), ["!$acc update self", "call synchronize()", "!$acc exit data"])
        for lib in ("cublas", "cusparse", "curand"):
            ordered(sub(lib, 2, "download"), ["else", "!$acc update self", "call synchronize()"])
        self.assertNotIn("copyin(mat_a,rhs_b,ipiv", sub("cusolver", 2, "setup"))
        self.assertIn("cudamemset", sub("cufft", 1, "restore"))
        self.assertIn("device(fft_in,fft_out)", sub("cufft", 2, "restore"))
        self.assertNotIn("mat_a", sub("cublas", 1, "restore"))
        self.assertNotIn("vec_x", sub("cusparse", 2, "restore"))
        self.assertNotIn("curandsetgeneratorordering", texts["curand", 1])
        self.assertNotIn("curandgenerateseeds", texts["curand", 1])
        for lib in ("cusparse", "cusolver"):
            self.assertIn("workspace=c_null_ptr", sub(lib, 1, "setup"))
            self.assertNotIn("max(1", sub(lib, 1, "setup"))
        self.assertIn("create_context(values,o%n)", sub("thrust", 1, "setup"))
        for impl in (1, 2):
            self.assertNotIn("cudamemcpy", sub("thrust", impl, "restore"))
            self.assertNotIn("!$acc update", sub("thrust", impl, "restore"))

@unittest.skipUnless(shutil.which("cmake"), "CMake unavailable")
class NvidiaFortranBuildOptionsTests(unittest.TestCase):
    def setUp(self):
        base = Path("/tmp/gpu-library-suite-local-build")
        base.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="fortran-options-", dir=base)
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.script = self.work / "profile.cmake"
        self.script.write_text(
            'cmake_minimum_required(VERSION 3.20)\n'
            'include("' + str(ROOT / "cmake/GpuSuiteFortran.cmake") + '")\n'
            'gpu_suite_fortran_numerical_options(options)\n'
            'file(WRITE "' + str(self.work / "options.txt") + '" "${options}")\n')

    def profile(self, overrides, expected=None):
        variables = {
            "GPU_SUITE_VENDOR": "nvidia", "GPU_SUITE_BUILD_OPENACC": "OFF",
            "CMAKE_Fortran_COMPILER_ID": "NVHPC",
            "CMAKE_C_COMPILER_ID": "GNU", "CMAKE_CXX_COMPILER_ID": "GNU",
            "CMAKE_Fortran_FLAGS_RELEASE": "-O3",
            "CMAKE_C_FLAGS_RELEASE": "-O3 -DNDEBUG",
            "CMAKE_CXX_FLAGS_RELEASE": "-O3 -DNDEBUG",
        }
        variables.update(overrides)
        result = subprocess.run(
            [shutil.which("cmake"), *["-D" + k + "=" + v for k, v in variables.items()],
             "-P", str(self.script)], cwd=self.work, capture_output=True, text=True)
        if expected is None:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Error", result.stderr)
        else:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((self.work / "options.txt").read_text(), expected)

    @staticmethod
    def openacc():
        return {
            "GPU_SUITE_BUILD_OPENACC": "ON",
            "CMAKE_Fortran_FLAGS_RELEASE": "-fast -O3",
            "CMAKE_C_COMPILER_ID": "NVHPC", "CMAKE_CXX_COMPILER_ID": "NVHPC",
            "CMAKE_C_FLAGS_RELEASE": "-fast -O3 -DNDEBUG",
            "CMAKE_CXX_FLAGS_RELEASE": "-fast -O3 -DNDEBUG",
        }

    def test_matching_profiles_keep_cpu_cuda_and_amd_separate(self):
        # Executes the real CMake policy, not an NVHPC/GPU compilation.
        cases = [
            ({}, "-Kieee;-Mnoflushz;-Mnodaz"),
            ({"GPU_SUITE_BUILD_CUDA": "ON"}, "-Kieee;-Mnoflushz;-Mnodaz"),
            (self.openacc(), ""),
            ({"CMAKE_Fortran_COMPILER_ID": "GNU"}, "-fno-fast-math"),
            ({"GPU_SUITE_VENDOR": "amd", "CMAKE_Fortran_COMPILER_ID": "GNU"},
             "-fno-fast-math;-ffp-contract=off"),
            ({"GPU_SUITE_VENDOR": "amd", "CMAKE_Fortran_COMPILER_ID": "LLVMFlang"},
             "-fno-fast-math;-ffp-contract=off"),
        ]
        for variables, expected in cases:
            with self.subTest(variables=variables):
                self.profile(variables, expected)

    def test_fast_exception_rejects_leaks_overrides_and_stale_cache(self):
        cases = [
            {"CMAKE_Fortran_FLAGS_RELEASE": "-fast -O3"},
            {"GPU_SUITE_VENDOR": "amd", "CMAKE_Fortran_COMPILER_ID": "GNU",
             "CMAKE_Fortran_FLAGS_RELEASE": "-fast -O3"},
            {**self.openacc(), "CMAKE_Fortran_FLAGS_RELEASE": "-O3"},
            {**self.openacc(), "CMAKE_CXX_COMPILER_ID": "GNU"},
            {**self.openacc(), "CMAKE_C_FLAGS_RELEASE": "-O3 -DNDEBUG"},
            {**self.openacc(), "CMAKE_Fortran_FLAGS_RELEASE": "-fast -O3 -ffast-math"},
            {**self.openacc(), "CMAKE_Fortran_FLAGS": "-Kieee"},
            {**self.openacc(), "CMAKE_CXX_FLAGS": "-Mnoflushz"},
            {**self.openacc(), "CMAKE_Fortran_FLAGS_DEBUG": "-fast"},
            {**self.openacc(), "CMAKE_Fortran_FLAGS": "-gpu=fastmath"},
        ]
        for variables in cases:
            with self.subTest(variables=variables):
                self.profile(variables)


if __name__ == "__main__":
    unittest.main()

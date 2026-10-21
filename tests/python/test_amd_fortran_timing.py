"""Host-only checks of AMD timing order; no ROCm calls or GPU execution."""
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
class AmdFortranTimingTests(unittest.TestCase):
    def setUp(self):
        base = Path("/tmp/gpu-library-suite-local-build")
        base.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="fortran-timing-", dir=base)
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.environment = dict(os.environ, TMPDIR=str(self.work))

    def command(self, argv, env=None):
        result = subprocess.run(list(map(str, argv)), cwd=self.work,
                                env=env or self.environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_driver_lifetimes_and_repeat_segments(self):
        # Compile the real driver against tracing callbacks/CLI-clock stubs.
        # These are lifecycle assertions, not measured GPU performance.
        (self.work / "bridge.c").write_text(r'''
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
static int library;
void gpu_suite_f_initialize(int lib, int impl) {
    library = lib; (void)impl; setvbuf(stdout, NULL, _IONBF, 0);
}
void gpu_suite_f_argument(const char *arg) { (void)arg; }
int gpu_suite_f_parse(void) { return 0; }
int64_t gpu_suite_f_integer(int key) {
    if (key == 11 || key == 13) return 2; /* warm-up, trials */
    if (key == 12) return library == 3 ? 1 : 3;
    if (key == 14) return atoi(getenv("TIMING_TEST_SCOPE"));
    return 1;
}
double gpu_suite_f_real(int key) { (void)key; return 1.; }
void gpu_suite_f_trial(int index) { printf("trial %d\n", index); }
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
subroutine operation(repetitions,timed)
 integer, intent(in) :: repetitions
 logical, intent(in) :: timed
 integer :: i
 if (timed) call start_timer()
 do i=1,repetitions
   print '(A)', 'task'
 end do
 if (timed) call end_timer()
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
        self.command(["gfortran", ROOT / "amd/fortran/common/benchmark_support.f90",
                      "trace.f90", "bridge.o", "-o", "trace"])
        for library in range(6):
            repeat = 1 if library == 3 else 3
            for implementation in range(3):
                for scope in (0, 1):
                    with self.subTest(library=library, implementation=implementation, scope=scope):
                        env = dict(self.environment, TIMING_TEST_SCOPE=str(scope),
                                   GFORTRAN_UNBUFFERED_ALL="1")
                        lines = self.command([self.work / "trace", library, implementation], env).splitlines()
                        self.assertEqual(lines.count("host_setup"), 1)
                        self.assertEqual(lines.count("host_cleanup"), 1)
                        self.assertEqual(lines.count("finish"), 2)
                        self.assertEqual(lines.count("task"), 2 + 2 * repeat)
                        segments = 2 * repeat if scope or library == 0 else 2
                        self.assertEqual(lines.count("start"), segments)
                        self.assertEqual(lines.count("end"), segments)
                        warm_setups = 0 if library == 1 and implementation == 0 else 2
                        self.assertEqual(lines.count("setup"), warm_setups + 2 * repeat if scope else 1)
                        self.assertEqual(lines.count("cleanup"), lines.count("setup"))
                        for index, line in enumerate(lines):
                            if line != "start":
                                continue
                            end = lines.index("end", index)
                            expected = (["setup", "task", "download"] if scope else
                                        ["task"] * (1 if library == 0 else repeat))
                            self.assertEqual(lines[index + 1:end], expected)
                            if scope:
                                self.assertEqual(lines[index - 1], "reset")
                                self.assertEqual(lines[end + 1], "cleanup")
                            else:
                                self.assertEqual(lines[index - 2:index], ["restore", "sync"])
                        warmup = lines[:lines.index("trial 0")]
                        self.assertEqual(warmup.count("task"), 2)
                        if not scope and library in (1, 2):
                            self.assertEqual(warmup.count("sync"), 1)
                            self.assertEqual(warmup.count("download"), 0)
                            self.assertEqual(warmup.count("restore"),
                                             2 if library == 1 and implementation == 1 else 0)

    def test_openmp_timer_and_completion_boundaries(self):
        # Preprocess the real OpenMP branches, including inactive #if removal.
        # No offload region is executed and no HIP module/library is required.
        stems = dict(rocfft="fft", rocblas="blas", rocsparse="sparse",
                     rocsolver="solver", rocrand="rand", rocthrust="reduce")
        texts = {lib: self.command(["gfortran", "-cpp", "-E", "-DGPU_SUITE_IMPL=2",
                                    ROOT / f"amd/fortran/{lib}/benchmarks/{stem}_workload.F90"]).lower()
                 for lib, stem in stems.items()}

        def sub(lib, name):
            match = re.search(r"(?ms)^\s*subroutine " + name + r"\(.*?^\s*end subroutine", texts[lib])
            self.assertIsNotNone(match, (lib, name))
            return match.group()

        def ordered(text, tokens):
            offset = 0
            for token in tokens:
                position = text.find(token, offset)
                self.assertGreaterEqual(position, 0, token)
                offset = position + len(token)

        ordered(sub("rocfft", "operation"), ["!$omp target data", "call start_timer()",
                "rocfft_execute(", "call synchronize()", "call end_timer()", "!$omp end target data"])
        ordered(sub("rocblas", "operation"), ["!$omp target data", "call start_timer()",
                "do repetition=1,repetitions", "rocblas_dgemm(", "end do", "!$omp end target data",
                "call synchronize()", "call end_timer()"])
        ordered(sub("rocrand", "operation"), ["call start_timer()", "!$omp target data",
                "do repetition=1,repetitions", "rocrand_generate_uniform_double(", "end do",
                "!$omp end target data", "hipdevicesynchronize()", "call end_timer()"])
        self.assertNotIn("target exit data", sub("rocrand", "download"))
        self.assertIn("target update from(rand_out)", sub("rocrand", "download"))
        self.assertIn("map(delete:rand_out)", sub("rocrand", "cleanup"))
        self.assertIn("target exit data map(from:vec_y)", sub("rocsparse", "download"))
        self.assertNotIn("hipdevicesynchronize", sub("rocsparse", "setup"))
        self.assertIn("descr,1_c_int", (ROOT / "amd/fortran/rocsparse/benchmarks/sparse_workload.F90").read_text())
        self.assertNotIn("hipdevicesynchronize(", texts["rocthrust"])
        ordered(sub("rocthrust", "operation"), ["call start_timer()", "do repetition=1,repetitions",
                "rocthrust_square_sum(", "end do", "call end_timer()"])
        ordered(sub("rocsolver", "operation"), ["call start_timer()", "!$omp target data",
                "call solve_device(", "!$omp end target data", "call end_timer()"])
        self.assertEqual(sub("rocsolver", "operation").count("!$omp target data"), 1)
        ordered(sub("rocsolver", "solve_device"), ["rocsolver_dgetrf(", "hipdevicesynchronize()",
                "hipmemcpy(", "if (host_info == 0)", "rocsolver_dgetrs(", "hipdevicesynchronize()"])
        self.assertNotIn("target update from(info_getrf)", texts["rocsolver"])


if __name__ == "__main__":
    unittest.main()

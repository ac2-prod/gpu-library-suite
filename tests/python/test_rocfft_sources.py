"""Source contracts only: these checks do not claim real HIP compilation."""
from pathlib import Path
import unittest
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ROCFFT = ROOT / "amd" / "c-cpp" / "rocfft"


class RocfftSourceTests(unittest.TestCase):
    def test_canonical_inventory_and_single_source_examples(self):
        expected = {"examples/fft_cpu.c", "examples/fft_gpu.cpp", "examples/openmp_rocfft.cpp",
                    "benchmarks/fft_cpu_bench.c", "benchmarks/fft_gpu_bench.cpp",
                    "benchmarks/openmp_rocfft_bench.cpp"}
        self.assertEqual({str(p.relative_to(ROCFFT)) for p in ROCFFT.rglob("*")
                          if p.suffix in {".c", ".cpp", ".hip"}}, expected)
        for path in (ROCFFT / "examples").iterdir():
            if path.suffix not in {".c", ".cpp"}:
                continue
            source = path.read_text()
            self.assertIn("nfft = 1024", source)
            self.assertIn("batch = 4096", source)
            self.assertIn("isfinite", source)
            self.assertNotIn('#include "', source)
            self.assertNotIn("gpu_suite_options", source)

    def test_explicit_workspace_and_repeated_completion(self):
        source = (ROCFFT / "benchmarks/rocfft_context.hpp").read_text()
        for token in ("rocfft_plan_get_work_buffer_size", "hipMalloc(&workspace",
                      "rocfft_execution_info_set_work_buffer", "rocfft_precision_single",
                      "rocfft_placement_notinplace", "rocfft_transform_type_complex_forward"):
            self.assertIn(token, source)
        execute = source[source.index("bool execute("):source.index("inline void canonical_host")]
        self.assertLess(execute.index("rocfft_execute"), execute.index("hipDeviceSynchronize"))
        self.assertNotIn("hipMalloc", execute)
        self.assertNotIn("hipFree", execute)
        hip = (ROCFFT / "benchmarks/fft_gpu_bench.cpp").read_text()
        compute = hip[hip.index("bool run_compute_trial"):hip.index("bool run_end_to_end_trial")]
        self.assertLess(compute.index("restore_device"), compute.index("gpu_suite_measurement_start"))
        self.assertIn("context.fft.execute(context.input, context.output, 1)", compute)
        self.assertIn("elapsed_total +=", compute)
        omp = (ROCFFT / "benchmarks/openmp_rocfft_bench.cpp").read_text()
        compute = omp[omp.index("bool run_compute("):omp.index("int main(")]
        self.assertLess(compute.index("target update"), compute.index("gpu_suite_measurement_start"))
        self.assertIn("context.execute(input_data, output_data, 1)", compute)
        self.assertIn("elapsed_total +=", compute)
        self.assertEqual(source.count("rocfft_setup()"), 1)
        self.assertNotIn("rocfft_setup", source[source.index("struct PlanContext"):])

    def test_openmp_managed_arrays_and_fallback_proof(self):
        for relative in ("examples/openmp_rocfft.cpp", "benchmarks/openmp_rocfft_bench.cpp"):
            source = (ROCFFT / relative).read_text()
            self.assertIn("map(to:", source)
            self.assertIn("map(from:", source)
            self.assertIn("use_device_ptr", source)
            self.assertIn("omp_is_initial_device", source)
            self.assertIn("hipPointerGetAttributes", source)
            self.assertIn("hipMemoryTypeDevice", source)
            self.assertNotIn("hipMalloc(&input", source)
            self.assertNotIn("hipMalloc(&output", source)
            self.assertNotIn("hipFree(input", source)
            self.assertNotIn("hipFree(output", source)
        source = (ROCFFT / "benchmarks/openmp_rocfft_bench.cpp").read_text()
        function = source[source.index("bool run_end_to_end_trial"):source.index("bool run_compute")]
        self.assertLess(function.index("gpu_suite_measurement_start"), function.index("context.create"))
        self.assertLess(function.index("map(from:"), function.index("gpu_suite_measurement_end"))
        self.assertLess(function.index("gpu_suite_measurement_end"), function.index("context.destroy"))

    def test_cpu_uses_existing_backend_and_result_contract(self):
        wrapper = (ROCFFT / "benchmarks/fft_cpu_bench.c").read_text()
        self.assertIn("GPU_SUITE_BENCHMARK_ROCFFT", wrapper)
        self.assertIn("gpu_suite/fftw_bench_impl.h", wrapper)
        self.assertIn("GPU_SUITE_FFT_SEGMENTED_COMPUTE 1", wrapper)
        source = (ROOT / "common/c-cpp/include/gpu_suite/fftw_bench_impl.h").read_text()
        for token in ( "cpu-fftw-threaded", "cpu-fftw-serial",
                      "fftwf_plan_many_dft", "fftwf_plan_with_nthreads",
                      "gpu_suite_measurement_start", "gpu_suite_result_write",
                      "gpu_suite_finite_absolute_error", "problem dimensions overflow"):
            self.assertIn(token, source)

    def test_declaration_fixture_host_syntax_only(self):
        compiler = shutil.which("clang++")
        if compiler is None:
            self.skipTest("Clang C++ frontend unavailable for declaration-only syntax test")
        command = [compiler, "-Xclang", "-fopenmp", "-std=c++17", "-fsyntax-only",
                   "-I", str(ROOT / "tests/fixtures/rocm/include"),
                   "-I", str(ROOT / "common/c-cpp/include"),
                   "-DGPU_SUITE_HAVE_HIP_RUNTIME_METADATA=1"]
        command.extend(str(ROCFFT / name) for name in (
            "examples/fft_gpu.cpp", "examples/openmp_rocfft.cpp",
            "benchmarks/fft_gpu_bench.cpp", "benchmarks/openmp_rocfft_bench.cpp"))
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_hip_metadata_is_separate_from_cuda(self):
        source = (ROOT / "common/c-cpp/include/gpu_suite/hip_metadata.hpp").read_text()
        self.assertIn('"hip_runtime_version"', source)
        self.assertIn('"hip_driver_version"', source)
        self.assertIn('"build_metadata_sha256"', source)
        self.assertIn('"runtime"', source)
        self.assertIn("result.cuda_driver_version = nullptr", source)
        self.assertIn("result.cuda_runtime_version = nullptr", source)
        self.assertNotIn("cudaGet", source)


if __name__ == "__main__":
    unittest.main()

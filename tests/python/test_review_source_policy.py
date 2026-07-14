import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LIBRARIES = {
    "cufft": ("fft", "cufft"),
    "cublas": ("blas", "cublas"),
    "cusparse": ("sparse", "cusparse"),
    "cusolver": ("solver", "cusolver"),
    "curand": ("rand", "curand"),
    "thrust": ("reduce", "thrust"),
}


def read(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


class ReviewSourcePolicyTests(unittest.TestCase):
    def benchmark_paths(self):
        paths = []
        for library, (stem, openacc_stem) in LIBRARIES.items():
            directory = ROOT / "nvidia" / "c-cpp" / library / "benchmarks"
            cpu_suffix = ".cpp" if library in {"curand", "thrust"} else ".c"
            paths.extend([
                directory / (stem + "_cpu_bench" + cpu_suffix),
                directory / (stem + "_gpu_bench.cu"),
                directory / ("openacc_" + openacc_stem + "_bench.cpp"),
            ])
        return paths

    def test_all_benchmarks_use_checked_measurement_boundaries(self):
        paths = self.benchmark_paths()
        self.assertEqual(len(paths), 18)
        for path in paths:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertIn("gpu_suite_measurement_start", text)
                self.assertIn("gpu_suite_measurement_end", text)
                self.assertNotIn("gpu_suite_utc_timestamp", text)
                self.assertNotIn("gpu_suite_monotonic_now", text)

        clock = read("common/c-cpp/src/clock.c")
        start = clock.index("int gpu_suite_measurement_start")
        end = clock.index("int gpu_suite_measurement_end")
        start_body = clock[start:end]
        end_body = clock[end:]
        self.assertLess(start_body.index("gpu_suite_utc_timestamp"),
                        start_body.index("gpu_suite_clock_now"))
        self.assertLess(end_body.index("gpu_suite_clock_now"),
                        end_body.index("gpu_suite_utc_timestamp"))
        self.assertIn("report_api_error", clock)
        self.assertIn('"clock_gettime(CLOCK_MONOTONIC)"', clock)

    def test_verification_failure_is_not_a_fatal_trial_barrier(self):
        for relative in (
            "nvidia/c-cpp/cublas/benchmarks/blas_gpu_bench.cu",
            "nvidia/c-cpp/cublas/benchmarks/openacc_cublas_bench.cpp",
            "nvidia/c-cpp/cusparse/benchmarks/sparse_gpu_bench.cu",
            "nvidia/c-cpp/cusparse/benchmarks/openacc_cusparse_bench.cpp",
        ):
            text = read(relative)
            with self.subTest(path=relative):
                self.assertIn("fatal_failure", text)
                self.assertIn("any_failure", text)
                self.assertRegex(
                    text,
                    r"any_failure\s*=\s*(?:any_failure\s*\|\||true)",
                )
                self.assertNotRegex(
                    text,
                    r"verification[^;]{0,160}fatal_failure\s*=\s*true",
                )

    def test_gpu_warmup_and_timing_calls_are_checked_and_diagnostic(self):
        gpu_paths = [
            path for path in self.benchmark_paths()
            if path.suffix in {".cu", ".cpp"}
            and (path.suffix == ".cu" or path.name.startswith("openacc_"))
        ]
        self.assertEqual(len(gpu_paths), 12)
        unchecked_sync = re.compile(
            r"^\s*cudaDeviceSynchronize\s*\(\s*\)\s*;", re.MULTILINE
        )
        for path in gpu_paths:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertIn("warmup", text.lower())
                self.assertIn("cudaGetErrorString", text)
                self.assertIsNone(unchecked_sync.search(text))
                self.assertNotIn("gpu_suite_utc_timestamp", text)
                self.assertNotIn("gpu_suite_monotonic_now", text)

        openacc_curand = read(
            "nvidia/c-cpp/curand/benchmarks/openacc_curand_bench.cpp"
        )
        self.assertNotRegex(
            openacc_curand,
            r"curandCreateGenerator\([^;]+CURAND_STATUS_SUCCESS",
        )
        sparse_cpu = read(
            "nvidia/c-cpp/cusparse/benchmarks/sparse_cpu_bench.c"
        )
        for api in (
            "mkl_sparse_d_create_csr", "mkl_sparse_set_mv_hint",
            "mkl_sparse_optimize", "mkl_sparse_d_mv",
        ):
            self.assertIn('"{0}"'.format(api), sparse_cpu)

    def test_openacc_solver_copies_nonzero_sentinel_info_back(self):
        benchmark = read(
            "nvidia/c-cpp/cusolver/benchmarks/openacc_cusolver_bench.cpp"
        )
        example = read(
            "nvidia/c-cpp/cusolver/examples/openacc_cusolver.cpp"
        )
        self.assertGreaterEqual(benchmark.count("getrf_info[0] = -1"), 3)
        self.assertGreaterEqual(benchmark.count("getrs_info[0] = -1"), 3)
        self.assertIn("update self(rhs_ptr", benchmark)
        self.assertIn("getrf_info_ptr[0 : 1]", benchmark)
        self.assertIn("getrs_info_ptr[0 : 1]", benchmark)
        self.assertIn("getrf_info(1, -1)", example)
        self.assertIn("getrs_info(1, -1)", example)
        self.assertIn("update self(getrf_ip[0 : 1], getrs_ip[0 : 1])", example)

    def test_cuda_sync_and_teaching_header_policy(self):
        for library, (stem, openacc_stem) in LIBRARIES.items():
            directory = ROOT / "nvidia" / "c-cpp" / library / "examples"
            direct = (directory / (stem + "_gpu.cu")).read_text(encoding="utf-8")
            openacc = (
                directory / ("openacc_" + openacc_stem + ".cpp")
            ).read_text(encoding="utf-8")
            with self.subTest(library=library):
                self.assertNotIn("cudaDeviceSynchronize", direct)
                self.assertIn("cudaDeviceSynchronize", openacc)

        blas = read("nvidia/c-cpp/cublas/examples/blas_cpu.c")
        solver = read("nvidia/c-cpp/cusolver/examples/solver_cpu.c")
        self.assertRegex(blas, r"#else\s*\n#include <cblas[.]h>")
        self.assertRegex(solver, r"#else\s*\n#include <lapacke[.]h>")

    def test_canonical_sparse_example_and_zero_workspace_policy(self):
        cpu = read("nvidia/c-cpp/cusparse/examples/sparse_cpu.c")
        cmake = read("nvidia/c-cpp/cusparse/CMakeLists.txt")
        self.assertIn("#include <mkl_spblas.h>", cpu)
        self.assertIn("mkl_sparse_d_create_csr", cpu)
        self.assertNotIn("GPU_SUITE_REFERENCE_SPARSE", cpu)
        self.assertIn("canonical sparse_cpu teaching example requires oneMKL Sparse",
                      cmake)

        conditions = {
            "nvidia/c-cpp/cusparse/examples/sparse_gpu.cu":
                "workspace_size > 0",
            "nvidia/c-cpp/cusparse/examples/openacc_cusparse.cpp":
                "workspace_size > 0",
            "nvidia/c-cpp/cusparse/benchmarks/sparse_gpu_bench.cu":
                "q.ws == 0",
            "nvidia/c-cpp/cusparse/benchmarks/openacc_cusparse_bench.cpp":
                "workspace_size == 0",
        }
        for relative, condition in conditions.items():
            text = read(relative)
            with self.subTest(path=relative):
                self.assertIn(condition, text)
                self.assertRegex(text, r"(?:void|auto) \*\w*(?:workspace|work)\w* = nullptr")

    def test_runtime_gpu_metadata_uses_device_and_library_apis(self):
        helper = read("common/c-cpp/include/gpu_suite/cuda_metadata.hpp")
        for symbol in (
            "cudaGetDeviceProperties", "cudaDriverGetVersion",
            "cudaRuntimeGetVersion", "cublasGetVersion", "cufftGetVersion",
            "cusparseGetVersion", "cusolverGetProperty", "curandGetVersion",
            "THRUST_VERSION",
        ):
            with self.subTest(symbol=symbol):
                self.assertIn(symbol, helper)
        self.assertRegex(helper, r"properties[.]uuid[.]bytes")
        targets = read("cmake/GpuSuiteTargets.cmake")
        self.assertIn("GPU_SUITE_HAVE_CUDA_RUNTIME_METADATA=1", targets)
        fft_result = read(
            "nvidia/c-cpp/cufft/benchmarks/fft_bench_result.hpp"
        )
        self.assertIn("apply_cuda_runtime_metadata", fft_result)
        for path in self.benchmark_paths():
            if path.suffix != ".cu" and not path.name.startswith("openacc_"):
                continue
            if path.parent.parent.name == "cufft":
                continue
            with self.subTest(runtime_metadata=path.relative_to(ROOT)):
                self.assertIn(
                    '#include "gpu_suite/benchmark.hpp"',
                    path.read_text(encoding="utf-8"),
                )

    def test_linux_c17_fixture_and_provider_cmake_policy(self):
        common = read("common/c-cpp/CMakeLists.txt")
        targets = read("cmake/GpuSuiteTargets.cmake")
        tests = read("tests/CMakeLists.txt")
        providers = read("cmake/GpuSuiteCpuDependencies.cmake")
        self.assertIn("UNIX AND NOT APPLE", common)
        self.assertIn("target_link_libraries(gpu_suite_math INTERFACE m)", common)
        self.assertIn("target_link_libraries(${target} PRIVATE gpu_suite::math)",
                      targets)
        self.assertIn("_POSIX_C_SOURCE=200809L", tests)
        self.assertIn('${CMAKE_BINARY_DIR}/test-fixtures', tests)
        self.assertNotIn('/tmp/gpu-library-suite', tests)
        for name in (
            "GPU_SUITE_ONEMKL_CBLAS_INCLUDE_DIR",
            "GPU_SUITE_OPENBLAS_CBLAS_INCLUDE_DIR",
            "GPU_SUITE_GENERIC_CBLAS_INCLUDE_DIR",
            "GPU_SUITE_ONEMKL_LAPACKE_INCLUDE_DIR",
            "GPU_SUITE_OPENBLAS_LAPACKE_INCLUDE_DIR",
            "GPU_SUITE_GENERIC_LAPACKE_INCLUDE_DIR",
        ):
            self.assertIn(name, providers)
        self.assertIn("cmake_cpu_provider_mixed_rejected", tests)
        self.assertIn("cmake_cpu_provider_switch_openblas", tests)

    def test_pegasus_mpi_and_optional_compiler_policy(self):
        template = read("jobs/pegasus/run_benchmarks.pbs.in")
        invocation = 'mpirun ${NQSV_MPIOPTS} "${MPI_OPTIONS[@]}"'
        self.assertEqual(template.count(invocation), 2)
        runtime = read("jobs/pegasus/collect_runtime_environment.py")
        self.assertIn('"optional-metadata"', runtime)
        self.assertIn('"required-site-policy"', runtime)
        self.assertIn("require_runtime_compilers", runtime)
        required_block = runtime[
            runtime.index("if require_gpu_tools:"):
            runtime.index("library_versions =", runtime.index("if require_gpu_tools:"))
        ]
        self.assertNotRegex(required_block, r"(?:nvcc|nvc|nvc_plus_plus).*status")
        node_tools = read("jobs/pegasus/node_tools.py")
        self.assertIn("ctypes.CDLL", node_tools)
        self.assertIn("cudaDriverGetVersion", node_tools)
        self.assertIn("cudaRuntimeGetVersion", node_tools)
        self.assertIn('"cuda_runtime_identity"', node_tools)
        self.assertIn('("cuda_driver_version", expected_driver', node_tools)
        self.assertIn('("cuda_runtime_version", expected_runtime', node_tools)

    def test_result_metadata_has_no_backend_name_inference(self):
        result = read("common/c-cpp/src/result.c")
        self.assertNotIn("strstr(options->cpu_backend", result)
        self.assertIn("options->cpu_backend_role", result)
        self.assertIn("options->series_role", result)
        self.assertIn("options->cpu_parallelism", result)
        self.assertIn("options->cpu_threads_effective", result)

        metadata = read("common/c-cpp/src/metadata.c")
        function_start = metadata.index(
            "gpu_suite_build_global_configure_flags"
        )
        openacc_branch = metadata[
            metadata.index(
                "implementation == GPU_SUITE_IMPLEMENTATION_OPENACC",
                function_start,
            ):
            metadata.index("return GPU_SUITE_BUILD_C_FLAGS", function_start)
        ]
        self.assertIn("return GPU_SUITE_BUILD_CXX_FLAGS", openacc_branch)
        self.assertNotIn("GPU_SUITE_BUILD_OPENACC", openacc_branch)


if __name__ == "__main__":
    unittest.main()

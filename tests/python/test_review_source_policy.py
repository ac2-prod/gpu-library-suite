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

    def test_all_benchmarks_check_api_ranges_products_and_byte_counts(self):
        paths = self.benchmark_paths()
        self.assertEqual(len(paths), 18)
        for path in paths:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertIn("gpu_suite_checked_bytes", text)
                self.assertNotRegex(text, r"malloc\([^)]*\*\s*sizeof")
                if path.suffix in {".cpp", ".cu"}:
                    self.assertIn("std::length_error", text)
                    self.assertIn("std::bad_alloc", text)

        for library in ("cufft", "cublas", "cusolver"):
            for path in paths:
                if path.parent.parent.name == library:
                    with self.subTest(api_integer=path.relative_to(ROOT)):
                        self.assertIn("gpu_suite_checked_u64_to_int",
                                      path.read_text(encoding="utf-8"))
        for path in paths:
            if path.parent.parent.name == "cusparse":
                with self.subTest(csr=path.relative_to(ROOT)):
                    self.assertIn(
                        "gpu_suite_checked_poisson2d_dimensions",
                        path.read_text(encoding="utf-8"),
                    )

    def test_end_to_end_warmup_uses_temporary_scope_pipeline(self):
        paths = self.benchmark_paths()
        self.assertEqual(len(paths), 18)
        warmup_loop = re.compile(
            r"(?:warmup|warm|w|iteration)\s*<\s*(?:options|o)[.]warmup"
        )
        for path in paths:
            text = path.read_text(encoding="utf-8")
            with self.subTest(warmup_count=path.relative_to(ROOT)):
                self.assertRegex(text, warmup_loop)

        direct_cuda = [
            path for path in paths if path.suffix == ".cu"
        ]
        self.assertEqual(len(direct_cuda), 6)
        for path in direct_cuda:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertRegex(text, r"run_end_to_end_(?:pipeline|trial)")
                if "warmup_end_to_end" in text:
                    self.assertGreaterEqual(text.count("warmup_end_to_end"), 2)
                else:
                    self.assertRegex(
                        text,
                        r"warmup[\s\S]{0,900}"
                        r"run_end_to_end_(?:pipeline|trial)|"
                        r"run_end_to_end_(?:pipeline|trial)"
                        r"[\s\S]{0,900}warmup",
                    )
                self.assertRegex(
                    text,
                    r"scope\s*==\s*GPU_SUITE_SCOPE_COMPUTE[\s\S]{0,500}"
                    r"(?:create|cudaMalloc|device_vector)",
                )

        openacc_blas = read(
            "nvidia/c-cpp/cublas/benchmarks/openacc_cublas_bench.cpp"
        )
        self.assertRegex(
            openacc_blas,
            r"warmup[\s\S]{0,900}cublasCreate[\s\S]{0,900}cublasDestroy",
        )
        openacc_solver = read(
            "nvidia/c-cpp/cusolver/benchmarks/openacc_cusolver_bench.cpp"
        )
        self.assertRegex(
            openacc_solver,
            r"warmup[\s\S]{0,1600}cusolverDnCreate[\s\S]{0,2200}"
            r"cusolverDnDestroy",
        )

        openacc_fft = read(
            "nvidia/c-cpp/cufft/benchmarks/openacc_cufft_bench.cpp"
        )
        fft_warmup = openacc_fft[
            openacc_fft.index("bool warmup_end_to_end"):
            openacc_fft.index("bool run_end_to_end_trial")
        ]
        self.assertIn("create_plan", fft_warmup)
        self.assertIn("#pragma acc data", fft_warmup)
        self.assertIn("cufftDestroy", fft_warmup)

        openacc_sparse = read(
            "nvidia/c-cpp/cusparse/benchmarks/openacc_cusparse_bench.cpp"
        )
        sparse_pipeline = openacc_sparse[
            openacc_sparse.index("static bool run_end_to_end_once"):
            openacc_sparse.index("int main")
        ]
        for marker in (
            "cusparseCreate", "cusparseCreateCsr", "#pragma acc data",
            "cudaFree", "cusparseDestroyDnVec", "cusparseDestroySpMat",
            "cusparseDestroy(handle)",
        ):
            self.assertIn(marker, sparse_pipeline)

        openacc_curand = read(
            "nvidia/c-cpp/curand/benchmarks/openacc_curand_bench.cpp"
        )
        curand_pipeline = openacc_curand[
            openacc_curand.index("bool run_end_to_end_pipeline"):
            openacc_curand.index("} // namespace")
        ]
        for marker in (
            "curandCreateGenerator", "#pragma acc data",
            "curandDestroyGenerator",
        ):
            self.assertIn(marker, curand_pipeline)
        self.assertRegex(
            openacc_curand,
            r"warmup\s*<\s*options[.]warmup[\s\S]{0,500}"
            r"run_end_to_end_pipeline",
        )

        openacc_thrust = read(
            "nvidia/c-cpp/thrust/benchmarks/openacc_thrust_bench.cpp"
        )
        thrust_pipeline = openacc_thrust[
            openacc_thrust.index("bool run_end_to_end_pipeline"):
            openacc_thrust.index("} // namespace")
        ]
        self.assertIn("#pragma acc data copyin", thrust_pipeline)
        self.assertIn("transform_reduce", thrust_pipeline)
        self.assertNotIn("device_vector", thrust_pipeline)
        self.assertRegex(
            openacc_thrust,
            r"warmup\s*<\s*options[.]warmup[\s\S]{0,500}"
            r"run_end_to_end_pipeline",
        )

    def test_solver_workspace_and_nonfinite_reduction_policy(self):
        for relative in (
            "nvidia/c-cpp/cusolver/benchmarks/solver_gpu_bench.cu",
            "nvidia/c-cpp/cusolver/benchmarks/openacc_cusolver_bench.cpp",
        ):
            text = read(relative)
            with self.subTest(path=relative):
                self.assertNotRegex(text, r"sqrt\s*\([^)]*matrix[.]size")
                self.assertIn("workspace_count", text)
                self.assertIn("gpu_suite_checked_bytes", text)
                self.assertRegex(text, r"workspace_(?:count|bytes)\s*(?:>=|!=|<)")

        benchmark_text = "\n".join(
            path.read_text(encoding="utf-8") for path in self.benchmark_paths()
        )
        helper_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "nvidia" / "c-cpp").glob(
                "*/benchmarks/*_bench_common.hpp"
            )
        )
        combined = benchmark_text + helper_text
        self.assertNotIn("fmax(", combined)
        self.assertNotIn("std::max(", combined)
        self.assertIn("gpu_suite_finite_absolute_error", combined)
        self.assertIn("gpu_suite_json_add_null", combined)
        self.assertIn('verification_status = "nonfinite"', combined)

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
        cuda_probe = read("jobs/pegasus/cuda_runtime_probe.py")
        self.assertIn("ctypes.CDLL", cuda_probe)
        self.assertIn("cudaDriverGetVersion", cuda_probe)
        self.assertIn("cudaRuntimeGetVersion", cuda_probe)
        self.assertIn('"cuda_runtime_identity"', node_tools)
        self.assertIn('("cuda_driver_version", expected_driver', node_tools)
        self.assertIn('("cuda_runtime_version", expected_runtime', node_tools)
        self.assertIn('"runtime_probe": cuda_runtime_identity', runtime)
        self.assertNotIn("GPU_SUITE_CUDA_RUNTIME_VERSION", runtime)

    def test_gpu_provenance_has_distinct_rank_local_producers(self):
        result = read("common/c-cpp/src/result.c")
        for obsolete in (
            "GPU_SUITE_CUDA_DRIVER_VERSION",
            "GPU_SUITE_CUDA_RUNTIME_VERSION",
            "GPU_SUITE_LIBRARY_VERSION",
        ):
            with self.subTest(obsolete=obsolete):
                self.assertNotIn(obsolete, result)
        run_node = read("jobs/pegasus/run_node.sh")
        node_tools = read("jobs/pegasus/node_tools.py")
        self.assertIn("GPU_SUITE_NVIDIA_DRIVER_VERSION", run_node)
        self.assertIn("GPU_SUITE_NVIDIA_DRIVER_VERSION", node_tools)
        self.assertIn('"nvidia_driver_version"', node_tools)
        self.assertIn('"cuda_driver_api_version"', node_tools)

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

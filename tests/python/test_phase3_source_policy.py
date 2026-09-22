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


class Phase3SourcePolicyTests(unittest.TestCase):
    def test_all_36_canonical_sources_and_target_names_exist(self):
        expected_paths = set()
        for library, names in LIBRARIES.items():
            stem, openacc_stem = names
            directory = ROOT / "nvidia" / "c-cpp" / library
            cpu_extension = ".cpp" if library in {"curand", "thrust"} else ".c"
            paths_and_targets = {
                directory / "examples" / (stem + "_cpu" + cpu_extension): stem + "_cpu",
                directory / "examples" / (stem + "_gpu.cu"): stem + "_gpu",
                directory / "examples" / ("openacc_" + openacc_stem + ".cpp"): "openacc_" + openacc_stem,
                directory / "benchmarks" / (stem + "_cpu_bench" + cpu_extension): stem + "_cpu_bench",
                directory / "benchmarks" / (stem + "_gpu_bench.cu"): stem + "_gpu_bench",
                directory / "benchmarks" / ("openacc_" + openacc_stem + "_bench.cpp"): "openacc_" + openacc_stem + "_bench",
            }
            cmake = (directory / "CMakeLists.txt").read_text(encoding="utf-8")
            for path, target in paths_and_targets.items():
                expected_paths.add(path)
                self.assertTrue(path.is_file(), str(path))
                self.assertEqual(path.stem, target)
                relative = path.relative_to(directory).as_posix()
                pattern = r"add_executable\(\s*{0}\s+{1}\s*\)".format(
                    re.escape(target), re.escape(relative)
                )
                self.assertRegex(cmake, pattern)
        actual_paths = {
            path
            for library in LIBRARIES
            for path in (ROOT / "nvidia" / "c-cpp" / library).glob("*/*")
            if path.suffix in {".c", ".cpp", ".cu"}
        }
        self.assertEqual(actual_paths, expected_paths)
        self.assertEqual(len(actual_paths), 36)

    def test_teaching_examples_are_direct_single_source_programs(self):
        for library in LIBRARIES:
            example_dir = ROOT / "nvidia" / "c-cpp" / library / "examples"
            for path in example_dir.iterdir():
                text = path.read_text(encoding="utf-8")
                self.assertIn("main(", text, str(path))
                self.assertNotIn("gpu_suite/", text, str(path))
                self.assertNotIn("--output", text, str(path))
                self.assertNotIn("TODO", text, str(path))
                self.assertNotIn("...", text, str(path))
        for path in (ROOT / "nvidia" / "c-cpp" / "cusparse" / "examples").iterdir():
            self.assertIn("make_poisson2d_csr", path.read_text(encoding="utf-8"))
        for path in (ROOT / "nvidia" / "c-cpp" / "cusolver" / "examples").iterdir():
            self.assertIn("make_dense_system", path.read_text(encoding="utf-8"))

    def test_openacc_managed_arrays_are_not_manually_allocated(self):
        for library in ("cublas", "curand", "thrust"):
            directory = ROOT / "nvidia" / "c-cpp" / library
            paths = list((directory / "examples").glob("openacc_*.cpp"))
            paths += list((directory / "benchmarks").glob("openacc_*.cpp"))
            for path in paths:
                self.assertNotIn(
                    "cudaMalloc", path.read_text(encoding="utf-8"), str(path)
                )
        for library in ("cusparse", "cusolver"):
            directory = ROOT / "nvidia" / "c-cpp" / library
            paths = list((directory / "examples").glob("openacc_*.cpp"))
            paths += list((directory / "benchmarks").glob("openacc_*.cpp"))
            for path in paths:
                text = path.read_text(encoding="utf-8")
                self.assertIn("#pragma acc data", text)
                self.assertIn("host_data", text)

    def test_library_readmes_record_required_interoperation_notes(self):
        for library in ("cublas", "cusparse", "cusolver", "curand", "thrust"):
            path = ROOT / "nvidia" / "c-cpp" / library / "README.md"
            self.assertTrue(path.is_file())
            text = path.read_text(encoding="utf-8")
            self.assertIn("BENCHMARK_PROTOCOL.md", text)
        curand = (ROOT / "nvidia" / "c-cpp" / "curand" / "README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("not of identical random", curand)
        thrust = (ROOT / "nvidia" / "c-cpp" / "thrust" / "README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("synchronous algorithms normally need no", thrust)

    def test_cuda_library_probes_use_runtime_and_exercised_symbols(self):
        expected = {
            "cufft": ("cudaFree", "cufftPlanMany", "cufftExecC2C"),
            "cublas": ("cudaFree", "cublasDgemm"),
            "cusparse": (
                "cudaFree",
                "cusparseCreateCsr",
                "cusparseCreateDnVec",
                "cusparseSpMV_bufferSize",
                "cusparseSpMV",
            ),
            "cusolver": (
                "cudaFree",
                "cusolverDnDgetrf_bufferSize",
                "cusolverDnDgetrf",
                "cusolverDnDgetrs",
            ),
            "curand": (
                "cudaFree",
                "curandCreateGenerator",
                "curandGenerateUniformDouble",
            ),
        }
        for library, symbols in expected.items():
            cmake = (
                ROOT / "nvidia" / "c-cpp" / library / "CMakeLists.txt"
            ).read_text(encoding="utf-8")
            for symbol in symbols:
                with self.subTest(library=library, symbol=symbol):
                    self.assertIn(symbol, cmake)

    def test_openacc_gpu_target_and_flags_are_applied_and_recorded(self):
        top_level = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        targets = (ROOT / "cmake" / "GpuSuiteTargets.cmake").read_text(
            encoding="utf-8"
        )
        probes = (ROOT / "cmake" / "GpuSuiteProbe.cmake").read_text(
            encoding="utf-8"
        )
        metadata = (
            ROOT / "cmake" / "gpu_suite_build_metadata.json.in"
        ).read_text(encoding="utf-8")
        self.assertIn("-gpu=${GPU_SUITE_NVHPC_GPU_TARGET}", top_level)
        self.assertIn(
            'message(FATAL_ERROR\n          "CUDAToolkit path differs',
            top_level,
        )
        self.assertIn("GPU_SUITE_OPENACC_COMPILE_OPTIONS", targets)
        self.assertIn("GPU_SUITE_OPENACC_LINK_OPTIONS", targets)
        self.assertIn("OpenACC::OpenACC_CXX", probes)
        self.assertIn("GPU_SUITE_OPENACC_COMPILE_OPTIONS", probes)
        self.assertIn("thrust_cuda_interop_compile_flags", metadata)
        self.assertIn("thrust_cuda_interop_link_flags", metadata)


if __name__ == "__main__":
    unittest.main()

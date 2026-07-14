import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CUFFT = ROOT / "nvidia" / "c-cpp" / "cufft"


class CufftSourcePolicyTests(unittest.TestCase):
    def test_canonical_source_inventory(self):
        expected = {
            "examples/fft_cpu.c",
            "examples/fft_gpu.cu",
            "examples/openacc_cufft.cpp",
            "benchmarks/fft_cpu_bench.c",
            "benchmarks/fft_gpu_bench.cu",
            "benchmarks/openacc_cufft_bench.cpp",
        }
        actual = {
            str(path.relative_to(CUFFT))
            for path in CUFFT.rglob("*")
            if path.suffix in {".c", ".cpp", ".cu"}
        }
        self.assertEqual(actual, expected)

    def test_target_names_equal_source_stems(self):
        cmake = (CUFFT / "CMakeLists.txt").read_text(encoding="utf-8")
        pairs = re.findall(r"add_executable\((\w+)\s+([^\s\)]+)", cmake)
        self.assertEqual(len(pairs), 6)
        for target, source in pairs:
            with self.subTest(target=target):
                self.assertEqual(target, Path(source).stem)

    def test_real_symbol_probes_are_present(self):
        finder = (ROOT / "cmake" / "FindFFTW3f.cmake").read_text(
            encoding="utf-8"
        )
        for symbol in (
            "fftwf_plan_many_dft",
            "fftwf_execute",
            "fftwf_init_threads",
            "fftwf_plan_with_nthreads",
            "fftwf_cleanup_threads",
        ):
            self.assertIn(symbol, finder)
        cmake = (CUFFT / "CMakeLists.txt").read_text(encoding="utf-8")
        for token in (
            "cudaFree",
            "cufftPlanMany",
            "cufftExecC2C",
            "OpenACC::OpenACC_CXX",
            "CUDA::cudart",
            "CUDA::cufft",
        ):
            self.assertIn(token, cmake)

    def test_openacc_managed_arrays_are_not_cuda_allocated(self):
        for relative in (
            "examples/openacc_cufft.cpp",
            "benchmarks/openacc_cufft_bench.cpp",
        ):
            source = (CUFFT / relative).read_text(encoding="utf-8")
            with self.subTest(source=relative):
                self.assertIn("#pragma acc data", source)
                self.assertIn("#pragma acc host_data use_device", source)
                self.assertNotIn("cudaMalloc", source)

    def test_primary_and_auxiliary_backend_names_are_explicit(self):
        source = (CUFFT / "benchmarks" / "fft_cpu_bench.c").read_text(
            encoding="utf-8"
        )
        self.assertIn("cpu-fftw-threaded", source)
        self.assertIn("cpu-fftw-serial", source)
        self.assertIn("threaded FFTW backend is unavailable", source)


if __name__ == "__main__":
    unittest.main()

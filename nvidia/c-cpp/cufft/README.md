# cuFFT C/C++ examples and benchmarks

This directory contains the six canonical cuFFT programs listed in the
[project specification](../../../docs/PROJECT_SPECIFICATION.md): a CPU, direct
CUDA, and OpenACC teaching example and their benchmark counterparts. The
teaching examples use the fixed batched FP32 forward transform from that
specification. Benchmark behavior, timing scopes, and verification are defined
by the [benchmark protocol](../../../docs/BENCHMARK_PROTOCOL.md).

The CPU programs require FFTW3's single-precision interface. `fft_cpu_bench`
contains the `cpu-fftw-threaded` series only when its thread initialization,
thread-count, and cleanup symbols compile and link. On Pegasus that threaded
series is the primary production denominator. `cpu-fftw-serial` remains an
auxiliary teaching-correspondence series; it never replaces a missing threaded
series and no primary cuFFT speedup is produced without the configured threaded
denominator.

Direct CUDA programs link `CUDA::cudart` and `CUDA::cufft`. OpenACC programs use
the same imported CUDA targets together with `OpenACC::OpenACC_CXX`; the
OpenACC-managed input/output arrays are not allocated with `cudaMalloc`.
Architecture and NVHPC GPU target selections must be supplied by the build
profile, not by these sources.

The canonical CMake targets and executable names are `fft_cpu`, `fft_gpu`,
`openacc_cufft`, `fft_cpu_bench`, `fft_gpu_bench`, and
`openacc_cufft_bench`. Missing optional dependencies disable only their affected
targets and appear with a reason in the configure summary.

GPU and NVHPC execution must be validated manually on Pegasus as described in
[the Pegasus execution guide](../../../docs/PEGASUS_EXECUTION.md); a local
CPU-only configure does not count as a GPU test.

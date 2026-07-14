# cuBLAS C/C++ examples and benchmarks

This directory pairs a column-major FP64 DGEMM teaching example with benchmark
implementations for a CBLAS CPU provider, direct CUDA, and OpenACC-to-cuBLAS
interoperation. The exact matrix problem is owned by
[`docs/PROJECT_SPECIFICATION.md`](../../../docs/PROJECT_SPECIFICATION.md), and
timing, restoration, CLI, and verification rules are owned by
[`docs/BENCHMARK_PROTOCOL.md`](../../../docs/BENCHMARK_PROTOCOL.md).

The CPU target is enabled only when a configured CBLAS provider compiles and
links a real `cblas_dgemm` call. `BLAS_FOUND` alone is not used. Direct CUDA and
OpenACC targets similarly require CUDA Runtime and cuBLAS symbol probes.
OpenACC owns A, B, and C device data; `host_data use_device` only exposes their
already-present device addresses to cuBLAS.

The canonical source-stem/target/executable names are `blas_cpu`, `blas_gpu`,
`openacc_cublas`, `blas_cpu_bench`, `blas_gpu_bench`, and
`openacc_cublas_bench`.

Benchmark executables send one machine-readable record per trial to stdout when
run with `--output -`; diagnostics go to stderr. Production runs are orchestrated
by `tools/run_suite.py`, which is the sole writer of node raw-result files.

# cuSOLVER C/C++ examples and benchmarks

These programs solve the canonical dense FP64 system described by
[`docs/PROJECT_SPECIFICATION.md`](../../../docs/PROJECT_SPECIFICATION.md).
The CPU teaching example intentionally uses `LAPACKE_dgesv`; the CPU benchmark
and both GPU benchmarks use separate `getrf` and `getrs` stages as required by
[`docs/BENCHMARK_PROTOCOL.md`](../../../docs/BENCHMARK_PROTOCOL.md).

The LAPACKE target is enabled only after `LAPACKE_dgesv`, `LAPACKE_dgetrf`, and
`LAPACKE_dgetrs` all compile and link. CUDA and OpenACC targets probe the CUDA
Runtime plus the cuSOLVER factorization and solve symbols. Benchmark records
store `getrf_info` and `getrs_info` separately and always require `repeat=1`.

The canonical source-stem/target/executable names are `solver_cpu`,
`solver_gpu`, `openacc_cusolver`, `solver_cpu_bench`, `solver_gpu_bench`, and
`openacc_cusolver_bench`.

OpenACC owns matrix, right-hand-side, pivot, and info arrays. Its only explicit
CUDA allocation is the cuSOLVER workspace; address translation occurs in small
`host_data use_device` regions after data entry.

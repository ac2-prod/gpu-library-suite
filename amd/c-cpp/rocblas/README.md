# rocBLAS C/C++ examples and benchmarks

[English](README.md) | [日本語](README.ja.md)

The CPU, HIP and OpenMP target sources are implemented. Local host API fixtures
exercise numerical recurrence and failure handling; they do not establish real
ROCm compilation, offload or GPU execution. On Sirius, all six BLAS artifacts
built successfully in `sirius-amd-build-20261017T104351Z-a3697cae`. The separate
BLAS run `sirius-blas-smoke-20261017T135452Z-a3697cae` passed all three teaching
examples and 96 numerical records, including OpenMP target and CPU thread
requests 1/2. Effective CPU workers and performance remain unverified; see the
[recorded source boundary and results](../../../docs/VALIDATION_REPORT.md#amd-focused-numerical-smoke).

| Implementation | Teaching source / target | Benchmark source / target |
| --- | --- | --- |
| CPU | `examples/blas_cpu.c` / `blas_cpu` | `benchmarks/blas_cpu_bench.c` / `blas_cpu_bench` |
| HIP | `examples/blas_gpu.cpp` / `blas_gpu` | `benchmarks/blas_gpu_bench.cpp` / `blas_gpu_bench` |
| OpenMP target | `examples/openmp_rocblas.cpp` / `openmp_rocblas` | `benchmarks/openmp_rocblas_bench.cpp` / `openmp_rocblas_bench` |

Each teaching example is one complete, independently compiled source. It uses
FP64 column-major DGEMM, `C = alpha A B + beta C`, with `m=n=k=1024`, `alpha=beta=1`, and all-one A/B/C. The expected value is 1025.
GPU calls use `rocblas_dgemm` with `rocblas_operation_none` for both operands.

CPU uses the configured CBLAS provider (`cpu-openblas`, `cpu-onemkl`, or `cpu-generic-cblas`). The requested backend must match the compiled binary.

The benchmark preserves the shared CLI and raw-result lifecycle. Compute keeps
allocations and library context alive, restores C/y after warm-up and before
each raw trial, then measures `repeat` successive updates and completion using
the host monotonic clock. Its expected result follows that recurrence.
End-to-end restores C/y before each repeat outside timing, then measures
context/descriptor preparation, allocation or OpenMP mapping, transfers,
execution, completion and retrieval. Explicit cleanup follows the measured
interval. Host input generation and verification are untimed. Numerical failure,
nonfinite output, invalid dimensions and unavailable dependencies are explicit.

OpenMP owns mapped matrix/vector storage and passes `use_device_ptr` addresses
to rocBLAS. `omp_is_initial_device()` must reject host fallback, and HIP pointer
attributes must match the selected HIP device. Device ordinals alone do not
establish correspondence. Set `OMP_TARGET_OFFLOAD=MANDATORY` for real execution.
Only the library workspace, where needed, is allocated with HIP in this path.

Configure a new vendor-specific build with `GPU_SUITE_VENDOR=amd` and
`GPU_SUITE_SOURCE_LANGUAGE=c-cpp`. CPU, HIP and OpenMP use separate build trees.
Set `GPU_SUITE_BUILD_CPU`, `GPU_SUITE_BUILD_HIP`, `GPU_SUITE_BUILD_OPENMP` for the
selected profile, and disable CUDA/OpenACC. Native CMake HIP uses a supported
Clang driver, not the hipcc wrapper; architecture and dependency roots are
explicit inputs established for the execution system. Missing optional
libraries disable only affected targets and are not installed automatically.

Manual teaching compilation may use `hipcc` with `-lrocblas`. OpenMP compilation
requires the supported compiler's `-fopenmp` and explicit target architecture,
plus `-lrocblas -lamdhip64` and ordinary installation-specific search flags.
These instructions do not authorize login-node builds: approved Sirius builds
and executions run through its scheduler. See [Sirius operations](../../../jobs/sirius/README.md).

Mathematics and canonical names belong to [the specification](../../../docs/PROJECT_SPECIFICATION.md).
Timing/CPU rules belong to [the protocol](../../../docs/BENCHMARK_PROTOCOL.md),
and campaign provenance to [the schema](../../../docs/RESULT_SCHEMA.md).
API signatures were checked against the [ROCm 7.2 rocBLAS reference](https://rocm.docs.amd.com/projects/rocBLAS/en/docs-7.2.0/reference/level-3.html).

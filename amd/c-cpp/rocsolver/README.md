# rocSOLVER: dense FP64 linear solve

This family implements the same column-major dense `A X = B` problem as the
NVIDIA solver family: diagonal entries `n + 1`, off-diagonal entries `1`, and
all right-hand sides `2n`, with exact solution `X = 1`. Teaching defaults are
`n = 1024`, `nrhs = 16`.

| Implementation | Standalone teaching source | Benchmark |
| --- | --- | --- |
| CPU / LAPACKE | `examples/solver_cpu.c` | `benchmarks/solver_cpu_bench.c` |
| HIP / rocSOLVER | `examples/solver_gpu.cpp` | `benchmarks/solver_gpu_bench.cpp` |
| OpenMP target / rocSOLVER | `examples/openmp_rocsolver.cpp` | `benchmarks/openmp_rocsolver_bench.cpp` |

The CPU teaching program uses `LAPACKE_dgesv`; benchmarks use explicit `dgetrf`
and `dgetrs` stages. rocSOLVER uses a rocBLAS handle, `rocsolver_dgetrf` and
`rocsolver_dgetrs`. Factorization `info` is retrieved and checked before the
solve. Unlike LAPACK/cuSOLVER, rocSOLVER `getrs` has no `info` pointer:
`getrs_info = 0` in a benchmark means that its API return and completion both
succeeded; it is null when no successful solve was observed.

Every teaching file is independently compilable. For example, a configured HIP
toolchain can compile `solver_gpu.cpp` with `hipcc ... -lrocsolver -lrocblas`.
OpenMP requires an actual AMD offload-capable compiler and explicit target
flags. Neither compiler paths nor architecture flags are embedded in sources.
Use native HIP's supported Clang compiler for CMake, not a fixed hipcc wrapper.
The CPU provider must expose LAPACKE; missing dependencies disable only the
corresponding targets. GPU library/API discovery is performed before metadata
capture.

Benchmarks require `--repeat 1`; amplify measurements using trials. Both scopes
restore the complete matrix, right-hand sides, pivots and status before every
operation. `compute` keeps the handle, explicitly queried/allocated workspace,
and device storage/mappings alive, and times factorization, the required info
check, solve and completion. The info check includes a small device-to-host
transfer to avoid solving a failed factorization. `end-to-end` additionally
times handle/workspace/storage creation and input/output transfers; input
generation/restoration, verification and explicit cleanup are outside timing.
The CPU backend uses existing LAPACKE provider/thread settings.

OpenMP owns mappings for the matrices, pivots and status. Mapped pointers must
resolve to HIP device memory on the requested device; actual offload is checked
and host fallback is rejected. Only the explicitly allocated rocSOLVER
workspace is HIP-owned in that path. No `hipFree` is applied to mapped arrays.

Verification examines every solution and residual component, rejects NaN/Inf,
and reports relative residual and solution error using the shared tolerances.
Runtime/library/build provenance follows the common schema; it must not be
mixed across campaigns even when mathematical keys match.

Host tests use synthetic API fixtures. They check numerical validation,
restoration, allocation/lifecycle, fault handling, emitted records and OpenMP
source syntax. They do **not** establish ROCm availability, real offload or
performance. The subsequent approved Sirius job verified the CPU, HIP and
OpenMP target builds, three teaching examples and all 96 numerical smoke
records. See [the recorded source snapshot and run evidence](../../../jobs/sirius/README.md#focused-numerical-smoke).
Performance, effective CPU worker count and later source changes remain unverified.

Authoritative API references:
[rocSOLVER LAPACK functions](https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html),
[workspace memory model](https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/howto/memory.html),
[library information](https://rocmdocs.amd.com/projects/rocSOLVER/en/latest/reference/helpers.html).
Repository contracts belong to `docs/PROJECT_SPECIFICATION.md`,
`docs/BENCHMARK_PROTOCOL.md` and `docs/RESULT_SCHEMA.md`.

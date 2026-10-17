# rocSPARSE C/C++ examples and benchmarks

[English](README.md) | [日本語](README.ja.md)

The CPU (AOCL Sparse), HIP and OpenMP target implementations passed Sirius
build and basic numerical smoke: three teaching examples and 96 records, all
with zero error. See [the recorded source snapshot and run evidence](../../../jobs/sirius/README.md#focused-numerical-smoke).
Performance and effective CPU worker count remain unmeasured; this evidence
does not validate later source changes.

| Implementation | Teaching source / target | Benchmark source / target |
| --- | --- | --- |
| CPU | `examples/sparse_cpu.c` / `sparse_cpu` | `benchmarks/sparse_cpu_bench.c` / `sparse_cpu_bench` |
| HIP | `examples/sparse_gpu.cpp` / `sparse_gpu` | `benchmarks/sparse_gpu_bench.cpp` / `sparse_gpu_bench` |
| OpenMP target | `examples/openmp_rocsparse.cpp` / `openmp_rocsparse` | `benchmarks/openmp_rocsparse_bench.cpp` / `openmp_rocsparse_bench` |

Each teaching example is one complete, independently compiled source. It uses
FP64 zero-based CSR SpMV on a 1024 by 1024 five-point Poisson grid: `y = alpha A x + beta y`, `alpha=beta=1`, diagonal 4, off-diagonal -1, and all-one x/y. Boundary/corner neighbor counts determine the expected output.
GPU calls use the generic `rocsparse_spmv` API with `rocsparse_spmv_alg_csr_adaptive`, 32-bit CSR indices and FP64 values. Workspace query, allocation and preprocess precede compute; persistent compute trials reuse those descriptors and workspace.

CPU production supports explicit oneMKL Sparse (`cpu-onemkl`, default ONEMKL)
or AOCL Sparse (`cpu-aocl-sparse`, `GPU_SUITE_CPU_SPARSE_BACKEND=AOCLSPARSE`).
`GPU_SUITE_CPU_SPARSE_BACKEND=REFERENCE` explicitly selects independent
`cpu-reference-csr` for teaching and auxiliary verification. No missing provider
selects another provider or reference code automatically.

AOCL Sparse uses the LP64 descriptor/CSR, hint/optimize and `aoclsparse_dmv`
APIs. The setup uses the same fixed hint of one call as the oneMKL path;
analysis precedes timed compute and belongs inside end-to-end setup.
Manual CPU teaching builds define `GPU_SUITE_USE_AOCL_SPARSE=1` and supply the
selected header/library paths and any required ordered link inputs, for example:

```sh
cc -std=c17 -O2 -DGPU_SUITE_USE_AOCL_SPARSE=1 -I"$AOCL_SPARSE_INCLUDE" \
  examples/sparse_cpu.c "$AOCL_SPARSE_LIBRARY" -lm -o sparse_cpu
```

For a static library, also provide its required runtime/dependency files after
the primary library; the example does not guess those inputs. With CMake, use
`GPU_SUITE_AOCL_SPARSE_INCLUDE_DIR`, `GPU_SUITE_AOCL_SPARSE_LIBRARY` and the
ordered `GPU_SUITE_AOCL_SPARSE_EXTRA_LIBRARIES`. API compile/link and LP64 header
checks must pass before enabling targets. See the
[backend contract](../../../docs/BENCHMARK_PROTOCOL.md#cpu-backends-and-parallelism).
The benchmark records API version and effective AOCL/OMP controls; observed
worker count remains null. Header/configured-library hashes do not prove actual
loaded DSO identity; the numerical smoke does not claim complete loaded-DSO evidence.

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
to rocSPARSE. `omp_is_initial_device()` must reject host fallback, and HIP pointer
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

Manual teaching compilation may use `hipcc` with `-lrocsparse`. OpenMP compilation
requires the supported compiler's `-fopenmp` and explicit target architecture,
plus `-lrocsparse -lamdhip64` and ordinary installation-specific search flags.
These instructions do not authorize login-node builds: approved Sirius builds
and executions run through its scheduler. See [Sirius operations](../../../jobs/sirius/README.md).

Mathematics and canonical names belong to [the specification](../../../docs/PROJECT_SPECIFICATION.md).
Timing/CPU rules belong to [the protocol](../../../docs/BENCHMARK_PROTOCOL.md),
and campaign provenance to [the schema](../../../docs/RESULT_SCHEMA.md).
API signatures were checked against the [ROCm 7.2 rocSPARSE reference](https://rocm.docs.amd.com/projects/rocSPARSE/en/docs-7.2.0/reference/generic.html).
AOCL CPU API signatures were checked against the
[AOCL Sparse 5.1 public headers](https://github.com/amd/aocl-sparse/tree/5.1/library/include).

# GPU Library Suite Project Specification

## Authority and scope

This document owns the repository hierarchy, repository-owned canonical
filenames, teaching examples, mathematical problems, default teaching-example
parameters, and example-level data-management rules. Benchmark CLI behavior,
timing, verification thresholds, and statistics are owned by
[`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md). Machine-readable formats are
owned by [`RESULT_SCHEMA.md`](RESULT_SCHEMA.md). Pegasus operations are owned by
[`PEGASUS_EXECUTION.md`](PEGASUS_EXECUTION.md); Sirius operations are owned by
[`jobs/sirius/README.md`](../jobs/sirius/README.md).

The canonical source for teaching code, filenames, and default problem settings
is the 2026-07-13 edition of *Library Edition (NVIDIA GPU, C/C++)* (「ライブラリ
編（NVIDIA GPU, C/C++）」). Material dated 2026-07-12 or earlier is historical
reference only and must not override the canonical edition. A current user
instruction takes precedence if it conflicts with that edition.

The approved NVIDIA Fortran addition follows the supplied text extraction of
*Library Edition (NVIDIA GPU, Fortran)*, 2026-09-17 (58 slides). This is a
source/provenance authority, not evidence of compilation or GPU execution.
The original extraction is retained separately in the ignored
`manual-validation/fortran-input/` directory. C/C++ defaults are unchanged.

The owner-approved AMD Fortran baseline is
`HAIRDESC_AMD_GPU_Library_F_20261020.pptx` (57 slides). Slides prioritize readable
library-call flow; they are not a literal completeness specification. The
owner decides which details to show. Distributed programs supply the API
arguments, explicit helper inputs, initialization, verification and cleanup
needed to run; this does not authorize rewriting the slides.

## Project identity and hierarchy

The repository name is `gpu-library-suite`. Users traverse source code in this
order:

1. GPU vendor;
2. programming language;
3. GPU library; and
4. purpose: `examples/` or `benchmarks/`.

The intended final hierarchy is:

```text
gpu-library-suite/
  README.md
  AGENTS.md
  CMakeLists.txt
  nvidia/
    c-cpp/
      cufft/
        examples/
        benchmarks/
        README.md
        CMakeLists.txt
      cublas/
        examples/
        benchmarks/
        README.md
        CMakeLists.txt
      cusparse/
        examples/
        benchmarks/
        README.md
        CMakeLists.txt
      cusolver/
        examples/
        benchmarks/
        README.md
        CMakeLists.txt
      curand/
        examples/
        benchmarks/
        README.md
        CMakeLists.txt
      thrust/
        examples/
        benchmarks/
        README.md
        CMakeLists.txt
    fortran/                 same six library/purpose subdirectories
  amd/
    c-cpp/
      rocfft/                CPU / HIP / OpenMP teaching and benchmarks
        examples/
        benchmarks/
        README.md
        README.ja.md
        CMakeLists.txt
      rocblas/               same examples/benchmarks layout
      rocsparse/              same layout; v2/csr_rowsplit GPU API
      rocsolver/              same examples/benchmarks layout
      rocrand/                same examples/benchmarks layout
      rocthrust/              OpenMP also links rocthrust_wrapper.hip
    fortran/                 same six AMD libraries; CPU / HIP / OpenMP
      common/                AMD Fortran benchmark driver and C ABI bridges
  common/
    c-cpp/
      include/
      src/
      CMakeLists.txt
    fortran/                 shared mathematical helpers and benchmark support
  configs/
    pilot.json
    benchmark.json
    executables.json.example
  tools/
  jobs/
    pegasus/
    sirius/                  build, smoke, measurement and preserved probe history
  docs/
  tests/
```

This tree is a target architecture, not permission to create empty directories.
Implemented source families are NVIDIA C/C++, NVIDIA Fortran with its common
support, all six AMD C/C++ families, and all six AMD Fortran families.
Recorded Pegasus and Sirius builds,
numerical checks, and measurements have distinct source identities; see
`VALIDATION_REPORT.md` and each library README for coverage and limits.
In particular, current rocSPARSE v2/csr_rowsplit evidence is documented in
`amd/c-cpp/rocsparse/README.md`; older adaptive measurements remain historical.
Recorded success does not establish another environment or later source edits.
AMD Fortran has its own [Sirius build, numerical and performance evidence](VALIDATION_REPORT.md#amd-fortran-production-validation),
including compiler/hipFORT compatibility and rocThrust interoperation.
Existing C/C++ measurements are not substituted for Fortran results. Do not
create empty future-area directories, dummy targets, or placeholder implementations.

NVIDIA Fortran uses the same library directories and stems as the table below, with
the `.f90` extension for all 18 examples and 18 benchmark entry points.
`nvidia/fortran/thrust/examples/thrust_wrapper.cu` is shared by both GPU
implementations. The incomplete p.13 stream-sharing fragment is not a target.

## Canonical teaching and benchmark filenames

All paths in this section are relative to the repository root. Benchmark names
are formed from the corresponding teaching-example stem plus `_bench`.

| Library | CPU example | CUDA example | OpenACC example | CPU benchmark | CUDA benchmark | OpenACC benchmark |
| --- | --- | --- | --- | --- | --- | --- |
| cuFFT | `nvidia/c-cpp/cufft/examples/fft_cpu.c` | `nvidia/c-cpp/cufft/examples/fft_gpu.cu` | `nvidia/c-cpp/cufft/examples/openacc_cufft.cpp` | `nvidia/c-cpp/cufft/benchmarks/fft_cpu_bench.c` | `nvidia/c-cpp/cufft/benchmarks/fft_gpu_bench.cu` | `nvidia/c-cpp/cufft/benchmarks/openacc_cufft_bench.cpp` |
| cuBLAS | `nvidia/c-cpp/cublas/examples/blas_cpu.c` | `nvidia/c-cpp/cublas/examples/blas_gpu.cu` | `nvidia/c-cpp/cublas/examples/openacc_cublas.cpp` | `nvidia/c-cpp/cublas/benchmarks/blas_cpu_bench.c` | `nvidia/c-cpp/cublas/benchmarks/blas_gpu_bench.cu` | `nvidia/c-cpp/cublas/benchmarks/openacc_cublas_bench.cpp` |
| cuSPARSE | `nvidia/c-cpp/cusparse/examples/sparse_cpu.c` | `nvidia/c-cpp/cusparse/examples/sparse_gpu.cu` | `nvidia/c-cpp/cusparse/examples/openacc_cusparse.cpp` | `nvidia/c-cpp/cusparse/benchmarks/sparse_cpu_bench.c` | `nvidia/c-cpp/cusparse/benchmarks/sparse_gpu_bench.cu` | `nvidia/c-cpp/cusparse/benchmarks/openacc_cusparse_bench.cpp` |
| cuSOLVER | `nvidia/c-cpp/cusolver/examples/solver_cpu.c` | `nvidia/c-cpp/cusolver/examples/solver_gpu.cu` | `nvidia/c-cpp/cusolver/examples/openacc_cusolver.cpp` | `nvidia/c-cpp/cusolver/benchmarks/solver_cpu_bench.c` | `nvidia/c-cpp/cusolver/benchmarks/solver_gpu_bench.cu` | `nvidia/c-cpp/cusolver/benchmarks/openacc_cusolver_bench.cpp` |
| cuRAND | `nvidia/c-cpp/curand/examples/rand_cpu.cpp` | `nvidia/c-cpp/curand/examples/rand_gpu.cu` | `nvidia/c-cpp/curand/examples/openacc_curand.cpp` | `nvidia/c-cpp/curand/benchmarks/rand_cpu_bench.cpp` | `nvidia/c-cpp/curand/benchmarks/rand_gpu_bench.cu` | `nvidia/c-cpp/curand/benchmarks/openacc_curand_bench.cpp` |
| Thrust | `nvidia/c-cpp/thrust/examples/reduce_cpu.cpp` | `nvidia/c-cpp/thrust/examples/reduce_gpu.cu` | `nvidia/c-cpp/thrust/examples/openacc_thrust.cpp` | `nvidia/c-cpp/thrust/benchmarks/reduce_cpu_bench.cpp` | `nvidia/c-cpp/thrust/benchmarks/reduce_gpu_bench.cu` | `nvidia/c-cpp/thrust/benchmarks/openacc_thrust_bench.cpp` |

Repository-owned canonical filenames must not carry manually maintained date,
revision, or release suffixes. Schema versions belong inside machine-readable
documents. An upstream-mandated spelling such as `cublas_v2.h` is not a
repository-owned filename and must remain unchanged.

### Approved AMD filename and teaching exception

The canonical HIP wrapper filename for rocThrust OpenMP interoperation is
`rocthrust_wrapper.hip`. Teaching material, distributed sources, CMake inputs,
tests, and documentation must use this same name. The inventory below fixes
the names and workloads of the six implemented AMD C/C++ families. The README files describe how to build and use them;
this document remains the authority for teaching filenames and problems.

Only the rocThrust OpenMP C/C++ teaching example may use two translation units:
the OpenMP caller and `rocthrust_wrapper.hip`. The complete caller and wrapper
must both be distributed and printed/documented as explicit compile/link
inputs, including the wrapper object in the final link. There is no hidden
helper source. The OpenMP caller owns its mapped arrays and passes their device
pointers to the HIP wrapper; the wrapper must not allocate/free those mapped
arrays or let a C++ exception escape its C ABI boundary. This exception does
not apply to another C/C++ teaching example, and does not alter the approved
Fortran helper model.

Manual teaching builds may use `hipcc`. CMake native HIP builds use supported
Clang rather than fixing the `hipcc` wrapper as `CMAKE_HIP_COMPILER`. Compiler
paths, ROCm versions, and GPU targets remain explicit installation inputs,
not portable source constants. Both translation units' compile
conditions and final link inputs are recorded by the build integration.

### rocFFT C/C++ inventory and teaching workload

The owner-approved first AMD library lives in `amd/c-cpp/rocfft/`:

| Implementation | Teaching source | Benchmark source |
| --- | --- | --- |
| CPU / FFTW3f | `examples/fft_cpu.c` | `benchmarks/fft_cpu_bench.c` |
| HIP / rocFFT | `examples/fft_gpu.cpp` | `benchmarks/fft_gpu_bench.cpp` |
| OpenMP target / rocFFT | `examples/openmp_rocfft.cpp` | `benchmarks/openmp_rocfft_bench.cpp` |

Every teaching program is one complete source: FP32 interleaved complex,
batched one-dimensional C2C forward, out-of-place, `nfft=1024`, `batch=4096`,
all input elements `1+0i`. CPU directly uses `fftwf_plan_many_dft` with
`FFTW_ESTIMATE`; HIP directly uses HIP Runtime and rocFFT. OpenMP uses
`target data map(to:...) map(from:...)` and `use_device_ptr`, waits for rocFFT
completion before reading output or leaving the data region, and never frees
OpenMP-owned arrays through HIP. Each batch expects DC `nfft+0i` and zero at
all other frequencies. These defaults do not prescribe benchmark thread counts.
CMake target/executable names match source stems; vendor-specific build trees
avoid collisions with the existing NVIDIA names. The additional AMD development inventory follows below; no placeholder targets
or empty examples are used.

### Additional AMD C/C++ development inventory

This expansion carries forward the mathematical teaching problems specified
below, with direct AMD library APIs and OpenMP target interoperability. It does
not change the NVIDIA source inventory. All filenames below are relative to
`amd/c-cpp/<library>/`; executable targets retain the source stem.

| Library | CPU teaching | HIP teaching | OpenMP teaching |
| --- | --- | --- | --- |
| rocblas | `examples/blas_cpu.c` | `examples/blas_gpu.cpp` | `examples/openmp_rocblas.cpp` |
| rocsparse | `examples/sparse_cpu.c` | `examples/sparse_gpu.cpp` | `examples/openmp_rocsparse.cpp` |
| rocsolver | `examples/solver_cpu.c` | `examples/solver_gpu.cpp` | `examples/openmp_rocsolver.cpp` |
| rocrand | `examples/rand_cpu.cpp` | `examples/rand_gpu.cpp` | `examples/openmp_rocrand.cpp` |
| rocthrust | `examples/reduce_cpu.cpp` | `examples/reduce_gpu.cpp` | `examples/openmp_rocthrust.cpp` |

Benchmark filenames use the same stems with `_bench` under `benchmarks/`.
rocThrust OpenMP additionally links `examples/rocthrust_wrapper.hip`; its C ABI
contains exceptions and does not allocate or free caller-managed arrays.

- rocBLAS: FP64 column-major DGEMM, m=n=k=1024, alpha=beta=1, A/B/C all ones;
  CPU CBLAS and HIP `rocblas_dgemm`.
- rocSPARSE: the identical zero-based FP64 1024x1024-grid Poisson CSR problem
  below; generic `rocsparse_spmv`, CSR adaptive algorithm, buffer-size,
  preprocessing and compute stages. CPU oneMKL, explicitly selected AOCL Sparse,
  and explicitly selected reference CSR backends remain distinct; an unavailable
  production provider never falls back to reference code. The AMD CPU teaching
  file remains `sparse_cpu.c`, independently compiled with its selected provider.
  AOCL uses direct descriptor/CSR, hint/optimize, SpMV and destruction calls in
  that file, without hidden helper translation units. NVIDIA's oneMKL teaching
  source is unchanged. Backend/build and thread semantics belong to
  [BENCHMARK_PROTOCOL](BENCHMARK_PROTOCOL.md#cpu-backends-and-parallelism).
- rocSOLVER: the identical FP64 n=1024, nrhs=16 dense system below; CPU teaching
  uses LAPACKE_dgesv, GPU uses rocsolver_dgetrf then rocsolver_dgetrs. Benchmark
  CPU uses staged LAPACKE getrf/getrs. Solver repeat remains one.
- rocRAND: 2^24 uniform doubles, seed 1234, offset zero,
  ROCRAND_RNG_PSEUDO_DEFAULT and default ordering. CPU std::mt19937_64 and GPU
  generators solve the same statistical task, not an identical bit sequence.
- rocThrust: FP64 transform_reduce of squared ones, 2^24 elements, synchronous
  HIP execution policy. CPU STL and explicitly configured OpenMP backends have
  distinct identities.

OpenMP target owns its mapped arrays and passes translated device pointers to
HIP libraries; actual offload and device agreement must be verified. Separate
library workspaces are explicit allocations and are never freed through two
owners. Only rocThrust uses the two-translation-unit teaching exception.

### AMD Fortran inventory and explicit helpers

All following files live under `amd/fortran/<library>/examples/`. The three
benchmark entry points use the same stems plus `_bench.f90` under
`benchmarks/`; each links its library's `<stem>_workload.F90`.

| Library | CPU example | HIP example | OpenMP target example | Workload stem |
| --- | --- | --- | --- | --- |
| rocfft | `fft_cpu.f90` | `fft_gpu.f90` | `openmp_rocfft.f90` | `fft` |
| rocblas | `blas_cpu.f90` | `blas_gpu.f90` | `openmp_rocblas.f90` | `blas` |
| rocsparse | `sparse_cpu.f90` | `sparse_gpu.f90` | `openmp_rocsparse.f90` | `sparse` |
| rocsolver | `solver_cpu.f90` | `solver_gpu.f90` | `openmp_rocsolver.f90` | `solver` |
| rocrand | `rand_cpu.f90` | `rand_gpu.f90` | `openmp_rocrand.f90` | `rand` |
| rocthrust | `reduce_cpu.f90` | `reduce_gpu.f90` | `openmp_rocthrust.f90` | `reduce` |

These 18 teaching programs retain the AMD C/C++ teaching sizes, precision and
mathematics. CPU FFT uses FFTW's Fortran interface; BLAS and Solver call native
LP64 Fortran BLAS/LAPACK. CPU RAND uses `random_seed`/`random_number`, and CPU
Thrust uses `sum(values*values)`, with distinct intrinsic backend identities.
GPU programs call hipFORT interfaces, except explicit ISO_C_BINDING declarations
for rocSPARSE v2, the rocThrust wrapper, rocSOLVER `getrf` and rocRAND ordering.
The latter two are declared in their callers against the public C ABI where
hipFORT declarations are incompatible or absent; no algorithm is substituted.
OpenMP uses separate mapped arrays,
`use_device_addr` and `c_loc`; complete HIP work before OpenMP retrieves results.
Physical shared HBM does not imply unified shared memory.

The following are explicit compile/link inputs, not hidden implementations:

- `amd/fortran/rocsparse/make_poisson2d_csr.f90` generates zero-based CSR offsets
  and column values for both AOCL Sparse and rocSPARSE. Fortran array subscripts
  remain one-based. NVIDIA Fortran's separate one-based CSR helper is unchanged.
- `amd/fortran/rocsparse/aocl_sparse_c.f90` declares the AOCL Sparse CPU C ABI;
  `amd/fortran/rocsparse/rocsparse_v2_c.f90` declares the GPU v2/rowsplit ABI.
  The respective binding module and the CSR generator precede their caller.
- `common/fortran/make_dense_system.f90` is linked by Solver examples and
  benchmarks; CPU teaching uses `dgesv`, benchmarks staged `dgetrf`/`dgetrs`.
- `amd/fortran/rocthrust/examples/rocthrust_wrapper.hip` is linked by both GPU
  implementations. Its three-argument C ABI returns integer status and writes
  a double result; exceptions do not cross the C boundary.
- `amd/fortran/common/benchmark_support.f90` and `benchmark_bridge.cpp` supply
  benchmark lifecycle, CLI, timer and serialization. The computational workload
  remains in Fortran; `hip_device_bridge.cpp` only checks mapped pointer/device
  agreement. These are benchmark inputs, not teaching-program dependencies.

hipFORT module files must match the selected Fortran compiler. Configure vendor,
language and execution modes in separate build trees; compiler/module ABI and
HIP-wrapper link compatibility are prerequisites, not inferred from C/C++ smoke.
See [AMD Fortran usage](../amd/fortran/README.md) for build inputs and current
limitations, and [Fortran timing](BENCHMARK_PROTOCOL.md#amd-fortran-binding) for
measurement behavior.

## Teaching-example policy

`examples/` is the source of truth for code printed in the teaching material.
Each example must:

- match its canonical teaching filename exactly;
- be a complete program (C/C++ examples are independently compilable single
  sources, with only the explicit rocThrust OpenMP two-translation-unit exception
  above);
- contain no ellipses, pseudocode, hidden helper source, benchmark CLI, repeated
  trials, CSV/JSON output, or environment metadata;
- preserve a direct, readable library-call sequence and avoid excessive
  abstraction;
- use concise CUDA Runtime and CUDA library error checks without obscuring that
  sequence; and
- contain no architecture-specific compiler option.

For C/C++, `make_poisson2d_csr` and `make_dense_system` must be defined in each
example that uses them. NVIDIA Fortran instead links the explicitly documented
`common/fortran/make_poisson2d_csr.f90` and `make_dense_system.f90` helpers,
as approved to fill the slide omissions without duplicating their bodies. AMD
Fortran uses the explicit helper/binding inventory above.

## Canonical teaching problems

### cuFFT

- Batched one-dimensional complex-to-complex forward FFT.
- FP32, `nfft = 1024`, and `batch = 4096`.
- Every input element is `1.0 + 0.0i`.
- The CPU plan uses `FFTW_ESTIMATE`.

### cuBLAS

- FP64 DGEMM: `C = alpha A B + beta C`.
- Column-major storage and no-transpose by no-transpose.
- `m = n = k = 1024`, `alpha = 1.0`, and `beta = 1.0`.
- Every element of A, B, and initial C is `1.0`.

### cuSPARSE

- FP64 CSR SpMV for a two-dimensional five-point Poisson operator:
  `y = alpha A x + beta y`.
- `nx = ny = 1024`, `alpha = 1.0`, and `beta = 1.0`.
- Every element of x and initial y is `1.0`.
- C/C++ CSR uses zero-based indexing; Fortran uses one-based row offsets and
  column indices for the identical matrix. Both use non-transpose operation and
  `CUSPARSE_SPMV_ALG_DEFAULT`.

For grid coordinates `(ix, iy)`, the row ordering is
`p = iy * nx + ix`. The matrix has `nrow = ncol = nx * ny`. Each row
contains diagonal value `4.0` and value `-1.0` for each valid left, right, up,
and down neighbor. Out-of-domain neighbors are not stored. Column indices in
each row are strictly ascending. The exact nonzero count is:

```text
nnz = 5 * nx * ny - 2 * nx - 2 * ny
```

### cuSOLVER

- FP64 dense solution of `A X = B` in column-major storage.
- `n = 1024` and `nrhs = 16`.
- The CPU teaching example calls `LAPACKE_dgesv` (C/C++) or `dgesv` (Fortran).
- The CUDA and OpenACC teaching examples call `getrf` followed by `getrs`.
- CPU benchmarks deliberately differ from the CPU teaching example; the staged
  benchmark API is specified in `BENCHMARK_PROTOCOL.md`.

For zero-based `(i, j)`, system generation is:

```text
A(i,j) = n + 1  when i == j
A(i,j) = 1      otherwise
X_true(i,rhs) = 1
B = A * X_true
```

Every element of B is therefore `2 * n`. Verification can use both known-solution
error and residual.

### cuRAND

- Uniform double-precision pseudorandom generation.
- `num_rand = 1 << 24`.
- `CURAND_RNG_PSEUDO_DEFAULT`, seed `1234`, offset `0`, and default order.
- The Fortran CPU example uses `random_seed` (all seed-vector elements set to
  1234) and `random_number`; its compiler-dependent sequence need not equal CUDA.

### Thrust

- FP64 `transform_reduce` computing `sum(values[i]^2)`.
- `num_elem = 1 << 24` and every `values[i] = 1.0`.
- Fortran CPU uses the intrinsic `sum(values*values)`, not the C++ STL backend.
- CUDA Fortran uses typed `device` allocatables and element-count `cudaMemcpy`;
  OpenACC Fortran retains separate memory, `data` and `host_data use_device`.
  Fortran library workspace uses typed device `allocate`/`deallocate` and is
  never simultaneously managed by an OpenACC data clause.

## OpenACC data management

Entering an `acc data` region or executing an `enter data` directive causes the
OpenACC runtime to allocate device storage for managed arrays and perform the
specified copyin/create operation. These are not three independent manual steps.
The same managed array must never also be allocated with `cudaMalloc`.

`host_data use_device` performs address translation: it exposes the device
address of data that is already present on the device so that a CUDA library can
consume it. It does not allocate storage or move data.

Explicit `cudaMalloc`/`cudaFree` is permitted only for memory not managed by an
OpenACC data clause, notably cuSPARSE or cuSOLVER library workspace. Such
workspace must not be named in `host_data use_device`.

For OpenACC cuSPARSE examples:

- enter the data region before obtaining any managed-array device address;
- restrict `host_data use_device` to obtaining the raw addresses passed to
  `cusparseCreateCsr` and `cusparseCreateDnVec`;
- create pointer-dependent descriptors only after data entry;
- query the workspace size after descriptor creation;
- allocate/free library workspace explicitly outside `host_data use_device` but
  inside the lifetime of the data region; and
- keep buffer-size query, `cusparseSpMV`, synchronization, workspace management,
  and descriptor destruction outside the address-translation region once the
  descriptors retain the needed addresses.

For OpenACC cuSOLVER examples, enter the data region first. Use a minimal
`host_data use_device(mat_A)` region for the buffer-size query and a separate
minimal region for `getrf`/`getrs` addresses (`mat_A`, `rhs_B`, `ipiv`, and
`info`). Workspace allocation, synchronization, and workspace release remain
outside those address-translation regions while the data region is active.

For OpenACC Thrust examples, translate the managed-array address only after data
entry and convert the known device address with `thrust::device_pointer_cast`.
A teaching-oriented `cudaDeviceSynchronize` may remain, but a comment and the
library README must explain that synchronous Thrust algorithms normally need no
additional synchronization.

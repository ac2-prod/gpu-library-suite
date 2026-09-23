# GPU Library Suite Project Specification

## Authority and scope

This document owns the repository hierarchy, repository-owned canonical
filenames, teaching examples, mathematical problems, default teaching-example
parameters, and example-level data-management rules. Benchmark CLI behavior,
timing, verification thresholds, and statistics are owned by
[`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md). Machine-readable formats are
owned by [`RESULT_SCHEMA.md`](RESULT_SCHEMA.md). Pegasus operations are owned by
[`PEGASUS_EXECUTION.md`](PEGASUS_EXECUTION.md).

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
    c-cpp/                   future; do not create as an empty placeholder
    fortran/                 future; do not create as an empty placeholder
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
  docs/
  tests/
```

This tree is a target architecture, not permission to create empty directories.
The initial implementation was limited to `nvidia/c-cpp`; the approved addition
is `nvidia/fortran` and its common support. AMD C/C++ and Fortran remain future work.
Do not create empty future-area directories, dummy targets, or placeholder
implementations.

Fortran uses the same library directories and stems as the table below, with
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

## Teaching-example policy

`examples/` is the source of truth for code printed in the teaching material.
Each example must:

- match its canonical teaching filename exactly;
- be a complete program (C/C++ examples are independently compilable single sources);
- contain no ellipses, pseudocode, hidden helper source, benchmark CLI, repeated
  trials, CSV/JSON output, or environment metadata;
- preserve a direct, readable library-call sequence and avoid excessive
  abstraction;
- use concise CUDA Runtime and CUDA library error checks without obscuring that
  sequence; and
- contain no architecture-specific compiler option.

For C/C++, `make_poisson2d_csr` and `make_dense_system` must be defined in each
example that uses them. Fortran instead links the explicitly documented
`common/fortran/make_poisson2d_csr.f90` and `make_dense_system.f90` helpers,
as approved to fill the slide omissions without duplicating their bodies.

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

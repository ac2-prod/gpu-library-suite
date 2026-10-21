# AMD GPU libraries from Fortran

[English](README.md) | [日本語](README.ja.md)

Compare native Fortran CPU code, direct HIP/roc-library calls through hipFORT,
and OpenMP target data management. Each library has three teaching programs in
`examples/` and three separate benchmark programs in `benchmarks/`.
Start with a CPU example, then compare the HIP and OpenMP versions of the same
operation. CMake supplies the required helper sources and mixed-language links.
The [validation report](../../docs/VALIDATION_REPORT.md#amd-fortran-production-validation)
records successful Sirius builds, numerical checks including rocThrust, and
6,480 performance records from six physical nodes over two waves.

## Source files

| Library | CPU example | HIP example | OpenMP example |
| --- | --- | --- | --- |
| [rocFFT](rocfft/examples) | `fft_cpu.f90` | `fft_gpu.f90` | `openmp_rocfft.f90` |
| [rocBLAS](rocblas/examples) | `blas_cpu.f90` | `blas_gpu.f90` | `openmp_rocblas.f90` |
| [rocSPARSE](rocsparse/examples) | `sparse_cpu.f90` | `sparse_gpu.f90` | `openmp_rocsparse.f90` |
| [rocSOLVER](rocsolver/examples) | `solver_cpu.f90` | `solver_gpu.f90` | `openmp_rocsolver.f90` |
| [rocRAND](rocrand/examples) | `rand_cpu.f90` | `rand_gpu.f90` | `openmp_rocrand.f90` |
| [rocThrust](rocthrust/examples) | `reduce_cpu.f90` | `reduce_gpu.f90` | `openmp_rocthrust.f90` |

Benchmark entry names add `_bench` before `.f90`. Numerical workloads are in
the corresponding `*_workload.F90`. The shared Fortran driver owns repetitions,
and its C++ bridge reuses the existing CLI, monotonic timer and result writer;
it does not replace the Fortran numerical workload.

The complete programs add initialization, error checks and verification to the
teaching flow. Sparse uses zero-based, sorted LP64 CSR, `rocsparse_spmv` v2 and
`csr_rowsplit`; the old SpMV API is not a fallback. Solver checks factorization
status before solving. RAND uses different CPU/GPU generators for the same
distribution. OpenMP uses explicit mappings, not implicit unified shared memory.

## Start with a CPU example

From the repository root, with a Fortran compiler:

```bash
mkdir -p build/amd-fortran-example
gfortran -O2 amd/fortran/rocthrust/examples/reduce_cpu.f90 \
  -o build/amd-fortran-example/reduce_cpu
build/amd-fortran-example/reduce_cpu
```

The expected square sum is `16777216`, with exit status zero. On a cluster,
compile and run within a compute allocation according to its operating rules.

For a CPU-only CMake tree:

```bash
cmake -S . -B build/amd-fortran-cpu \
  -DGPU_SUITE_VENDOR=amd -DGPU_SUITE_SOURCE_LANGUAGE=fortran \
  -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_BUILD_TYPE=Release \
  '-DGPU_SUITE_AMD_LIBRARIES=rocrand;rocthrust' \
  -DGPU_SUITE_BUILD_BENCHMARKS=OFF -DGPU_SUITE_BUILD_TESTS=OFF \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_HIP=OFF \
  -DGPU_SUITE_BUILD_OPENMP=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=OFF
cmake --build build/amd-fortran-cpu --target reduce_cpu rand_cpu
```

RAND and reduction need no numerical library. FFT needs single-precision FFTW
and `fftw3.f03`; threaded FFT is enabled only after the three thread APIs link.
BLAS and Solver use native LP64 `dgemm`, `dgesv`, `dgetrf` and `dgetrs` symbols.
Select the same explicit provider options as [AMD C/C++](../c-cpp/rocblas/README.md):
`GPU_SUITE_CPU_BLAS_BACKEND` and `GPU_SUITE_CPU_LAPACK_BACKEND`. For AOCL,
use `GENERIC_CBLAS` / `GENERIC_LAPACKE`, explicit header/library paths and all
ordered extra libraries; libFLAME must resolve BLAS from the intended BLIS.
For a multithreaded baseline, select the parallel BLIS provider (the Sirius
measurement used `libblis-mt`), not a sequential library with a similar name.
The `GENERIC_CBLAS` / `GENERIC_LAPACKE` option names are shared build identifiers;
these Fortran programs call Fortran BLAS/LAPACK, not CBLAS or LAPACKE entry points.
Sparse requires explicit `GPU_SUITE_CPU_SPARSE_BACKEND=AOCLSPARSE`.
CMake checks provider headers and complete Fortran example links. Read the
enabled/disabled target summary: configure success alone is not build success.
CPU-only configuration does not require ROCm, hipFORT or an AMD compiler.

## HIP and OpenMP builds

Use a hipFORT installation compiled for the selected Fortran compiler: `.mod`
files from another compiler are not interchangeable. Set
`GPU_SUITE_HIPFORT_MODULE_DIR`, `GPU_SUITE_HIPFORT_LIBRARY` and, if needed,
`GPU_SUITE_ROCM_ROOT` to the actual installation. No compiler/site paths are
hard-coded. The HIP tree uses `GPU_SUITE_BUILD_HIP=ON`; the OpenMP tree uses
`GPU_SUITE_BUILD_OPENMP=ON` with an AMD LLVM Flang compiler and an explicit
`GPU_SUITE_AMD_GPU_TARGET`. Keep them in separate build trees.

The first five HIP examples call C libraries through hipFORT and do not need
HIP compilation of the Fortran sources. rocThrust additionally compiles
`rocthrust/examples/rocthrust_wrapper.hip` with native CMake HIP: use a compatible
Clang as `CMAKE_HIP_COMPILER`, explicit `CMAKE_HIP_ARCHITECTURES`, and CMake 3.29+.
Do not set `CMAKE_HIP_COMPILER=hipcc`. Manual wrapper compilation may use hipcc.
The wrapper returns an integer status and writes the double result through its
third argument. CMake includes its object and the C++ runtime in the final link.
Both the mixed-language configure check and final target link with the Fortran
driver, including the selected OpenMP link options.

`rocsolver_dgetrf` and `rocrand_set_ordering` have explicit `bind(C)` declarations
in their callers for hipFORT versions whose interfaces are incompatible or
missing. They call the same public C symbols; RAND retains explicit DEFAULT
ordering, seed/offset restoration and generator initialization.

The following builds and runs only the HIP DGEMM teaching example. Set `FC` to
your compatible Fortran compiler, `HIPFORT_MOD_DIR` / `HIPFORT_LIBRARY` to its
hipFORT installation, and `ROCM_PATH` to the ROCm prefix. `HIP_CLANG` and
`GPU_ARCH` select the native HIP compiler and GPU architecture when building
rocThrust. Use values from your installation, not paths copied from another site.

```bash
cmake -S . -B build/amd-fortran-hip \
  -DGPU_SUITE_VENDOR=amd -DGPU_SUITE_SOURCE_LANGUAGE=fortran \
  -DCMAKE_Fortran_COMPILER="$FC" -DCMAKE_BUILD_TYPE=Release \
  -DGPU_SUITE_AMD_LIBRARIES=rocblas \
  -DGPU_SUITE_BUILD_BENCHMARKS=OFF -DGPU_SUITE_BUILD_TESTS=OFF \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_HIP=ON \
  -DGPU_SUITE_BUILD_OPENMP=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=OFF \
  -DGPU_SUITE_HIPFORT_MODULE_DIR="$HIPFORT_MOD_DIR" \
  -DGPU_SUITE_HIPFORT_LIBRARY="$HIPFORT_LIBRARY" \
  -DGPU_SUITE_ROCM_ROOT="$ROCM_PATH" \
  -DCMAKE_HIP_COMPILER="$HIP_CLANG" -DCMAKE_HIP_ARCHITECTURES="$GPU_ARCH" &&
cmake --build build/amd-fortran-hip --target blas_gpu &&
build/amd-fortran-hip/amd/fortran/rocblas/blas_gpu
```

For OpenMP, use a new tree, disable HIP targets, enable OpenMP targets and add
`-DGPU_SUITE_AMD_GPU_TARGET="$GPU_ARCH"`. Build target `openmp_rocblas` and run
`OMP_TARGET_OFFLOAD=MANDATORY build/amd-fortran-openmp/amd/fortran/rocblas/openmp_rocblas`.
For another library, change the library selection and target using the source
table above. Native HIP settings are still needed for the rocThrust wrapper.
The compile/link probes do not execute GPU code.
`--device` selects the HIP device; `OMP_DEFAULT_DEVICE` independently selects
the OpenMP device. Benchmarks reject host fallback and mapped allocations that
do not belong to the requested HIP device. Neither ordinal is silently remapped.

## Explicit additional link inputs

| Programs | Additional sources |
| --- | --- |
| Sparse, all implementations | `rocsparse/make_poisson2d_csr.f90` |
| Sparse CPU | `rocsparse/aocl_sparse_c.f90` |
| Sparse HIP / OpenMP | `rocsparse/rocsparse_v2_c.f90` |
| Solver, all implementations | `common/fortran/make_dense_system.f90` at repository root |
| Thrust HIP / OpenMP | `rocthrust/examples/rocthrust_wrapper.hip` |
| All benchmarks | matching workload, `common/benchmark_support.f90`, `common/benchmark_bridge.cpp`, common C support |
| GPU benchmarks | `common/hip_device_bridge.cpp` |

Compile Fortran modules before their users; CMake handles this ordering.
Teaching examples never link the benchmark driver.

## Benchmark inputs and interpretation

[Smoke config](../../configs/amd/fortran/smoke.json) requests 72 records:
6 libraries × 3 implementations × 2 scopes × 2 trials. It chooses FFTW serial,
generic BLAS/LAPACK providers, AOCL Sparse and intrinsic RNG/sum. Match these
backend identities to your built manifests; an unavailable backend is an error,
not permission to substitute another one. Use this small numerical check before
performance measurement; it is not a production config or a job submission script.

Use `--scope compute` for the displayed **library** scope and
`--scope end-to-end` for **end-to-end**. The
[AMD Fortran timing specification](../../docs/BENCHMARK_PROTOCOL.md#amd-fortran-binding)
defines the exact boundaries. They follow the corresponding AMD C/C++ timer,
synchronization and repetition sequence. This alignment does not make compiler,
language binding or RNG-algorithm differences disappear.
All AMD FFT implementations restore canonical input outside each timed execute.
Solver uses `repeat=1`; other mutable state is restored after warm-up and before
each trial. CPU thread settings are requests: external-library effective worker
counts remain unknown; intrinsic RAND and reduction remain serial.

For measurement, configure new trees with `GPU_SUITE_BUILD_BENCHMARKS=ON`.
Generate the partial manifest with `gpu_suite_partial_manifest` in each tree,
then use the [shared measurement workflow](../../docs/PORTABILITY.md) with the
Fortran config and only Fortran manifests. `source_language=fortran` separates
these records from C/C++; do not merge existing C/C++ measurements into them.

## Tested configuration and interpretation limits

The recorded configuration used GNU Fortran 14.2.1 for CPU code, AMD LLVM Flang
22.0.0 and matching hipFORT from ROCm 7.2.0 for GPU code, and AOCL 5.1.0.
These are tested versions, not universal installation requirements. The
[Sirius procedure](../../jobs/sirius/README.md#amd-fortran-current-workflow)
describes site-specific modules, runtime inputs and allocations.

| Operation | CPU implementation used in the recorded comparison |
| --- | --- |
| FFT | AOCL FFTW, single precision, OpenMP thread library |
| BLAS | AOCL BLIS, native Fortran `dgemm`, linked to multithreaded BLIS |
| Sparse | AOCL Sparse LP64, called through its C binding |
| Solver | AOCL libFLAME `dgetrf` / `dgetrs`, linked with the same multithreaded BLIS |
| RAND | Serial Fortran `random_number` |
| Reduction | Serial Fortran `sum(values*values)` |

The first four series request 24 CPU threads; their effective worker counts
were not measured. A label such as “24 C” does not establish 24 active workers.
RAND and reduction are serial baselines. Recorded link inputs and resolved
library files do not establish every library actually loaded during execution.

Performance figures use five trials per condition, node medians within each
wave, then equally weighted wave medians. Both waves used the same six hosts;
they are repeated observations, not twelve different machines. The latest
6,480-row campaign replaces the earlier 5,400-row provisional comparison for
this source version; old data and failure records remain separate. Software or
hardware changes require a new comparison rather than relabelling saved results.

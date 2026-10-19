# cuSOLVER — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

Solve a dense FP64 system `A*X=B` with LU factorization and multiple right-hand sides. The example shows pivot storage and solver status checks as well as matrix storage.

<a id="problem"></a>

## Problem and expected result

`N=1024`, `nrhs=16`, `A=N*I+ones`, and every B element `2*N`. The solution is all ones. Require successful API returns and `info=0`; numerical checks must also pass.

Fortran examples must exit 0 and print `verification PASS`. Teaching sizes are source constants.

<a id="dependencies"></a>

## Dependencies

CPU uses the oneMKL LP64 Fortran LAPACK interface. Set `MKLROOT` and supply `mkl_rt` and `mkl_service.h` from that installation. The C/C++ LAPACKE interface is not required.

CPU supports GNU Fortran / NVHPC; GPU uses NVHPC `nvfortran` and CUDA Toolkit. Do not share `.mod` files between compilers or reuse a C/C++ tree. The [Fortran guide](../README.md) covers ABI and numerical requirements.

<a id="select-a-minimal-build"></a>

## Select a minimal build

Set these variables in Bash, then follow only the CPU, direct GPU, or directive section you need in the [shared build guide](../../../docs/BUILDING.md#select). It builds the selected example; benchmarks and the full test suite are not needed.
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=fortran
LIBRARY=cusolver
CPU_TARGET=solver_cpu
GPU_TARGET=solver_gpu
DIRECTIVE_TARGET=openacc_cusolver
CPU_OPTIONS=()
```

<a id="sources"></a>

## Sources and explicit link inputs

| Variant | Example | Benchmark |
| --- | --- | --- |
| CPU | [solver_cpu.f90](examples/solver_cpu.f90) / `solver_cpu` | [solver_cpu_bench.f90](benchmarks/solver_cpu_bench.f90) / `solver_cpu_bench` |
| CUDA | [solver_gpu.f90](examples/solver_gpu.f90) / `solver_gpu` | [solver_gpu_bench.f90](benchmarks/solver_gpu_bench.f90) / `solver_gpu_bench` |
| OpenACC | [openacc_cusolver.f90](examples/openacc_cusolver.f90) / `openacc_cusolver` | [openacc_cusolver_bench.f90](benchmarks/openacc_cusolver_bench.f90) / `openacc_cusolver_bench` |

All three examples explicitly link the matrix helper [make_dense_system.f90](../../../common/fortran/make_dense_system.f90); CMake includes it automatically.

<a id="adapting-the-example"></a>

## Adapt this example

Factorization overwrites A and solving overwrites B. Preserve originals for a residual check or another trial. Distinguish an API failure from singularity reported through info. If A is unchanged and only B changes, factor once and reuse the LU factors/pivots with getrs. This has a different cost from refactorizing every solve.

Distinguish Fortran column-major arrays from C ABI types. CUDA Fortran uses device arrays; OpenACC manages a data region while calling the same GPU library. Include required helpers/wrappers in the application link.

<a id="direct-compilation"></a>

## Direct compilation (NVHPC alternative)

Prefer the common CMake route. These optional commands run in Bash from the repository root with dependency search paths already configured. CPU and GPU blocks are separate; use only the variants you need.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" -module "$DIRECT_BUILD" \
  -I"${MKLROOT:?Set the installed oneMKL root}/include" \
  nvidia/fortran/cusolver/examples/solver_cpu.f90 common/fortran/make_dense_system.f90 \
  -L"${MKL_LIBDIR:?Set the installed oneMKL library directory}" -lmkl_rt -lpthread -lm -ldl \
  -o "$DIRECT_BUILD/solver_cpu" &&
"$DIRECT_BUILD/solver_cpu"
```

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-gpu.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
export NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set your NVHPC GPU target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=cusolver \
  nvidia/fortran/cusolver/examples/solver_gpu.f90 common/fortran/make_dense_system.f90 -o "$DIRECT_BUILD/solver_gpu" &&
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=cusolver \
  nvidia/fortran/cusolver/examples/openacc_cusolver.f90 common/fortran/make_dense_system.f90 -o "$DIRECT_BUILD/openacc_cusolver" &&
"$DIRECT_BUILD/solver_gpu" &&
"$DIRECT_BUILD/openacc_cusolver"
```

<a id="examples-and-a-small-benchmark"></a>

## After the example: optional measurement

The minimal build excludes benchmarks. To measure, enable them using the [Fortran measurement workflow](../README.md#first-checks-and-own-measurements), with the language-specific config and manifest. Library uses internal CLI scope `compute`; the other scope is `end-to-end`. These intervals differ from timing the entire teaching program.

CPU teaching uses dgesv; benchmarks call getrf / getrs separately. repeat is 1 in both scopes.

See the [problem specification](../../../docs/PROJECT_SPECIFICATION.md) and [timing protocol](../../../docs/BENCHMARK_PROTOCOL.md). Do not mix Fortran and C/C++ results; intrinsic CPU RNG / sum are serial series.

<a id="validation"></a>

## Recorded validation

Saved Pegasus evidence covers NVHPC builds, numerical smoke, and production measurements; see the [versions and limits](../../../docs/VALIDATION_REPORT.md#fortran-production-validation). This does not certify another compiler / GPU. Distinguish requested/effective CPU threads and linked-library evidence from complete actual load provenance.

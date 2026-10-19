# cuBLAS — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

Multiply dense matrices with FP64 DGEMM: `C = alpha*A*B + beta*C`. Compare the CPU BLAS call with direct GPU use and directive-managed arrays.

<a id="problem"></a>

## Problem and expected result

Column-major square matrices, `m=n=k=1024`, `alpha=beta=1`, and all-one A/B/C. Every output element must be `1025`.

Fortran examples must exit 0 and print `verification PASS`. Teaching sizes are source constants.

<a id="dependencies"></a>

## Dependencies

CPU uses oneMKL LP64. Set `MKLROOT` and supply `mkl_rt` and `mkl_service.h` from that installation.

CPU supports GNU Fortran / NVHPC; GPU uses NVHPC `nvfortran` and CUDA Toolkit. Do not share `.mod` files between compilers or reuse a C/C++ tree. The [Fortran guide](../README.md) covers ABI and numerical requirements.

<a id="select-a-minimal-build"></a>

## Select a minimal build

Set these variables in Bash, then follow only the CPU, direct GPU, or directive section you need in the [shared build guide](../../../docs/BUILDING.md#select). It builds the selected example; benchmarks and the full test suite are not needed.
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=fortran
LIBRARY=cublas
CPU_TARGET=blas_cpu
GPU_TARGET=blas_gpu
DIRECTIVE_TARGET=openacc_cublas
CPU_OPTIONS=()
```

<a id="sources"></a>

## Sources and explicit link inputs

| Variant | Example | Benchmark |
| --- | --- | --- |
| CPU | [blas_cpu.f90](examples/blas_cpu.f90) / `blas_cpu` | [blas_cpu_bench.f90](benchmarks/blas_cpu_bench.f90) / `blas_cpu_bench` |
| CUDA | [blas_gpu.f90](examples/blas_gpu.f90) / `blas_gpu` | [blas_gpu_bench.f90](benchmarks/blas_gpu_bench.f90) / `blas_gpu_bench` |
| OpenACC | [openacc_cublas.f90](examples/openacc_cublas.f90) / `openacc_cublas` | [openacc_cublas_bench.f90](benchmarks/openacc_cublas_bench.f90) / `openacc_cublas_bench` |

<a id="adapting-the-example"></a>

## Adapt this example

Match matrix layout and leading dimensions (`lda`, `ldb`, `ldc`) to your storage. With nonzero beta, C is an input as well as an output: copying only A/B is insufficient. Keep arrays and a handle across repeated multiplies when their lifetime permits it. CPU parallelism belongs to the selected BLAS provider; a requested thread count is not an observed worker count.

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
  nvidia/fortran/cublas/examples/blas_cpu.f90 \
  -L"${MKL_LIBDIR:?Set the installed oneMKL library directory}" -lmkl_rt -lpthread -lm -ldl \
  -o "$DIRECT_BUILD/blas_cpu" &&
"$DIRECT_BUILD/blas_cpu"
```

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-gpu.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
export NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set your NVHPC GPU target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=cublas \
  nvidia/fortran/cublas/examples/blas_gpu.f90 -o "$DIRECT_BUILD/blas_gpu" &&
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=cublas \
  nvidia/fortran/cublas/examples/openacc_cublas.f90 -o "$DIRECT_BUILD/openacc_cublas" &&
"$DIRECT_BUILD/blas_gpu" &&
"$DIRECT_BUILD/openacc_cublas"
```

<a id="examples-and-a-small-benchmark"></a>

## After the example: optional measurement

The minimal build excludes benchmarks. To measure, enable them using the [Fortran measurement workflow](../README.md#first-checks-and-own-measurements), with the language-specific config and manifest. Library uses internal CLI scope `compute`; the other scope is `end-to-end`. These intervals differ from timing the entire teaching program.

See the [problem specification](../../../docs/PROJECT_SPECIFICATION.md) and [timing protocol](../../../docs/BENCHMARK_PROTOCOL.md). Do not mix Fortran and C/C++ results; intrinsic CPU RNG / sum are serial series.

<a id="validation"></a>

## Recorded validation

Saved Pegasus evidence covers NVHPC builds, numerical smoke, and production measurements; see the [versions and limits](../../../docs/VALIDATION_REPORT.md#fortran-production-validation). This does not certify another compiler / GPU. Distinguish requested/effective CPU threads and linked-library evidence from complete actual load provenance.

# cuSPARSE — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

Compute FP64 sparse matrix-vector multiplication in CSR form: `y = alpha*A*x + beta*y`. The sample matrix is a regular 2D five-point Poisson stencil.

<a id="problem"></a>

## Problem and expected result

A 1024×1024 grid gives 1,048,576 rows. Diagonal entries are 4, neighbors are −1, and `alpha=beta=1`, `x=y=1`. Expected y is 1 in the interior, 2 on an edge, and 3 at a corner. CSR indices are one-based.

Fortran examples must exit 0 and print `verification PASS`. Teaching sizes are source constants.

<a id="dependencies"></a>

## Dependencies

CPU requires oneMKL Sparse LP64. Set `MKLROOT` and supply `mkl_rt`, `mkl_service.h`, and `mkl_spblas.f90` from that installation; do not reuse another compiler’s `.mod` files.

CPU supports GNU Fortran / NVHPC; GPU uses NVHPC `nvfortran` and CUDA Toolkit. Do not share `.mod` files between compilers or reuse a C/C++ tree. The [Fortran guide](../README.md) covers ABI and numerical requirements.

<a id="select-a-minimal-build"></a>

## Select a minimal build

Set these variables in Bash, then follow only the CPU, direct GPU, or directive section you need in the [shared build guide](../../../docs/BUILDING.md#select). It builds the selected example; benchmarks and the full test suite are not needed.
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=fortran
LIBRARY=cusparse
CPU_TARGET=sparse_cpu
GPU_TARGET=sparse_gpu
DIRECTIVE_TARGET=openacc_cusparse
CPU_OPTIONS=()
```

<a id="sources"></a>

## Sources and explicit link inputs

| Variant | Example | Benchmark |
| --- | --- | --- |
| CPU | [sparse_cpu.f90](examples/sparse_cpu.f90) / `sparse_cpu` | [sparse_cpu_bench.f90](benchmarks/sparse_cpu_bench.f90) / `sparse_cpu_bench` |
| CUDA | [sparse_gpu.f90](examples/sparse_gpu.f90) / `sparse_gpu` | [sparse_gpu_bench.f90](benchmarks/sparse_gpu_bench.f90) / `sparse_gpu_bench` |
| OpenACC | [openacc_cusparse.f90](examples/openacc_cusparse.f90) / `openacc_cusparse` | [openacc_cusparse_bench.f90](benchmarks/openacc_cusparse_bench.f90) / `openacc_cusparse_bench` |

All three examples explicitly link the matrix helper [make_poisson2d_csr.f90](../../../common/fortran/make_poisson2d_csr.f90); CMake includes it automatically.

<a id="adapting-the-example"></a>

## Adapt this example

Replace CSR row pointers, column indices, and values together. Preserve index width/base, dimensions, and nonzero ordering. Initial y matters when beta is nonzero. Reuse descriptors and analysis only when the API permits it for the matrix/pointers you retain. Regular five-point rows do not predict performance for highly irregular application matrices.

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
  nvidia/fortran/cusparse/examples/sparse_cpu.f90 common/fortran/make_poisson2d_csr.f90 \
  -L"${MKL_LIBDIR:?Set the installed oneMKL library directory}" -lmkl_rt -lpthread -lm -ldl \
  -o "$DIRECT_BUILD/sparse_cpu" &&
"$DIRECT_BUILD/sparse_cpu"
```

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-gpu.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
export NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set your NVHPC GPU target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=cusparse \
  nvidia/fortran/cusparse/examples/sparse_gpu.f90 common/fortran/make_poisson2d_csr.f90 -o "$DIRECT_BUILD/sparse_gpu" &&
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=cusparse \
  nvidia/fortran/cusparse/examples/openacc_cusparse.f90 common/fortran/make_poisson2d_csr.f90 -o "$DIRECT_BUILD/openacc_cusparse" &&
"$DIRECT_BUILD/sparse_gpu" &&
"$DIRECT_BUILD/openacc_cusparse"
```

<a id="examples-and-a-small-benchmark"></a>

## After the example: optional measurement

The minimal build excludes benchmarks. To measure, enable them using the [Fortran measurement workflow](../README.md#first-checks-and-own-measurements), with the language-specific config and manifest. Library uses internal CLI scope `compute`; the other scope is `end-to-end`. These intervals differ from timing the entire teaching program.

See the [problem specification](../../../docs/PROJECT_SPECIFICATION.md) and [timing protocol](../../../docs/BENCHMARK_PROTOCOL.md). Do not mix Fortran and C/C++ results; intrinsic CPU RNG / sum are serial series.

<a id="validation"></a>

## Recorded validation

Saved Pegasus evidence covers NVHPC builds, numerical smoke, and production measurements; see the [versions and limits](../../../docs/VALIDATION_REPORT.md#fortran-production-validation). This does not certify another compiler / GPU. Distinguish requested/effective CPU threads and linked-library evidence from complete actual load provenance.

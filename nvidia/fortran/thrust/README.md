# Thrust — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

Compute `sum(values[i]^2)` with transform-reduce. Combining the transform with reduction avoids storing an intermediate array of squared values.

<a id="problem"></a>

## Problem and expected result

Input is `2^24` FP64 ones; the expected square sum is `16777216`. CPU uses serial intrinsic `sum(values*values)`. The result returned to the host is one double.

Fortran examples must exit 0 and print `verification PASS`. Teaching sizes are source constants.

<a id="dependencies"></a>

## Dependencies

CPU needs only its Fortran compiler and intrinsic sum. GPU needs Thrust/CCCL and CUDA. The CMake profiles compile C++ interoperation with NVCC for CUDA and nvc++ for OpenACC; see the [profile requirements](../README.md#numerical-and-cpu-runtime-conditions).

CPU supports GNU Fortran / NVHPC; GPU uses NVHPC `nvfortran` and CUDA Toolkit. Do not share `.mod` files between compilers or reuse a C/C++ tree. The [Fortran guide](../README.md) covers ABI and numerical requirements.

<a id="select-a-minimal-build"></a>

## Select a minimal build

Set these variables in Bash, then follow only the CPU, direct GPU, or directive section you need in the [shared build guide](../../../docs/BUILDING.md#select). It builds the selected example; benchmarks and the full test suite are not needed.
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=fortran
LIBRARY=thrust
CPU_TARGET=reduce_cpu
GPU_TARGET=reduce_gpu
DIRECTIVE_TARGET=openacc_thrust
CPU_OPTIONS=()
```

<a id="sources"></a>

## Sources and explicit link inputs

| Variant | Example | Benchmark |
| --- | --- | --- |
| CPU | [reduce_cpu.f90](examples/reduce_cpu.f90) / `reduce_cpu` | [reduce_cpu_bench.f90](benchmarks/reduce_cpu_bench.f90) / `reduce_cpu_bench` |
| CUDA | [reduce_gpu.f90](examples/reduce_gpu.f90) / `reduce_gpu` | [reduce_gpu_bench.f90](benchmarks/reduce_gpu_bench.f90) / `reduce_gpu_bench` |
| OpenACC | [openacc_thrust.f90](examples/openacc_thrust.f90) / `openacc_thrust` | [openacc_thrust_bench.f90](benchmarks/openacc_thrust_bench.f90) / `openacc_thrust_bench` |

Both GPU teaching callers and the OpenACC benchmark use [thrust_wrapper.cu](examples/thrust_wrapper.cu). Its `bind(C)` function is `double thrust_square_sum(const double*, int)`; the element count uses Fortran `integer(c_int)`. C++ exceptions become NaN, rejected by the Fortran caller. This is a different ABI from the AMD rocThrust wrapper (three arguments, int return).

The CUDA benchmark instead links [reduce_context.cu](benchmarks/reduce_context.cu).
It owns a `thrust::device_vector` across library-scope repeats, so allocation and
input copying stay outside that interval. End-to-end includes creating/copying
that vector and returning the scalar result; destruction follows timing.
The teaching interface is unchanged. See the [timing contract](../../../docs/BENCHMARK_PROTOCOL.md#fortran-binding-of-this-protocol).

<a id="adapting-the-example"></a>

## Adapt this example

Replace the transform, associative reduction, and identity value together. Floating-point reductions can differ with order, so use a suitable tolerance for general inputs. Keep input resident when applying repeated GPU operations. A directive-managed device pointer must not be freed by a wrapper; the caller owns its allocation.

Distinguish Fortran column-major arrays from C ABI types. CUDA Fortran uses device arrays; OpenACC manages a data region while calling the same GPU library. Include required helpers/wrappers in the application link.

<a id="direct-compilation"></a>

## Direct compilation (NVHPC alternative)

Prefer the common CMake route. These optional commands run in Bash from the repository root with dependency search paths already configured. CPU and GPU blocks are separate; use only the variants you need. These teaching-only alternatives do not reproduce the measured OpenACC build profile; use the CMake workflow above for benchmark comparisons.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" nvidia/fortran/thrust/examples/reduce_cpu.f90 \
  -o "$DIRECT_BUILD/reduce_cpu" &&
"$DIRECT_BUILD/reduce_cpu"
```

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-gpu.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
export NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set your NVHPC GPU target},mem:separate")
unset NVCOMPILER_FPU_STATE
"$CUDA_TOOLKIT_ROOT/bin/nvcc" -std=c++17 -O3 -arch="${NVCC_ARCH:?Set the nvcc architecture for your GPU}" \
  -I"${CUDA_CCCL_INCLUDE_DIR:?Set the external Toolkit directory containing thrust headers}" \
  -c nvidia/fortran/thrust/examples/thrust_wrapper.cu -o "$DIRECT_BUILD/thrust_wrapper.o" &&
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -c++libs \
  nvidia/fortran/thrust/examples/reduce_gpu.f90 "$DIRECT_BUILD/thrust_wrapper.o" -o "$DIRECT_BUILD/reduce_gpu" &&
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -c++libs \
  nvidia/fortran/thrust/examples/openacc_thrust.f90 "$DIRECT_BUILD/thrust_wrapper.o" -o "$DIRECT_BUILD/openacc_thrust" &&
"$DIRECT_BUILD/reduce_gpu" &&
"$DIRECT_BUILD/openacc_thrust"
```

<a id="examples-and-a-small-benchmark"></a>

## After the example: optional measurement

The minimal build excludes benchmarks. To measure, enable them using the [Fortran measurement workflow](../README.md#first-checks-and-own-measurements), with the language-specific config and manifest. Library uses internal CLI scope `compute`; the other scope is `end-to-end`. These intervals differ from timing the entire teaching program.

See the [problem specification](../../../docs/PROJECT_SPECIFICATION.md) and [timing protocol](../../../docs/BENCHMARK_PROTOCOL.md). Do not mix Fortran and C/C++ results; intrinsic CPU RNG / sum are serial series.

<a id="validation"></a>

## Recorded validation

Saved Pegasus evidence covers NVHPC builds, numerical smoke, and production measurements; see the [versions and limits](../../../docs/VALIDATION_REPORT.md#fortran-production-validation). This does not certify another compiler / GPU. Distinguish requested/effective CPU threads and linked-library evidence from complete actual load provenance.

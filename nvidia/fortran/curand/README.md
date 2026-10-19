# cuRAND — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

Generate uniform FP64 random values. This demonstrates a persistent generator and output buffer, useful for simulation inputs and stochastic algorithms.

<a id="problem"></a>

## Problem and expected result

Generate `2^24` values with seed 1234. CPU uses the Fortran intrinsic RNG. GPU uses the library’s default pseudorandom generator. CPU uniform values are in [0,1), GPU values in (0,1]. These are comparable distributions, not identical streams.

Fortran examples must exit 0 and print `verification PASS`. Teaching sizes are source constants.

<a id="dependencies"></a>

## Dependencies

CPU uses its Fortran compiler and intrinsic RNG only. GPU requires cuRAND.

CPU supports GNU Fortran / NVHPC; GPU uses NVHPC `nvfortran` and CUDA Toolkit. Do not share `.mod` files between compilers or reuse a C/C++ tree. The [Fortran guide](../README.md) covers ABI and numerical requirements.

<a id="select-a-minimal-build"></a>

## Select a minimal build

Set these variables in Bash, then follow only the CPU, direct GPU, or directive section you need in the [shared build guide](../../../docs/BUILDING.md#select). It builds the selected example; benchmarks and the full test suite are not needed.
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=fortran
LIBRARY=curand
CPU_TARGET=rand_cpu
GPU_TARGET=rand_gpu
DIRECTIVE_TARGET=openacc_curand
CPU_OPTIONS=()
```

<a id="sources"></a>

## Sources and explicit link inputs

| Variant | Example | Benchmark |
| --- | --- | --- |
| CPU | [rand_cpu.f90](examples/rand_cpu.f90) / `rand_cpu` | [rand_cpu_bench.f90](benchmarks/rand_cpu_bench.f90) / `rand_cpu_bench` |
| CUDA | [rand_gpu.f90](examples/rand_gpu.f90) / `rand_gpu` | [rand_gpu_bench.f90](benchmarks/rand_gpu_bench.f90) / `rand_gpu_bench` |
| OpenACC | [openacc_curand.f90](examples/openacc_curand.f90) / `openacc_curand` | [openacc_curand_bench.f90](benchmarks/openacc_curand_bench.f90) / `openacc_curand_bench` |

<a id="adapting-the-example"></a>

## Adapt this example

Choose a generator and reproducible seed/stream policy for the application. Do not reset the seed on every call unless restarting the same stream is intended. Reuse the generator and output buffer for a continuing stream. Compare range and distribution statistics, not CPU/GPU element identity; CPU and GPU algorithms differ. The CPU examples are serial.

Distinguish Fortran column-major arrays from C ABI types. CUDA Fortran uses device arrays; OpenACC manages a data region while calling the same GPU library. Include required helpers/wrappers in the application link.

<a id="direct-compilation"></a>

## Direct compilation (NVHPC alternative)

Prefer the common CMake route. These optional commands run in Bash from the repository root with dependency search paths already configured. CPU and GPU blocks are separate; use only the variants you need.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" nvidia/fortran/curand/examples/rand_cpu.f90 \
  -o "$DIRECT_BUILD/rand_cpu" &&
"$DIRECT_BUILD/rand_cpu"
```

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-gpu.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
export NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set your NVHPC GPU target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=curand \
  nvidia/fortran/curand/examples/rand_gpu.f90 -o "$DIRECT_BUILD/rand_gpu" &&
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=curand \
  nvidia/fortran/curand/examples/openacc_curand.f90 -o "$DIRECT_BUILD/openacc_curand" &&
"$DIRECT_BUILD/rand_gpu" &&
"$DIRECT_BUILD/openacc_curand"
```

<a id="examples-and-a-small-benchmark"></a>

## After the example: optional measurement

The minimal build excludes benchmarks. To measure, enable them using the [Fortran measurement workflow](../README.md#first-checks-and-own-measurements), with the language-specific config and manifest. Library uses internal CLI scope `compute`; the other scope is `end-to-end`. These intervals differ from timing the entire teaching program.

See the [problem specification](../../../docs/PROJECT_SPECIFICATION.md) and [timing protocol](../../../docs/BENCHMARK_PROTOCOL.md). Do not mix Fortran and C/C++ results; intrinsic CPU RNG / sum are serial series.

<a id="validation"></a>

## Recorded validation

Saved Pegasus evidence covers NVHPC builds, numerical smoke, and production measurements; see the [versions and limits](../../../docs/VALIDATION_REPORT.md#fortran-production-validation). This does not certify another compiler / GPU. Distinguish requested/effective CPU threads and linked-library evidence from complete actual load provenance.

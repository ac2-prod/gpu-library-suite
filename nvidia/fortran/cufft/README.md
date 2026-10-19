# cuFFT — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

Compute 4,096 independent length-1,024 FP32 complex forward FFTs. This is useful when an application already has many signals or transforms to process as a batch.

<a id="problem"></a>

## Problem and expected result

Out-of-place 1D C2C, contiguous interleaved complex input, all values `1+0i`. Each transform must produce DC `1024+0i` and zero elsewhere, within the source checks.

Fortran examples must exit 0 and print `verification PASS`. Teaching sizes are source constants.

<a id="dependencies"></a>

## Dependencies

CPU needs FP32 FFTW: `fftw3.h`, `libfftw3f`, and `fftw3.f03`. Set `FFTW_ROOT` for a prefix outside the normal search path. Threaded FFTW is an optional benchmark dependency, not needed by the serial teaching example.

CPU supports GNU Fortran / NVHPC; GPU uses NVHPC `nvfortran` and CUDA Toolkit. Do not share `.mod` files between compilers or reuse a C/C++ tree. The [Fortran guide](../README.md) covers ABI and numerical requirements.

<a id="select-a-minimal-build"></a>

## Select a minimal build

Set these variables in Bash, then follow only the CPU, direct GPU, or directive section you need in the [shared build guide](../../../docs/BUILDING.md#select). It builds the selected example; benchmarks and the full test suite are not needed.
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=fortran
LIBRARY=cufft
CPU_TARGET=fft_cpu
GPU_TARGET=fft_gpu
DIRECTIVE_TARGET=openacc_cufft
CPU_OPTIONS=()
```

<a id="sources"></a>

## Sources and explicit link inputs

| Variant | Example | Benchmark |
| --- | --- | --- |
| CPU | [fft_cpu.f90](examples/fft_cpu.f90) / `fft_cpu` | [fft_cpu_bench.f90](benchmarks/fft_cpu_bench.f90) / `fft_cpu_bench` |
| CUDA | [fft_gpu.f90](examples/fft_gpu.f90) / `fft_gpu` | [fft_gpu_bench.f90](benchmarks/fft_gpu_bench.f90) / `fft_gpu_bench` |
| OpenACC | [openacc_cufft.f90](examples/openacc_cufft.f90) / `openacc_cufft` | [openacc_cufft_bench.f90](benchmarks/openacc_cufft_bench.f90) / `openacc_cufft_bench` |

<a id="adapting-the-example"></a>

## Adapt this example

Keep precision, transform direction, stride, distance, and batch consistent with your buffers. Reuse a plan only while its layout and sizes remain valid. FFTW teaching uses a serial `FFTW_ESTIMATE` plan; threaded benchmarks are a separate choice.

Distinguish Fortran column-major arrays from C ABI types. CUDA Fortran uses device arrays; OpenACC manages a data region while calling the same GPU library. Include required helpers/wrappers in the application link.

<a id="direct-compilation"></a>

## Direct compilation (NVHPC alternative)

Prefer the common CMake route. These optional commands run in Bash from the repository root with dependency search paths already configured. CPU and GPU blocks are separate; use only the variants you need.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" -module "$DIRECT_BUILD" \
  -I"${FFTW_INCLUDE_DIR:?Set the directory containing fftw3.f03}" \
  nvidia/fortran/cufft/examples/fft_cpu.f90 \
  -L"${FFTW_LIBDIR:?Set the directory containing libfftw3f}" -lfftw3f \
  -o "$DIRECT_BUILD/fft_cpu" &&
"$DIRECT_BUILD/fft_cpu"
```

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-gpu.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
export NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set your NVHPC GPU target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=cufft \
  nvidia/fortran/cufft/examples/fft_gpu.f90 -o "$DIRECT_BUILD/fft_gpu" &&
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=cufft \
  nvidia/fortran/cufft/examples/openacc_cufft.f90 -o "$DIRECT_BUILD/openacc_cufft" &&
"$DIRECT_BUILD/fft_gpu" &&
"$DIRECT_BUILD/openacc_cufft"
```

<a id="examples-and-a-small-benchmark"></a>

## After the example: optional measurement

The minimal build excludes benchmarks. To measure, enable them using the [Fortran measurement workflow](../README.md#first-checks-and-own-measurements), with the language-specific config and manifest. Library uses internal CLI scope `compute`; the other scope is `end-to-end`. These intervals differ from timing the entire teaching program.

See the [problem specification](../../../docs/PROJECT_SPECIFICATION.md) and [timing protocol](../../../docs/BENCHMARK_PROTOCOL.md). Do not mix Fortran and C/C++ results; intrinsic CPU RNG / sum are serial series.

<a id="validation"></a>

## Recorded validation

Saved Pegasus evidence covers NVHPC builds, numerical smoke, and production measurements; see the [versions and limits](../../../docs/VALIDATION_REPORT.md#fortran-production-validation). This does not certify another compiler / GPU. Distinguish requested/effective CPU threads and linked-library evidence from complete actual load provenance.

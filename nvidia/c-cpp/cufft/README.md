# cuFFT C/C++: build, run, and adapt

[English](README.md) | [日本語](README.ja.md)

Compute 4,096 independent length-1,024 FP32 complex forward FFTs. This is useful when an application already has many signals or transforms to process as a batch.

<a id="teaching-default"></a>

## Teaching default

Out-of-place 1D C2C, contiguous interleaved complex input, all values `1+0i`. Each transform must produce DC `1024+0i` and zero elsewhere, within the source checks.

<a id="dependencies"></a>

## Dependencies

CPU needs FP32 FFTW: `fftw3.h`, `libfftw3f`. Set `FFTW_ROOT` for a prefix outside the normal search path. Threaded FFTW is an optional benchmark dependency, not needed by the serial teaching example.

GPU variants need a compatible NVIDIA GPU/driver and CUDA Toolkit; OpenACC additionally needs NVHPC nvc++.

<a id="select-a-minimal-build"></a>

## Select a minimal build

Set these variables in Bash, then follow only the CPU, direct GPU, or directive section you need in the [shared build guide](../../../docs/BUILDING.md#select). It builds the selected example; benchmarks and the full test suite are not needed.
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=c-cpp
LIBRARY=cufft
CPU_TARGET=fft_cpu
GPU_TARGET=fft_gpu
DIRECTIVE_TARGET=openacc_cufft
CPU_OPTIONS=()
```

<a id="canonical-sources-and-targets"></a>

## Canonical sources and targets

| Role | Source | CMake target/executable |
| --- | --- | --- |
| CPU teaching | `examples/fft_cpu.c` | `fft_cpu` |
| CUDA teaching | `examples/fft_gpu.cu` | `fft_gpu` |
| OpenACC teaching | `examples/openacc_cufft.cpp` | `openacc_cufft` |
| CPU benchmark | `benchmarks/fft_cpu_bench.c` | `fft_cpu_bench` |
| CUDA benchmark | `benchmarks/fft_gpu_bench.cu` | `fft_gpu_bench` |
| OpenACC benchmark | `benchmarks/openacc_cufft_bench.cpp` | `openacc_cufft_bench` |

<a id="adapting-the-example"></a>

## Adapt this example

Keep precision, transform direction, stride, distance, and batch consistent with your buffers. Reuse a plan only while its layout and sizes remain valid. FFTW teaching uses a serial `FFTW_ESTIMATE` plan; threaded benchmarks are a separate choice.

Direct CUDA manages device arrays; OpenACC uses a data region and `host_data use_device` to call the same GPU library. Keep allocation ownership clear and ensure completion before reading results.

<a id="direct-compile"></a>

## Direct compile

An alternative to CMake, from the repository root in Bash with provider header/library search paths already configured. Choose the variant you need; run it only if compilation succeeds.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
cc -std=c17 -O2 nvidia/c-cpp/cufft/examples/fft_cpu.c -lfftw3f -lm -o "$DIRECT_BUILD/fft_cpu" &&
"$DIRECT_BUILD/fft_cpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVCC_ARCH:?Set the nvcc architecture supported by your GPU}"
"$CUDA_TOOLKIT_ROOT/bin/nvcc" -std=c++17 -O2 -arch="$NVCC_ARCH" \
  nvidia/c-cpp/cufft/examples/fft_gpu.cu -lcufft \
  -o "$DIRECT_BUILD/fft_gpu" &&
"$DIRECT_BUILD/fft_gpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVHPC_GPU_TARGET:?Set your NVHPC GPU target}"
NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT" \
nvc++ -std=c++17 -O2 -acc -cuda -gpu="$NVHPC_GPU_TARGET" \
  nvidia/c-cpp/cufft/examples/openacc_cufft.cpp -cudalib=cufft \
  -o "$DIRECT_BUILD/openacc_cufft" &&
"$DIRECT_BUILD/openacc_cufft"
```

<a id="cmake-configure-and-build"></a>

## CMake configure and build

Use the selection above with the [shared guide](../../../docs/BUILDING.md). It creates separate CPU, CUDA, and OpenACC trees and builds only the chosen target. Use the GPU sections alone if you do not need CPU numerical dependencies.

<a id="first-build-and-run"></a>

## First build and run

Follow selection → build → run and require exit status 0 plus the expected result under Teaching default. A missing target or nonzero exit is not success. Use a compute allocation on shared clusters.

<a id="run-the-examples"></a>
For Fortran, use the [separate guide](../../fortran/cufft/README.md).

<a id="from-example-to-benchmark"></a>

## From example to benchmark

Teaching examples have no timing CLI. For performance use the matching `_bench`, separating library (internal scope `compute`) from end-to-end. The sections below are benchmark reference material. Enable benchmarks using the [measurement workflow](../../../docs/PORTABILITY.md#measuring-on-your-own-system) and use its build paths; the minimal example build does not include benchmarks.

<a id="benchmark-cli"></a>

## Benchmark CLI

This small check requests one CPU thread and uses the build from the linked measurement workflow. BLAS / Sparse / Solver use its oneMKL configuration. If you built another provider, match `--cpu-backend` to that build.

A standalone threaded CPU smoke run is:

```bash
"$CPU_CUDA_BUILD/nvidia/c-cpp/cufft/fft_cpu_bench" \
  --size 256 --batch 8 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --abs-tolerance 1e-4 \
  --rel-tolerance 1e-5 --output - --format jsonl \
  --cpu-backend cpu-fftw-threaded --cpu-threads 1 \
  --cpu-threads-effective 1 --cpu-backend-role production \
  --series-role primary --cpu-parallelism threaded
```

Production suite execution obtains all counts, roles, and verification
thresholds from the effective canonical configuration and uses
`tools/run_suite.py` as the only node raw-file writer.

<a id="timing-scopes"></a>

## Timing scopes

The display term library corresponds to internal scope `compute`.

`compute` times only repeated transforms after plan creation, allocation,
transfer/restoration, and pre-timing synchronization. `end-to-end` times each
repeat's allocation, plan creation, input transfer, transform, completion, and
output transfer; cleanup follows the end timestamp. Canonical state is restored
before every repeat. Warm-up and verification remain outside timed intervals.

<a id="verification"></a>

## Verification

The output is compared with the canonical analytical transform using the
effective absolute and relative tolerances. A verification failure records that
trial as failed but does not prevent a restored later trial from running.

<a id="openacc-notes"></a>

## OpenACC notes

OpenACC owns input and output array storage; `host_data use_device` exposes the
already-present device addresses to cuFFT. The target links
`OpenACC::OpenACC_CXX`, `CUDA::cudart`, and `CUDA::cufft` in its own build tree.

<a id="cpu-backend-and-role"></a>

## CPU backend and role

Select the production CPU backend explicitly for measurements; do not replace it because a dependency is missing. Distinguish requested threads from effective workers. FFTW teaching is serial; threaded FFTW is a separate benchmark choice.

<a id="known-limitations-and-local-validation"></a>

## Known limitations and local validation

Saved Pegasus evidence includes NVIDIA C/C++ builds, numerical smoke, and performance measurements; see the [versions and coverage](../../../docs/VALIDATION_REPORT.md#current-coexistence-status). GPU/toolchain behavior is locally unverified by host-only fixtures; saved real-machine runs are distinct evidence and do not certify another environment or later edits. See the [protocol](../../../docs/BENCHMARK_PROTOCOL.md) for numerical/statistical verification and CPU parallelism limits.

# rocFFT C/C++: build, run, and adapt

[English](README.md) | [日本語](README.ja.md)

Compute 4,096 independent length-1,024 FP32 complex forward FFTs. This is useful when an application already has many signals or transforms to process as a batch.

<a id="teaching-default"></a>

## Problem and expected result

Out-of-place 1D C2C, contiguous interleaved complex input, all values `1+0i`. Each transform must produce DC `1024+0i` and zero elsewhere, within the source checks.

Require exit status 0 and the result checks above. Teaching sizes are source constants; do not pass benchmark CLI options to these examples.

<a id="dependencies"></a>

## Dependencies

CPU needs FP32 FFTW: `fftw3.h`, `libfftw3f`. Set `FFTW_ROOT` for a prefix outside the normal search path. Threaded FFTW is an optional benchmark dependency, not needed by the serial teaching example.

HIP needs the corresponding ROCm libraries; OpenMP target also needs compatible AMD Clang. Native CMake HIP uses HIP-capable Clang, not `hipcc`. CPU-only usage does not require ROCm.

<a id="select-a-minimal-build"></a>

## Select a minimal build

Set these variables in Bash, then follow only the CPU, direct GPU, or directive section you need in the [shared build guide](../../../docs/BUILDING.md#select). It builds the selected example; benchmarks and the full test suite are not needed.
```bash
VENDOR=amd
SOURCE_LANGUAGE=c-cpp
LIBRARY=rocfft
CPU_TARGET=fft_cpu
GPU_TARGET=fft_gpu
DIRECTIVE_TARGET=openmp_rocfft
CPU_OPTIONS=()
```

<a id="canonical-sources-and-targets"></a>

## Sources and targets

| Variant | Teaching source / target | Benchmark source / target |
| --- | --- | --- |
| CPU | [`fft_cpu.c`](examples/fft_cpu.c) / `fft_cpu` | [`fft_cpu_bench.c`](benchmarks/fft_cpu_bench.c) / `fft_cpu_bench` |
| HIP | [`fft_gpu.cpp`](examples/fft_gpu.cpp) / `fft_gpu` | [`fft_gpu_bench.cpp`](benchmarks/fft_gpu_bench.cpp) / `fft_gpu_bench` |
| OpenMP target | [`openmp_rocfft.cpp`](examples/openmp_rocfft.cpp) / `openmp_rocfft` | [`openmp_rocfft_bench.cpp`](benchmarks/openmp_rocfft_bench.cpp) / `openmp_rocfft_bench` |

<a id="adapting-the-example"></a>

## Adapt this example

Keep precision, transform direction, stride, distance, and batch consistent with your buffers. Reuse a plan only while its layout and sizes remain valid. FFTW teaching uses a serial `FFTW_ESTIMATE` plan; threaded benchmarks are a separate choice. For rocFFT, preserve original input if it is needed again: do not assume the input survives execution unchanged.

The HIP variant manages arrays with `hipMalloc` / `hipMemcpy`. In OpenMP target, `map` owns storage and `use_device_ptr` exposes it to the GPU library. Ensure completion before result retrieval/map exit; do not free mapped storage with HIP. Shared physical memory does not imply USM or omitted copies.

<a id="direct-compile"></a>

## Direct compile (optional)

Run from the repository root in Bash with the provider header/library search paths available. Prefer CMake for complex CPU dependencies or static linking. Run only the CPU, HIP, or OpenMP blocks you need.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
cc -std=c17 -O2 amd/c-cpp/rocfft/examples/fft_cpu.c -lfftw3f -lm -o "$DIRECT_BUILD/fft_cpu" &&
"$DIRECT_BUILD/fft_cpu"
```
Set the GPU target for your installation and use `hipcc` from the selected ROCm stack.

```bash
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
hipcc -x hip -std=c++17 -O2 --offload-arch="$AMD_GPU_TARGET" \
  amd/c-cpp/rocfft/examples/fft_gpu.cpp -lrocfft \
  -o "$DIRECT_BUILD/fft_gpu" &&
"$DIRECT_BUILD/fft_gpu"
```

```bash
: "${AMD_CLANGXX:?Set the AMD Clang OpenMP C++ compiler}"
: "${ROCM_ROOT:?Set the ROCm prefix containing include}"
: "${ROCM_LIBDIR:?Set the ROCm shared-library directory}"
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
"$AMD_CLANGXX" -std=c++17 -O2 -fopenmp --offload-arch="$AMD_GPU_TARGET" \
  -I"$ROCM_ROOT/include" amd/c-cpp/rocfft/examples/openmp_rocfft.cpp \
  -L"$ROCM_LIBDIR" -Wl,-rpath,"$ROCM_LIBDIR" -lrocfft -lamdhip64 \
  -o "$DIRECT_BUILD/openmp_rocfft" &&
OMP_TARGET_OFFLOAD=MANDATORY "$DIRECT_BUILD/openmp_rocfft"
```

<a id="timing-scopes"></a>

## Optional performance measurement

Figures use library / end-to-end; internal CLI scopes are `compute` / `end-to-end`. These are controlled intervals, not simply the whole teaching program.

library reuses the plan, execution info, workspace, and mapped/device arrays. Required input restoration is outside each individually timed FFT plus completion; elapsed intervals are summed and divided by repeat. NVIDIA FFT times its repeat sequence differently. End-to-end includes plan/workspace setup, device allocation or mapping, input movement, FFT/completion, and result retrieval. HIP also clears output inside this scope. OpenMP map exit is timed; explicit plan/workspace and HIP array cleanup are outside. The accepted method is retained for the AMD CPU/GPU comparison, not an identical NVIDIA/AMD timing claim.

Host input generation and numerical verification are outside timing. See the [benchmark protocol](../../../docs/BENCHMARK_PROTOCOL.md) for details and [Sirius procedures](../../../jobs/sirius/README.md) for suite execution/aggregation. Distributed configs do not automatically adapt to another installation.

<a id="validation-and-limits"></a>

## Recorded validation and limits

Saved Sirius evidence includes CPU / HIP / OpenMP builds, numerical checks, and performance measurements. Source versions and coverage are recorded in the [validation report](../../../docs/VALIDATION_REPORT.md#amd-production-comparison).

Requested CPU threads are not observed worker counts. Saved RAND / Thrust CPU measurements use serial backends. Complete actually loaded-library identity remains unverified. Recorded success is not a guarantee for another environment or later edits; local fixtures do not prove GPU execution.

[Problem/source specification](../../../docs/PROJECT_SPECIFICATION.md) · [Result format](../../../docs/RESULT_SCHEMA.md)

# rocThrust C/C++: build, run, and adapt

[English](README.md) | [日本語](README.ja.md)

Compute `sum(values[i]^2)` with transform-reduce. Combining the transform with reduction avoids storing an intermediate array of squared values.

<a id="teaching-default"></a>

## Problem and expected result

Input is `2^24` FP64 ones; the expected square sum is `16777216`. CPU uses serial STL `std::transform_reduce`. The result returned to the host is one double.

Require exit status 0 and the result checks above. Teaching sizes are source constants; do not pass benchmark CLI options to these examples.

<a id="dependencies"></a>

## Dependencies

Serial CPU needs only C++17 and its standard library. GPU needs rocThrust/rocPRIM and HIP.

HIP needs the corresponding ROCm libraries; OpenMP target also needs compatible AMD Clang. Native CMake HIP uses HIP-capable Clang, not `hipcc`. CPU-only usage does not require ROCm.

<a id="select-a-minimal-build"></a>

## Select a minimal build

Set these variables in Bash, then follow only the CPU, direct GPU, or directive section you need in the [shared build guide](../../../docs/BUILDING.md#select). It builds the selected example; benchmarks and the full test suite are not needed.
```bash
VENDOR=amd
SOURCE_LANGUAGE=c-cpp
LIBRARY=rocthrust
CPU_TARGET=reduce_cpu
GPU_TARGET=reduce_gpu
DIRECTIVE_TARGET=openmp_rocthrust
CPU_OPTIONS=(-DGPU_SUITE_CPU_THRUST_BACKEND=STL)
```

<a id="canonical-sources-and-targets"></a>

## Sources and targets

| Variant | Teaching source / target | Benchmark source / target |
| --- | --- | --- |
| CPU | [`reduce_cpu.cpp`](examples/reduce_cpu.cpp) / `reduce_cpu` | [`reduce_cpu_bench.cpp`](benchmarks/reduce_cpu_bench.cpp) / `reduce_cpu_bench` |
| HIP | [`reduce_gpu.cpp`](examples/reduce_gpu.cpp) / `reduce_gpu` | [`reduce_gpu_bench.cpp`](benchmarks/reduce_gpu_bench.cpp) / `reduce_gpu_bench` |
| OpenMP target | [`openmp_rocthrust.cpp`](examples/openmp_rocthrust.cpp) / `openmp_rocthrust` | [`openmp_rocthrust_bench.cpp`](benchmarks/openmp_rocthrust_bench.cpp) / `openmp_rocthrust_bench` |

OpenMP teaching and benchmarks explicitly compile/link [rocthrust_wrapper.hip](examples/rocthrust_wrapper.hip). Its C ABI is `int rocthrust_square_sum(const double*, size_t, double*) noexcept`: status is returned and the scalar is written through argument 3. This differs from the NVIDIA Fortran wrapper ABI. It neither retains/frees the caller’s mapping nor lets exceptions escape the C ABI.

<a id="adapting-the-example"></a>

## Adapt this example

Replace the transform, associative reduction, and identity value together. Floating-point reductions can differ with order, so use a suitable tolerance for general inputs. Keep input resident when applying repeated GPU operations. A directive-managed device pointer must not be freed by a wrapper; the caller owns its allocation.

The direct HIP example allocates and initializes `thrust::device_vector<double>` on the device; it does not explicitly copy an input host array with `hipMemcpy`. In OpenMP target, `map` owns storage and `use_device_ptr` exposes it to the GPU library. Ensure completion before result retrieval/map exit; do not free mapped storage with HIP. Shared physical memory does not imply USM or omitted copies.

<a id="direct-compile"></a>

## Direct compile (optional)

Run from the repository root in Bash with the provider header/library search paths available. Prefer CMake for complex CPU dependencies or static linking. Run only the CPU, HIP, or OpenMP blocks you need.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
c++ -std=c++17 -O2 amd/c-cpp/rocthrust/examples/reduce_cpu.cpp -o "$DIRECT_BUILD/reduce_cpu" &&
"$DIRECT_BUILD/reduce_cpu"
```
Set the GPU target for your installation and use `hipcc` from the selected ROCm stack.

```bash
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
hipcc -x hip -std=c++17 -O2 --offload-arch="$AMD_GPU_TARGET" \
  amd/c-cpp/rocthrust/examples/reduce_gpu.cpp \
  -o "$DIRECT_BUILD/reduce_gpu" &&
"$DIRECT_BUILD/reduce_gpu"
```

```bash
: "${AMD_CLANGXX:?Set the AMD Clang OpenMP C++ compiler}"
: "${ROCM_ROOT:?Set the ROCm prefix containing include}"
: "${ROCM_LIBDIR:?Set the ROCm shared-library directory}"
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
hipcc -x hip -std=c++17 -O2 --offload-arch="$AMD_GPU_TARGET" \
  -c amd/c-cpp/rocthrust/examples/rocthrust_wrapper.hip \
  -o "$DIRECT_BUILD/rocthrust_wrapper.o" &&
"$AMD_CLANGXX" -std=c++17 -O2 -fopenmp --offload-arch="$AMD_GPU_TARGET" \
  -I"$ROCM_ROOT/include" -c amd/c-cpp/rocthrust/examples/openmp_rocthrust.cpp \
  -o "$DIRECT_BUILD/openmp_rocthrust.o" &&
"$AMD_CLANGXX" -fopenmp --offload-arch="$AMD_GPU_TARGET" \
  "$DIRECT_BUILD/openmp_rocthrust.o" "$DIRECT_BUILD/rocthrust_wrapper.o" \
  -L"$ROCM_LIBDIR" -Wl,-rpath,"$ROCM_LIBDIR" -lamdhip64 \
  -o "$DIRECT_BUILD/openmp_rocthrust" &&
OMP_TARGET_OFFLOAD=MANDATORY "$DIRECT_BUILD/openmp_rocthrust"
```

<a id="timing-scopes"></a>

## Optional performance measurement

Figures use library / end-to-end; internal CLI scopes are `compute` / `end-to-end`. These are controlled intervals, not simply the whole teaching program.

library keeps input on the GPU and times repeated synchronous transform_reduce calls. End-to-end adds allocation/map and input copy, then reduction and the host scalar result. Explicit frees/data deletion are outside timing. The unchanged input needs no per-repeat restoration; each reduction starts at identity 0.0. The recorded CPU production series is serial STL.

Host input generation and numerical verification are outside timing. See the [benchmark protocol](../../../docs/BENCHMARK_PROTOCOL.md) for details and [Sirius procedures](../../../jobs/sirius/README.md) for suite execution/aggregation. Distributed configs do not automatically adapt to another installation.

<a id="validation-and-limits"></a>

## Recorded validation and limits

Saved Sirius evidence includes CPU / HIP / OpenMP builds, numerical checks, and performance measurements. Source versions and coverage are recorded in the [validation report](../../../docs/VALIDATION_REPORT.md#amd-production-comparison).

Requested CPU threads are not observed worker counts. Saved RAND / Thrust CPU measurements use serial backends. Complete actually loaded-library identity remains unverified. Recorded success is not a guarantee for another environment or later edits; local fixtures do not prove GPU execution.

[Problem/source specification](../../../docs/PROJECT_SPECIFICATION.md) · [Result format](../../../docs/RESULT_SCHEMA.md)

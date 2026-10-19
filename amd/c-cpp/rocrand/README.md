# rocRAND C/C++: build, run, and adapt

[English](README.md) | [日本語](README.ja.md)

Generate uniform FP64 random values. This demonstrates a persistent generator and output buffer, useful for simulation inputs and stochastic algorithms.

<a id="teaching-default"></a>

## Problem and expected result

Generate `2^24` values with seed 1234. CPU uses serial `std::mt19937_64`. GPU uses the library’s default pseudorandom generator. CPU uniform values are in [0,1), GPU values in (0,1]. These are comparable distributions, not identical streams.

Require exit status 0 and the result checks above. Teaching sizes are source constants; do not pass benchmark CLI options to these examples.

<a id="dependencies"></a>

## Dependencies

CPU needs only C++17 and its standard library. GPU requires rocRAND.

HIP needs the corresponding ROCm libraries; OpenMP target also needs compatible AMD Clang. Native CMake HIP uses HIP-capable Clang, not `hipcc`. CPU-only usage does not require ROCm.

<a id="select-a-minimal-build"></a>

## Select a minimal build

Set these variables in Bash, then follow only the CPU, direct GPU, or directive section you need in the [shared build guide](../../../docs/BUILDING.md#select). It builds the selected example; benchmarks and the full test suite are not needed.
```bash
VENDOR=amd
SOURCE_LANGUAGE=c-cpp
LIBRARY=rocrand
CPU_TARGET=rand_cpu
GPU_TARGET=rand_gpu
DIRECTIVE_TARGET=openmp_rocrand
CPU_OPTIONS=()
```

<a id="canonical-sources-and-targets"></a>

## Sources and targets

| Variant | Teaching source / target | Benchmark source / target |
| --- | --- | --- |
| CPU | [`rand_cpu.cpp`](examples/rand_cpu.cpp) / `rand_cpu` | [`rand_cpu_bench.cpp`](benchmarks/rand_cpu_bench.cpp) / `rand_cpu_bench` |
| HIP | [`rand_gpu.cpp`](examples/rand_gpu.cpp) / `rand_gpu` | [`rand_gpu_bench.cpp`](benchmarks/rand_gpu_bench.cpp) / `rand_gpu_bench` |
| OpenMP target | [`openmp_rocrand.cpp`](examples/openmp_rocrand.cpp) / `openmp_rocrand` | [`openmp_rocrand_bench.cpp`](benchmarks/openmp_rocrand_bench.cpp) / `openmp_rocrand_bench` |

<a id="adapting-the-example"></a>

## Adapt this example

Choose a generator and reproducible seed/stream policy for the application. Do not reset the seed on every call unless restarting the same stream is intended. Reuse the generator and output buffer for a continuing stream. Compare range and distribution statistics, not CPU/GPU element identity; CPU and GPU algorithms differ. The CPU examples are serial.

The HIP variant manages arrays with `hipMalloc` / `hipMemcpy`. In OpenMP target, `map` owns storage and `use_device_ptr` exposes it to the GPU library. Ensure completion before result retrieval/map exit; do not free mapped storage with HIP. Shared physical memory does not imply USM or omitted copies.

<a id="direct-compile"></a>

## Direct compile (optional)

Run from the repository root in Bash with the provider header/library search paths available. Prefer CMake for complex CPU dependencies or static linking. Run only the CPU, HIP, or OpenMP blocks you need.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
c++ -std=c++17 -O2 amd/c-cpp/rocrand/examples/rand_cpu.cpp -o "$DIRECT_BUILD/rand_cpu" &&
"$DIRECT_BUILD/rand_cpu"
```
Set the GPU target for your installation and use `hipcc` from the selected ROCm stack.

```bash
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
hipcc -x hip -std=c++17 -O2 --offload-arch="$AMD_GPU_TARGET" \
  amd/c-cpp/rocrand/examples/rand_gpu.cpp -lrocrand \
  -o "$DIRECT_BUILD/rand_gpu" &&
"$DIRECT_BUILD/rand_gpu"
```

```bash
: "${AMD_CLANGXX:?Set the AMD Clang OpenMP C++ compiler}"
: "${ROCM_ROOT:?Set the ROCm prefix containing include}"
: "${ROCM_LIBDIR:?Set the ROCm shared-library directory}"
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
"$AMD_CLANGXX" -std=c++17 -O2 -fopenmp --offload-arch="$AMD_GPU_TARGET" \
  -I"$ROCM_ROOT/include" amd/c-cpp/rocrand/examples/openmp_rocrand.cpp \
  -L"$ROCM_LIBDIR" -Wl,-rpath,"$ROCM_LIBDIR" -lrocrand -lamdhip64 \
  -o "$DIRECT_BUILD/openmp_rocrand" &&
OMP_TARGET_OFFLOAD=MANDATORY "$DIRECT_BUILD/openmp_rocrand"
```

<a id="timing-scopes"></a>

## Optional performance measurement

Figures use library / end-to-end; internal CLI scopes are `compute` / `end-to-end`. These are controlled intervals, not simply the whole teaching program.

library keeps a generator/output allocation, resets and initializes the generator outside timing after warm-up and before each trial, then measures repeated generation plus completion. End-to-end creates/initializes a generator and output allocation, generates, completes, and retrieves every output value. Generator destruction and allocation release are outside timing. CPU and GPU RNG algorithms differ; neither elementwise stream equality nor parallel CPU execution is implied.

Host input generation and numerical verification are outside timing. See the [benchmark protocol](../../../docs/BENCHMARK_PROTOCOL.md) for details and [Sirius procedures](../../../jobs/sirius/README.md) for suite execution/aggregation. Distributed configs do not automatically adapt to another installation.

<a id="validation-and-limits"></a>

## Recorded validation and limits

Saved Sirius evidence includes CPU / HIP / OpenMP builds, numerical checks, and performance measurements. Source versions and coverage are recorded in the [validation report](../../../docs/VALIDATION_REPORT.md#amd-production-comparison).

Requested CPU threads are not observed worker counts. Saved RAND / Thrust CPU measurements use serial backends. Complete actually loaded-library identity remains unverified. Recorded success is not a guarantee for another environment or later edits; local fixtures do not prove GPU execution.

[Problem/source specification](../../../docs/PROJECT_SPECIFICATION.md) · [Result format](../../../docs/RESULT_SCHEMA.md)

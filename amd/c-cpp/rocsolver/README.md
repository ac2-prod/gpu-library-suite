# rocSOLVER C/C++: build, run, and adapt

[English](README.md) | [日本語](README.ja.md)

Solve a dense FP64 system `A*X=B` with LU factorization and multiple right-hand sides. The example shows pivot storage and solver status checks as well as matrix storage.

<a id="teaching-default"></a>

## Problem and expected result

`N=1024`, `nrhs=16`, `A=N*I+ones`, and every B element `2*N`. The solution is all ones. Require successful API returns and `info=0`; numerical checks must also pass.

Require exit status 0 and the result checks above. Teaching sizes are source constants; do not pass benchmark CLI options to these examples.

<a id="dependencies"></a>

## Dependencies

CPU needs the LAPACKE API: choose oneMKL, OpenBLAS with LAPACKE, or generic LAPACKE explicitly. The selection below uses generic LAPACKE. A BLAS-only installation is insufficient.

HIP needs the corresponding ROCm libraries; OpenMP target also needs compatible AMD Clang. Native CMake HIP uses HIP-capable Clang, not `hipcc`. CPU-only usage does not require ROCm.

<a id="select-a-minimal-build"></a>

## Select a minimal build

Set these variables in Bash, then follow only the CPU, direct GPU, or directive section you need in the [shared build guide](../../../docs/BUILDING.md#select). It builds the selected example; benchmarks and the full test suite are not needed.
```bash
VENDOR=amd
SOURCE_LANGUAGE=c-cpp
LIBRARY=rocsolver
CPU_TARGET=solver_cpu
GPU_TARGET=solver_gpu
DIRECTIVE_TARGET=openmp_rocsolver
CPU_OPTIONS=(-DGPU_SUITE_CPU_LAPACK_BACKEND=GENERIC_LAPACKE)
```

<a id="canonical-sources-and-targets"></a>

## Sources and targets

| Variant | Teaching source / target | Benchmark source / target |
| --- | --- | --- |
| CPU | [`solver_cpu.c`](examples/solver_cpu.c) / `solver_cpu` | [`solver_cpu_bench.c`](benchmarks/solver_cpu_bench.c) / `solver_cpu_bench` |
| HIP | [`solver_gpu.cpp`](examples/solver_gpu.cpp) / `solver_gpu` | [`solver_gpu_bench.cpp`](benchmarks/solver_gpu_bench.cpp) / `solver_gpu_bench` |
| OpenMP target | [`openmp_rocsolver.cpp`](examples/openmp_rocsolver.cpp) / `openmp_rocsolver` | [`openmp_rocsolver_bench.cpp`](benchmarks/openmp_rocsolver_bench.cpp) / `openmp_rocsolver_bench` |

<a id="adapting-the-example"></a>

## Adapt this example

Factorization overwrites A and solving overwrites B. Preserve originals for a residual check or another trial. Distinguish an API failure from singularity reported through info. If A is unchanged and only B changes, factor once and reuse the LU factors/pivots with getrs. This has a different cost from refactorizing every solve.

The HIP variant manages arrays with `hipMalloc` / `hipMemcpy`. In OpenMP target, `map` owns storage and `use_device_ptr` exposes it to the GPU library. Ensure completion before result retrieval/map exit; do not free mapped storage with HIP. Shared physical memory does not imply USM or omitted copies.

<a id="direct-compile"></a>

## Direct compile (optional)

Run from the repository root in Bash with the provider header/library search paths available. Prefer CMake for complex CPU dependencies or static linking. Run only the CPU, HIP, or OpenMP blocks you need.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
cc -std=c17 -O2 amd/c-cpp/rocsolver/examples/solver_cpu.c \
  -llapacke -llapack -lblas -lm -o "$DIRECT_BUILD/solver_cpu" &&
"$DIRECT_BUILD/solver_cpu"
```
Set the GPU target for your installation and use `hipcc` from the selected ROCm stack.

```bash
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
hipcc -x hip -std=c++17 -O2 --offload-arch="$AMD_GPU_TARGET" \
  amd/c-cpp/rocsolver/examples/solver_gpu.cpp -lrocsolver -lrocblas \
  -o "$DIRECT_BUILD/solver_gpu" &&
"$DIRECT_BUILD/solver_gpu"
```

```bash
: "${AMD_CLANGXX:?Set the AMD Clang OpenMP C++ compiler}"
: "${ROCM_ROOT:?Set the ROCm prefix containing include}"
: "${ROCM_LIBDIR:?Set the ROCm shared-library directory}"
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
"$AMD_CLANGXX" -std=c++17 -O2 -fopenmp --offload-arch="$AMD_GPU_TARGET" \
  -I"$ROCM_ROOT/include" amd/c-cpp/rocsolver/examples/openmp_rocsolver.cpp \
  -L"$ROCM_LIBDIR" -Wl,-rpath,"$ROCM_LIBDIR" -lrocsolver -lrocblas -lamdhip64 \
  -o "$DIRECT_BUILD/openmp_rocsolver" &&
OMP_TARGET_OFFLOAD=MANDATORY "$DIRECT_BUILD/openmp_rocsolver"
```

<a id="timing-scopes"></a>

## Optional performance measurement

Figures use library / end-to-end; internal CLI scopes are `compute` / `end-to-end`. These are controlled intervals, not simply the whole teaching program.

library restores A/B outside timing and includes getrf, its completion and info retrieval/check, then getrs and completion. repeat is 1. End-to-end additionally includes handle/workspace and array preparation, restoration copies, and final solution retrieval. Cleanup, including explicit OpenMP data deletion, is outside. CPU teaching calls dgesv; benchmarks split getrf/getrs. Keep original A/B for residual verification.

Host input generation and numerical verification are outside timing. See the [benchmark protocol](../../../docs/BENCHMARK_PROTOCOL.md) for details and [Sirius procedures](../../../jobs/sirius/README.md) for suite execution/aggregation. Distributed configs do not automatically adapt to another installation.

<a id="validation-and-limits"></a>

## Recorded validation and limits

Saved Sirius evidence includes CPU / HIP / OpenMP builds, numerical checks, and performance measurements. Source versions and coverage are recorded in the [validation report](../../../docs/VALIDATION_REPORT.md#amd-production-comparison).

Requested CPU threads are not observed worker counts. Saved RAND / Thrust CPU measurements use serial backends. Complete actually loaded-library identity remains unverified. Recorded success is not a guarantee for another environment or later edits; local fixtures do not prove GPU execution.

[Problem/source specification](../../../docs/PROJECT_SPECIFICATION.md) · [Result format](../../../docs/RESULT_SCHEMA.md)

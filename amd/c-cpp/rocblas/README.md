# rocBLAS C/C++: build, run, and adapt

[English](README.md) | [日本語](README.ja.md)

Multiply dense matrices with FP64 DGEMM: `C = alpha*A*B + beta*C`. Compare the CPU BLAS call with direct GPU use and directive-managed arrays.

<a id="teaching-default"></a>

## Problem and expected result

Column-major square matrices, `m=n=k=1024`, `alpha=beta=1`, and all-one A/B/C. Every output element must be `1025`; successful C/C++ examples report maximum error at most `1e-10`.

Require exit status 0 and the result checks above. Teaching sizes are source constants; do not pass benchmark CLI options to these examples.
CPU calls CBLAS; GPU calls `rocblas_dgemm`.

<a id="dependencies"></a>

## Dependencies

Select oneMKL, OpenBLAS, or a generic CBLAS provider explicitly for CPU. The selection below uses OpenBLAS (`cblas.h`, `libopenblas`).

HIP needs the corresponding ROCm libraries; OpenMP target also needs compatible AMD Clang. Native CMake HIP uses HIP-capable Clang, not `hipcc`. CPU-only usage does not require ROCm.

<a id="select-a-minimal-build"></a>

## Select a minimal build

Set these variables in Bash, then follow only the CPU, direct GPU, or directive section you need in the [shared build guide](../../../docs/BUILDING.md#select). It builds the selected example; benchmarks and the full test suite are not needed.
```bash
VENDOR=amd
SOURCE_LANGUAGE=c-cpp
LIBRARY=rocblas
CPU_TARGET=blas_cpu
GPU_TARGET=blas_gpu
DIRECTIVE_TARGET=openmp_rocblas
CPU_OPTIONS=(-DGPU_SUITE_CPU_BLAS_BACKEND=OPENBLAS)
```

<a id="canonical-sources-and-targets"></a>

## Sources and targets

| Variant | Teaching source / target | Benchmark source / target |
| --- | --- | --- |
| CPU | [`blas_cpu.c`](examples/blas_cpu.c) / `blas_cpu` | [`blas_cpu_bench.c`](benchmarks/blas_cpu_bench.c) / `blas_cpu_bench` |
| HIP | [`blas_gpu.cpp`](examples/blas_gpu.cpp) / `blas_gpu` | [`blas_gpu_bench.cpp`](benchmarks/blas_gpu_bench.cpp) / `blas_gpu_bench` |
| OpenMP target | [`openmp_rocblas.cpp`](examples/openmp_rocblas.cpp) / `openmp_rocblas` | [`openmp_rocblas_bench.cpp`](benchmarks/openmp_rocblas_bench.cpp) / `openmp_rocblas_bench` |

<a id="adapting-the-example"></a>

## Adapt this example

Match matrix layout and leading dimensions (`lda`, `ldb`, `ldc`) to your storage. With nonzero beta, C is an input as well as an output: copying only A/B is insufficient. Keep arrays and a handle across repeated multiplies when their lifetime permits it. CPU parallelism belongs to the selected BLAS provider; a requested thread count is not an observed worker count.

The HIP variant manages arrays with `hipMalloc` / `hipMemcpy`. In OpenMP target, `map` owns storage and `use_device_ptr` exposes it to the GPU library. Ensure completion before result retrieval/map exit; do not free mapped storage with HIP. Shared physical memory does not imply USM or omitted copies.

<a id="direct-compile"></a>

## Direct compile (optional)

Run from the repository root in Bash with the provider header/library search paths available. Prefer CMake for complex CPU dependencies or static linking. Run only the CPU, HIP, or OpenMP blocks you need.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
cc -std=c17 -O2 amd/c-cpp/rocblas/examples/blas_cpu.c -lopenblas -lm -o "$DIRECT_BUILD/blas_cpu" &&
"$DIRECT_BUILD/blas_cpu"
```
Set the GPU target for your installation and use `hipcc` from the selected ROCm stack.

```bash
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
hipcc -x hip -std=c++17 -O2 --offload-arch="$AMD_GPU_TARGET" \
  amd/c-cpp/rocblas/examples/blas_gpu.cpp -lrocblas \
  -o "$DIRECT_BUILD/blas_gpu" &&
"$DIRECT_BUILD/blas_gpu"
```

```bash
: "${AMD_CLANGXX:?Set the AMD Clang OpenMP C++ compiler}"
: "${ROCM_ROOT:?Set the ROCm prefix containing include}"
: "${ROCM_LIBDIR:?Set the ROCm shared-library directory}"
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
"$AMD_CLANGXX" -std=c++17 -O2 -fopenmp --offload-arch="$AMD_GPU_TARGET" \
  -I"$ROCM_ROOT/include" amd/c-cpp/rocblas/examples/openmp_rocblas.cpp \
  -L"$ROCM_LIBDIR" -Wl,-rpath,"$ROCM_LIBDIR" -lrocblas -lamdhip64 \
  -o "$DIRECT_BUILD/openmp_rocblas" &&
OMP_TARGET_OFFLOAD=MANDATORY "$DIRECT_BUILD/openmp_rocblas"
```

<a id="timing-scopes"></a>

## Optional performance measurement

Figures use library / end-to-end; internal CLI scopes are `compute` / `end-to-end`. These are controlled intervals, not simply the whole teaching program.

library restores C outside timing, then times repeated DGEMM updates and completion using persistent arrays/handle. End-to-end restores host C outside timing, then includes device allocation/map, input movement, handle creation, DGEMM/completion, and C retrieval. Explicit HIP frees and handle destruction are outside; OpenMP structured map exit is inside. With beta=1, library verification accounts for all repeat updates.

Host input generation and numerical verification are outside timing. See the [benchmark protocol](../../../docs/BENCHMARK_PROTOCOL.md) for details and [Sirius procedures](../../../jobs/sirius/README.md) for suite execution/aggregation. Distributed configs do not automatically adapt to another installation.

<a id="validation-and-limits"></a>

## Recorded validation and limits

Saved Sirius evidence includes CPU / HIP / OpenMP builds, numerical checks, and performance measurements. Source versions and coverage are recorded in the [validation report](../../../docs/VALIDATION_REPORT.md#amd-production-comparison).

Requested CPU threads are not observed worker counts. Saved RAND / Thrust CPU measurements use serial backends. Complete actually loaded-library identity remains unverified. Recorded success is not a guarantee for another environment or later edits; local fixtures do not prove GPU execution.

[Problem/source specification](../../../docs/PROJECT_SPECIFICATION.md) · [Result format](../../../docs/RESULT_SCHEMA.md)

# rocSPARSE C/C++: build, run, and adapt

[English](README.md) | [日本語](README.ja.md)

Compute FP64 sparse matrix-vector multiplication in CSR form: `y = alpha*A*x + beta*y`. The sample matrix is a regular 2D five-point Poisson stencil.

<a id="teaching-default"></a>

## Problem and expected result

A 1024×1024 grid gives 1,048,576 rows. Diagonal entries are 4, neighbors are −1, and `alpha=beta=1`, `x=y=1`. Expected y is 1 in the interior, 2 on an edge, and 3 at a corner. CSR indices are zero-based.

Require exit status 0 and the result checks above. Teaching sizes are source constants; do not pass benchmark CLI options to these examples.
GPU uses `rocsparse_v2_spmv` with `rocsparse_spmv_alg_csr_rowsplit`. It needs the v2 API checked with ROCm 7.2; there is no old-API fallback. Analysis and compute workspace sizes are queried separately.

<a id="dependencies"></a>

## Dependencies

Select AOCL Sparse (`aoclsparse.h`, `libaoclsparse`, LP64) or oneMKL Sparse explicitly for CPU. The selection below uses AOCL Sparse. REFERENCE is an auxiliary benchmark backend, not a substitute for the teaching provider.

HIP needs the corresponding ROCm libraries; OpenMP target also needs compatible AMD Clang. Native CMake HIP uses HIP-capable Clang, not `hipcc`. CPU-only usage does not require ROCm.
The CPU selection below explicitly uses AOCL Sparse. If you intend oneMKL instead, set `CPU_OPTIONS=(-DGPU_SUITE_CPU_SPARSE_BACKEND=ONEMKL)`.

<a id="select-a-minimal-build"></a>

## Select a minimal build

Set these variables in Bash, then follow only the CPU, direct GPU, or directive section you need in the [shared build guide](../../../docs/BUILDING.md#select). It builds the selected example; benchmarks and the full test suite are not needed.
```bash
VENDOR=amd
SOURCE_LANGUAGE=c-cpp
LIBRARY=rocsparse
CPU_TARGET=sparse_cpu
GPU_TARGET=sparse_gpu
DIRECTIVE_TARGET=openmp_rocsparse
CPU_OPTIONS=(-DGPU_SUITE_CPU_SPARSE_BACKEND=AOCLSPARSE)
```

<a id="canonical-sources-and-targets"></a>

## Sources and targets

| Variant | Teaching source / target | Benchmark source / target |
| --- | --- | --- |
| CPU | [`sparse_cpu.c`](examples/sparse_cpu.c) / `sparse_cpu` | [`sparse_cpu_bench.c`](benchmarks/sparse_cpu_bench.c) / `sparse_cpu_bench` |
| HIP | [`sparse_gpu.cpp`](examples/sparse_gpu.cpp) / `sparse_gpu` | [`sparse_gpu_bench.cpp`](benchmarks/sparse_gpu_bench.cpp) / `sparse_gpu_bench` |
| OpenMP target | [`openmp_rocsparse.cpp`](examples/openmp_rocsparse.cpp) / `openmp_rocsparse` | [`openmp_rocsparse_bench.cpp`](benchmarks/openmp_rocsparse_bench.cpp) / `openmp_rocsparse_bench` |

<a id="adapting-the-example"></a>

## Adapt this example

Replace CSR row pointers, column indices, and values together. Preserve index width/base, dimensions, and nonzero ordering. Initial y matters when beta is nonzero. Reuse descriptors and analysis only when the API permits it for the matrix/pointers you retain. Regular five-point rows do not predict performance for highly irregular application matrices.

The HIP variant manages arrays with `hipMalloc` / `hipMemcpy`. In OpenMP target, `map` owns storage and `use_device_ptr` exposes it to the GPU library. Ensure completion before result retrieval/map exit; do not free mapped storage with HIP. Shared physical memory does not imply USM or omitted copies.

<a id="direct-compile"></a>

## Direct compile (optional)

Run from the repository root in Bash with the provider header/library search paths available. Prefer CMake for complex CPU dependencies or static linking. Run only the CPU, HIP, or OpenMP blocks you need.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```
If AOCL needs additional dependencies, set `AOCL_EXTRA_LIBRARIES` as a Bash array of absolute files in link order; otherwise use an empty array.

```bash
AOCL_EXTRA_LIBRARIES=()
```

```bash
cc -std=c17 -O2 -DGPU_SUITE_USE_AOCL_SPARSE=1 \
  -I"${AOCL_SPARSE_INCLUDE:?Set the aoclsparse.h directory}" \
  amd/c-cpp/rocsparse/examples/sparse_cpu.c \
  "${AOCL_SPARSE_LIBRARY:?Set the AOCL Sparse shared library file}" \
  "${AOCL_EXTRA_LIBRARIES[@]}" -lm -o "$DIRECT_BUILD/sparse_cpu" &&
"$DIRECT_BUILD/sparse_cpu"
```
Set the GPU target for your installation and use `hipcc` from the selected ROCm stack.

```bash
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
hipcc -x hip -std=c++17 -O2 --offload-arch="$AMD_GPU_TARGET" \
  amd/c-cpp/rocsparse/examples/sparse_gpu.cpp -lrocsparse \
  -o "$DIRECT_BUILD/sparse_gpu" &&
"$DIRECT_BUILD/sparse_gpu"
```

```bash
: "${AMD_CLANGXX:?Set the AMD Clang OpenMP C++ compiler}"
: "${ROCM_ROOT:?Set the ROCm prefix containing include}"
: "${ROCM_LIBDIR:?Set the ROCm shared-library directory}"
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
"$AMD_CLANGXX" -std=c++17 -O2 -fopenmp --offload-arch="$AMD_GPU_TARGET" \
  -I"$ROCM_ROOT/include" amd/c-cpp/rocsparse/examples/openmp_rocsparse.cpp \
  -L"$ROCM_LIBDIR" -Wl,-rpath,"$ROCM_LIBDIR" -lrocsparse -lamdhip64 \
  -o "$DIRECT_BUILD/openmp_rocsparse" &&
OMP_TARGET_OFFLOAD=MANDATORY "$DIRECT_BUILD/openmp_rocsparse"
```

<a id="timing-scopes"></a>

## Optional performance measurement

Figures use library / end-to-end; internal CLI scopes are `compute` / `end-to-end`. These are controlled intervals, not simply the whole teaching program.

library reuses descriptors and compute workspace, restores y before the trial, and times repeated v2 SpMV compute calls plus completion. End-to-end includes CPU descriptor/hint/optimize or GPU copies/maps, descriptors, v2 analysis, workspace setup, compute, and y retrieval. Analysis workspace release is part of setup; final explicit resource cleanup is outside timing, while OpenMP structured map exit is inside. Do not mix older adaptive measurements with v2/csr_rowsplit results.

Host input generation and numerical verification are outside timing. See the [benchmark protocol](../../../docs/BENCHMARK_PROTOCOL.md) for details and [Sirius procedures](../../../jobs/sirius/README.md) for suite execution/aggregation. Distributed configs do not automatically adapt to another installation.

<a id="validation-and-limits"></a>

## Recorded validation and limits

Sirius revision `a974467` verified six v2/csr_rowsplit builds, three teaching examples, 96 smoke records, and 1,080 production records in `sirius-rocsparse-v2-20261019T010457Z-a974467`. Earlier adaptive results remain separate historical evidence.

Requested CPU threads are not observed worker counts. Saved RAND / Thrust CPU measurements use serial backends. Complete actually loaded-library identity remains unverified. Recorded success is not a guarantee for another environment or later edits; local fixtures do not prove GPU execution.

[Problem/source specification](../../../docs/PROJECT_SPECIFICATION.md) · [Result format](../../../docs/RESULT_SCHEMA.md)

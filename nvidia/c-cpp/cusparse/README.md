# cuSPARSE C/C++: build, run, and adapt

[English](README.md) | [日本語](README.ja.md)

Compute FP64 sparse matrix-vector multiplication in CSR form: `y = alpha*A*x + beta*y`. The sample matrix is a regular 2D five-point Poisson stencil.

<a id="teaching-default"></a>

## Teaching default

A 1024×1024 grid gives 1,048,576 rows. Diagonal entries are 4, neighbors are −1, and `alpha=beta=1`, `x=y=1`. Expected y is 1 in the interior, 2 on an edge, and 3 at a corner. CSR indices are zero-based.

<a id="dependencies"></a>

## Dependencies

The CPU teaching example uses oneMKL Sparse LP64 (`mkl_spblas.h`, `mkl_rt`). REFERENCE is an auxiliary benchmark backend, not a substitute for the teaching provider.

GPU variants need a compatible NVIDIA GPU/driver and CUDA Toolkit; OpenACC additionally needs NVHPC nvc++.

<a id="select-a-minimal-build"></a>

## Select a minimal build

Set these variables in Bash, then follow only the CPU, direct GPU, or directive section you need in the [shared build guide](../../../docs/BUILDING.md#select). It builds the selected example; benchmarks and the full test suite are not needed.
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=c-cpp
LIBRARY=cusparse
CPU_TARGET=sparse_cpu
GPU_TARGET=sparse_gpu
DIRECTIVE_TARGET=openacc_cusparse
CPU_OPTIONS=(-DGPU_SUITE_CPU_SPARSE_BACKEND=ONEMKL)
```

<a id="canonical-sources-and-targets"></a>

## Canonical sources and targets

| Role | Source | CMake target/executable |
| --- | --- | --- |
| CPU teaching | `examples/sparse_cpu.c` | `sparse_cpu` |
| CUDA teaching | `examples/sparse_gpu.cu` | `sparse_gpu` |
| OpenACC teaching | `examples/openacc_cusparse.cpp` | `openacc_cusparse` |
| CPU benchmark | `benchmarks/sparse_cpu_bench.c` | `sparse_cpu_bench` |
| CUDA benchmark | `benchmarks/sparse_gpu_bench.cu` | `sparse_gpu_bench` |
| OpenACC benchmark | `benchmarks/openacc_cusparse_bench.cpp` | `openacc_cusparse_bench` |

<a id="adapting-the-example"></a>

## Adapt this example

Replace CSR row pointers, column indices, and values together. Preserve index width/base, dimensions, and nonzero ordering. Initial y matters when beta is nonzero. Reuse descriptors and analysis only when the API permits it for the matrix/pointers you retain. Regular five-point rows do not predict performance for highly irregular application matrices.

Direct CUDA manages device arrays; OpenACC uses a data region and `host_data use_device` to call the same GPU library. Keep allocation ownership clear and ensure completion before reading results.

<a id="direct-compile"></a>

## Direct compile

An alternative to CMake, from the repository root in Bash with provider header/library search paths already configured. Choose the variant you need; run it only if compilation succeeds.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
cc -std=c17 -O2 -I"${MKLROOT:?Set the installed oneMKL prefix}/include" \
  nvidia/c-cpp/cusparse/examples/sparse_cpu.c -L"${MKL_LIBDIR:?Set the oneMKL library directory}" \
  -lmkl_rt -lpthread -ldl -lm -o "$DIRECT_BUILD/sparse_cpu" &&
"$DIRECT_BUILD/sparse_cpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVCC_ARCH:?Set the nvcc architecture supported by your GPU}"
"$CUDA_TOOLKIT_ROOT/bin/nvcc" -std=c++17 -O2 -arch="$NVCC_ARCH" \
  nvidia/c-cpp/cusparse/examples/sparse_gpu.cu -lcusparse \
  -o "$DIRECT_BUILD/sparse_gpu" &&
"$DIRECT_BUILD/sparse_gpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVHPC_GPU_TARGET:?Set your NVHPC GPU target}"
NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT" \
nvc++ -std=c++17 -O2 -acc -cuda -gpu="$NVHPC_GPU_TARGET" \
  nvidia/c-cpp/cusparse/examples/openacc_cusparse.cpp -cudalib=cusparse \
  -o "$DIRECT_BUILD/openacc_cusparse" &&
"$DIRECT_BUILD/openacc_cusparse"
```

<a id="cmake-configure-and-build"></a>

## CMake configure and build

Use the selection above with the [shared guide](../../../docs/BUILDING.md). It creates separate CPU, CUDA, and OpenACC trees and builds only the chosen target. Use the GPU sections alone if you do not need CPU numerical dependencies.

<a id="first-build-and-run"></a>

## First build and run

Follow selection → build → run and require exit status 0 plus the expected result under Teaching default. A missing target or nonzero exit is not success. Use a compute allocation on shared clusters.

<a id="run-the-examples"></a>
For Fortran, use the [separate guide](../../fortran/cusparse/README.md).

<a id="from-example-to-benchmark"></a>

## From example to benchmark

Teaching examples have no timing CLI. For performance use the matching `_bench`, separating library (internal scope `compute`) from end-to-end. The sections below are benchmark reference material. Enable benchmarks using the [measurement workflow](../../../docs/PORTABILITY.md#measuring-on-your-own-system) and use its build paths; the minimal example build does not include benchmarks.

<a id="benchmark-cli"></a>

## Benchmark CLI

This small check requests one CPU thread and uses the build from the linked measurement workflow. BLAS / Sparse / Solver use its oneMKL configuration. If you built another provider, match `--cpu-backend` to that build.

```bash
"$CPU_CUDA_BUILD/nvidia/c-cpp/cusparse/sparse_cpu_bench" \
  --size 4096 --alpha 1 --beta 1 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --abs-tolerance 1e-12 \
  --rel-tolerance 1e-10 --output - --format jsonl \
  --cpu-backend cpu-onemkl --cpu-threads 1 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism threaded
```

The square `--size` determines `nx`, `ny`, and the exact nonzero count. Suite
runs pass config-owned roles and thresholds and keep the runner as the sole raw
result-file writer.

<a id="timing-scopes"></a>

## Timing scopes

The display term library corresponds to internal scope `compute`.

`compute` constructs descriptors and storage before timing, restores y, and
applies the SpMV state update continuously across repeats; verification checks
the repeat-folded output. `end-to-end` times each repeat's storage, transfers,
descriptor creation, workspace query/allocation, SpMV, completion, and copyout;
cleanup follows the end timestamp, and canonical y is restored before every
repeat.

<a id="verification"></a>

## Verification

Maximum absolute error is checked against
`abs_tolerance + rel_tolerance * reference_scale`. Verification failure is a
recoverable trial failure; fatal descriptor, allocation, API, or clock failure
alone skips the remaining trials.

<a id="openacc-notes"></a>

## OpenACC notes

The data region owns CSR and vector storage, and `host_data use_device` exposes
those addresses to cuSPARSE. Only a nonzero cuSPARSE workspace is allocated with
CUDA; a zero-byte workspace remains `nullptr`. The target links
`OpenACC::OpenACC_CXX`, `CUDA::cudart`, and `CUDA::cusparse`.

<a id="cpu-backend-and-role"></a>

## CPU backend and role

Select the production CPU backend explicitly for measurements; do not replace it because a dependency is missing. Distinguish requested threads from effective workers.

<a id="known-limitations-and-local-validation"></a>

## Known limitations and local validation

Saved Pegasus evidence includes NVIDIA C/C++ builds, numerical smoke, and performance measurements; see the [versions and coverage](../../../docs/VALIDATION_REPORT.md#current-coexistence-status). GPU/toolchain behavior is locally unverified by host-only fixtures; saved real-machine runs are distinct evidence and do not certify another environment or later edits. See the [protocol](../../../docs/BENCHMARK_PROTOCOL.md) for numerical/statistical verification and CPU parallelism limits.

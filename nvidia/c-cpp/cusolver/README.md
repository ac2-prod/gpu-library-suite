# cuSOLVER C/C++: build, run, and adapt

[English](README.md) | [日本語](README.ja.md)

Solve a dense FP64 system `A*X=B` with LU factorization and multiple right-hand sides. The example shows pivot storage and solver status checks as well as matrix storage.

<a id="teaching-default"></a>

## Teaching default

`N=1024`, `nrhs=16`, `A=N*I+ones`, and every B element `2*N`. The solution is all ones. Require successful API returns and `info=0`; numerical checks must also pass.

<a id="dependencies"></a>

## Dependencies

CPU needs the LAPACKE API: choose oneMKL, OpenBLAS with LAPACKE, or generic LAPACKE explicitly. The selection below uses generic LAPACKE. A BLAS-only installation is insufficient.

GPU variants need a compatible NVIDIA GPU/driver and CUDA Toolkit; OpenACC additionally needs NVHPC nvc++.

<a id="select-a-minimal-build"></a>

## Select a minimal build

Set these variables in Bash, then follow only the CPU, direct GPU, or directive section you need in the [shared build guide](../../../docs/BUILDING.md#select). It builds the selected example; benchmarks and the full test suite are not needed.
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=c-cpp
LIBRARY=cusolver
CPU_TARGET=solver_cpu
GPU_TARGET=solver_gpu
DIRECTIVE_TARGET=openacc_cusolver
CPU_OPTIONS=(-DGPU_SUITE_CPU_LAPACK_BACKEND=GENERIC_LAPACKE)
```

<a id="canonical-sources-and-targets"></a>

## Canonical sources and targets

| Role | Source | CMake target/executable |
| --- | --- | --- |
| CPU teaching | `examples/solver_cpu.c` | `solver_cpu` |
| CUDA teaching | `examples/solver_gpu.cu` | `solver_gpu` |
| OpenACC teaching | `examples/openacc_cusolver.cpp` | `openacc_cusolver` |
| CPU benchmark | `benchmarks/solver_cpu_bench.c` | `solver_cpu_bench` |
| CUDA benchmark | `benchmarks/solver_gpu_bench.cu` | `solver_gpu_bench` |
| OpenACC benchmark | `benchmarks/openacc_cusolver_bench.cpp` | `openacc_cusolver_bench` |

<a id="adapting-the-example"></a>

## Adapt this example

Factorization overwrites A and solving overwrites B. Preserve originals for a residual check or another trial. Distinguish an API failure from singularity reported through info. If A is unchanged and only B changes, factor once and reuse the LU factors/pivots with getrs. This has a different cost from refactorizing every solve.

Direct CUDA manages device arrays; OpenACC uses a data region and `host_data use_device` to call the same GPU library. Keep allocation ownership clear and ensure completion before reading results.

<a id="direct-compile"></a>

## Direct compile

An alternative to CMake, from the repository root in Bash with provider header/library search paths already configured. Choose the variant you need; run it only if compilation succeeds.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
cc -std=c17 -O2 nvidia/c-cpp/cusolver/examples/solver_cpu.c \
  -llapacke -llapack -lblas -lm -o "$DIRECT_BUILD/solver_cpu" &&
"$DIRECT_BUILD/solver_cpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVCC_ARCH:?Set the nvcc architecture supported by your GPU}"
"$CUDA_TOOLKIT_ROOT/bin/nvcc" -std=c++17 -O2 -arch="$NVCC_ARCH" \
  nvidia/c-cpp/cusolver/examples/solver_gpu.cu -lcusolver \
  -o "$DIRECT_BUILD/solver_gpu" &&
"$DIRECT_BUILD/solver_gpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVHPC_GPU_TARGET:?Set your NVHPC GPU target}"
NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT" \
nvc++ -std=c++17 -O2 -acc -cuda -gpu="$NVHPC_GPU_TARGET" \
  nvidia/c-cpp/cusolver/examples/openacc_cusolver.cpp -cudalib=cusolver \
  -o "$DIRECT_BUILD/openacc_cusolver" &&
"$DIRECT_BUILD/openacc_cusolver"
```

<a id="cmake-configure-and-build"></a>

## CMake configure and build

Use the selection above with the [shared guide](../../../docs/BUILDING.md). It creates separate CPU, CUDA, and OpenACC trees and builds only the chosen target. Use the GPU sections alone if you do not need CPU numerical dependencies.

<a id="first-build-and-run"></a>

## First build and run

Follow selection → build → run and require exit status 0 plus the expected result under Teaching default. A missing target or nonzero exit is not success. Use a compute allocation on shared clusters.

<a id="run-the-examples"></a>
For Fortran, use the [separate guide](../../fortran/cusolver/README.md).

<a id="from-example-to-benchmark"></a>

## From example to benchmark

Teaching examples have no timing CLI. For performance use the matching `_bench`, separating library (internal scope `compute`) from end-to-end. The sections below are benchmark reference material. Enable benchmarks using the [measurement workflow](../../../docs/PORTABILITY.md#measuring-on-your-own-system) and use its build paths; the minimal example build does not include benchmarks.

<a id="benchmark-cli"></a>

## Benchmark CLI

This small check requests one CPU thread and uses the build from the linked measurement workflow. BLAS / Sparse / Solver use its oneMKL configuration. If you built another provider, match `--cpu-backend` to that build.

```bash
"$CPU_CUDA_BUILD/nvidia/c-cpp/cusolver/solver_cpu_bench" \
  --size 64 --nrhs 2 --warmup 1 --repeat 1 --trials 1 \
  --scope end-to-end --verify true --abs-tolerance 1e-12 \
  --rel-tolerance 1e-10 --output - --format jsonl \
  --cpu-backend cpu-onemkl --cpu-threads 1 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism threaded
```

Every cuSOLVER scope requires `repeat=1`. Use the backend name matching the
selected LAPACKE provider, and leave effective threads null unless the runtime
can establish them. Production suite thresholds come only from the effective
configuration.

<a id="timing-scopes"></a>

## Timing scopes

The display term library corresponds to internal scope `compute`.

`compute` allocates, transfers, and queries workspace before timing, restores
the canonical matrix/RHS, then times one `getrf` plus `getrs` pipeline.
`end-to-end` includes allocation, copies or OpenACC updates, handle/workspace
creation, solve, completion, solution, and both info-code retrievals. Cleanup
follows the end timestamp; warm-up, restoration, and verification are outside
timing.

<a id="verification"></a>

## Verification

Both solution relative error and relative residual are checked with effective
absolute/relative tolerances. `getrf_info` and `getrs_info` must each be zero;
OpenACC initializes them to `-1` and explicitly copies them to the host, so a
missing copyback cannot masquerade as success.

<a id="openacc-notes"></a>

## OpenACC notes

The OpenACC data region owns matrix, RHS, pivots, and info arrays. Small
`host_data use_device` regions expose them to cuSOLVER; only the workspace is a
manual CUDA allocation. End-to-end result retrieval includes explicit
`acc update self` for RHS and both info arrays before the measurement end.

<a id="cpu-backend-and-role"></a>

## CPU backend and role

Select the production CPU backend explicitly for measurements; do not replace it because a dependency is missing. Distinguish requested threads from effective workers.

<a id="known-limitations-and-local-validation"></a>

## Known limitations and local validation

Saved Pegasus evidence includes NVIDIA C/C++ builds, numerical smoke, and performance measurements; see the [versions and coverage](../../../docs/VALIDATION_REPORT.md#current-coexistence-status). GPU/toolchain behavior is locally unverified by host-only fixtures; saved real-machine runs are distinct evidence and do not certify another environment or later edits. See the [protocol](../../../docs/BENCHMARK_PROTOCOL.md) for numerical/statistical verification and CPU parallelism limits.

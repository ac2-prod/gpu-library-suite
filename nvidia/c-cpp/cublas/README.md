# cuBLAS C/C++: build, run, and adapt

[English](README.md) | [日本語](README.ja.md)

Multiply dense matrices with FP64 DGEMM: `C = alpha*A*B + beta*C`. Compare the CPU BLAS call with direct GPU use and directive-managed arrays.

<a id="teaching-default"></a>

## Teaching default

Column-major square matrices, `m=n=k=1024`, `alpha=beta=1`, and all-one A/B/C. Every output element must be `1025`; successful C/C++ examples report maximum error at most `1e-10`.

<a id="dependencies"></a>

## Dependencies

Select oneMKL, OpenBLAS, or a generic CBLAS provider explicitly for CPU. The selection below uses OpenBLAS (`cblas.h`, `libopenblas`).

GPU variants need a compatible NVIDIA GPU/driver and CUDA Toolkit; OpenACC additionally needs NVHPC nvc++.

<a id="select-a-minimal-build"></a>

## Select a minimal build

Set these variables in Bash, then follow only the CPU, direct GPU, or directive section you need in the [shared build guide](../../../docs/BUILDING.md#select). It builds the selected example; benchmarks and the full test suite are not needed.
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=c-cpp
LIBRARY=cublas
CPU_TARGET=blas_cpu
GPU_TARGET=blas_gpu
DIRECTIVE_TARGET=openacc_cublas
CPU_OPTIONS=(-DGPU_SUITE_CPU_BLAS_BACKEND=OPENBLAS)
```

<a id="canonical-sources-and-targets"></a>

## Canonical sources and targets

| Role | Source | CMake target/executable |
| --- | --- | --- |
| CPU teaching | `examples/blas_cpu.c` | `blas_cpu` |
| CUDA teaching | `examples/blas_gpu.cu` | `blas_gpu` |
| OpenACC teaching | `examples/openacc_cublas.cpp` | `openacc_cublas` |
| CPU benchmark | `benchmarks/blas_cpu_bench.c` | `blas_cpu_bench` |
| CUDA benchmark | `benchmarks/blas_gpu_bench.cu` | `blas_gpu_bench` |
| OpenACC benchmark | `benchmarks/openacc_cublas_bench.cpp` | `openacc_cublas_bench` |

<a id="adapting-the-example"></a>

## Adapt this example

Match matrix layout and leading dimensions (`lda`, `ldb`, `ldc`) to your storage. With nonzero beta, C is an input as well as an output: copying only A/B is insufficient. Keep arrays and a handle across repeated multiplies when their lifetime permits it. CPU parallelism belongs to the selected BLAS provider; a requested thread count is not an observed worker count.

Direct CUDA manages device arrays; OpenACC uses a data region and `host_data use_device` to call the same GPU library. Keep allocation ownership clear and ensure completion before reading results.

<a id="direct-compile"></a>

## Direct compile

An alternative to CMake, from the repository root in Bash with provider header/library search paths already configured. Choose the variant you need; run it only if compilation succeeds.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
cc -std=c17 -O2 nvidia/c-cpp/cublas/examples/blas_cpu.c -lopenblas -lm -o "$DIRECT_BUILD/blas_cpu" &&
"$DIRECT_BUILD/blas_cpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVCC_ARCH:?Set the nvcc architecture supported by your GPU}"
"$CUDA_TOOLKIT_ROOT/bin/nvcc" -std=c++17 -O2 -arch="$NVCC_ARCH" \
  nvidia/c-cpp/cublas/examples/blas_gpu.cu -lcublas \
  -o "$DIRECT_BUILD/blas_gpu" &&
"$DIRECT_BUILD/blas_gpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVHPC_GPU_TARGET:?Set your NVHPC GPU target}"
NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT" \
nvc++ -std=c++17 -O2 -acc -cuda -gpu="$NVHPC_GPU_TARGET" \
  nvidia/c-cpp/cublas/examples/openacc_cublas.cpp -cudalib=cublas \
  -o "$DIRECT_BUILD/openacc_cublas" &&
"$DIRECT_BUILD/openacc_cublas"
```

<a id="cmake-configure-and-build"></a>

## CMake configure and build

Use the selection above with the [shared guide](../../../docs/BUILDING.md). It creates separate CPU, CUDA, and OpenACC trees and builds only the chosen target. Use the GPU sections alone if you do not need CPU numerical dependencies.

<a id="first-build-and-run"></a>

## First build and run

Follow selection → build → run and require exit status 0 plus the expected result under Teaching default. A missing target or nonzero exit is not success. Use a compute allocation on shared clusters.

<a id="run-the-examples"></a>
For Fortran, use the [separate guide](../../fortran/cublas/README.md).

<a id="from-example-to-benchmark"></a>

## From example to benchmark

Teaching examples have no timing CLI. For performance use the matching `_bench`, separating library (internal scope `compute`) from end-to-end. The sections below are benchmark reference material. Enable benchmarks using the [measurement workflow](../../../docs/PORTABILITY.md#measuring-on-your-own-system) and use its build paths; the minimal example build does not include benchmarks.

<a id="benchmark-cli"></a>

## Benchmark CLI

This small check requests one CPU thread and uses the build from the linked measurement workflow. BLAS / Sparse / Solver use its oneMKL configuration. If you built another provider, match `--cpu-backend` to that build.

```bash
"$CPU_CUDA_BUILD/nvidia/c-cpp/cublas/blas_cpu_bench" \
  --size 128 --alpha 1 --beta 1 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --abs-tolerance 1e-12 \
  --rel-tolerance 1e-10 --output - --format jsonl \
  --cpu-backend cpu-onemkl --cpu-threads 1 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism threaded
```

Use the CPU backend name matching the selected provider. If an effective thread
count has been independently established, pass it with
`--cpu-threads-effective`; otherwise it remains null. Production suite runs
must take all threshold and role values from the effective configuration.

<a id="timing-scopes"></a>

## Timing scopes

The display term library corresponds to internal scope `compute`.

`compute` creates storage/handles and restores C before timing, then applies the
DGEMM update continuously across repeats. Verification therefore checks the
repeat-folded result. `end-to-end` times allocation, copies, handle lifecycle,
DGEMM, completion, and result retrieval for each repeat; cleanup follows the
end timestamp. It restores canonical C before every repeat so state is never
inherited.

<a id="verification"></a>

## Verification

The maximum absolute error is checked against
`abs_tolerance + rel_tolerance * reference_scale` using the effective
configuration. Verification failures are retained as raw failures; after state
restoration, later trials may continue.

<a id="openacc-notes"></a>

## OpenACC notes

The OpenACC data region owns A, B, and C. `host_data use_device` exposes those
addresses to cuBLAS; no second CUDA allocation owns the arrays. The canonical
link set is `OpenACC::OpenACC_CXX`, `CUDA::cudart`, and `CUDA::cublas`.

<a id="cpu-backend-and-role"></a>

## CPU backend and role

Select the production CPU backend explicitly for measurements; do not replace it because a dependency is missing. Distinguish requested threads from effective workers.

<a id="known-limitations-and-local-validation"></a>

## Known limitations and local validation

Saved Pegasus evidence includes NVIDIA C/C++ builds, numerical smoke, and performance measurements; see the [versions and coverage](../../../docs/VALIDATION_REPORT.md#current-coexistence-status). GPU/toolchain behavior is locally unverified by host-only fixtures; saved real-machine runs are distinct evidence and do not certify another environment or later edits. See the [protocol](../../../docs/BENCHMARK_PROTOCOL.md) for numerical/statistical verification and CPU parallelism limits.

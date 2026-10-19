# Thrust C/C++: build, run, and adapt

[English](README.md) | [日本語](README.ja.md)

Compute `sum(values[i]^2)` with transform-reduce. Combining the transform with reduction avoids storing an intermediate array of squared values.

<a id="teaching-default"></a>

## Teaching default

Input is `2^24` FP64 ones; the expected square sum is `16777216`. CPU uses serial STL `std::transform_reduce`. The result returned to the host is one double.

<a id="dependencies"></a>

## Dependencies

Serial CPU needs only C++17 and its standard library. GPU needs Thrust/CCCL and CUDA.

GPU variants need a compatible NVIDIA GPU/driver and CUDA Toolkit; OpenACC additionally needs NVHPC nvc++.

<a id="select-a-minimal-build"></a>

## Select a minimal build

Set these variables in Bash, then follow only the CPU, direct GPU, or directive section you need in the [shared build guide](../../../docs/BUILDING.md#select). It builds the selected example; benchmarks and the full test suite are not needed.
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=c-cpp
LIBRARY=thrust
CPU_TARGET=reduce_cpu
GPU_TARGET=reduce_gpu
DIRECTIVE_TARGET=openacc_thrust
CPU_OPTIONS=(-DGPU_SUITE_CPU_THRUST_BACKEND=STL)
```

<a id="canonical-sources-and-targets"></a>

## Canonical sources and targets

| Role | Source | CMake target/executable |
| --- | --- | --- |
| CPU teaching | `examples/reduce_cpu.cpp` | `reduce_cpu` |
| CUDA teaching | `examples/reduce_gpu.cu` | `reduce_gpu` |
| OpenACC teaching | `examples/openacc_thrust.cpp` | `openacc_thrust` |
| CPU benchmark | `benchmarks/reduce_cpu_bench.cpp` | `reduce_cpu_bench` |
| CUDA benchmark | `benchmarks/reduce_gpu_bench.cu` | `reduce_gpu_bench` |
| OpenACC benchmark | `benchmarks/openacc_thrust_bench.cpp` | `openacc_thrust_bench` |

<a id="adapting-the-example"></a>

## Adapt this example

Replace the transform, associative reduction, and identity value together. Floating-point reductions can differ with order, so use a suitable tolerance for general inputs. Keep input resident when applying repeated GPU operations. A directive-managed device pointer must not be freed by a wrapper; the caller owns its allocation.

Direct CUDA manages device arrays; OpenACC uses a data region and `host_data use_device` to call the same GPU library. Keep allocation ownership clear and ensure completion before reading results.

<a id="direct-compile"></a>

## Direct compile

An alternative to CMake, from the repository root in Bash with provider header/library search paths already configured. Choose the variant you need; run it only if compilation succeeds.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
c++ -std=c++17 -O2 nvidia/c-cpp/thrust/examples/reduce_cpu.cpp -o "$DIRECT_BUILD/reduce_cpu" &&
"$DIRECT_BUILD/reduce_cpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVCC_ARCH:?Set the nvcc architecture supported by your GPU}"
"$CUDA_TOOLKIT_ROOT/bin/nvcc" -std=c++17 -O2 -arch="$NVCC_ARCH" -I"${CUDA_CCCL_INCLUDE_DIR:?Set the directory containing thrust headers}" \
  nvidia/c-cpp/thrust/examples/reduce_gpu.cu \
  -o "$DIRECT_BUILD/reduce_gpu" &&
"$DIRECT_BUILD/reduce_gpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVHPC_GPU_TARGET:?Set your NVHPC GPU target}"
NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT" \
nvc++ -std=c++17 -O2 -acc -cuda -gpu="$NVHPC_GPU_TARGET" -I"${CUDA_CCCL_INCLUDE_DIR:?Set the directory containing thrust headers}" \
  nvidia/c-cpp/thrust/examples/openacc_thrust.cpp \
  -o "$DIRECT_BUILD/openacc_thrust" &&
"$DIRECT_BUILD/openacc_thrust"
```

<a id="cmake-configure-and-build"></a>

## CMake configure and build

Use the selection above with the [shared guide](../../../docs/BUILDING.md). It creates separate CPU, CUDA, and OpenACC trees and builds only the chosen target. Use the GPU sections alone if you do not need CPU numerical dependencies.

<a id="first-build-and-run"></a>

## First build and run

Follow selection → build → run and require exit status 0 plus the expected result under Teaching default. A missing target or nonzero exit is not success. Use a compute allocation on shared clusters.

<a id="run-the-examples"></a>
For Fortran, use the [separate guide](../../fortran/thrust/README.md).

<a id="from-example-to-benchmark"></a>

## From example to benchmark

Teaching examples have no timing CLI. For performance use the matching `_bench`, separating library (internal scope `compute`) from end-to-end. The sections below are benchmark reference material. Enable benchmarks using the [measurement workflow](../../../docs/PORTABILITY.md#measuring-on-your-own-system) and use its build paths; the minimal example build does not include benchmarks.

<a id="benchmark-cli"></a>

## Benchmark CLI

This small check requests one CPU thread and uses the build from the linked measurement workflow. BLAS / Sparse / Solver use its oneMKL configuration. If you built another provider, match `--cpu-backend` to that build.

```bash
"$CPU_CUDA_BUILD/nvidia/c-cpp/thrust/reduce_cpu_bench" \
  --size 65536 --warmup 1 --repeat 2 --trials 1 --scope compute \
  --verify true --abs-tolerance 1e-12 --rel-tolerance 1e-10 \
  --output - --format jsonl --cpu-backend cpu-stl-serial \
  --cpu-threads 1 --cpu-threads-effective 1 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism serial
```

Production runs obtain thresholds and roles from the effective configuration,
and `tools/run_suite.py` remains the sole writer of node raw-result files.

<a id="timing-scopes"></a>

## Timing scopes

The display term library corresponds to internal scope `compute`.

`compute` prepares input/device storage before timing and times only repeated
transform-reduce operations. GPU `end-to-end` includes device storage creation,
input transfer, reduction, and completion/result retrieval for each repeat;
cleanup follows the end timestamp. The CPU vector and canonical all-ones host
input are prepared outside timing; the CPU path has no device transfer.
Canonical restoration and verification stay outside timing.

<a id="verification"></a>

## Verification

The measured result is compared with the canonical analytical square sum using
`abs_tolerance + rel_tolerance * reference_scale`. A verification failure is
retained but does not become a fatal barrier to a restored later trial.

<a id="openacc-notes"></a>

## OpenACC notes

OpenACC owns the input allocation and transfer, then
`thrust::device_pointer_cast` translates the known device address. Thrust's
synchronous algorithms normally need no additional completion call for host
result retrieval; the teaching OpenACC example retains an explicit
`cudaDeviceSynchronize` to show the interoperation boundary. Compile and link
both carry NVHPC CUDA C++ interoperation flags.

<a id="cpu-backend-and-role"></a>

## CPU backend and role

The recorded CPU series is serial. A larger requested thread count does not make it parallel. Label your own CPU/backend and distinguish requested from effective threads.

<a id="known-limitations-and-local-validation"></a>

## Known limitations and local validation

Saved Pegasus evidence includes NVIDIA C/C++ builds, numerical smoke, and performance measurements; see the [versions and coverage](../../../docs/VALIDATION_REPORT.md#current-coexistence-status). GPU/toolchain behavior is locally unverified by host-only fixtures; saved real-machine runs are distinct evidence and do not certify another environment or later edits. See the [protocol](../../../docs/BENCHMARK_PROTOCOL.md) for numerical/statistical verification and CPU parallelism limits.

# cuRAND C/C++: build, run, and adapt

[English](README.md) | [日本語](README.ja.md)

Generate uniform FP64 random values. This demonstrates a persistent generator and output buffer, useful for simulation inputs and stochastic algorithms.

<a id="teaching-default"></a>

## Teaching default

Generate `2^24` values with seed 1234. CPU uses serial `std::mt19937_64`. GPU uses the library’s default pseudorandom generator. CPU uniform values are in [0,1), GPU values in (0,1]. These are comparable distributions, not identical streams.
This is a comparison of the distribution/output task, not of identical random number algorithms.

<a id="dependencies"></a>

## Dependencies

CPU needs only C++17 and its standard library. GPU requires cuRAND.

GPU variants need a compatible NVIDIA GPU/driver and CUDA Toolkit; OpenACC additionally needs NVHPC nvc++.

<a id="select-a-minimal-build"></a>

## Select a minimal build

Set these variables in Bash, then follow only the CPU, direct GPU, or directive section you need in the [shared build guide](../../../docs/BUILDING.md#select). It builds the selected example; benchmarks and the full test suite are not needed.
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=c-cpp
LIBRARY=curand
CPU_TARGET=rand_cpu
GPU_TARGET=rand_gpu
DIRECTIVE_TARGET=openacc_curand
CPU_OPTIONS=()
```

<a id="canonical-sources-and-targets"></a>

## Canonical sources and targets

| Role | Source | CMake target/executable |
| --- | --- | --- |
| CPU teaching | `examples/rand_cpu.cpp` | `rand_cpu` |
| CUDA teaching | `examples/rand_gpu.cu` | `rand_gpu` |
| OpenACC teaching | `examples/openacc_curand.cpp` | `openacc_curand` |
| CPU benchmark | `benchmarks/rand_cpu_bench.cpp` | `rand_cpu_bench` |
| CUDA benchmark | `benchmarks/rand_gpu_bench.cu` | `rand_gpu_bench` |
| OpenACC benchmark | `benchmarks/openacc_curand_bench.cpp` | `openacc_curand_bench` |

<a id="adapting-the-example"></a>

## Adapt this example

Choose a generator and reproducible seed/stream policy for the application. Do not reset the seed on every call unless restarting the same stream is intended. Reuse the generator and output buffer for a continuing stream. Compare range and distribution statistics, not CPU/GPU element identity; CPU and GPU algorithms differ. The CPU examples are serial.

Direct CUDA manages device arrays; OpenACC uses a data region and `host_data use_device` to call the same GPU library. Keep allocation ownership clear and ensure completion before reading results.

<a id="direct-compile"></a>

## Direct compile

An alternative to CMake, from the repository root in Bash with provider header/library search paths already configured. Choose the variant you need; run it only if compilation succeeds.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
c++ -std=c++17 -O2 nvidia/c-cpp/curand/examples/rand_cpu.cpp -o "$DIRECT_BUILD/rand_cpu" &&
"$DIRECT_BUILD/rand_cpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVCC_ARCH:?Set the nvcc architecture supported by your GPU}"
"$CUDA_TOOLKIT_ROOT/bin/nvcc" -std=c++17 -O2 -arch="$NVCC_ARCH" \
  nvidia/c-cpp/curand/examples/rand_gpu.cu -lcurand \
  -o "$DIRECT_BUILD/rand_gpu" &&
"$DIRECT_BUILD/rand_gpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVHPC_GPU_TARGET:?Set your NVHPC GPU target}"
NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT" \
nvc++ -std=c++17 -O2 -acc -cuda -gpu="$NVHPC_GPU_TARGET" \
  nvidia/c-cpp/curand/examples/openacc_curand.cpp -cudalib=curand \
  -o "$DIRECT_BUILD/openacc_curand" &&
"$DIRECT_BUILD/openacc_curand"
```

<a id="cmake-configure-and-build"></a>

## CMake configure and build

Use the selection above with the [shared guide](../../../docs/BUILDING.md). It creates separate CPU, CUDA, and OpenACC trees and builds only the chosen target. Use the GPU sections alone if you do not need CPU numerical dependencies.

<a id="first-build-and-run"></a>

## First build and run

Follow selection → build → run and require exit status 0 plus the expected result under Teaching default. A missing target or nonzero exit is not success. Use a compute allocation on shared clusters.

<a id="run-the-examples"></a>
For Fortran, use the [separate guide](../../fortran/curand/README.md).

<a id="from-example-to-benchmark"></a>

## From example to benchmark

Teaching examples have no timing CLI. For performance use the matching `_bench`, separating library (internal scope `compute`) from end-to-end. The sections below are benchmark reference material. Enable benchmarks using the [measurement workflow](../../../docs/PORTABILITY.md#measuring-on-your-own-system) and use its build paths; the minimal example build does not include benchmarks.

<a id="benchmark-cli"></a>

## Benchmark CLI

This small check requests one CPU thread and uses the build from the linked measurement workflow. BLAS / Sparse / Solver use its oneMKL configuration. If you built another provider, match `--cpu-backend` to that build.

```bash
"$CPU_CUDA_BUILD/nvidia/c-cpp/curand/rand_cpu_bench" \
  --size 65536 --warmup 1 --repeat 2 --trials 1 --scope compute \
  --verify true --sigma-multiplier 6 --expected-mean 0.5 \
  --expected-second-central-moment 0.08333333333333333 \
  --output - --format jsonl --cpu-backend cpu-std-random-serial \
  --cpu-threads 1 --cpu-threads-effective 1 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism serial
```

Production suite execution passes all statistical constants from the effective
configuration and uses the runner as the only node raw-file writer.

<a id="timing-scopes"></a>

## Timing scopes

The display term library corresponds to internal scope `compute`.

`compute` creates and seeds the persistent generator before timing, restores
the configured offset, and times repeated generation into existing storage.
`end-to-end` includes storage and generator creation, configuration, generation,
completion, and host result retrieval for every repeat; destruction follows the
end timestamp. Canonical generator state is restored before each raw
trial/repeat as the scope requires.

<a id="verification"></a>

## Verification

Both intervals share the inclusive range check `0 <= x <= 1`; metadata records
the CPU `[0,1)` and cuRAND `(0,1]` contracts. Required metrics are
`observed_min`, `observed_max`, `sample_mean`, and
`second_central_moment_about_half`. The configured bounds are
`sigma_multiplier*sqrt(1/(12*N))` about mean 0.5 and
`sigma_multiplier*sqrt(1/(180*N))` about second central moment 1/12.

<a id="openacc-notes"></a>

## OpenACC notes

The data region owns output storage and `host_data use_device` exposes it to
cuRAND. The target links `OpenACC::OpenACC_CXX`, `CUDA::cudart`, and
`CUDA::curand`; portable source does not hard-code a GPU architecture.

<a id="cpu-backend-and-role"></a>

## CPU backend and role

The recorded CPU series is serial. A larger requested thread count does not make it parallel. Label your own CPU/backend and distinguish requested from effective threads.

<a id="known-limitations-and-local-validation"></a>

## Known limitations and local validation

Saved Pegasus evidence includes NVIDIA C/C++ builds, numerical smoke, and performance measurements; see the [versions and coverage](../../../docs/VALIDATION_REPORT.md#current-coexistence-status). GPU/toolchain behavior is locally unverified by host-only fixtures; saved real-machine runs are distinct evidence and do not certify another environment or later edits. See the [protocol](../../../docs/BENCHMARK_PROTOCOL.md) for numerical/statistical verification and CPU parallelism limits.

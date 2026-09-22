# Thrust C/C++ examples and benchmarks

[English](README.md) | [日本語](README.ja.md)

This directory compares the canonical FP64 square-sum reduction through C++17
STL, direct Thrust, and OpenACC/Thrust interoperation. Workload ownership belongs
to [`docs/PROJECT_SPECIFICATION.md`](../../../docs/PROJECT_SPECIFICATION.md),
and CLI, timing, restoration, and verification belong to
[`docs/BENCHMARK_PROTOCOL.md`](../../../docs/BENCHMARK_PROTOCOL.md).

## Canonical sources and targets

| Role | Source | CMake target/executable |
| --- | --- | --- |
| CPU teaching | `examples/reduce_cpu.cpp` | `reduce_cpu` |
| CUDA teaching | `examples/reduce_gpu.cu` | `reduce_gpu` |
| OpenACC teaching | `examples/openacc_thrust.cpp` | `openacc_thrust` |
| CPU benchmark | `benchmarks/reduce_cpu_bench.cpp` | `reduce_cpu_bench` |
| CUDA benchmark | `benchmarks/reduce_gpu_bench.cu` | `reduce_gpu_bench` |
| OpenACC benchmark | `benchmarks/openacc_thrust_bench.cpp` | `openacc_thrust_bench` |

## Teaching default

Each teaching program creates `2^24` FP64 values and evaluates
`sum(values[i]^2)` using a transform-reduce flow. The programs remain direct,
single-source examples without benchmark infrastructure.

## Dependencies

The serial CPU targets need only a C++17 standard library. An explicitly
selected OpenMP CPU variant requires `OpenMP::OpenMP_CXX`. CUDA targets require
CUDA Runtime and Thrust headers. OpenACC targets require NVHPC OpenACC C++, CUDA
Runtime, and Thrust, including `-cuda`-equivalent flags at both compile and link
time. CMake resolves `include/cccl/thrust/version.h` from
`CUDAToolkit_INCLUDE_DIRS` and prepends that external Toolkit's `include/cccl`
for both CUDA and OpenACC probes, examples, and benchmarks.

## Direct compile

```bash
mkdir -p /tmp/gpu-library-suite-local-build
c++ -std=c++17 nvidia/c-cpp/thrust/examples/reduce_cpu.cpp \
  -o /tmp/gpu-library-suite-local-build/reduce_cpu-direct
nvcc -std=c++17 nvidia/c-cpp/thrust/examples/reduce_gpu.cu \
  -o /tmp/gpu-library-suite-local-build/reduce_gpu-direct
nvc++ -std=c++17 -acc -cuda \
  nvidia/c-cpp/thrust/examples/openacc_thrust.cpp \
  -o /tmp/gpu-library-suite-local-build/openacc_thrust-direct
```

## CMake configure and build

The commands assume that `CUDA_TOOLKIT_ROOT`, `CUDA_ARCHITECTURES`, and
`NVHPC_GPU_TARGET` were set to explicit site-approved values.

```bash
cmake -S . -B /tmp/gpu-library-suite-local-build/cpu-cuda \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_CUDA=ON \
  -DGPU_SUITE_BUILD_OPENACC=OFF
cmake --build /tmp/gpu-library-suite-local-build/cpu-cuda \
  --target reduce_cpu reduce_gpu reduce_cpu_bench reduce_gpu_bench

NVHPC_CUDA_HOME="${CUDA_TOOLKIT_ROOT}" \
cmake -S . -B /tmp/gpu-library-suite-local-build/openacc \
  -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="${NVHPC_GPU_TARGET}" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=ON
cmake --build /tmp/gpu-library-suite-local-build/openacc \
  --target openacc_thrust openacc_thrust_bench
```

## Run the examples

From the repository root, after the corresponding targets above built:

```bash
CPU_CUDA_BUILD="${CPU_CUDA_BUILD:-/tmp/gpu-library-suite-local-build/cpu-cuda}"
OPENACC_BUILD="${OPENACC_BUILD:-/tmp/gpu-library-suite-local-build/openacc}"
"$CPU_CUDA_BUILD/nvidia/c-cpp/thrust/reduce_cpu"
printf 'CPU exit status: %s\n' "$?"
"$CPU_CUDA_BUILD/nvidia/c-cpp/thrust/reduce_gpu"
printf 'CUDA exit status: %s\n' "$?"
"$OPENACC_BUILD/nvidia/c-cpp/thrust/openacc_thrust"
printf 'OpenACC exit status: %s\n' "$?"
```

Inspect the square-sum result against the all-ones input count and each exit
status. A missing target or nonzero exit leaves the comparison incomplete.
Continue with [measurement and result processing](../../../docs/PORTABILITY.md#measuring-on-your-own-system).

## Benchmark CLI

```bash
/tmp/gpu-library-suite-local-build/cpu-cuda/nvidia/c-cpp/thrust/reduce_cpu_bench \
  --size 65536 --warmup 1 --repeat 2 --trials 1 --scope compute \
  --verify true --abs-tolerance 1e-12 --rel-tolerance 1e-10 \
  --output - --format jsonl --cpu-backend cpu-stl-serial \
  --cpu-threads 48 --cpu-threads-effective 1 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism serial
```

Production runs obtain thresholds and roles from the effective configuration,
and `tools/run_suite.py` remains the sole writer of node raw-result files.

## Timing scopes

`compute` prepares input/device storage before timing and times only repeated
transform-reduce operations. GPU `end-to-end` includes device storage creation,
input transfer, reduction, and completion/result retrieval for each repeat;
cleanup follows the end timestamp. The CPU vector and canonical all-ones host
input are prepared outside timing; the CPU path has no device transfer.
Canonical restoration and verification stay outside timing.

## Verification

The measured result is compared with the canonical analytical square sum using
`abs_tolerance + rel_tolerance * reference_scale`. A verification failure is
retained but does not become a fatal barrier to a restored later trial.

## CPU backend and role

`cpu-stl-serial` is the configured production primary serial CPU backend and is
plotted as **Intel Xeon Platinum 8468, STL (single thread)** in the approved
legend, not as a parallel baseline. Requested threads
may be 48 while effective threads are 1. No speedup is plotted from this series.
An optional `cpu-openmp` build is a separately named backend and is never a
silent replacement.

## OpenACC notes

OpenACC owns the input allocation and transfer, then
`thrust::device_pointer_cast` translates the known device address. Thrust's
synchronous algorithms normally need no additional completion call for host
result retrieval; the teaching OpenACC example retains an explicit
`cudaDeviceSynchronize` to show the interoperation boundary. Compile and link
both carry NVHPC CUDA C++ interoperation flags.

## Known limitations and local validation

Reduction order may differ across implementations, so verification uses the
declared floating-point tolerance rather than bitwise equality. Local tests run
the serial CPU implementation and fake-header GPU syntax; real Thrust/CUDA,
NVHPC, GPU, OpenMP production configuration, and Pegasus execution remain
locally unverified.
Later saved real execution records are distinguished from these local tests in
[the validation report](../../../docs/VALIDATION_REPORT.md#saved-execution-evidence-reviewed-for-publication).

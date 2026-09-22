# cuBLAS C/C++ examples and benchmarks

[English](README.md) | [日本語](README.ja.md)

This directory compares canonical column-major FP64 DGEMM through CBLAS,
direct cuBLAS, and OpenACC/cuBLAS. Problem mathematics belong to
[`docs/PROJECT_SPECIFICATION.md`](../../../docs/PROJECT_SPECIFICATION.md), and
CLI, timing, restoration, and verification belong to
[`docs/BENCHMARK_PROTOCOL.md`](../../../docs/BENCHMARK_PROTOCOL.md).

## Canonical sources and targets

| Role | Source | CMake target/executable |
| --- | --- | --- |
| CPU teaching | `examples/blas_cpu.c` | `blas_cpu` |
| CUDA teaching | `examples/blas_gpu.cu` | `blas_gpu` |
| OpenACC teaching | `examples/openacc_cublas.cpp` | `openacc_cublas` |
| CPU benchmark | `benchmarks/blas_cpu_bench.c` | `blas_cpu_bench` |
| CUDA benchmark | `benchmarks/blas_gpu_bench.cu` | `blas_gpu_bench` |
| OpenACC benchmark | `benchmarks/openacc_cublas_bench.cpp` | `openacc_cublas_bench` |

## Teaching default

All three teaching programs compute a 1,024 by 1,024 DGEMM with
`alpha=1` and `beta=1`, using column-major FP64 matrices. They are direct,
complete single-source programs without benchmark infrastructure.

## Dependencies

CPU targets require a selected ONEMKL, OPENBLAS, or GENERIC_CBLAS provider that
compiles and links `cblas_dgemm`; `BLAS_FOUND` alone is not accepted. The
provider-specific cache keys and expected header/library names prevent mixing
oneMKL headers with OpenBLAS libraries. CUDA targets require CUDA Runtime and
cuBLAS, while OpenACC targets additionally require NVHPC OpenACC C++.

## Direct compile

The default CPU source includes `<cblas.h>` unless CMake supplies a provider
override. These representative commands assume ordinary search paths are set:

```bash
mkdir -p /tmp/gpu-library-suite-local-build
cc -std=c17 nvidia/c-cpp/cublas/examples/blas_cpu.c \
  -lcblas -lm -o /tmp/gpu-library-suite-local-build/blas_cpu-direct
nvcc -std=c++17 nvidia/c-cpp/cublas/examples/blas_gpu.cu \
  -lcublas -o /tmp/gpu-library-suite-local-build/blas_gpu-direct
nvc++ -std=c++17 -acc -cuda \
  nvidia/c-cpp/cublas/examples/openacc_cublas.cpp -cudalib=cublas \
  -o /tmp/gpu-library-suite-local-build/openacc_cublas-direct
```

## First build and run

Use a shell whose working directory is the repository root. You need CMake
3.20+, a C17/C++17 toolchain, and a CBLAS implementation already available to
the compiler/linker. No CUDA installation or Pegasus account is needed for
this first CPU step. The example below selects `GENERIC_CBLAS` (`cblas.h` and
`libcblas`); if you have oneMKL or OpenBLAS instead, select `ONEMKL` or
`OPENBLAS` explicitly. Do not substitute a provider without recording the change.
oneMKL discovery uses the installed `MKLROOT`; ordinary include/library search
paths or CMake cache entries must identify your actual provider installation.

```bash
EXAMPLE_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-library-suite-example.XXXXXX")"
cmake -S . -B "$EXAMPLE_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=OFF -DGPU_SUITE_BUILD_TESTS=OFF \
  -DGPU_SUITE_CPU_BLAS_BACKEND=GENERIC_CBLAS
cmake --build "$EXAMPLE_BUILD" --target blas_cpu
"$EXAMPLE_BUILD/nvidia/c-cpp/cublas/blas_cpu"
printf 'example exit status: %s\n' "$?"
```

Stop if configure disables `blas_cpu` or the build fails; the absence of an
optional dependency is not a successful CPU/GPU comparison. On success the
program prints `CBLAS DGEMM complete; max error = ...` and exits 0. Its all-ones
inputs give every output element `1025` (`k + 1`); it accepts maximum absolute
error at most `1e-10`. A nonzero exit or a larger error requires investigation.

Next provide a compatible NVIDIA GPU/driver, CUDA Toolkit, and, for OpenACC,
NVHPC. Set the explicit Toolkit/architecture variables described in the next
section and build its two profiles. Those commands use the following default
directories; if you followed the general workflow, keep its exported directory
values instead. Run only after the corresponding target built successfully:

```bash
CPU_CUDA_BUILD="${CPU_CUDA_BUILD:-/tmp/gpu-library-suite-local-build/cpu-cuda}"
OPENACC_BUILD="${OPENACC_BUILD:-/tmp/gpu-library-suite-local-build/openacc}"
"$CPU_CUDA_BUILD/nvidia/c-cpp/cublas/blas_cpu"
printf 'CPU exit status: %s\n' "$?"
"$CPU_CUDA_BUILD/nvidia/c-cpp/cublas/blas_gpu"
printf 'CUDA exit status: %s\n' "$?"
"$OPENACC_BUILD/nvidia/c-cpp/cublas/openacc_cublas"
printf 'OpenACC exit status: %s\n' "$?"
```

Inspect each command's exit status immediately, not just the last command's.
The GPU programs report the same `max error` expectation, with `cuBLAS` or
`OpenACC-managed cuBLAS` in the message. They do not report benchmark timings.
If a GPU dependency is unavailable, the CPU example is still useful, but the
three-way execution check remains incomplete.

## CMake configure and build

For example, a generic CBLAS CPU/CUDA tree and a separate OpenACC tree are shown
below. The commands assume that `CUDA_TOOLKIT_ROOT`, `CUDA_ARCHITECTURES`, and
`NVHPC_GPU_TARGET` were set to explicit site-approved values.

```bash
cmake -S . -B /tmp/gpu-library-suite-local-build/cpu-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_CUDA=ON \
  -DGPU_SUITE_BUILD_OPENACC=OFF \
  -DGPU_SUITE_CPU_BLAS_BACKEND=GENERIC_CBLAS
cmake --build /tmp/gpu-library-suite-local-build/cpu-cuda \
  --target blas_cpu blas_gpu blas_cpu_bench blas_gpu_bench

NVHPC_CUDA_HOME="${CUDA_TOOLKIT_ROOT}" \
cmake -S . -B /tmp/gpu-library-suite-local-build/openacc \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="${NVHPC_GPU_TARGET}" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=ON
cmake --build /tmp/gpu-library-suite-local-build/openacc \
  --target openacc_cublas openacc_cublas_bench
```

## Benchmark CLI

```bash
/tmp/gpu-library-suite-local-build/cpu-cuda/nvidia/c-cpp/cublas/blas_cpu_bench \
  --size 128 --alpha 1 --beta 1 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --abs-tolerance 1e-12 \
  --rel-tolerance 1e-10 --output - --format jsonl \
  --cpu-backend cpu-generic-cblas --cpu-threads 48 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism threaded
```

Use the CPU backend name matching the selected provider. If an effective thread
count has been independently established, pass it with
`--cpu-threads-effective`; otherwise it remains null. Production suite runs
must take all threshold and role values from the effective configuration.

## Timing scopes

`compute` creates storage/handles and restores C before timing, then applies the
DGEMM update continuously across repeats. Verification therefore checks the
repeat-folded result. `end-to-end` times allocation, copies, handle lifecycle,
DGEMM, completion, and result retrieval for each repeat; cleanup follows the
end timestamp. It restores canonical C before every repeat so state is never
inherited.

## From example to benchmark

Read the three short examples before their corresponding `_bench` sources.
They solve the same all-ones DGEMM problem, but a benchmark adds controlled
measurement rather than timing the whole `main` function.

| Part of the program | Teaching example | Benchmark counterpart |
| --- | --- | --- |
| Input and initialization | Fixed `m=n=k=1024`, host A/B/C filled with ones | CLI/configured dimensions; checked sizes; the same canonical input generation outside timing |
| CPU computation | One `cblas_dgemm` call | Same provider call; parallelism is inside the CPU library, not an application OpenMP loop |
| Direct CUDA storage | Explicit `cudaMalloc`, H2D and D2H | Persistent allocation/handle for compute; per-repeat pipeline for E2E |
| OpenACC storage | `acc data` owns arrays; `host_data use_device` passes device pointers to cuBLAS | The same ownership, with data entry/exit moved relative to timestamps according to scope |
| Calculation | One `cublasDgemm` or `cblas_dgemm` | Repeated calls for compute; one-shot E2E in the publication configuration |
| Restoration and verification | Check the one-call expectation `k+1` | Restore C after warm-up and before every trial; compute verification accounts for all repeat updates; E2E restores C before each repeat |
| Timing and output | No timer or machine-readable record | Shared monotonic wall clock, GPU completion synchronization, per-trial JSONL/CSV, and verification outside timing |

The authoritative [timing boundaries](../../../docs/BENCHMARK_PROTOCOL.md#timing-fields)
explain exactly which setup, transfer, synchronization, and cleanup steps are
included. Continue with the [small measurement and full data flow](../../../docs/PORTABILITY.md#measuring-on-your-own-system),
not by adding a stopwatch around the teaching program.

## Verification

The maximum absolute error is checked against
`abs_tolerance + rel_tolerance * reference_scale` using the effective
configuration. Verification failures are retained as raw failures; after state
restoration, later trials may continue.

## CPU backend and role

The configured CBLAS provider is the production primary CPU backend. Supported
names are `cpu-onemkl`, `cpu-openblas`, and `cpu-generic-cblas`; one is selected
for a campaign, and another provider is never a silent fallback.

## OpenACC notes

The OpenACC data region owns A, B, and C. `host_data use_device` exposes those
addresses to cuBLAS; no second CUDA allocation owns the arrays. The canonical
link set is `OpenACC::OpenACC_CXX`, `CUDA::cudart`, and `CUDA::cublas`.

## Known limitations and local validation

Provider thread counts may be unknown unless the runtime exposes a trustworthy
value, so requested threads are not relabeled as effective threads. Local tests
exercise fake CBLAS providers and GPU syntax only. A real cuBLAS runtime,
NVHPC compiler, GPU, production oneMKL/OpenBLAS installation, and Pegasus run
remain locally unverified.
Later saved real execution records are distinguished from these local tests in
[the validation report](../../../docs/VALIDATION_REPORT.md#saved-execution-evidence-reviewed-for-publication).

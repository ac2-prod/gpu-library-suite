# cuSPARSE C/C++ examples and benchmarks

[English](README.md) | [日本語](README.ja.md)

This directory compares the canonical FP64 five-point Poisson CSR SpMV through
oneMKL Sparse, direct cuSPARSE, and OpenACC/cuSPARSE. Matrix construction belongs
to [`docs/PROJECT_SPECIFICATION.md`](../../../docs/PROJECT_SPECIFICATION.md),
and CLI, timing, restoration, and verification belong to
[`docs/BENCHMARK_PROTOCOL.md`](../../../docs/BENCHMARK_PROTOCOL.md).

## Canonical sources and targets

| Role | Source | CMake target/executable |
| --- | --- | --- |
| CPU teaching | `examples/sparse_cpu.c` | `sparse_cpu` |
| CUDA teaching | `examples/sparse_gpu.cu` | `sparse_gpu` |
| OpenACC teaching | `examples/openacc_cusparse.cpp` | `openacc_cusparse` |
| CPU benchmark | `benchmarks/sparse_cpu_bench.c` | `sparse_cpu_bench` |
| CUDA benchmark | `benchmarks/sparse_gpu_bench.cu` | `sparse_gpu_bench` |
| OpenACC benchmark | `benchmarks/openacc_cusparse_bench.cpp` | `openacc_cusparse_bench` |

## Teaching default

The teaching programs build a 1,024 by 1,024 grid, so the matrix has
1,048,576 rows and `5*n - 2*nx - 2*ny` nonzeros. They execute FP64 SpMV with
`alpha=1` and `beta=1`. The canonical CPU teaching example is specifically a
oneMKL Sparse example; it does not contain a handwritten reference fallback.

## Dependencies

`sparse_cpu` requires oneMKL Sparse and a successful descriptor/create,
hint/optimize, SpMV, and destroy compile-and-link probe. The CPU benchmark may
instead be built as the separately named `cpu-reference-csr` backend when
`GPU_SUITE_CPU_SPARSE_BACKEND=REFERENCE`; that choice disables the canonical
CPU teaching target with an explicit reason. CUDA and OpenACC targets require
CUDA Runtime and cuSPARSE descriptor/SpMV symbols.

## Direct compile

These representative commands assume the selected libraries are on compiler
search paths:

```bash
mkdir -p /tmp/gpu-library-suite-local-build
cc -std=c17 nvidia/c-cpp/cusparse/examples/sparse_cpu.c \
  -lmkl_rt -lpthread -ldl -lm \
  -o /tmp/gpu-library-suite-local-build/sparse_cpu-direct
nvcc -std=c++17 nvidia/c-cpp/cusparse/examples/sparse_gpu.cu \
  -lcusparse -o /tmp/gpu-library-suite-local-build/sparse_gpu-direct
nvc++ -std=c++17 -acc -cuda \
  nvidia/c-cpp/cusparse/examples/openacc_cusparse.cpp -cudalib=cusparse \
  -o /tmp/gpu-library-suite-local-build/openacc_cusparse-direct
```

## CMake configure and build

The commands assume that `CUDA_TOOLKIT_ROOT`, `CUDA_ARCHITECTURES`, and
`NVHPC_GPU_TARGET` were set to explicit site-approved values.

```bash
cmake -S . -B /tmp/gpu-library-suite-local-build/cpu-cuda \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_CUDA=ON \
  -DGPU_SUITE_BUILD_OPENACC=OFF \
  -DGPU_SUITE_CPU_SPARSE_BACKEND=ONEMKL
cmake --build /tmp/gpu-library-suite-local-build/cpu-cuda \
  --target sparse_cpu sparse_gpu sparse_cpu_bench sparse_gpu_bench

NVHPC_CUDA_HOME="${CUDA_TOOLKIT_ROOT}" \
cmake -S . -B /tmp/gpu-library-suite-local-build/openacc \
  -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="${NVHPC_GPU_TARGET}" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=ON
cmake --build /tmp/gpu-library-suite-local-build/openacc \
  --target openacc_cusparse openacc_cusparse_bench
```

## Run the examples

From the repository root, after the corresponding targets above built:

```bash
CPU_CUDA_BUILD="${CPU_CUDA_BUILD:-/tmp/gpu-library-suite-local-build/cpu-cuda}"
OPENACC_BUILD="${OPENACC_BUILD:-/tmp/gpu-library-suite-local-build/openacc}"
"$CPU_CUDA_BUILD/nvidia/c-cpp/cusparse/sparse_cpu"
printf 'CPU exit status: %s\n' "$?"
"$CPU_CUDA_BUILD/nvidia/c-cpp/cusparse/sparse_gpu"
printf 'CUDA exit status: %s\n' "$?"
"$OPENACC_BUILD/nvidia/c-cpp/cusparse/openacc_cusparse"
printf 'OpenACC exit status: %s\n' "$?"
```

Inspect the Poisson SpMV error and each exit status. A missing target or nonzero
exit leaves the three-way check incomplete. Continue with
[measurement and result processing](../../../docs/PORTABILITY.md#measuring-on-your-own-system).

## Benchmark CLI

```bash
/tmp/gpu-library-suite-local-build/cpu-cuda/nvidia/c-cpp/cusparse/sparse_cpu_bench \
  --size 4096 --alpha 1 --beta 1 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --abs-tolerance 1e-12 \
  --rel-tolerance 1e-10 --output - --format jsonl \
  --cpu-backend cpu-onemkl --cpu-threads 48 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism threaded
```

The square `--size` determines `nx`, `ny`, and the exact nonzero count. Suite
runs pass config-owned roles and thresholds and keep the runner as the sole raw
result-file writer.

## Timing scopes

`compute` constructs descriptors and storage before timing, restores y, and
applies the SpMV state update continuously across repeats; verification checks
the repeat-folded output. `end-to-end` times each repeat's storage, transfers,
descriptor creation, workspace query/allocation, SpMV, completion, and copyout;
cleanup follows the end timestamp, and canonical y is restored before every
repeat.

## Verification

Maximum absolute error is checked against
`abs_tolerance + rel_tolerance * reference_scale`. Verification failure is a
recoverable trial failure; fatal descriptor, allocation, API, or clock failure
alone skips the remaining trials.

## CPU backend and role

`cpu-onemkl` is the production primary backend. `cpu-reference-csr` is an
explicit reference backend for benchmark work only; it cannot replace oneMKL
inside the canonical teaching example or silently become a production series.

## OpenACC notes

The data region owns CSR and vector storage, and `host_data use_device` exposes
those addresses to cuSPARSE. Only a nonzero cuSPARSE workspace is allocated with
CUDA; a zero-byte workspace remains `nullptr`. The target links
`OpenACC::OpenACC_CXX`, `CUDA::cudart`, and `CUDA::cusparse`.

## Known limitations and local validation

Only square problem sizes are accepted by the canonical suite configuration,
and allocation failure is reported without automatic downsizing. Local tests
use a fake oneMKL provider and fake CUDA headers; a production oneMKL Sparse
runtime, real cuSPARSE, NVHPC, GPU, and Pegasus execution remain locally unverified.
Later saved real execution records are distinguished from these local tests in
[the validation report](../../../docs/VALIDATION_REPORT.md#saved-execution-evidence-reviewed-for-publication).

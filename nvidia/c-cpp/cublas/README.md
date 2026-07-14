# cuBLAS C/C++ examples and benchmarks

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
cc -std=c17 nvidia/c-cpp/cublas/examples/blas_cpu.c \
  -lcblas -lm -o /tmp/gpu-library-suite-local-build/blas_cpu-direct
nvcc -std=c++17 nvidia/c-cpp/cublas/examples/blas_gpu.cu \
  -lcublas -o /tmp/gpu-library-suite-local-build/blas_gpu-direct
nvc++ -std=c++17 -acc -cuda \
  nvidia/c-cpp/cublas/examples/openacc_cublas.cpp -cudalib=cublas \
  -o /tmp/gpu-library-suite-local-build/openacc_cublas-direct
```

## CMake configure and build

For example, a generic CBLAS CPU/CUDA tree and a separate OpenACC tree are shown
below. The commands assume that `CUDA_TOOLKIT_ROOT`, `CUDA_ARCHITECTURES`, and
`NVHPC_GPU_TARGET` were set to explicit site-approved values.

```bash
cmake -S . -B /tmp/gpu-library-suite-local-build/cpu-cuda \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_CUDA=ON \
  -DGPU_SUITE_BUILD_OPENACC=OFF \
  -DGPU_SUITE_CPU_BLAS_BACKEND=GENERIC_CBLAS
cmake --build /tmp/gpu-library-suite-local-build/cpu-cuda \
  --target blas_cpu blas_gpu blas_cpu_bench blas_gpu_bench

NVHPC_CUDA_HOME="${CUDA_TOOLKIT_ROOT}" \
cmake -S . -B /tmp/gpu-library-suite-local-build/openacc \
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

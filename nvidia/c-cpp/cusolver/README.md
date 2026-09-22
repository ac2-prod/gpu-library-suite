# cuSOLVER C/C++ examples and benchmarks

This directory solves the canonical dense FP64 system through LAPACKE, direct
cuSOLVER, and OpenACC/cuSOLVER. System construction belongs to
[`docs/PROJECT_SPECIFICATION.md`](../../../docs/PROJECT_SPECIFICATION.md), and
CLI, staging, timing, and verification belong to
[`docs/BENCHMARK_PROTOCOL.md`](../../../docs/BENCHMARK_PROTOCOL.md).

## Canonical sources and targets

| Role | Source | CMake target/executable |
| --- | --- | --- |
| CPU teaching | `examples/solver_cpu.c` | `solver_cpu` |
| CUDA teaching | `examples/solver_gpu.cu` | `solver_gpu` |
| OpenACC teaching | `examples/openacc_cusolver.cpp` | `openacc_cusolver` |
| CPU benchmark | `benchmarks/solver_cpu_bench.c` | `solver_cpu_bench` |
| CUDA benchmark | `benchmarks/solver_gpu_bench.cu` | `solver_gpu_bench` |
| OpenACC benchmark | `benchmarks/openacc_cusolver_bench.cpp` | `openacc_cusolver_bench` |

## Teaching default

All teaching variants solve a canonical `n=1024`, `nrhs=16` FP64 system. The
CPU teaching example deliberately shows `LAPACKE_dgesv`; benchmark variants use
separate factorization and solve stages so `getrf_info` and `getrs_info` remain
independently visible.

## Dependencies

CPU targets require an ONEMKL, OPENBLAS, or GENERIC_LAPACKE provider whose
`LAPACKE_dgesv`, `LAPACKE_dgetrf`, and `LAPACKE_dgetrs` all compile and link.
Provider-specific cache keys prevent stale cross-provider paths. CUDA programs
require CUDA Runtime and cuSOLVER dense APIs. OpenACC programs additionally
require NVHPC OpenACC C++.

## Direct compile

The default CPU source includes `<lapacke.h>` unless CMake supplies its
provider header. Representative direct commands are:

```bash
mkdir -p /tmp/gpu-library-suite-local-build
cc -std=c17 nvidia/c-cpp/cusolver/examples/solver_cpu.c \
  -llapacke -llapack -lblas -lm \
  -o /tmp/gpu-library-suite-local-build/solver_cpu-direct
nvcc -std=c++17 nvidia/c-cpp/cusolver/examples/solver_gpu.cu \
  -lcusolver -o /tmp/gpu-library-suite-local-build/solver_gpu-direct
nvc++ -std=c++17 -acc -cuda \
  nvidia/c-cpp/cusolver/examples/openacc_cusolver.cpp -cudalib=cusolver \
  -o /tmp/gpu-library-suite-local-build/openacc_cusolver-direct
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
  -DGPU_SUITE_CPU_LAPACK_BACKEND=GENERIC_LAPACKE
cmake --build /tmp/gpu-library-suite-local-build/cpu-cuda \
  --target solver_cpu solver_gpu solver_cpu_bench solver_gpu_bench

NVHPC_CUDA_HOME="${CUDA_TOOLKIT_ROOT}" \
cmake -S . -B /tmp/gpu-library-suite-local-build/openacc \
  -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="${NVHPC_GPU_TARGET}" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=ON
cmake --build /tmp/gpu-library-suite-local-build/openacc \
  --target openacc_cusolver openacc_cusolver_bench
```

## Run the examples

From the repository root, after the corresponding targets above built:

```bash
CPU_CUDA_BUILD="${CPU_CUDA_BUILD:-/tmp/gpu-library-suite-local-build/cpu-cuda}"
OPENACC_BUILD="${OPENACC_BUILD:-/tmp/gpu-library-suite-local-build/openacc}"
"$CPU_CUDA_BUILD/nvidia/c-cpp/cusolver/solver_cpu"
printf 'CPU exit status: %s\n' "$?"
"$CPU_CUDA_BUILD/nvidia/c-cpp/cusolver/solver_gpu"
printf 'CUDA exit status: %s\n' "$?"
"$OPENACC_BUILD/nvidia/c-cpp/cusolver/openacc_cusolver"
printf 'OpenACC exit status: %s\n' "$?"
```

Inspect solver `info` values, the solution error and each exit status. A missing
target or nonzero exit leaves the comparison incomplete. The CPU teaching call
is `LAPACKE_dgesv`; the benchmark separates factorization and solve as explained
below. Continue with [measurement and result processing](../../../docs/PORTABILITY.md#measuring-on-your-own-system).

## Benchmark CLI

```bash
/tmp/gpu-library-suite-local-build/cpu-cuda/nvidia/c-cpp/cusolver/solver_cpu_bench \
  --size 64 --nrhs 2 --warmup 1 --repeat 1 --trials 1 \
  --scope end-to-end --verify true --abs-tolerance 1e-12 \
  --rel-tolerance 1e-10 --output - --format jsonl \
  --cpu-backend cpu-generic-lapacke --cpu-threads 48 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism threaded
```

Every cuSOLVER scope requires `repeat=1`. Use the backend name matching the
selected LAPACKE provider, and leave effective threads null unless the runtime
can establish them. Production suite thresholds come only from the effective
configuration.

## Timing scopes

`compute` allocates, transfers, and queries workspace before timing, restores
the canonical matrix/RHS, then times one `getrf` plus `getrs` pipeline.
`end-to-end` includes allocation, copies or OpenACC updates, handle/workspace
creation, solve, completion, solution, and both info-code retrievals. Cleanup
follows the end timestamp; warm-up, restoration, and verification are outside
timing.

## Verification

Both solution relative error and relative residual are checked with effective
absolute/relative tolerances. `getrf_info` and `getrs_info` must each be zero;
OpenACC initializes them to `-1` and explicitly copies them to the host, so a
missing copyback cannot masquerade as success.

## CPU backend and role

The configured provider is the production primary backend, named
`cpu-onemkl`, `cpu-openblas`, or `cpu-generic-lapacke`. Providers do not silently
replace one another, and requested thread count is not assumed to be effective.

## OpenACC notes

The OpenACC data region owns matrix, RHS, pivots, and info arrays. Small
`host_data use_device` regions expose them to cuSOLVER; only the workspace is a
manual CUDA allocation. End-to-end result retrieval includes explicit
`acc update self` for RHS and both info arrays before the measurement end.

## Known limitations and local validation

The algorithm does not switch numerical mode after hardware detection and will
report allocation or factorization failure directly. Local tests use fake
LAPACKE/CUDA interfaces and do not execute a production LAPACKE installation,
cuSOLVER runtime, NVHPC compiler, real GPU, or Pegasus job; those remain locally unverified.
Later saved real execution records are distinguished from these local tests in
[the validation report](../../../docs/VALIDATION_REPORT.md#saved-execution-evidence-reviewed-for-publication).

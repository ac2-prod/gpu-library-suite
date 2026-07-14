# cuRAND C/C++ examples and benchmarks

This directory compares uniform-double generation through C++17
`std::mt19937_64`, direct cuRAND, and OpenACC/cuRAND. Workload ownership belongs
to [`docs/PROJECT_SPECIFICATION.md`](../../../docs/PROJECT_SPECIFICATION.md),
and CLI, timing, and statistical verification belong to
[`docs/BENCHMARK_PROTOCOL.md`](../../../docs/BENCHMARK_PROTOCOL.md).

## Canonical sources and targets

| Role | Source | CMake target/executable |
| --- | --- | --- |
| CPU teaching | `examples/rand_cpu.cpp` | `rand_cpu` |
| CUDA teaching | `examples/rand_gpu.cu` | `rand_gpu` |
| OpenACC teaching | `examples/openacc_curand.cpp` | `openacc_curand` |
| CPU benchmark | `benchmarks/rand_cpu_bench.cpp` | `rand_cpu_bench` |
| CUDA benchmark | `benchmarks/rand_gpu_bench.cu` | `rand_gpu_bench` |
| OpenACC benchmark | `benchmarks/openacc_curand_bench.cpp` | `openacc_curand_bench` |

## Teaching default

Each teaching program generates `2^24` FP64 uniform values. The CPU uses
`std::mt19937_64`; GPU variants use `CURAND_RNG_PSEUDO_DEFAULT`. This compares
the same distribution/output-type task, not of identical random number
algorithms, so element identity across CPU and GPU is neither expected nor
verified.

## Dependencies

CPU targets need only a C++17 standard library. CUDA targets require CUDA
Runtime and cuRAND generator APIs. OpenACC targets require NVHPC OpenACC C++,
CUDA Runtime, and cuRAND. CMake compile-and-link probes exercise generator
creation and uniform-double generation before enabling GPU targets.

## Direct compile

```bash
c++ -std=c++17 nvidia/c-cpp/curand/examples/rand_cpu.cpp \
  -o /tmp/gpu-library-suite-local-build/rand_cpu-direct
nvcc -std=c++17 nvidia/c-cpp/curand/examples/rand_gpu.cu \
  -lcurand -o /tmp/gpu-library-suite-local-build/rand_gpu-direct
nvc++ -std=c++17 -acc -cuda \
  nvidia/c-cpp/curand/examples/openacc_curand.cpp -cudalib=curand \
  -o /tmp/gpu-library-suite-local-build/openacc_curand-direct
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
  --target rand_cpu rand_gpu rand_cpu_bench rand_gpu_bench

NVHPC_CUDA_HOME="${CUDA_TOOLKIT_ROOT}" \
cmake -S . -B /tmp/gpu-library-suite-local-build/openacc \
  -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="${NVHPC_GPU_TARGET}" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=ON
cmake --build /tmp/gpu-library-suite-local-build/openacc \
  --target openacc_curand openacc_curand_bench
```

## Benchmark CLI

```bash
/tmp/gpu-library-suite-local-build/cpu-cuda/nvidia/c-cpp/curand/rand_cpu_bench \
  --size 65536 --warmup 1 --repeat 2 --trials 1 --scope compute \
  --verify true --sigma-multiplier 6 --expected-mean 0.5 \
  --expected-second-central-moment 0.08333333333333333 \
  --output - --format jsonl --cpu-backend cpu-std-random-serial \
  --cpu-threads 48 --cpu-threads-effective 1 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism serial
```

Production suite execution passes all statistical constants from the effective
configuration and uses the runner as the only node raw-file writer.

## Timing scopes

`compute` creates and seeds the persistent generator before timing, restores
the configured offset, and times repeated generation into existing storage.
`end-to-end` includes storage and generator creation, configuration, generation,
completion, and host result retrieval for every repeat; destruction follows the
end timestamp. Canonical generator state is restored before each raw
trial/repeat as the scope requires.

## Verification

Both intervals share the inclusive range check `0 <= x <= 1`; metadata records
the CPU `[0,1)` and cuRAND `(0,1]` contracts. Required metrics are
`observed_min`, `observed_max`, `sample_mean`, and
`second_central_moment_about_half`. The configured bounds are
`sigma_multiplier*sqrt(1/(12*N))` about mean 0.5 and
`sigma_multiplier*sqrt(1/(180*N))` about second central moment 1/12.

## CPU backend and role

`cpu-std-random-serial` is the production primary serial CPU baseline. It is
plotted as **Serial CPU baseline**; a campaign may request 48 threads while the
effective count remains 1.

## OpenACC notes

The data region owns output storage and `host_data use_device` exposes it to
cuRAND. The target links `OpenACC::OpenACC_CXX`, `CUDA::cudart`, and
`CUDA::curand`; portable source does not hard-code a GPU architecture.

## Known limitations and local validation

The statistical test compares distribution behavior, not sequence equivalence,
and finite samples may fail according to the configured bound. Local tests run
the serial CPU generator and fake GPU syntax only. A real cuRAND runtime,
NVHPC compiler, GPU, and Pegasus campaign remain locally unverified.

# cuRAND C/C++ examples and benchmarks

The benchmark compares uniform-double generation with the production serial CPU
baseline (`std::mt19937_64`) and cuRAND's `CURAND_RNG_PSEUDO_DEFAULT`. It is a
comparison of the same distribution/output-type task, not of identical random
number algorithms; CPU/GPU element identity is neither expected nor verified.

The CPU distribution contract is `[0,1)` and cuRAND's is `(0,1]`. Both use the
common inclusive range check plus the configured mean and second-central-moment
statistical bounds owned by
[`docs/BENCHMARK_PROTOCOL.md`](../../../docs/BENCHMARK_PROTOCOL.md). Raw records
store the generator, seed, offset, interval contract, four required metrics, and
the verification sample count.

The CUDA and OpenACC targets require real CUDA Runtime and cuRAND generator
symbol probes. In the OpenACC implementation, the data region owns output
storage and `host_data use_device` exposes it to cuRAND.

The canonical source-stem/target/executable names are `rand_cpu`, `rand_gpu`,
`openacc_curand`, `rand_cpu_bench`, `rand_gpu_bench`, and
`openacc_curand_bench`. The dependency-free CPU teaching example can be compiled
directly from the repository root:

```bash
c++ -std=c++17 -O2 nvidia/c-cpp/curand/examples/rand_cpu.cpp \
  -o /tmp/gpu-library-suite-local-build/rand_cpu-direct
/tmp/gpu-library-suite-local-build/rand_cpu-direct
```

The CPU benchmark also has no external numerical-library dependency, but it
uses the shared benchmark support and generated build metadata. Build and run
it through the CPU-only CMake tree:

```bash
cmake --build /tmp/gpu-library-suite-local-build/cpu-only-validation \
  --target rand_cpu_bench
/tmp/gpu-library-suite-local-build/cpu-only-validation/nvidia/c-cpp/curand/rand_cpu_bench \
  --size 65536 --warmup 1 --repeat 2 --trials 1 --scope compute \
  --verify true --output - --format jsonl \
  --cpu-backend cpu-std-random-serial --cpu-threads 48
```

# Thrust C/C++ examples and benchmarks

The examples compute `sum(values[i]^2)` in FP64 with `transform_reduce`, as
defined by [`docs/PROJECT_SPECIFICATION.md`](../../../docs/PROJECT_SPECIFICATION.md).
The production CPU baseline is the serial C++17 STL implementation; an OpenMP
variant, when explicitly selected and available, remains a separately named
backend.

Direct CUDA uses `thrust::device_vector`. OpenACC owns the input allocation and
transfer, then translates the known device address with
`thrust::device_pointer_cast`. Thrust's synchronous algorithms normally need no
extra synchronization. The teaching example retains an explicit
`cudaDeviceSynchronize` only to make the device-completion boundary visible;
benchmark timing follows the shared protocol in
[`docs/BENCHMARK_PROTOCOL.md`](../../../docs/BENCHMARK_PROTOCOL.md).

NVHPC compilation and linking both receive the `-cuda` interoperation option,
and the target must pass a Thrust compile-and-link probe before it is enabled.

The canonical source-stem/target/executable names are `reduce_cpu`,
`reduce_gpu`, `openacc_thrust`, `reduce_cpu_bench`, `reduce_gpu_bench`, and
`openacc_thrust_bench`. The dependency-free serial CPU teaching example can be
compiled directly from the repository root:

```bash
c++ -std=c++17 -O2 nvidia/c-cpp/thrust/examples/reduce_cpu.cpp \
  -o /tmp/gpu-library-suite-local-build/reduce_cpu-direct
/tmp/gpu-library-suite-local-build/reduce_cpu-direct
```

The benchmark uses shared support and generated build metadata, so build and
run it through the CPU-only CMake tree:

```bash
cmake --build /tmp/gpu-library-suite-local-build/cpu-only-validation \
  --target reduce_cpu_bench
/tmp/gpu-library-suite-local-build/cpu-only-validation/nvidia/c-cpp/thrust/reduce_cpu_bench \
  --size 65536 --warmup 1 --repeat 2 --trials 1 --scope compute \
  --verify true --output - --format jsonl \
  --cpu-backend cpu-stl-serial --cpu-threads 48
```

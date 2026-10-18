# rocThrust C/C++

This family implements FP64 transform-reduce of `sum(values[i]^2)` for CPU,
HIP and OpenMP target. Sirius builds, three teaching examples and all 96
numerical smoke records passed, with zero reduction error. See
[the recorded source snapshot and run evidence](../../../jobs/sirius/README.md#focused-numerical-smoke).
The tested CPU backend is serial STL; performance and later source changes
remain unverified.

| Purpose | CPU | HIP | OpenMP target |
| --- | --- | --- | --- |
| Teaching | `examples/reduce_cpu.cpp` | `examples/reduce_gpu.cpp` | `examples/openmp_rocthrust.cpp` + `examples/rocthrust_wrapper.hip` |
| Measurement | `benchmarks/reduce_cpu_bench.cpp` | `benchmarks/reduce_gpu_bench.cpp` | `benchmarks/openmp_rocthrust_bench.cpp` + `examples/rocthrust_wrapper.hip` |

Teaching uses `1 << 24` doubles, all 1.0, so the expected result is the element
count. The CPU backend is C++17 STL `std::transform_reduce`; an explicitly
selected `GPU_SUITE_CPU_THRUST_BACKEND=OPENMP` build instead exposes `cpu-openmp`.
It does not silently substitute a different CPU backend. HIP uses
`thrust::hip_rocprim::par`, which synchronously returns a host scalar.

OpenMP interoperation is the sole two-translation-unit C/C++ teaching exception.
Compile `rocthrust_wrapper.hip` with HIP and the caller with OpenMP target support,
then explicitly link both objects with the matching HIP runtime. For example,
after supplying the actual target flags, include paths and link flags:

```sh
hipcc -std=c++17 $HIP_TARGET_FLAGS -c examples/rocthrust_wrapper.hip -o rocthrust_wrapper.o
"$OMP_CXX" -std=c++17 $OMP_TARGET_FLAGS $HIP_INCLUDE_FLAGS -c examples/openmp_rocthrust.cpp -o openmp_rocthrust.o
"$OMP_CXX" $OMP_TARGET_FLAGS openmp_rocthrust.o rocthrust_wrapper.o $HIP_LINK_FLAGS -o openmp_rocthrust
```

`HIP_TARGET_FLAGS` must explicitly select the same externally chosen GPU target
as `OMP_TARGET_FLAGS`; do not leave architecture selection to device discovery.

`rocthrust_wrapper.hip` is the canonical filename everywhere. Its C ABI accepts
an existing device mapping and a host result address, catches all exceptions,
and never frees or retains the caller's buffer. The OpenMP caller rejects host
fallback and validates that the mapping belongs to the requested HIP device.
Native CMake compiles the wrapper as a distinct HIP object and explicitly links
it into the OpenMP caller with the OpenMP C++ compiler as the final link driver.
Its configure check compiles the canonical wrapper and an OpenMP caller together
and verifies this mixed link without running the executable. The probe log is
`CMakeFiles/GPU_SUITE_ROCTHRUST_OPENMP_LINKS/output.log` in the build tree.
The two translation units require separate recorded compile conditions.
`hipcc` is permitted for manual builds, not fixed as the native CMake HIP compiler.

Benchmark compute timing retains device input and measures repeated synchronous
reductions. End-to-end timing includes allocation/mapping, input transfer and
the scalar-producing reduction; explicit deallocation occurs after timing.
Input is immutable, and each reduction starts with the identity 0.0. Verification
checks the expected count with configured absolute/relative tolerances and
rejects nonfinite output. Sizes must fit host containers, byte counts and
iterator difference types before allocation.

See [PROJECT_SPECIFICATION](../../../docs/PROJECT_SPECIFICATION.md),
[BENCHMARK_PROTOCOL](../../../docs/BENCHMARK_PROTOCOL.md), and
[Sirius operations](../../../jobs/sirius/README.md) for their respective contracts.
The [official HIP policy documentation](https://rocm.docs.amd.com/projects/rocThrust/en/docs-7.1.0/reference/rocThrust-hip-policies.html)
defines the synchronous `par` policy used here. Local host doubles test ownership,
exception handling and verification; they do not demonstrate GPU execution.

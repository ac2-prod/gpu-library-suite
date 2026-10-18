# rocRAND C/C++

This family implements the CPU, HIP and OpenMP target forms of the canonical
FP64 uniform-generation task. Sirius builds, three teaching examples and all
96 numerical smoke records passed for CPU, HIP and OpenMP target. See
[the recorded source snapshot and run evidence](../../../jobs/sirius/README.md#focused-numerical-smoke).
The tested CPU backend is serial; performance and later source changes remain
unverified.

| Purpose | CPU | HIP | OpenMP target |
| --- | --- | --- | --- |
| Teaching | `examples/rand_cpu.cpp` | `examples/rand_gpu.cpp` | `examples/openmp_rocrand.cpp` |
| Measurement | `benchmarks/rand_cpu_bench.cpp` | `benchmarks/rand_gpu_bench.cpp` | `benchmarks/openmp_rocrand_bench.cpp` |

Each teaching program is standalone. It generates `1 << 24` doubles with seed
1234 and offset 0. CPU uses `std::mt19937_64` with `[0,1)` uniform distribution;
GPU uses `ROCRAND_RNG_PSEUDO_DEFAULT` and default ordering, with `(0,1]` output.
**Same distribution and output type task; different RNG algorithms.** No
elementwise equality between CPU and GPU streams is required.

The benchmark records seed, offset, ordering, actual generator selector and
distribution interval. It checks range, mean and second central moment using
the configuration's statistical thresholds, over the final retrieved sample.
Compute scope keeps generator and output allocation, explicitly resets and
initializes generator state after warm-up and before each trial, then measures
generation plus completion. Repeats advance that trial's stream. End-to-end
creates, seeds, initializes, generates and retrieves on each repeat; explicit
generator destruction and allocation release occur after the timing endpoint.

OpenMP owns its mapped output; `use_device_ptr` supplies its actual device
pointer to rocRAND. A target-region check rejects host fallback and HIP pointer
attributes must match the requested GPU. The library never frees OpenMP-owned
memory. One HIP completion boundary follows a generation sequence.

Manual teaching builds may use `hipcc` (for example,
`hipcc examples/rand_gpu.cpp -lrocrand -o rand_gpu`). Native CMake HIP instead
uses a supported Clang. OpenMP requires the compiler's target flags and matching
HIP/rocRAND libraries; architecture and installation paths are explicit build
inputs, never embedded in portable code. CPU-only builds need only C++17.

The authoritative problem and timing contracts are
[PROJECT_SPECIFICATION](../../../docs/PROJECT_SPECIFICATION.md) and
[BENCHMARK_PROTOCOL](../../../docs/BENCHMARK_PROTOCOL.md). Site approval and
scheduler requirements belong to [Sirius operations](../../../jobs/sirius/README.md).

API references: [rocRAND host API](https://rocm.docs.amd.com/projects/rocRAND/en/docs-7.1.0/doxygen/html/group__rocrandhost.html),
including explicit generator initialization after seed/offset resets.
Local tests use synthetic API doubles and are not GPU execution evidence.

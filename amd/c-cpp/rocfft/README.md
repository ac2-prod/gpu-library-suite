# rocFFT C/C++ examples and benchmarks

[English](README.md) | [日本語](README.ja.md)

This first AMD implementation compares FFTW3f CPU, HIP/rocFFT and OpenMP
target/rocFFT. Its recorded Sirius baseline passed all three builds, teaching
examples, and 96 basic smoke records including serial and two-thread CPU.
See [the exact evidence and limits](../../../docs/VALIDATION_REPORT.md#amd-rocfft-sirius-baseline).
Later source changes and the other five libraries require their own validation;
no performance/scaling claim follows from this small numerical smoke.

## Sources and targets

| Implementation | Teaching source / target | Benchmark source / target |
| --- | --- | --- |
| CPU | `examples/fft_cpu.c` / `fft_cpu` | `benchmarks/fft_cpu_bench.c` / `fft_cpu_bench` |
| HIP | `examples/fft_gpu.cpp` / `fft_gpu` | `benchmarks/fft_gpu_bench.cpp` / `fft_gpu_bench` |
| OpenMP target | `examples/openmp_rocfft.cpp` / `openmp_rocfft` | `benchmarks/openmp_rocfft_bench.cpp` / `openmp_rocfft_bench` |

Each example is a complete single source with direct library calls, FP32
interleaved complex out-of-place batched 1D C2C forward, `nfft=1024`,
`batch=4096`, and all-one input. CPU uses `fftwf_plan_many_dft`; OpenMP owns
mapped arrays and passes `use_device_ptr` addresses to rocFFT, waiting before
`map(from:)` completes. Canonical mathematics and filenames are owned by
[the project specification](../../../docs/PROJECT_SPECIFICATION.md#rocfft-cc-inventory-and-teaching-workload).

## Dependencies and build separation

CPU requires FFTW3f; `cpu-fftw-threaded` additionally requires its threads API
and library. CPU-only configuration needs no ROCm. HIP requires a native CMake
HIP-capable Clang, HIP Runtime and rocFFT. OpenMP requires the AMD compiler's
target-offload support plus HIP Runtime and rocFFT. Enablement uses real API
compile/link tests and reports unavailable targets with reasons. A header-only
or synthetic fixture pass is not a real toolchain or GPU pass.

Configure three **new, separate** build trees with common options
`-DGPU_SUITE_VENDOR=amd -DGPU_SUITE_SOURCE_LANGUAGE=c-cpp`,
`-DGPU_SUITE_BUILD_CUDA=OFF -DGPU_SUITE_BUILD_OPENACC=OFF`:

| Tree | CPU / HIP / OpenMP options | External compiler/target inputs |
| --- | --- | --- |
| CPU | `ON` / `OFF` / `OFF` | `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER`, `FFTW_ROOT` |
| HIP | `OFF` / `ON` / `OFF` | `CMAKE_HIP_COMPILER` (supported Clang), `CMAKE_HIP_ARCHITECTURES`, `GPU_SUITE_ROCM_ROOT` |
| OpenMP | `OFF` / `OFF` / `ON` | `CMAKE_CXX_COMPILER` (AMD Clang), `GPU_SUITE_AMD_GPU_TARGET`, `GPU_SUITE_ROCM_ROOT` |

The option names are `GPU_SUITE_BUILD_CPU`, `GPU_SUITE_BUILD_HIP`, and
`GPU_SUITE_BUILD_OPENMP`. Native HIP `.cpp` sources are explicitly compiled as
HIP. Do not set `CMAKE_HIP_COMPILER=hipcc`, use an unknown machine's `native`
architecture, or guess installation paths. Compiler choices, dependency roots,
architecture, runtime paths and thread counts remain explicit system inputs.
Targets/executables reside under each build's `amd/c-cpp/rocfft/` directory.

Manual teaching compilation may use `hipcc`, for example the following after
setting ordinary include/library search paths for the actual installation:

```sh
cc -std=c17 amd/c-cpp/rocfft/examples/fft_cpu.c -lfftw3f -lm -o "$CPU_EXAMPLE"
hipcc -std=c++17 amd/c-cpp/rocfft/examples/fft_gpu.cpp -lrocfft -o "$HIP_EXAMPLE"
"$OPENMP_CXX" -std=c++17 -fopenmp --offload-arch="$GPU_ARCH" \
  amd/c-cpp/rocfft/examples/openmp_rocfft.cpp -lrocfft -lamdhip64 -o "$OPENMP_EXAMPLE"
```

These generic local-toolchain examples are not permission to build on a Sirius
login node. Sirius uses the separately approved, allocated-compute-node
[build/smoke body](../../../jobs/sirius/README.md#next-rocfft-build-and-smoke-compute-node-only).

## Benchmark behavior

Use the existing CLI with `--size` (FFT length), `--batch`, `--scope compute`
or `end-to-end`, `--warmup`, `--repeat`, `--trials`, and explicit verification
thresholds. CPU backends are `cpu-fftw-serial` and `cpu-fftw-threaded`; no missing
threaded dependency is replaced by serial under its name. The existing common
runner owns node-level JSONL/CSV, configuration and manifest validation.
[The benchmark protocol](../../../docs/BENCHMARK_PROTOCOL.md#rocfft-cpuhipopenmp-contract)
owns timing and verification, and [the result schema](../../../docs/RESULT_SCHEMA.md#rocfft-additive-representation-and-campaign-identity)
owns AMD representation and provenance.

Compute prepares persistent plan, execution info, explicit rocFFT workspace
and arrays. rocFFT may overwrite out-of-place input, so CPU/HIP/OpenMP each
restore input/output outside timing before every repeat, time one FFT plus
completion, and sum those intervals. Wall timestamps may also span untimed
restoration. Warm-up restores before each call. End-to-end times each repeat's
plan/info setup, allocation, transfer, execution, completion and retrieval;
OpenMP includes data-region entry/exit and `map(from:)`. Host generation,
verification and explicit cleanup follow the existing untimed rules;
`rocfft_setup`/`rocfft_cleanup` run once per program outside both scopes. All
batches' DC/non-DC values and every nonfinite output are checked.

`--device` selects a HIP index; `OMP_DEFAULT_DEVICE` independently selects an
OpenMP index. Matching integers do not prove physical identity. The OpenMP
benchmark requires actual device execution and checks the mapped HIP pointer
attributes against the selected HIP device before timing. Set
`OMP_TARGET_OFFLOAD=MANDATORY` in real smoke. Mapped arrays must not be freed
with HIP. USM and stream interoperability are not assumed.

HIP driver/runtime evidence belongs in `parameters.runtime`; CUDA fields remain
null. The actual rocFFT library API supplies its version. Mathematical FFT
keys exclude runtime evidence, but campaign validation still binds source,
configuration, artifact compiler/build flags, binary, manifest and runtime
identity; different provenance cannot be mixed.

# GPU Library Suite

[English](README.md) | [日本語](README.ja.md)

Learn how to use GPU libraries in your application by comparing CPU, direct GPU-library calls, and directive-managed GPU data. The examples cover FFTs, matrix multiplication, sparse matrix-vector multiplication, linear solves, random generation, and reductions.

The repository includes NVIDIA C/C++, NVIDIA Fortran, AMD C/C++, and AMD Fortran. Start with `examples/`: small programs showing input preparation, library calls, and result checking. Use the separate `benchmarks/` when you need controlled performance measurements.

<a id="start-here"></a>

## Choose your implementation

Choose the column matching your GPU and language, then run one example. CPU-only examples need no GPU. General Linux usage does not require Pegasus or Sirius access. AMD Fortran has completed build, numerical validation and a six-node, two-wave performance evaluation on Sirius; see the [tested versions and limits](docs/VALIDATION_REPORT.md#amd-fortran-production-validation).

| Operation | NVIDIA C/C++: CUDA / OpenACC | NVIDIA Fortran: CUDA Fortran / OpenACC | AMD C/C++: HIP / OpenMP target | AMD Fortran: HIP / OpenMP target |
| --- | --- | --- | --- | --- |
| FFT | [cufft](nvidia/c-cpp/cufft/README.md) | [cufft](nvidia/fortran/cufft/README.md) | [rocfft](amd/c-cpp/rocfft/README.md) | [rocfft](amd/fortran/rocfft/examples) |
| BLAS | [cublas](nvidia/c-cpp/cublas/README.md) | [cublas](nvidia/fortran/cublas/README.md) | [rocblas](amd/c-cpp/rocblas/README.md) | [rocblas](amd/fortran/rocblas/examples) |
| Sparse | [cusparse](nvidia/c-cpp/cusparse/README.md) | [cusparse](nvidia/fortran/cusparse/README.md) | [rocsparse](amd/c-cpp/rocsparse/README.md) | [rocsparse](amd/fortran/rocsparse/examples) |
| Solver | [cusolver](nvidia/c-cpp/cusolver/README.md) | [cusolver](nvidia/fortran/cusolver/README.md) | [rocsolver](amd/c-cpp/rocsolver/README.md) | [rocsolver](amd/fortran/rocsolver/examples) |
| RAND | [curand](nvidia/c-cpp/curand/README.md) | [curand](nvidia/fortran/curand/README.md) | [rocrand](amd/c-cpp/rocrand/README.md) | [rocrand](amd/fortran/rocrand/examples) |
| Thrust | [thrust](nvidia/c-cpp/thrust/README.md) | [thrust](nvidia/fortran/thrust/README.md) | [rocthrust](amd/c-cpp/rocthrust/README.md) | [rocthrust](amd/fortran/rocthrust/examples) |

The [build and run guide](docs/BUILDING.md) builds only your chosen teaching target. See the [NVIDIA Fortran](nvidia/fortran/README.md) and [AMD Fortran](amd/fortran/README.md) guides for language-specific requirements.

<a id="first-example"></a>

## Try one CPU example

Run this in Bash from the repository root. It needs only a C++17 compiler and sums the squares of 2²⁴ ones. The input array uses 128 MiB.
```bash
EXAMPLE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-example.XXXXXX")" || exit 1
c++ -std=c++17 -O2 nvidia/c-cpp/thrust/examples/reduce_cpu.cpp \
  -o "$EXAMPLE_DIR/reduce_cpu" && "$EXAMPLE_DIR/reduce_cpu"
```
Success means a reported square sum of `16777216` and exit status 0. Stop on compilation failure or a nonzero exit. Choose GPU or Fortran examples from the table above. On a shared cluster, perform builds and execution inside a compute allocation as required by that site.

<a id="using-the-examples"></a>

## Use the example in your application

1. Compare CPU and GPU `examples/` and identify the input format, precision, and operation.
2. Replace the sample input with your own data; preserve the required array layout, dimensions, leading dimensions, and CSR index base.
3. Decide who owns allocations and ensure completion before retrieving results. OpenACC / OpenMP target variants still call the GPU libraries; their main distinction is data management.
4. For repeated operations with compatible shapes, reuse arrays, plans, handles, and descriptors where the API permits it.
5. Check return codes, solver `info`, and numerical errors. The all-ones sample reference is not a reference for arbitrary application data.

Each library README explains the application-specific choices. C/C++ examples normally use one source file; rocThrust OpenMP also needs `rocthrust_wrapper.hip`. Fortran matrix helpers and the Thrust wrapper are explicit compile/link inputs.

<a id="requirements-and-cpu-only-build"></a>

## Build requirements

Use CMake 3.20+ (3.21+ for native HIP and rocThrust OpenMP), C17, and C++17. Fortran CPU supports GNU Fortran or NVHPC; CUDA Fortran requires NVHPC. GPU execution needs a compatible GPU, driver, and CUDA or ROCm installation; directive variants also need the corresponding offload compiler.

CPU numerical dependencies depend on the operation. Serial RAND / Thrust CPU examples need no external numerical library. Missing optional dependencies disable affected targets, so check the target status and build result rather than configure exit alone. See [build instructions](docs/BUILDING.md) for dependencies and explicit CPU backend choices.

<a id="benchmark-configuration-and-output"></a>

## Measure after the example works

Teaching examples have no benchmark CLI. `benchmarks/` controls size, warm-up, repeats, trials, and verification. Figures show a **data-resident library panel** and a **one-shot end-to-end panel**. The internal CLI/schema scope `compute` means library. End-to-end includes preparation, data management, and result retrieval, but does not necessarily include first runtime startup or every cleanup operation; consult the [timing boundaries](docs/BENCHMARK_PROTOCOL.md).

Use the [NVIDIA measurement workflow](docs/PORTABILITY.md#measuring-on-your-own-system), [Fortran measurement inputs](nvidia/fortran/README.md#first-checks-and-own-measurements), or [AMD Sirius procedure](jobs/sirius/README.md) as appropriate. Adapt paths, modules, and GPU targets to your actual system.

<a id="cpu-baselines-and-verification-notes"></a>

## Interpret the comparison

Check the CPU backend and actual parallelism. Requested threads do not prove effective worker count. C/C++ RAND / Thrust reference series use a **single thread**; Fortran uses serial intrinsic RNG / sum. RNG comparisons use the **same distribution and output type**, but **RNG algorithms differ**, so they are not identical-algorithm speedups.

AMD FFT timing accommodates required input restoration and differs from NVIDIA FFT within the repeat loop; see [rocFFT timing](amd/c-cpp/rocfft/README.md#timing-scopes).

<a id="validation-status"></a>
<a id="publication-scope"></a>

## What is included and verified

Published material includes NVIDIA and AMD sources in both C/C++ and Fortran, together with configurations, tools, tests, and documentation. Saved raw measurements, execution logs, PPTX files, and private review material are not part of the checkout. Using the examples does not require the authors’ saved data or conversation history.

Builds, numerical checks, and measurements have been recorded for NVIDIA on Pegasus and AMD on Sirius. See the [validation report](docs/VALIDATION_REPORT.md) and each family’s recorded coverage. These establish evidence for those versions/environments, not a guarantee for another installation or later edits. Unobserved effective CPU workers and complete loaded-library identity remain explicit limitations.

Maintained under `ac2-prod`; this is not an official HAIRDESC repository.

<a id="documentation-map"></a>
<a id="pegasus-execution-boundary"></a>
<a id="reference"></a>

## Further reading

- [Build and run](docs/BUILDING.md): minimal examples, dependencies, and troubleshooting.
- [Measurement workflow](docs/PORTABILITY.md), [benchmark protocol](docs/BENCHMARK_PROTOCOL.md): measure and interpret your own application costs.
- [Problem/source specification](docs/PROJECT_SPECIFICATION.md), [result schema](docs/RESULT_SCHEMA.md): detailed references.
- [Pegasus](jobs/pegasus/README.md), [Sirius](jobs/sirius/README.md): operations for users of those systems.
- [Validation history](docs/VALIDATION_REPORT.md), [implementation plan](docs/IMPLEMENTATION_PLAN.md), [decisions](docs/DECISIONS.md): maintenance evidence.

<a id="what-is-implemented"></a>

<details>
<summary>NVIDIA C/C++ source inventory</summary>

| Library | CPU | CUDA | OpenACC | CPU benchmark | CUDA benchmark | OpenACC benchmark |
| --- | --- | --- | --- | --- | --- | --- |
| cufft | [fft_cpu](nvidia/c-cpp/cufft/examples/fft_cpu.c) | [fft_gpu](nvidia/c-cpp/cufft/examples/fft_gpu.cu) | [openacc_cufft](nvidia/c-cpp/cufft/examples/openacc_cufft.cpp) | [fft_cpu_bench](nvidia/c-cpp/cufft/benchmarks/fft_cpu_bench.c) | [fft_gpu_bench](nvidia/c-cpp/cufft/benchmarks/fft_gpu_bench.cu) | [openacc_cufft_bench](nvidia/c-cpp/cufft/benchmarks/openacc_cufft_bench.cpp) |
| cublas | [blas_cpu](nvidia/c-cpp/cublas/examples/blas_cpu.c) | [blas_gpu](nvidia/c-cpp/cublas/examples/blas_gpu.cu) | [openacc_cublas](nvidia/c-cpp/cublas/examples/openacc_cublas.cpp) | [blas_cpu_bench](nvidia/c-cpp/cublas/benchmarks/blas_cpu_bench.c) | [blas_gpu_bench](nvidia/c-cpp/cublas/benchmarks/blas_gpu_bench.cu) | [openacc_cublas_bench](nvidia/c-cpp/cublas/benchmarks/openacc_cublas_bench.cpp) |
| cusparse | [sparse_cpu](nvidia/c-cpp/cusparse/examples/sparse_cpu.c) | [sparse_gpu](nvidia/c-cpp/cusparse/examples/sparse_gpu.cu) | [openacc_cusparse](nvidia/c-cpp/cusparse/examples/openacc_cusparse.cpp) | [sparse_cpu_bench](nvidia/c-cpp/cusparse/benchmarks/sparse_cpu_bench.c) | [sparse_gpu_bench](nvidia/c-cpp/cusparse/benchmarks/sparse_gpu_bench.cu) | [openacc_cusparse_bench](nvidia/c-cpp/cusparse/benchmarks/openacc_cusparse_bench.cpp) |
| cusolver | [solver_cpu](nvidia/c-cpp/cusolver/examples/solver_cpu.c) | [solver_gpu](nvidia/c-cpp/cusolver/examples/solver_gpu.cu) | [openacc_cusolver](nvidia/c-cpp/cusolver/examples/openacc_cusolver.cpp) | [solver_cpu_bench](nvidia/c-cpp/cusolver/benchmarks/solver_cpu_bench.c) | [solver_gpu_bench](nvidia/c-cpp/cusolver/benchmarks/solver_gpu_bench.cu) | [openacc_cusolver_bench](nvidia/c-cpp/cusolver/benchmarks/openacc_cusolver_bench.cpp) |
| curand | [rand_cpu](nvidia/c-cpp/curand/examples/rand_cpu.cpp) | [rand_gpu](nvidia/c-cpp/curand/examples/rand_gpu.cu) | [openacc_curand](nvidia/c-cpp/curand/examples/openacc_curand.cpp) | [rand_cpu_bench](nvidia/c-cpp/curand/benchmarks/rand_cpu_bench.cpp) | [rand_gpu_bench](nvidia/c-cpp/curand/benchmarks/rand_gpu_bench.cu) | [openacc_curand_bench](nvidia/c-cpp/curand/benchmarks/openacc_curand_bench.cpp) |
| thrust | [reduce_cpu](nvidia/c-cpp/thrust/examples/reduce_cpu.cpp) | [reduce_gpu](nvidia/c-cpp/thrust/examples/reduce_gpu.cu) | [openacc_thrust](nvidia/c-cpp/thrust/examples/openacc_thrust.cpp) | [reduce_cpu_bench](nvidia/c-cpp/thrust/benchmarks/reduce_cpu_bench.cpp) | [reduce_gpu_bench](nvidia/c-cpp/thrust/benchmarks/reduce_gpu_bench.cu) | [openacc_thrust_bench](nvidia/c-cpp/thrust/benchmarks/openacc_thrust_bench.cpp) |

</details>

<details>
<summary>Reference details for existing measurements and tools</summary>

The existing Pegasus CPU FFT series is `cpu-fftw-threaded`. The one-node Pegasus pilot and production conditions are recorded in the [operating specification](docs/PEGASUS_EXECUTION.md). `tools/run_suite.py` is the only writer of the node-level raw-result file. No speedup, throughput, reuse-count, amortized, or generic extra figure is produced by the existing publication plotter. No real CUDA GPU execution is established by Mac local/fixture checks; those checks are distinct from saved real-machine validation.

</details>

<a id="license"></a>

## License

Original content in this repository is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Ryohei Kobayashi

Existing third-party copyright notices and license terms remain applicable.
This license does not relicense external libraries or separately distributed
teaching slides and videos.

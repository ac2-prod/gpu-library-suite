# cuRAND — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

See the [shared Fortran guide](../README.md) for dependencies, separate CMake builds, numerical flags and validation limits. GPU execution remains unverified.

| Implementation | Example | Benchmark |
| --- | --- | --- |
| CPU | [rand_cpu.f90](examples/rand_cpu.f90) | [rand_cpu_bench.f90](benchmarks/rand_cpu_bench.f90) |
| CUDA | [rand_gpu.f90](examples/rand_gpu.f90) | [rand_gpu_bench.f90](benchmarks/rand_gpu_bench.f90) |
| OpenACC | [openacc_curand.f90](examples/openacc_curand.f90) | [openacc_curand_bench.f90](benchmarks/openacc_curand_bench.f90) |

FP64 uniform random output. CPU uses random_seed(size=...) then a seed vector filled with 1234 and random_number, never the C++ MT engine. The algorithm/stream depends on the Fortran compiler; only the distribution task is compared with CURAND_RNG_PSEUDO_DEFAULT. CPU values are in [0,1), GPU in (0,1]. Verification checks finite range, mean and second central moment with configured sigma bounds. Compute advances the stream across repeats and resets seed/offset before each trial; E2E creates/resets each pipeline. The CPU benchmark requires seed<=INT_MAX and implements offset by consuming the intrinsic stream in bounded chunks.

## Direct compilation

Prefer the [shared CMake recipe](../README.md#separate-gpu-build-trees). These are NVHPC/Linux commands from the repository root in the same Toolkit environment, not a claim of an executed GPU build. Supply real provider include/library directories. Helpers/wrappers are explicit link inputs; no dependency installation is performed.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")"
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set the actual target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" nvidia/fortran/curand/examples/rand_cpu.f90 \
  -o "$DIRECT_BUILD/rand_cpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=curand \
  nvidia/fortran/curand/examples/rand_gpu.f90 -o "$DIRECT_BUILD/rand_gpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=curand \
  nvidia/fortran/curand/examples/openacc_curand.f90 -o "$DIRECT_BUILD/openacc_curand"
```

## Examples and a small benchmark

Use the two build variables from the CMake recipe. Require exit 0 and verification PASS for each example. For direct compilation use the same names inside DIRECT_BUILD.

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/curand/rand_cpu"
"$CPU_CUDA_BUILD/nvidia/fortran/curand/rand_gpu"
"$OPENACC_BUILD/nvidia/fortran/curand/openacc_curand"
```

RUN_DIR must be fresh. This standalone CPU diagnostic is not a campaign input:

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/curand/rand_cpu_bench" \
  --size 65536 --seed 1234 --offset 0 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --cpu-threads 1 --cpu-backend cpu-fortran-random-serial \
  --cpu-parallelism serial --cpu-threads-effective 1 \
  --output "$RUN_DIR/standalone-curand-cpu-compute.jsonl" --format jsonl
```

For E2E use --scope end-to-end and a fresh output name. For GPU diagnostics select the corresponding benchmark path in the table (CUDA under CPU_CUDA_BUILD, OpenACC under OPENACC_BUILD), remove CPU-only arguments and preserve the same problem/verification/repetition settings. Start E2E diagnostics with repeat=1.

[Own measurement → verification → aggregation → figures](../README.md#first-checks-and-own-measurements) uses the existing runner. The [benchmark protocol](../../../docs/BENCHMARK_PROTOCOL.md) owns scope, restoration and tolerances. Retain the Fortran language identity and do not mix in C/C++ measurements.

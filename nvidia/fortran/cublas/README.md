# cuBLAS — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

See the [shared Fortran guide](../README.md) for dependencies, separate CMake builds, numerical flags and validation limits. GPU execution remains unverified.

| Implementation | Example | Benchmark |
| --- | --- | --- |
| CPU | [blas_cpu.f90](examples/blas_cpu.f90) | [blas_cpu_bench.f90](benchmarks/blas_cpu_bench.f90) |
| CUDA | [blas_gpu.f90](examples/blas_gpu.f90) | [blas_gpu_bench.f90](benchmarks/blas_gpu_bench.f90) |
| OpenACC | [openacc_cublas.f90](examples/openacc_cublas.f90) | [openacc_cublas_bench.f90](benchmarks/openacc_cublas_bench.f90) |

FP64 column-major C=alpha*A*B+beta*C, with all-ones A/B/C. The CPU calls Fortran dgemm through oneMKL; GPU variants use cublas_v2. C is an input as well as an output, so OpenACC uses copy, not copyout-only. Compute repeats accumulate the specified recurrence; E2E restores the original C outside every interval. Verification uses the matching recurrence. Benchmark --m/--n/--k also supports nonsquare problems.

## Direct compilation

Prefer the [shared CMake recipe](../README.md#separate-gpu-build-trees). These are NVHPC/Linux commands from the repository root in the same Toolkit environment, not a claim of an executed GPU build. Supply real provider include/library directories. Helpers/wrappers are explicit link inputs; no dependency installation is performed.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")"
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set the actual target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" -module "$DIRECT_BUILD" \
  -I"${MKLROOT:?Set the installed oneMKL root}/include" \
  nvidia/fortran/cublas/examples/blas_cpu.f90 \
  -L"${MKL_LIBDIR:?Set the installed oneMKL library directory}" -lmkl_rt -lpthread -lm -ldl \
  -o "$DIRECT_BUILD/blas_cpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=cublas \
  nvidia/fortran/cublas/examples/blas_gpu.f90 -o "$DIRECT_BUILD/blas_gpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=cublas \
  nvidia/fortran/cublas/examples/openacc_cublas.f90 -o "$DIRECT_BUILD/openacc_cublas"
```

## Examples and a small benchmark

Use the two build variables from the CMake recipe. Require exit 0 and verification PASS for each example. For direct compilation use the same names inside DIRECT_BUILD.

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/cublas/blas_cpu"
"$CPU_CUDA_BUILD/nvidia/fortran/cublas/blas_gpu"
"$OPENACC_BUILD/nvidia/fortran/cublas/openacc_cublas"
```

RUN_DIR must be fresh. This standalone CPU diagnostic is not a campaign input:

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/cublas/blas_cpu_bench" \
  --size 32 --alpha 1 --beta 1 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --cpu-threads 1 --cpu-backend cpu-onemkl \
  --cpu-parallelism threaded \
  --output "$RUN_DIR/standalone-cublas-cpu-compute.jsonl" --format jsonl
```

For E2E use --scope end-to-end and a fresh output name. For GPU diagnostics select the corresponding benchmark path in the table (CUDA under CPU_CUDA_BUILD, OpenACC under OPENACC_BUILD), remove CPU-only arguments and preserve the same problem/verification/repetition settings. Start E2E diagnostics with repeat=1.

[Own measurement → verification → aggregation → figures](../README.md#first-checks-and-own-measurements) uses the existing runner. The [benchmark protocol](../../../docs/BENCHMARK_PROTOCOL.md) owns scope, restoration and tolerances. Retain the Fortran language identity and do not mix in C/C++ measurements.

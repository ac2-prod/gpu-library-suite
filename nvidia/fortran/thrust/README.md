# Thrust — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

See the [shared Fortran guide](../README.md) for dependencies, separate CMake builds, numerical flags and validation limits. GPU execution remains unverified.

| Implementation | Example | Benchmark |
| --- | --- | --- |
| CPU | [reduce_cpu.f90](examples/reduce_cpu.f90) | [reduce_cpu_bench.f90](benchmarks/reduce_cpu_bench.f90) |
| CUDA | [reduce_gpu.f90](examples/reduce_gpu.f90) | [reduce_gpu_bench.f90](benchmarks/reduce_gpu_bench.f90) |
| OpenACC | [openacc_thrust.f90](examples/openacc_thrust.f90) | [openacc_thrust_bench.f90](benchmarks/openacc_thrust_bench.f90) |

FP64 sum of squares of all-ones input. CPU keeps Fortran sum(values*values), without an OpenMP loop, STL substitution or automatic GPU offload. Both GPU callers use the supplied extern-C thrust_square_sum through bind(C); the wrapper contains transform_reduce and is shared, not duplicated as a second algorithm. A thrown C++ exception is diagnosed and becomes NaN, which Fortran rejects. Benchmark synchronization is at timing boundaries. Teachings retain their original synchronization style; a host scalar return is not justification to remove benchmark completion checks.

## Direct compilation

Prefer the [shared CMake recipe](../README.md#separate-gpu-build-trees). These are NVHPC/Linux commands from the repository root in the same Toolkit environment, not a claim of an executed GPU build. Supply real provider include/library directories. Helpers/wrappers are explicit link inputs; no dependency installation is performed.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")"
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set the actual target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" nvidia/fortran/thrust/examples/reduce_cpu.f90 \
  -o "$DIRECT_BUILD/reduce_cpu"
nvcc -std=c++17 -O3 -arch="${NVCC_ARCH:?Set the NVCC sm target}" \
  -I"${CUDA_CCCL_INCLUDE_DIR:?Set the external Toolkit directory containing thrust headers}" \
  -c nvidia/fortran/thrust/examples/thrust_wrapper.cu -o "$DIRECT_BUILD/thrust_wrapper.o"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -c++libs \
  nvidia/fortran/thrust/examples/reduce_gpu.f90 "$DIRECT_BUILD/thrust_wrapper.o" -o "$DIRECT_BUILD/reduce_gpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -c++libs \
  nvidia/fortran/thrust/examples/openacc_thrust.f90 "$DIRECT_BUILD/thrust_wrapper.o" -o "$DIRECT_BUILD/openacc_thrust"
```

## Examples and a small benchmark

Use the two build variables from the CMake recipe. Require exit 0 and verification PASS for each example. For direct compilation use the same names inside DIRECT_BUILD.

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/thrust/reduce_cpu"
"$CPU_CUDA_BUILD/nvidia/fortran/thrust/reduce_gpu"
"$OPENACC_BUILD/nvidia/fortran/thrust/openacc_thrust"
```

RUN_DIR must be fresh. This standalone CPU diagnostic is not a campaign input:

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/thrust/reduce_cpu_bench" \
  --size 4096 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --cpu-threads 1 --cpu-backend cpu-fortran-sum-serial \
  --cpu-parallelism serial --cpu-threads-effective 1 \
  --output "$RUN_DIR/standalone-thrust-cpu-compute.jsonl" --format jsonl
```

For E2E use --scope end-to-end and a fresh output name. For GPU diagnostics select the corresponding benchmark path in the table (CUDA under CPU_CUDA_BUILD, OpenACC under OPENACC_BUILD), remove CPU-only arguments and preserve the same problem/verification/repetition settings. Start E2E diagnostics with repeat=1.

[Own measurement → verification → aggregation → figures](../README.md#first-checks-and-own-measurements) uses the existing runner. The [benchmark protocol](../../../docs/BENCHMARK_PROTOCOL.md) owns scope, restoration and tolerances. Retain the Fortran language identity and do not mix in C/C++ measurements.

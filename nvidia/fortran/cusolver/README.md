# cuSOLVER — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

See the [shared Fortran guide](../README.md) for dependencies, separate CMake builds, numerical flags, and the scope and limits of completed Pegasus validation.

| Implementation | Example | Benchmark |
| --- | --- | --- |
| CPU | [solver_cpu.f90](examples/solver_cpu.f90) | [solver_cpu_bench.f90](benchmarks/solver_cpu_bench.f90) |
| CUDA | [solver_gpu.f90](examples/solver_gpu.f90) | [solver_gpu_bench.f90](benchmarks/solver_gpu_bench.f90) |
| OpenACC | [openacc_cusolver.f90](examples/openacc_cusolver.f90) | [openacc_cusolver_bench.f90](benchmarks/openacc_cusolver_bench.f90) |

FP64 AX=B, A=N*I+ones and every B element 2*N, so the known solution is one. The external helper completes the omitted teaching input without changing the problem. CPU teaching uses dgesv; the benchmark deliberately uses dgetrf then dgetrs to match GPU stages. CUDA/OpenACC retain separate getrf/getrs info values. Examples inspect factorization info before solve; benchmarks keep the two infos and inspect after the timed synchronization/download, without adding a per-stage timing barrier. Verification checks solution error and original-system residual. Repeat must be 1 in both scopes.

## Direct compilation

Prefer the [shared CMake recipe](../README.md#separate-gpu-build-trees). These are NVHPC/Linux commands from the repository root in the same Toolkit environment, not a claim of an executed GPU build. Supply real provider include/library directories. Helpers/wrappers are explicit link inputs; no dependency installation is performed.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")"
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set the actual target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" -module "$DIRECT_BUILD" \
  -I"${MKLROOT:?Set the installed oneMKL root}/include" \
  nvidia/fortran/cusolver/examples/solver_cpu.f90 common/fortran/make_dense_system.f90 \
  -L"${MKL_LIBDIR:?Set the installed oneMKL library directory}" -lmkl_rt -lpthread -lm -ldl \
  -o "$DIRECT_BUILD/solver_cpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=cusolver \
  nvidia/fortran/cusolver/examples/solver_gpu.f90 common/fortran/make_dense_system.f90 -o "$DIRECT_BUILD/solver_gpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=cusolver \
  nvidia/fortran/cusolver/examples/openacc_cusolver.f90 common/fortran/make_dense_system.f90 -o "$DIRECT_BUILD/openacc_cusolver"
```

## Examples and a small benchmark

Use the two build variables from the CMake recipe. Require exit 0 and verification PASS for each example. For direct compilation use the same names inside DIRECT_BUILD.

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/cusolver/solver_cpu"
"$CPU_CUDA_BUILD/nvidia/fortran/cusolver/solver_gpu"
"$OPENACC_BUILD/nvidia/fortran/cusolver/openacc_cusolver"
```

RUN_DIR must be fresh. This standalone CPU diagnostic is not a campaign input:

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/cusolver/solver_cpu_bench" \
  --size 32 --nrhs 2 --warmup 1 --repeat 1 --trials 1 \
  --scope compute --verify true --cpu-threads 1 --cpu-backend cpu-onemkl \
  --cpu-parallelism threaded \
  --output "$RUN_DIR/standalone-cusolver-cpu-compute.jsonl" --format jsonl
```

For E2E use --scope end-to-end and a fresh output name. For GPU diagnostics select the corresponding benchmark path in the table (CUDA under CPU_CUDA_BUILD, OpenACC under OPENACC_BUILD), remove CPU-only arguments and preserve the same problem/verification/repetition settings. Repeat must always be 1.

[Own measurement → verification → aggregation → figures](../README.md#first-checks-and-own-measurements) uses the existing runner. The [benchmark protocol](../../../docs/BENCHMARK_PROTOCOL.md) owns scope, restoration and tolerances. Retain the Fortran language identity and do not mix in C/C++ measurements.

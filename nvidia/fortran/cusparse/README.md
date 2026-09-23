# cuSPARSE — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

See the [shared Fortran guide](../README.md) for dependencies, separate CMake builds, numerical flags and validation limits. GPU execution remains unverified.

| Implementation | Example | Benchmark |
| --- | --- | --- |
| CPU | [sparse_cpu.f90](examples/sparse_cpu.f90) | [sparse_cpu_bench.f90](benchmarks/sparse_cpu_bench.f90) |
| CUDA | [sparse_gpu.f90](examples/sparse_gpu.f90) | [sparse_gpu_bench.f90](benchmarks/sparse_gpu_bench.f90) |
| OpenACC | [openacc_cusparse.f90](examples/openacc_cusparse.f90) | [openacc_cusparse_bench.f90](benchmarks/openacc_cusparse_bench.f90) |

FP64 2D five-point Poisson SpMV. The external helper implements the existing diagonal-4/neighbor-minus-1 problem, ordered one-based CSR, N=nx*ny and nnz=5*nx*ny-2*nx-2*ny. It handles boundaries without adding wraparound edges. CPU uses oneMKL Sparse BLAS and GPU uses generic cuSPARSE descriptors with INDEX_BASE_ONE. Initial x=y=1 is preserved; y is copied in and out under OpenACC. Workspace is explicitly device allocated. The benchmark times descriptor analysis/workspace setup only in E2E.

## Direct compilation

Prefer the [shared CMake recipe](../README.md#separate-gpu-build-trees). These are NVHPC/Linux commands from the repository root in the same Toolkit environment, not a claim of an executed GPU build. Supply real provider include/library directories. Helpers/wrappers are explicit link inputs; no dependency installation is performed.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")"
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set the actual target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" -module "$DIRECT_BUILD" \
  -I"${MKLROOT:?Set the installed oneMKL root}/include" \
  nvidia/fortran/cusparse/examples/sparse_cpu.f90 common/fortran/make_poisson2d_csr.f90 \
  -L"${MKL_LIBDIR:?Set the installed oneMKL library directory}" -lmkl_rt -lpthread -lm -ldl \
  -o "$DIRECT_BUILD/sparse_cpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=cusparse \
  nvidia/fortran/cusparse/examples/sparse_gpu.f90 common/fortran/make_poisson2d_csr.f90 -o "$DIRECT_BUILD/sparse_gpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=cusparse \
  nvidia/fortran/cusparse/examples/openacc_cusparse.f90 common/fortran/make_poisson2d_csr.f90 -o "$DIRECT_BUILD/openacc_cusparse"
```

## Examples and a small benchmark

Use the two build variables from the CMake recipe. Require exit 0 and verification PASS for each example. For direct compilation use the same names inside DIRECT_BUILD.

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/cusparse/sparse_cpu"
"$CPU_CUDA_BUILD/nvidia/fortran/cusparse/sparse_gpu"
"$OPENACC_BUILD/nvidia/fortran/cusparse/openacc_cusparse"
```

RUN_DIR must be fresh. This standalone CPU diagnostic is not a campaign input:

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/cusparse/sparse_cpu_bench" \
  --nx 8 --ny 8 --alpha 1 --beta 1 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --cpu-threads 1 --cpu-backend cpu-onemkl \
  --cpu-parallelism threaded \
  --output "$RUN_DIR/standalone-cusparse-cpu-compute.jsonl" --format jsonl
```

For E2E use --scope end-to-end and a fresh output name. For GPU diagnostics select the corresponding benchmark path in the table (CUDA under CPU_CUDA_BUILD, OpenACC under OPENACC_BUILD), remove CPU-only arguments and preserve the same problem/verification/repetition settings. Start E2E diagnostics with repeat=1.

[Own measurement → verification → aggregation → figures](../README.md#first-checks-and-own-measurements) uses the existing runner. The [benchmark protocol](../../../docs/BENCHMARK_PROTOCOL.md) owns scope, restoration and tolerances. Retain the Fortran language identity and do not mix in C/C++ measurements.

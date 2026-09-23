# cuFFT — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

See the [shared Fortran guide](../README.md) for dependencies, separate CMake builds, numerical flags and validation limits. GPU execution remains unverified.

| Implementation | Example | Benchmark |
| --- | --- | --- |
| CPU | [fft_cpu.f90](examples/fft_cpu.f90) | [fft_cpu_bench.f90](benchmarks/fft_cpu_bench.f90) |
| CUDA | [fft_gpu.f90](examples/fft_gpu.f90) | [fft_gpu_bench.f90](benchmarks/fft_gpu_bench.f90) |
| OpenACC | [openacc_cufft.f90](examples/openacc_cufft.f90) | [openacc_cufft_bench.f90](benchmarks/openacc_cufft_bench.f90) |

FP32 complex, out-of-place batched 1D C2C. Each input is one; DC must equal the FFT length and all other bins must be zero within the declared tolerances. The teaching CPU example uses serial FFTW_ESTIMATE. The benchmark additionally supports cpu-fftw-threaded when its symbols are available; the thread API is initialized before plans. Plans and transfers are outside compute and inside each E2E pipeline. Page 13's incomplete stream supplement is excluded.

## Direct compilation

Prefer the [shared CMake recipe](../README.md#separate-gpu-build-trees). These are NVHPC/Linux commands from the repository root in the same Toolkit environment, not a claim of an executed GPU build. Supply real provider include/library directories. Helpers/wrappers are explicit link inputs; no dependency installation is performed.

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")"
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set the actual target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" -module "$DIRECT_BUILD" \
  -I"${FFTW_INCLUDE_DIR:?Set the directory containing fftw3.f03}" \
  nvidia/fortran/cufft/examples/fft_cpu.f90 \
  -L"${FFTW_LIBDIR:?Set the directory containing libfftw3f}" -lfftw3f \
  -o "$DIRECT_BUILD/fft_cpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=cufft \
  nvidia/fortran/cufft/examples/fft_gpu.f90 -o "$DIRECT_BUILD/fft_gpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=cufft \
  nvidia/fortran/cufft/examples/openacc_cufft.f90 -o "$DIRECT_BUILD/openacc_cufft"
```

## Examples and a small benchmark

Use the two build variables from the CMake recipe. Require exit 0 and verification PASS for each example. For direct compilation use the same names inside DIRECT_BUILD.

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/cufft/fft_cpu"
"$CPU_CUDA_BUILD/nvidia/fortran/cufft/fft_gpu"
"$OPENACC_BUILD/nvidia/fortran/cufft/openacc_cufft"
```

RUN_DIR must be fresh. This standalone CPU diagnostic is not a campaign input:

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/cufft/fft_cpu_bench" \
  --size 64 --batch 4 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --cpu-threads 1 --cpu-backend cpu-fftw-serial \
  --cpu-parallelism serial --cpu-threads-effective 1 \
  --output "$RUN_DIR/standalone-cufft-cpu-compute.jsonl" --format jsonl
```

For E2E use --scope end-to-end and a fresh output name. For GPU diagnostics select the corresponding benchmark path in the table (CUDA under CPU_CUDA_BUILD, OpenACC under OPENACC_BUILD), remove CPU-only arguments and preserve the same problem/verification/repetition settings. Start E2E diagnostics with repeat=1.

[Own measurement → verification → aggregation → figures](../README.md#first-checks-and-own-measurements) uses the existing runner. The [benchmark protocol](../../../docs/BENCHMARK_PROTOCOL.md) owns scope, restoration and tolerances. Retain the Fortran language identity and do not mix in C/C++ measurements.

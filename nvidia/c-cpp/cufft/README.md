# cuFFT C/C++ examples and benchmarks

This directory compares the canonical batched FP32 forward complex transform
through FFTW3f, direct cuFFT, and OpenACC/cuFFT. Problem mathematics belong to
[`docs/PROJECT_SPECIFICATION.md`](../../../docs/PROJECT_SPECIFICATION.md), and
CLI, timing, restoration, and verification belong to
[`docs/BENCHMARK_PROTOCOL.md`](../../../docs/BENCHMARK_PROTOCOL.md).

## Canonical sources and targets

| Role | Source | CMake target/executable |
| --- | --- | --- |
| CPU teaching | `examples/fft_cpu.c` | `fft_cpu` |
| CUDA teaching | `examples/fft_gpu.cu` | `fft_gpu` |
| OpenACC teaching | `examples/openacc_cufft.cpp` | `openacc_cufft` |
| CPU benchmark | `benchmarks/fft_cpu_bench.c` | `fft_cpu_bench` |
| CUDA benchmark | `benchmarks/fft_gpu_bench.cu` | `fft_gpu_bench` |
| OpenACC benchmark | `benchmarks/openacc_cufft_bench.cpp` | `openacc_cufft_bench` |

## Teaching default

All three teaching programs perform 4,096 independent length-1,024 C2C
forward transforms in single precision. They are complete single-source
programs and intentionally contain no benchmark CLI or result framework.

## Dependencies

The CPU programs require FFTW3f; threaded benchmark support additionally
requires the FFTW3f threads library and its initialization, thread-count, and
cleanup symbols. CUDA programs require CUDA Runtime and cuFFT. OpenACC programs
require NVHPC OpenACC C++, CUDA Runtime, and cuFFT. CMake uses compile-and-link
probes and reports why an affected target is disabled.

## Direct compile

From the repository root, adjust ordinary compiler search paths for the local
installation:

```bash
mkdir -p /tmp/gpu-library-suite-local-build
cc -std=c17 nvidia/c-cpp/cufft/examples/fft_cpu.c \
  -lfftw3f -lm -o /tmp/gpu-library-suite-local-build/fft_cpu-direct
nvcc -std=c++17 nvidia/c-cpp/cufft/examples/fft_gpu.cu \
  -lcufft -o /tmp/gpu-library-suite-local-build/fft_gpu-direct
nvc++ -std=c++17 -acc -cuda \
  nvidia/c-cpp/cufft/examples/openacc_cufft.cpp -cudalib=cufft \
  -o /tmp/gpu-library-suite-local-build/openacc_cufft-direct
```

## CMake configure and build

CPU and CUDA belong to one tree; OpenACC belongs to a separate NVHPC tree. The
commands assume that `CUDA_TOOLKIT_ROOT`, `CUDA_ARCHITECTURES`, and
`NVHPC_GPU_TARGET` were set to explicit site-approved values.

```bash
cmake -S . -B /tmp/gpu-library-suite-local-build/cpu-cuda \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_CUDA=ON \
  -DGPU_SUITE_BUILD_OPENACC=OFF
cmake --build /tmp/gpu-library-suite-local-build/cpu-cuda \
  --target fft_cpu fft_gpu fft_cpu_bench fft_gpu_bench

NVHPC_CUDA_HOME="${CUDA_TOOLKIT_ROOT}" \
cmake -S . -B /tmp/gpu-library-suite-local-build/openacc \
  -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="${NVHPC_GPU_TARGET}" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=ON
cmake --build /tmp/gpu-library-suite-local-build/openacc \
  --target openacc_cufft openacc_cufft_bench
```

CUDA architecture, Toolkit root, and NVHPC GPU target remain explicit external
inputs; portable sources do not select them.

## Run the examples

From the repository root, after the corresponding targets above built:

```bash
CPU_CUDA_BUILD="${CPU_CUDA_BUILD:-/tmp/gpu-library-suite-local-build/cpu-cuda}"
OPENACC_BUILD="${OPENACC_BUILD:-/tmp/gpu-library-suite-local-build/openacc}"
"$CPU_CUDA_BUILD/nvidia/c-cpp/cufft/fft_cpu"
printf 'CPU exit status: %s\n' "$?"
"$CPU_CUDA_BUILD/nvidia/c-cpp/cufft/fft_gpu"
printf 'CUDA exit status: %s\n' "$?"
"$OPENACC_BUILD/nvidia/c-cpp/cufft/openacc_cufft"
printf 'OpenACC exit status: %s\n' "$?"
```

Inspect the reported transform error and require each available example to exit
0. Do not treat a missing target or nonzero exit as a passed comparison. The CPU
teaching program uses serial FFTW, not the threaded benchmark backend. Continue
with [measurement and result processing](../../../docs/PORTABILITY.md#measuring-on-your-own-system).

## Benchmark CLI

A standalone threaded CPU smoke run is:

```bash
/tmp/gpu-library-suite-local-build/cpu-cuda/nvidia/c-cpp/cufft/fft_cpu_bench \
  --size 256 --batch 8 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --abs-tolerance 1e-4 \
  --rel-tolerance 1e-5 --output - --format jsonl \
  --cpu-backend cpu-fftw-threaded --cpu-threads 48 \
  --cpu-threads-effective 48 --cpu-backend-role production \
  --series-role primary --cpu-parallelism threaded
```

Production suite execution obtains all counts, roles, and verification
thresholds from the effective canonical configuration and uses
`tools/run_suite.py` as the only node raw-file writer.

## Timing scopes

`compute` times only repeated transforms after plan creation, allocation,
transfer/restoration, and pre-timing synchronization. `end-to-end` times each
repeat's allocation, plan creation, input transfer, transform, completion, and
output transfer; cleanup follows the end timestamp. Canonical state is restored
before every repeat. Warm-up and verification remain outside timed intervals.

## Verification

The output is compared with the canonical analytical transform using the
effective absolute and relative tolerances. A verification failure records that
trial as failed but does not prevent a restored later trial from running.

## CPU backend and role

`cpu-fftw-threaded` is the only CPU series in the Pegasus publication
configuration and is labeled **Intel Xeon Platinum 8468, FFTW (48 C)** in the
approved figures. It uses the FFTW threads API; the saved publication library
was built with pthread-based `--enable-threads`. CMake may
retain a separately invoked `cpu-fftw-serial` teaching-correspondence backend,
but it is not a publication series and is never substituted for threaded FFTW.

## OpenACC notes

OpenACC owns input and output array storage; `host_data use_device` exposes the
already-present device addresses to cuFFT. The target links
`OpenACC::OpenACC_CXX`, `CUDA::cudart`, and `CUDA::cufft` in its own build tree.

## Known limitations and local validation

Very large batch/length combinations may fail allocation and are reported as
failures rather than silently resized. Local CPU and fake-header syntax tests do
not exercise a real CUDA GPU, cuFFT runtime, NVHPC compiler, or Pegasus run;
those production dependencies remain locally unverified and require the manual
Pegasus checks in
[`docs/PEGASUS_EXECUTION.md`](../../../docs/PEGASUS_EXECUTION.md).
Later saved real execution records are distinguished from these local tests in
[the validation report](../../../docs/VALIDATION_REPORT.md#saved-execution-evidence-reviewed-for-publication).

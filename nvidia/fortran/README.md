# NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

Six families show CPU library calls, CUDA Fortran, and OpenACC from Fortran. Start by running a small teaching example, then substitute your own arrays and operation.

The [minimal build/run guide](../../docs/BUILDING.md) builds only your selected example. GPU variants require NVHPC. CPU RAND / Thrust can also start with GNU Fortran alone. Benchmark measurement and aggregation are optional next steps.

## Measurement compatibility and verified results

The benchmarks follow the published NVIDIA C/C++ timing boundaries,
synchronization, warm-up/restoration and build profiles. The revised version
passed an 18-benchmark Pegasus build, 72 numerical smoke cases and all 6,480
production records in two six-node waves. See the [tested version, workloads and
results](../../docs/VALIDATION_REPORT.md#fortran-production-validation).

Use the library panel for data already resident on the GPU and the end-to-end
panel for a one-shot operation including setup and result retrieval. Matching
measurement conditions helps compare implementations; it does not make native
Fortran RNG/reduction, compiler-generated code or language bindings identical.
Historical measurements remain separate from this aligned campaign.

## Source map

| Library | CPU / direct CUDA / OpenACC stems | CPU calculation | Teaching problem |
| --- | --- | --- | --- |
| [cuFFT](cufft/README.md) | `fft_cpu` / `fft_gpu` / `openacc_cufft` | serial FFTW Fortran interface | FP32 complex C2C, length 1024, batch 4096 |
| [cuBLAS](cublas/README.md) | `blas_cpu` / `blas_gpu` / `openacc_cublas` | Fortran DGEMM, oneMKL | FP64, M=N=K=1024, alpha=beta=1 |
| [cuSPARSE](cusparse/README.md) | `sparse_cpu` / `sparse_gpu` / `openacc_cusparse` | oneMKL Sparse BLAS | FP64, 1024x1024 Poisson grid, one-based CSR |
| [cuSOLVER](cusolver/README.md) | `solver_cpu` / `solver_gpu` / `openacc_cusolver` | Fortran DGESV, oneMKL | FP64, N=1024, 16 RHS, known solution 1 |
| [cuRAND](curand/README.md) | `rand_cpu` / `rand_gpu` / `openacc_curand` | `random_seed` / `random_number` | FP64, 2^24 values, seed 1234 |
| [Thrust](thrust/README.md) | `reduce_cpu` / `reduce_gpu` / `openacc_thrust` | `sum(values*values)` | FP64, 2^24 ones |

Teaching sources are `<library>/examples/<stem>.f90`. Benchmarks are
`<library>/benchmarks/<stem>_bench.f90`; each calls that library's Fortran
`*_workload.F90`, compiled separately for CPU, CUDA or OpenACC. Target and
executable names equal source stems, in a separate Fortran build tree.
[Shared support](../../common/fortran) contains mathematical helpers and a C
interop bridge to the existing CLI, monotonic clock and strict result writer.
The matrix/library computation is Fortran; no C/C++ benchmark executable is
launched. Thrust interoperation uses CUDA C++: both teaching GPU callers share
[thrust_wrapper.cu](thrust/examples/thrust_wrapper.cu), while the CUDA benchmark
uses a separate [device-vector lifetime adapter](thrust/benchmarks/reduce_context.cu)
to match C/C++ allocation and timing. The OpenACC benchmark retains the teaching
wrapper and directive-managed memory.

## Build requirements

Use CMake 3.20+, C17/C++17 and Fortran 2008 features. CPU-only builds support
GNU Fortran or NVHPC. CUDA Fortran/OpenACC requires NVHPC `nvfortran`.
Pegasus builds and execution were validated with NVHPC 25.11. This does not
certify other compiler versions or GPU environments.
Use the same external Toolkit with CMake, NVHPC and NVCC. Thrust uses NVCC
for CUDA-direct and nvc++ for OpenACC, with that Toolkit's Thrust/CCCL headers. GNU Fortran cannot compile the device
extensions. Do not share compiler-generated `.mod` files or build trees across
compilers, profiles or source languages.

FFTW requires FP32 `fftw3f` and `fftw3.f03`. Threaded benchmarks additionally
need a supported thread API provider (pthread, OpenMP, or a combined library);
see the [FFTW discovery options](../../docs/BUILDING.md#cpu-providers). oneMKL requires the LP64 Fortran BLAS/LAPACK ABI,
`mkl_rt`, `mkl_service.h`, and `mkl_spblas.f90` for Sparse BLAS. The supplied
module source is compiled with the selected compiler; do not copy an Intel
compiler's `.mod` into NVHPC/GNU builds. Export actual `MKLROOT` and optionally
`FFTW_ROOT`; CMake probes interfaces and linkable symbols and disables only
affected targets. It never installs providers.

The default `GPU_SUITE_SOURCE_LANGUAGE=c-cpp` does not enable or require
Fortran. Select `fortran` in a **new** tree:

```bash
FORTRAN_CPU_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-cpu.XXXXXX")" || exit 1
cmake -S . -B "$FORTRAN_CPU_BUILD" \
  -DGPU_SUITE_VENDOR=nvidia -DGPU_SUITE_SOURCE_LANGUAGE=fortran \
  -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_BUILD_TYPE=Release \
  -DGPU_SUITE_BUILD_EXAMPLES=ON -DGPU_SUITE_BUILD_BENCHMARKS=OFF \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=OFF -DGPU_SUITE_BUILD_TESTS=OFF &&
cmake --build "$FORTRAN_CPU_BUILD" --target reduce_cpu --parallel 2 &&
"$FORTRAN_CPU_BUILD/nvidia/fortran/thrust/reduce_cpu"
```

Without FFTW/oneMKL, the CPU random and reduction programs remain available.
An enabled target is not proof it has run. Compile/link logs are under
`fortran-probes/`; final build flags are in `compile_commands.json` and verbose
build output. Inspect the configure summary, not only the configure exit code.

Require square sum `16777216`, `verification PASS`, and exit 0. Input uses 128 MiB. The following full benchmark workflow is optional and is not required to run your first teaching example.

<details>
<summary>Benchmark builds, configuration, and records (optional)</summary>

### Separate GPU build trees

Run the following from the repository root **on an authorized GPU build/run
host**, with installed toolchains already selected. On Pegasus first follow the
[human-only Fortran procedure](../../docs/PEGASUS_EXECUTION.md#fortran-first-validation).
No login-node GPU work is intended. Set the three externally supplied variables
below from the actual machine, never by copying an unrelated site's values.

```bash
: "${CUDA_TOOLKIT_ROOT:?Set the installed external CUDA Toolkit root}"
: "${CUDA_ARCHITECTURES:?Set the CMake CUDA architecture}"
: "${NVHPC_GPU_TARGET:?Set the NVHPC GPU target}"
export CUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT" NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
export CUDA_HOME="$CUDA_TOOLKIT_ROOT" CUDA_PATH="$CUDA_TOOLKIT_ROOT"
export PATH="$CUDA_TOOLKIT_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset NVCOMPILER_FPU_STATE
mkdir -p build results
BUILD_ROOT="$(mktemp -d "$PWD/build/fortran.XXXXXX")"
RUN_DIR="$(mktemp -d "$PWD/results/fortran-pilot.XXXXXX")"
export CPU_CUDA_BUILD="$BUILD_ROOT/cpu-cuda" OPENACC_BUILD="$BUILD_ROOT/openacc"
set -o noclobber
SOURCE_HASH="$(python3 tools/hash_source_snapshot.py "$PWD")"
printf '%s\n' "$SOURCE_HASH" > "$RUN_DIR/source-snapshot.sha256"

cmake -S . -B "$CPU_CUDA_BUILD" -G "Unix Makefiles" \
  -DGPU_SUITE_SOURCE_LANGUAGE=fortran -DCMAKE_Fortran_COMPILER=nvfortran \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=Release -DGPU_SUITE_BUILD_TESTS=OFF \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_CUDA=ON -DGPU_SUITE_BUILD_OPENACC=OFF \
  -DCUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT" \
  -DCMAKE_CUDA_COMPILER="$CUDA_TOOLKIT_ROOT/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="$NVHPC_GPU_TARGET"
cmake --build "$CPU_CUDA_BUILD" --target gpu_suite_partial_manifest --parallel 2 --verbose

cmake -S . -B "$OPENACC_BUILD" -G "Unix Makefiles" \
  -DGPU_SUITE_SOURCE_LANGUAGE=fortran -DCMAKE_Fortran_COMPILER=nvfortran \
  -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ \
  "-DCMAKE_C_FLAGS_RELEASE=-fast -O3 -DNDEBUG" \
  "-DCMAKE_CXX_FLAGS_RELEASE=-fast -O3 -DNDEBUG" \
  -DCMAKE_BUILD_TYPE=Release -DGPU_SUITE_BUILD_TESTS=OFF \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF -DGPU_SUITE_BUILD_OPENACC=ON \
  -DCUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="$NVHPC_GPU_TARGET"
cmake --build "$OPENACC_BUILD" --target gpu_suite_partial_manifest --parallel 2 --verbose
test "$SOURCE_HASH" = "$(python3 tools/hash_source_snapshot.py "$PWD")"
python3 tools/merge_manifests.py --output "$RUN_DIR/executables.json" \
  "$CPU_CUDA_BUILD/partial-manifest.json" "$OPENACC_BUILD/partial-manifest.json"
```

Stop on a failed command; save stdout/stderr, module/compiler versions and all
build/manifest files in a new validation directory. Do not rebuild or change
source between the snapshot and a run. The snapshot includes untracked
non-ignored sources. For a dirty smoke/pilot tree, use
`PROVENANCE_ARGS=(--source-snapshot-sha256 "$SOURCE_HASH")` throughout preparation
and execution, as illustrated below. A clean committed build may use an empty
array instead; production requires a clean worktree.
Inspect both build metadata documents: CMake Toolkit root and
`nvhpc_cuda_home` must match. A complete build has 36 manifest entries
(18 examples + 18 benchmarks); missing optional targets are not a complete
six-library check.

### Numerical and CPU runtime conditions

The NVIDIA Fortran build follows the corresponding published C/C++ profile.
CPU/CUDA uses `-O3`, GCC/G++ host support, and NVCC for CUDA Thrust.
NVHPC CPU/CUDA preserves subnormals with `-Kieee -Mnoflushz -Mnodaz`;
GNU CPU uses `-fno-fast-math` with its ordinary contraction default.
OpenACC uses nvfortran `-fast -O3`, nvc/nvc++ host support with
`-fast -O3 -DNDEBUG`, and nvc++ `-cuda` for its Thrust wrapper. It does not
add the CPU/CUDA strict options. Use a new build tree for each profile.
Separate GPU memory, explicit verification and the existing CUDA numerical
modes are retained. Build metadata records global and target flags separately.
Unrelated fast-math and automatic parallel/offload overrides remain rejected.
See the [numerical contract](../../docs/BENCHMARK_PROTOCOL.md#fortran-binding-of-this-protocol).

Unset `NVCOMPILER_FPU_STATE` before examples and benchmarks. NVHPC benchmarks
reject the variable even when empty, because either form overrides compile-time
FPU settings. This does not reconstruct unrecorded historical FPU state.
The revised binaries completed that build, smoke and measurement sequence on
Pegasus; the [validation record](../../docs/VALIDATION_REPORT.md#fortran-production-validation)
identifies the exact version. Validate a new build in your own environment before
using its timings. The published C/C++ binaries and measurements remain the reference.

FFTW teaching examples are serial. Benchmarks initialize the threaded API
before plans when `cpu-fftw-threaded` is selected. oneMKL internally controls
parallelism; the bridge requests threads without claiming the observed count.
Intrinsic RNG and sum builds do not enable automatic parallelization/offload:
their backend IDs are `cpu-fortran-random-serial` and `cpu-fortran-sum-serial`,
effective count 1. Record compiler identity: the intrinsic RNG algorithm is
compiler-dependent, not C++ MT19937 and not stream-equivalent to cuRAND.
The distributed small pilot requests one thread for diagnostics. Choose and record production CPU settings explicitly; it is not the saved production configuration.

## First checks and own measurements

Each library README gives helper-aware direct compile commands, example
execution paths and a small benchmark command. Require exit 0 and
`verification PASS` for examples. Teaching defaults are unchanged; benchmarks
accept runtime sizes. Start with a small cuBLAS CPU/CUDA/OpenACC check in both
scopes before running all six. Standalone files are diagnostics, not complete
campaign data; do not mix them into runner output.

For suite measurement reuse [the existing reader workflow](../../docs/PORTABILITY.md#measuring-on-your-own-system)
with the build trees above, the Fortran manifest, a fresh run ID/directory and
these inputs instead of the C/C++ inputs:

```bash
MANIFEST="$RUN_DIR/executables.json"
cp configs/fortran/pilot.json "$RUN_DIR/config-input.json"
CONFIG_INPUT="$RUN_DIR/config-input.json"
CONFIG="$RUN_DIR/effective-config.json"
PROVENANCE_ARGS=(--source-snapshot-sha256 "$SOURCE_HASH")
```

`configs/fortran/pilot.json` supplies three **small diagnostic** cases per library,
two trials and both scopes. It preserves problem/precision/verification
semantics, selects Fortran intrinsic backend names and sets
`source_language="fortran"`. It is not the historical C/C++ publication workload
and is not the separately saved production configuration. Choose measurement
settings for your own environment. For the first six-row
check, edit the new copy: enable only cuBLAS, keep its first case, set
`run_mode="smoke"` and each scope's warmup/repeat/trials to 1. Retain the other
five library objects disabled. Do not overwrite the canonical configurations.

Then follow the existing commands to
[serialize the effective input](../../docs/PORTABILITY.md#2-make-a-small-measurement-check-then-choose-the-suite-settings),
[capture runtime/prepare/run](../../docs/PORTABILITY.md#3-capture-the-real-runtime-environment-and-launch-one-local-block),
[validate and aggregate](../../docs/PORTABILITY.md#validate-and-aggregate-a-completed-run),
and [render the six two-panel figures](../../docs/PORTABILITY.md#figures-from-your-own-measurements).
Use the module-aware collector on module systems. Keep the dirty-source
`PROVENANCE_ARGS` from above instead of resetting it to the clean example's empty
array. The local `--local` route is for a true single-host non-PBS run; for
Pegasus use its PBS provenance/launcher procedure, not a disguised non-PBS run.

Raw schema version 1 is retained. Fortran rows carry
`parameters.source_language="fortran"`; old C/C++ rows keep their original shape.
Manifests have `compiler_language="fortran"`. Mixed source families are rejected
in a manifest/run and in a single figure; aggregation retains language in the
parameter signature. Figure legends identify Fortran and its intrinsic
backends. A complete six-figure input still requires three cases and all three
implementations in both scopes, successful numerical verification and matching
CUDA/OpenACC Thrust library versions. A six-row smoke cannot produce six figures.
Never relabel old C/C++ results as Fortran or infer rankings from synthetic tests.

## Teaching extraction changes and validation limits

| Pages / library | Necessary distribution changes |
| --- | --- |
| 9, 12 / cuFFT | Allocation/plan/API checks and finite/DC/non-DC verification; original plan and synchronization flow retained |
| 18, 20 / cuBLAS | Allocation/API checks and all-ones DGEMM verification; initial C and FP64 preserved |
| 25–30 / cuSPARSE | Complete external one-based Poisson helper; initialize all descriptor fields; API/result checks |
| 35–38 / cuSOLVER | Complete external A=N*I+ones, B=2N helper; preserve separate getrf/getrs info and check factorization before solve in examples; solution/residual checks |
| 43, 45 / cuRAND | Allocation/API and distribution checks; original Fortran RNG and cuRAND generator retained |
| 50–53 / Thrust | Finite/reduction checks; catch C++ exceptions and return NaN for the caller to reject; one shared wrapper function |

The page-13 ellipsis-containing cuFFT stream fragment is not a build target.
Its different synchronization style was not substituted into all main examples.
Reference extraction files remain unchanged in ignored local validation inputs;
the separately distributed teaching slides are not included in this checkout.

Local GNU Fortran CPU checks cover intrinsic programs and helper mathematics.
Controlled test providers additionally cover all four external CPU call paths,
both scopes, state restoration and nonfinite/info failures; they are **not**
oneMKL/FFTW installation or NVHPC ABI validation. See
the [local validation record](../../docs/VALIDATION_REPORT.md#fortran-local-validation)
and the later [Pegasus production record](../../docs/VALIDATION_REPORT.md#fortran-production-validation)
for their distinct evidence. Pegasus validation used the CMake profiles; the
direct-compile recipes and arbitrary other environments are not claimed as
separately executed. Use the CMake profiles above, not the optional teaching
compile recipes, to reproduce the recorded benchmark build policy.
The current aligned campaign passed all 6,480 numerical checks on 12 distinct
hosts across two waves; aggregation and six figures were checked against those
records. The saved build, smoke and wave scripts each returned 0. Their script
completion records are distinct from scheduler accounting evidence.
oneMKL effective worker counts remain unobserved; FFTW records the configured
48 threads and the intrinsic baselines record 1. Complete loaded-library
identity and component-level overheads were not measured. Earlier results and
failed attempts remain separate. Saved raw data and figures are not distributed.

Interface references: [NVIDIA Fortran CUDA Interfaces 25.11](https://docs.nvidia.com/hpc-sdk/archive/25.11/compilers/fortran-cuda-interfaces/index.html),
[CUDA Fortran Guide 25.11](https://docs.nvidia.com/hpc-sdk/archive/25.11/compilers/cuda-fortran-prog-guide/index.html),
[HPC Compilers Reference 25.11](https://docs.nvidia.com/hpc-sdk/archive/25.11/compilers/hpc-compilers-ref-guide/index.html).
Static agreement with these references is not evidence of an executed GPU test.

</details>

NVHPC builds, real-machine validation, and production measurements were completed on Pegasus. See the [versions and coverage](../../docs/VALIDATION_REPORT.md#fortran-production-validation); they do not guarantee another environment or later edits. Saved measurements, logs, and PPTX files are not distributed.

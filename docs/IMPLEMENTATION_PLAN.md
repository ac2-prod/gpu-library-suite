# GPU Library Suite Implementation Plan

## Authority and scope

This document owns implementation phases, deliverables, dependencies,
acceptance criteria, validation activities, risks, and unresolved decisions. It
does not redefine command-line behavior, mathematical problems, result schemas,
or Pegasus operating procedures. Those specifications are authoritative in:

- [`PROJECT_SPECIFICATION.md`](PROJECT_SPECIFICATION.md)
- [`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md)
- [`RESULT_SCHEMA.md`](RESULT_SCHEMA.md)
- [`PEGASUS_EXECUTION.md`](PEGASUS_EXECUTION.md)

The decision-complete details below are implementation requirements and inputs
to the Phase 1 specification-delta gate. Once synchronized, each owning
specification remains normative; this plan does not become a competing schema,
protocol, or operating manual.

## Planning baseline and cross-phase constraints

### Planning status and local toolchain preflight

The approved objective is the complete first implementation described by this
decision-complete plan. The repository now contains the resulting source and
user-facing documentation. The implementation preserves unrelated user changes
and each phase advances only after its required gate passes.

Local preflight has established that Apple Clang 21 supports strict C17 and
C++17, `CLOCK_MONOTONIC`, and `std::transform_reduce`. CMake and CTest 4.4.0 can
configure, build, and run two synthetic toolchain checks with Unix Makefiles.
Those two checks are environment preflight only and are not project tests.

### Execution safety boundary

Implementation and validation stay inside the repository, except for small
artifacts under `/tmp/gpu-library-suite-local-build`. They do not use remote
access, scheduler/accounting commands, network access, package installation, or
Git-writing commands. They do not access Pegasus filesystems from the local
host. Job submission and site inspection remain human actions, and all
unavailable tests are reported rather than inferred successful.

### Language enablement and CMake compatibility

The top-level project starts with only CPU languages:

```cmake
cmake_minimum_required(VERSION 3.20)
project(gpu_library_suite LANGUAGES C CXX)
```

The project sets all of the following without vendor extensions:

```cmake
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
```

Only when `GPU_SUITE_BUILD_CUDA=ON` does CMake include `CheckLanguage`, call
`check_language(CUDA)`, and call `enable_language(CUDA)` when a CUDA compiler is
actually available. A missing compiler, Toolkit component, or externally
supplied `CMAKE_CUDA_ARCHITECTURES` disables only CUDA targets with a clear
reason. CPU-only configure must remain successful.

The portable common C target compiles with `_POSIX_C_SOURCE=200809L`. Its public
headers do not expose POSIX implementation types, so the definition remains a
target-private implementation requirement. Direct benchmark compile examples
include the same definition explicitly.

OpenACC discovery uses `find_package(OpenACC COMPONENTS CXX)` and tests
`OpenACC_CXX_FOUND` or `TARGET OpenACC::OpenACC_CXX`; it never relies on the
generic `OpenACC_FOUND` variable. OpenACC targets link
`OpenACC::OpenACC_CXX`.

The implementation may use only commands, properties, and modules available in
CMake 3.20. In particular, it uses `CheckLanguage`,
`CheckCSourceCompiles`/`CheckCXXSourceCompiles`, `CMakePushCheckState`,
`FindOpenACC`, and `FindCUDAToolkit`, and avoids later-only find validators,
link-group properties, and presets as required build behavior.

### Build options and profiles

The implementation exposes:

- `GPU_SUITE_BUILD_EXAMPLES`
- `GPU_SUITE_BUILD_BENCHMARKS`
- `GPU_SUITE_BUILD_CPU`
- `GPU_SUITE_BUILD_CUDA`
- `GPU_SUITE_BUILD_OPENACC`
- `GPU_SUITE_BUILD_TESTS`
- `GPU_SUITE_CPU_BLAS_BACKEND`
- `GPU_SUITE_CPU_SPARSE_BACKEND`
- `GPU_SUITE_CPU_LAPACK_BACKEND`
- `GPU_SUITE_CPU_THRUST_BACKEND`
- `GPU_SUITE_NVHPC_GPU_TARGET`

Exactly two production build profiles are supported:

| Profile | Required CMake settings |
| --- | --- |
| `cpu-cuda` | `GPU_SUITE_BUILD_CPU=ON`, `GPU_SUITE_BUILD_CUDA=ON`, `GPU_SUITE_BUILD_OPENACC=OFF` |
| `openacc` | `CMAKE_C_COMPILER=nvc`, `CMAKE_CXX_COMPILER=nvc++`, `GPU_SUITE_BUILD_CPU=OFF`, `GPU_SUITE_BUILD_CUDA=OFF`, `GPU_SUITE_BUILD_OPENACC=ON` |

Production artifacts use one of those two profile names. The additional local
validation-only tree is
`/tmp/gpu-library-suite-local-build/cpu-only-validation`, configured with CPU
ON and both GPU modes OFF; it is never merged into a production campaign
manifest. Local production-profile checks use
`/tmp/gpu-library-suite-local-build/cpu-cuda` and `.../openacc` when their
toolchains are available. Pegasus build paths are human-supplied external
paths. Generated files never enter portable source directories.

The Pegasus configuration keeps `cpu_cuda_build_modules`,
`openacc_build_modules`, and `benchmark_runtime_modules` separate. Build jobs
load only their profile's modules. Benchmark jobs load only the human-validated
runtime module set that can execute all prebuilt binaries together.

### OpenACC and CUDA Toolkit selection

All five CUDA-library OpenACC targets use one canonical link model rather than
mixing link strategies:

| Target family | Link interface |
| --- | --- |
| cuFFT | `OpenACC::OpenACC_CXX`, `CUDA::cudart`, `CUDA::cufft` |
| cuBLAS | `OpenACC::OpenACC_CXX`, `CUDA::cudart`, `CUDA::cublas` |
| cuSPARSE | `OpenACC::OpenACC_CXX`, `CUDA::cudart`, `CUDA::cusparse` |
| cuSOLVER | `OpenACC::OpenACC_CXX`, `CUDA::cudart`, `CUDA::cusolver` |
| cuRAND | `OpenACC::OpenACC_CXX`, `CUDA::cudart`, `CUDA::curand` |

The implementation does not use the alternative NVHPC `-cudalib=<library>`
path for these five targets. OpenACC Thrust links
`OpenACC::OpenACC_CXX` and `CUDA::cudart` and receives NVHPC's `-cuda`
equivalent in both compile and link options.

The OpenACC build requires either an explicit `NVHPC_CUDA_HOME` or an
unambiguous CUDA Toolkit path reported by the selected NVHPC compiler. That path
is also supplied as `CUDAToolkit_ROOT`. A mismatch between the Toolkit imported
by CMake and the Toolkit selected by NVHPC is a configure failure for the
OpenACC profile, not a silent substitution.

Build metadata stores `CUDAToolkit` path/version, the NVHPC-selected CUDA path,
`NVHPC_CUDA_HOME` when set, NVHPC compiler/runtime version, GPU target, and
compile/link flags. Portable source never hard-codes `cc90`, `sm_90`, H100, or a
site path.

### Compile-and-link dependency probes

Discovery of a header or library filename is never sufficient. Each backend is
enabled only after a compile-and-link probe with the actual candidate include
directories, definitions, options, and libraries:

- FFTW3f calls `fftwf_plan_many_dft` and `fftwf_execute`.
- FFTW3f threads calls `fftwf_init_threads`,
  `fftwf_plan_with_nthreads`, and `fftwf_cleanup_threads`, linked with the base
  FFTW3f library and required system Threads.
- CBLAS calls `cblas_dgemm`; `BLAS_FOUND` alone is never evidence of CBLAS.
- LAPACKE links one probe containing `LAPACKE_dgesv`,
  `LAPACKE_dgetrf`, and `LAPACKE_dgetrs`.
- oneMKL Sparse creates a CSR descriptor with the exact planned API, performs
  `mkl_sparse_d_mv` with `matrix_descr`, and destroys the descriptor.

Each OpenACC CUDA-library target also has an `nvc++` compile-and-link probe that
includes its CUDA library header and calls `cudaGetDeviceCount` plus these real
symbols from the implemented operation:

| Library | Probe symbols |
| --- | --- |
| cuFFT | `cufftPlanMany`, `cufftExecC2C` |
| cuBLAS | `cublasCreate_v2`, `cublasDgemm_v2` |
| cuSPARSE | `cusparseCreateCsr`, `cusparseSpMV_bufferSize`, `cusparseSpMV` |
| cuSOLVER | `cusolverDnDgetrf_bufferSize`, `cusolverDnDgetrf`, `cusolverDnDgetrs` |
| cuRAND | `curandCreateGenerator`, `curandGenerateUniformDouble` |

Each probe uses the exact target link interface shown above.

Configure summary output states requested state, selected backend, probe
result, enabled targets, and a precise disable reason. Local tests use small
positive and negative fixture libraries to test probe logic without claiming
that an actual optional dependency is installed.

### Canonical target and executable names

CMake target names and output executable names exactly match source stems:

| Library | Example targets | Benchmark targets |
| --- | --- | --- |
| cuFFT | `fft_cpu`, `fft_gpu`, `openacc_cufft` | `fft_cpu_bench`, `fft_gpu_bench`, `openacc_cufft_bench` |
| cuBLAS | `blas_cpu`, `blas_gpu`, `openacc_cublas` | `blas_cpu_bench`, `blas_gpu_bench`, `openacc_cublas_bench` |
| cuSPARSE | `sparse_cpu`, `sparse_gpu`, `openacc_cusparse` | `sparse_cpu_bench`, `sparse_gpu_bench`, `openacc_cusparse_bench` |
| cuSOLVER | `solver_cpu`, `solver_gpu`, `openacc_cusolver` | `solver_cpu_bench`, `solver_gpu_bench`, `openacc_cusolver_bench` |
| cuRAND | `rand_cpu`, `rand_gpu`, `openacc_curand` | `rand_cpu_bench`, `rand_gpu_bench`, `openacc_curand_bench` |
| Thrust | `reduce_cpu`, `reduce_gpu`, `openacc_thrust` | `reduce_cpu_bench`, `reduce_gpu_bench`, `openacc_thrust_bench` |

No target name encodes a manual version, date, GPU architecture, provider, or
site. Separate build profiles and manifest metadata distinguish artifacts.

### Build metadata and executable manifests

Every manifest entry contains at least `artifact_id`, `target_name`,
`build_profile`, `backend_variant`, library, implementation, executable role,
path, build type, binary SHA-256, compiler metadata, Git provenance, and build
metadata SHA-256.

`artifact_id` is the SHA-256 of the deterministic JSON identity object
containing library, implementation, executable role, target name, build
profile, backend variant, and binary SHA-256. The semantic key
`(library, implementation, executable_role)` and the build identity key
`(target_name, build_profile, backend_variant)` must each be unique. Any
duplicate is an error even when paths or hashes happen to match. Conflicting
paths, hashes, providers, Git state, compiler metadata, or build types receive
specific diagnostics.

`backend_variant` identifies the compiled provider/capability, such as
`fftw3f-serial+threaded`, `fftw3f-serial`, `onemkl`, `openblas`, or
`nvhpc-cuda-interop`; it does not rename the executable. A manifest also records
the stable CPU backend names supported by that artifact. Builds from different
providers cannot be merged ambiguously into one campaign manifest.

The manifest builder hashes binaries only after build completion. `run_suite.py`
checks generated build metadata against the manifest before launch and checks
raw compiler/Git fields against the selected entry after launch. Production
runs require Release artifacts and clean source. Dirty smoke/pilot provenance
uses a deterministic source-snapshot hash that includes tracked and untracked,
non-ignored build inputs.

### Initial calibration configurations

Both canonical configurations encode `warmup`, `repeat`, and `trials` under each
scope. Implementations cannot override them. Equality is enforced across CPU,
CUDA, and OpenACC for each benchmark + normalized problem + scope; compute and
end-to-end scopes remain independent.

Both files begin with `config_schema_version=1`; ordinary calibration changes
do not change that value.

Each case has the structural shape `problem` plus a `scopes` object containing
separate `compute` and `end-to-end` objects. Both canonical configurations use
JSON Lines as the node raw default, enable verification, and set
`continue_on_failure=true`; failures still make final campaign validation fail.
Both canonical initial calibration configurations request 48 CPU threads. A
serial backend still records effective thread count 1; requested and effective
thread counts are separate provenance fields.

Phase 1 formalizes this exact configuration shape in `RESULT_SCHEMA.md`:

```text
config_schema_version
run_mode
output_format
continue_on_failure
cpu_threads
benchmarks
  <benchmark>
    enabled
    precision
    series[]
      implementation
      cpu_backend
      cpu_backend_role
      series_role
    default_speedup_cpu_backend
    verification
    cases[]
      parameters
      scopes
        compute
          warmup
          repeat
          trials
        end-to-end
          warmup
          repeat
          trials
```

All numeric case parameters are positive integers except explicitly floating
library parameters such as alpha/beta. Unknown keys, missing required keys,
duplicate keys, non-standard numeric constants, and implementation-local timing
overrides are errors.

`configs/pilot.json` starts with:

| Benchmark | Problem |
| --- | --- |
| cuFFT | `nfft=256`, `batch=8` |
| cuBLAS | `size=128` |
| cuSPARSE | `size=4096` |
| cuSOLVER | `size=64`, `nrhs=2` |
| cuRAND | `size=65536` |
| Thrust | `size=65536` |

Pilot compute settings are `warmup=1`, `trials=1`, and `repeat=2`; pilot
end-to-end settings are `warmup=1`, `trials=1`, and `repeat=1`. cuSOLVER uses
`repeat=1` in both scopes.

`configs/benchmark.json` starts with:

| Benchmark | Problems | Compute repeat |
| --- | --- | --- |
| cuFFT | `nfft=[256,1024,4096,16384]`, `batch=4096` | 3 |
| cuBLAS | `size=[512,1024,2048,4096]` | 3 |
| cuSPARSE | `size=[65536,262144,1048576,4194304]` | 10 |
| cuSOLVER | `size=[256,512,1024,2048]`, `nrhs=16` | 1 |
| cuRAND | `size=[1048576,4194304,16777216,67108864]` | 3 |
| Thrust | `size=[1048576,4194304,16777216,67108864]` | 10 |

Benchmark compute settings are `warmup=1` and `trials=5`. End-to-end settings
are `warmup=1`, `repeat=1`, and `trials=5` for every library. These are initial
calibration candidates, not production-fixed values. After pilot measurement,
the same canonical `configs/benchmark.json` is updated without renaming it or
incrementing `config_schema_version`.

The initial Pegasus primary CPU selections are:

| Benchmark | Primary backend |
| --- | --- |
| cuFFT | `cpu-fftw-threaded` |
| cuBLAS | `cpu-onemkl` |
| cuSPARSE | `cpu-onemkl` |
| cuSOLVER | `cpu-onemkl` |
| cuRAND | `cpu-std-random-serial` |
| Thrust | `cpu-stl-serial` |

Production build scripts therefore select oneMKL for BLAS, Sparse, and LAPACK
and STL for Thrust. OpenBLAS, generic CBLAS/LAPACKE, reference CSR, or OpenMP
Thrust require a separately identified build/campaign or an explicitly
auxiliary series; none silently replaces a listed primary backend.

### Primary CPU series and repeat-state semantics

Pegasus cuFFT uses `cpu-fftw-threaded` as the primary production CPU backend.
`cpu-fftw-serial` is auxiliary or teaching-correspondence only. If the threaded
probe fails, serial does not replace it and no primary cuFFT speedup is emitted.

cuRAND's `cpu-std-random-serial` and Thrust's `cpu-stl-serial` remain production
serial baselines. Plots label both exactly **Serial CPU baseline**, even when
the campaign requested 48 CPU threads.

cuBLAS and cuSPARSE compute repeats intentionally carry C or y state through
the repeat loop, and verification accounts for every update. End-to-end repeats
restore canonical C or y before each timed pipeline and never inherit state
from the preceding repeat. cuSOLVER requires `repeat=1` in every scope at
configuration, runner, and executable validation layers.

### Benchmark output ownership and dry-run behavior

With `--output -`, a benchmark writes only machine-readable records to stdout;
all diagnostics go to stderr. With `--output <path>`, it exclusively creates a
new file and never overwrites or implicitly appends.

During suite execution, every benchmark is launched with `--output -`.
`run_suite.py` is the only writer of each node-level `raw-results.jsonl` or
`raw-results.csv`. It validates subprocess output before accepting rows and
writes the CSV header exactly once. Benchmarks never append to a shared raw
file.

`run_suite.py --dry-run` validates and displays the resolved configuration,
manifest selection, implementation/size order, and argv arrays as deterministic
JSON. It creates no directory, campaign/wave/node metadata, raw file, log, or
subprocess.

### Trial outcome model

The initial `result_schema_version=1` adds required `attempted` and
`failure_origin` fields before any machine-readable artifact exists.
`failure_origin` is null or one of `benchmark`, `verification`, `subprocess`,
`output-validation`, `prior-failure`, or `prerequisite`.

- A started successful trial has `attempted=true`, `status=success`, and null
  origin.
- A started execution or verification failure has `attempted=true`,
  `status=failure`, and the corresponding origin.
- A remaining trial not started after a fatal predecessor has
  `attempted=false`, `status=skipped`, and `prior-failure`.
- A missing prerequisite produces unattempted `prerequisite` skipped rows.
- For a crash, signal, missing output, or invalid stdout, the first unresolved
  trial is a synthetic attempted failure; later unresolved trials are synthetic
  skipped rows.

Every skipped row has null measurement timestamps and elapsed fields,
`verification_status=skipped`, and null `exit_code`. A synthetic failure has
null timing fields when timing never began and records the actual subprocess
exit code or signal-derived exit code when available.

The runner preserves validated completed rows, rejects duplicate or out-of-range
trial indices, and materializes each expected index exactly once. Recoverable
verification failures may continue after canonical restoration; fatal execution
failures skip only the trials that truly did not start. Aggregation counts
success, failure, and skipped separately and includes only valid successes in
performance statistics.

### Measurement and telemetry timestamps

Every raw row carries `record_timestamp`, `measurement_start_timestamp`, and
`measurement_end_timestamp`, all UTC with exactly millisecond precision.
End-to-end repeats are timed individually, so the UTC measurement span can
exceed `elapsed_total_sec` because restoration and cleanup gaps are untimed.

Telemetry metadata carries `telemetry_start_timestamp_utc`,
`telemetry_end_timestamp_utc`, `sample_interval_sec`, `timezone`, `utc_offset`,
and `midnight_rollover_count`. Telemetry samples are associated with each
trial's complete measurement interval, not with `record_timestamp` or a single
nearest sample.

### Runtime environment provenance

Before campaign preflight, the job master deterministically writes
`runtime-environment.json` and hashes its exact bytes as
`runtime_environment_sha256`. It records at least:

- normalized module list;
- `PATH` and `LD_LIBRARY_PATH`;
- NVIDIA driver and CUDA runtime/Toolkit versions;
- NVHPC compiler/runtime version;
- FFTW, oneMKL, OpenBLAS, and LAPACKE versions;
- `ldd` output and resolved shared-library paths for every manifest binary;
- all required CPU runtime variables; and
- `NVHPC_CUDA_HOME` or the actual NVHPC-selected CUDA Toolkit.

Additional waves require an identical runtime-environment hash in addition to
configuration, manifest/binary, Git, dirty-source, and build provenance. A
mismatch requires a new run ID and cannot be forced into the existing campaign.

### CPU runtime variables

Pegasus configuration, rendered PBS, runtime environment, node metadata, and
CPU telemetry all record:

```text
OMP_NUM_THREADS
OMP_PROC_BIND
OMP_PLACES
OMP_DYNAMIC=FALSE
MKL_NUM_THREADS
MKL_DYNAMIC=FALSE
MKL_THREADING_LAYER
OPENBLAS_NUM_THREADS
```

Binding and threading-layer values are explicit human inputs, not guesses.
Requested threads and effective backend threads remain distinct, so a serial
backend can correctly record requested 48, effective 1, and `serial`.

### cuRAND verification

All implementations check `0.0 <= x <= 1.0` while recording that the C++
standard distribution has interval `[0,1)` and cuRAND has `(0,1]`. Configuration
stores `sigma_multiplier=6.0`, `expected_mean=0.5`, and
`expected_second_central_moment=1/12`.

Required metrics are `observed_min`, `observed_max`, `sample_mean`, and
`second_central_moment_about_half`, where the latter is
`mean((x_i - 0.5)^2)`. The mean bound is
`sigma_multiplier * sqrt(1/(12*N))`; the second-central-moment bound is
`sigma_multiplier * sqrt(1/(180*N))`. Verification uses the last retrieved N
values and records `verification_sample_count=N`.

No CPU/GPU element identity is expected. README, aggregate metadata, and plot
metadata state: **Same distribution and output type task; different RNG
algorithms.**

### Locale-independent C/C++ serialization

The benchmark process fixes `LC_NUMERIC` to `C`. The common writer uses
round-trip precision, normalizes exponent spelling, removes redundant exponent
sign/zero padding, and canonicalizes negative zero to `0`. It checks every
floating value with `isfinite`; NaN and infinities become null plus a
failure/nonfinite status and never appear as JSON constants.

Tests cover decimal points, small/large exponents, round-trip values, negative
zero, NaN, and both infinities. When a non-C locale is installed, a subprocess
test proves the output remains C-locale; otherwise the test is reported as
unexecuted with its reason. Python JSON loading always supplies both
`object_pairs_hook` and `parse_constant` and rejects invalid UTF-8.

### Python and optional plotting

Python tools target Python 3.9 or newer and otherwise use only the standard
library. Every Python source is checked with
`ast.parse(feature_version=(3, 9))` and normal `py_compile` or `compileall`, with
bytecode redirected under `/tmp/gpu-library-suite-local-build`. Python 3.10+
syntax and APIs such as structural pattern matching, PEP 604 unions,
`zip(strict=...)`, `itertools.pairwise`, and `tomllib` are prohibited.

Local unit tests run with the installed Python version; a real Python 3.9 run is
part of manual Pegasus validation. Matplotlib is lazy-imported only by plotting.
Its absence produces a clear diagnostic and never triggers installation.

### Phase gate policy

Each required gate must pass before work begins on the next phase. The agent
fixes in-scope gate failures without waiting for user input. A correctly
diagnosed optional dependency absence and target disable is not a gate failure;
failure of an unaffected required target or of the disable logic is a gate
failure.

## Phase 1 — Common benchmark infrastructure

### Specification delta synchronization gate

Before writing any related source, update each authoritative owner and the
decision register so the initial implementation has one consistent normative
baseline:

| Delta | Authoritative owner to update before code |
| --- | --- |
| `attempted` and `failure_origin`; raw timestamp fields; manifest identity fields | `RESULT_SCHEMA.md` |
| Scope-specific config; output ownership; trial failure/skipped semantics; cuFFT primary CPU backend; cuRAND formulas; elapsed/span distinction | `BENCHMARK_PROTOCOL.md` |
| Runtime-environment content/hash; module group separation; NVHPC Toolkit selection; timestamp correlation; `prepare_wave.py`; node failure isolation | `PEGASUS_EXECUTION.md` |
| Build phases, CMake constraints, gates, validation, and risks | this document |
| Accepted rationale and links, without duplicated detail | `DECISIONS.md` |

The gate compares all cross-references and terminology after those edits.
Because no project machine-readable artifact exists yet, all compatible schema
clarifications are incorporated into initial schema version 1. No common,
benchmark, runner, or Pegasus source is written until this gate passes. Phase 6
does not introduce these specifications; it audits implemented behavior and
finishes user-facing documentation.

### Deliverables

- Synchronized authoritative specifications and decision entries.
- Root and common CMake structure with conditional CUDA language enablement.
- Common C17 CLI parsing, checked arithmetic, monotonic clock, timestamp, result,
  verification, provenance, and exclusive-output APIs with C++ guards.
- Locale-independent deterministic JSON Lines and CSV serialization.
- Initial raw schema support, including attempted/failure/skipped records.
- Strict Python JSON/config/schema/hash utilities.
- Generated build metadata and partial-manifest foundations.
- Positive/negative dependency-probe fixtures.
- C, C++, CMake integration, and Python unit tests for common behavior.

### Dependencies

- C/C++ standard libraries and POSIX monotonic clock.
- Python 3.9 standard library.
- CMake 3.20-compatible features and the local C/C++ toolchain.
- Completion of the specification synchronization gate.

### Required gate

- Every listed specification delta is present in its owner, cross-links resolve,
  and no owned rule is redefined elsewhere.
- CPU-only CMake configure succeeds without probing or enabling CUDA.
- Strict C17 and C++17 build succeeds with extensions disabled; the common C
  target uses `_POSIX_C_SOURCE=200809L`.
- Enabling CUDA with no CUDA compiler disables only CUDA targets with a precise
  summary reason.
- Common CLI accepts every specified option and rejects invalid enum, numeric,
  run-ID, dimension, and implementation-order input.
- JSONL and CSV preserve nested objects, quoting, finite round-trip numbers, and
  one logical row per trial.
- stdout/path ownership, exclusive create, and stderr-only diagnostics pass
  direct common-output tests.
- Result validation distinguishes attempted failure from unattempted skipped and
  rejects duplicate/out-of-range trial indices.
- Locale tests cover decimal point, exponent normalization, negative zero,
  nonfinite values, and an available non-C locale.
- Strict Python loading rejects duplicate keys, `NaN`, `Infinity`,
  `-Infinity`, invalid UTF-8, and nondeterministic output.
- UTC timestamps have exactly millisecond precision, and monotonic resolution is
  finite and positive.
- Generated metadata and manifest identities are deterministic and reject
  conflicting semantic/build keys.
- Positive and negative fixture probes prove that header discovery alone cannot
  enable a backend.
- Every Python file parses with `feature_version=(3, 9)` and passes normal
  byte-compilation.
- All Phase 1 CTest and Python unit tests pass.
- `git diff --check` passes.

### Local validation

- Configure and build in
  `/tmp/gpu-library-suite-local-build/cpu-only-validation` for the Phase 1
  CPU-only check.
- Run CTest with `--output-on-failure`.
- Run Python unittest, AST 3.9 parsing, and byte-compilation with cache output
  under `/tmp/gpu-library-suite-local-build`.
- Run output and schema fixtures only in memory or temporary storage; do not
  create repository sample results.
- Run `git diff --check` and inspect `git status --short` before opening Phase 2.

### Manual Pegasus validation

- None required until executable implementations exist.

### Risks

- C and Python serialization can drift if either bypasses its single common
  implementation; schema equivalence tests mitigate this.
- Global numeric locale changes are process-wide; benchmarks are single-process
  executables, and tests must establish locale before exercising the writer.
- CMake 4.4 local validation cannot alone prove 3.20 compatibility; the plan
  restricts the feature inventory and retains a real-site CMake check.

## Phase 2 — cuFFT examples and benchmarks

### Deliverables

- Three canonical cuFFT teaching examples.
- Three corresponding cuFFT benchmarks.
- `fft_cpu_bench` with distinct `cpu-fftw-threaded` primary and
  `cpu-fftw-serial` auxiliary series when threaded support is linked.
- FFTW3f base/threaded and OpenACC cuFFT compile-and-link probes.
- CMake targets with canonical stem names in both build profiles.
- DC/non-DC verification, state restoration, and both timing scopes.
- cuFFT README, build metadata, manifest entries, and tests.

### Dependencies

- Phase 1 infrastructure.
- Optional FFTW3 single-precision interface and threaded interface.
- Optional CUDA Toolkit and NVHPC for GPU targets.

### Required gate

- Filenames, teaching defaults, and plan choice match the project specification.
- All six CMake target and executable names exactly match their source stems.
- FFTW3f is enabled only when its actual plan/execute probe links; threaded mode
  is enabled only when all three threaded calls link with the base library and
  system Threads.
- Missing FFTW disables only affected CPU targets and reports the failed probe.
- The canonical CPU, CUDA, and OpenACC examples are complete single-source
  teaching programs with checked allocation, API diagnostics, and cleanup.
- Benchmark compute and end-to-end scope, warm-up, state restoration, timestamps,
  and DC/non-DC verification match the synchronized benchmark protocol.
- OpenACC data regions perform managed allocation/movement exactly once; no
  managed array is also explicitly allocated.
- The OpenACC probe includes `<cufft.h>`, a CUDA Runtime call, and a cuFFT symbol
  and links through `OpenACC::OpenACC_CXX`, `CUDA::cudart`, and `CUDA::cufft`.
- Pegasus configuration selects `cpu-fftw-threaded` as the primary denominator
  and `cpu-fftw-serial` as auxiliary/teaching correspondence.
- If threaded FFTW is unavailable, serial never becomes primary and aggregate
  output contains no primary cuFFT speedup.
- CPU-only configure remains valid when every GPU dependency is absent.
- Manifest entries contain the required identity, profile, variant, metadata,
  and binary hashes and pass runner-side consistency fixtures.
- Available CPU examples/benchmarks pass numerical verification; unavailable
  optional execution is reported rather than claimed.
- All Phase 2 tests and `git diff --check` pass.

### Local validation

- Exercise missing and synthetic-positive FFTW probe configurations.
- Compile and run CPU examples/benchmarks only when a real FFTW dependency is
  already available.
- Perform syntax/static checks for GPU sources without claiming GPU execution.
- Verify canonical target names, source inventory, disable diagnostics, primary
  backend selection, and no-fallback behavior.
- Re-run the complete Phase 1 gate before entering Phase 3.

### Manual Pegasus validation

- Build CPU/CUDA and OpenACC trees separately on an appropriate compute node.
- Run the teaching examples, a pilot benchmark, verification, and sanitizer.

### Risks

- FFTW threaded configuration may differ among installations.
- OpenACC/CUDA interoperability may depend on compiler-library compatibility.
- Local absence of FFTW prevents a real CPU run but does not excuse probe,
  target-disable, source-policy, or no-fallback test failures.

## Phase 3 — cuBLAS, cuSPARSE, cuSOLVER, cuRAND, and Thrust

### Deliverables

- Canonical examples and benchmarks for cuBLAS, cuSPARSE, cuSOLVER, cuRAND,
  and Thrust, implemented in that order.
- Provider-specific CBLAS, oneMKL Sparse, and LAPACKE probes and target logic.
- Production/reference CPU backends with explicit roles and no silent fallback.
- Library-specific verification, canonical-state restoration, and timing scopes.
- OpenACC CUDA-library probes and canonical imported-target link interfaces.
- CMake targets, READMEs, build metadata, manifests, and tests.

### Dependencies

- Phases 1 and 2 patterns.
- Optional CBLAS, oneMKL/OpenBLAS, oneMKL Sparse BLAS, LAPACKE, CUDA Toolkit,
  NVHPC, and OpenMP support.

### Required gate

- All canonical filenames and teaching problems match the project specification.
- All 36 target/executable names exactly match their source stems.
- CBLAS, LAPACKE, and oneMKL Sparse targets require successful real-symbol
  compile-and-link probes; `BLAS_FOUND` alone cannot enable DGEMM.
- Negative and synthetic-positive probe fixtures pass for every CPU provider.
- cuSOLVER's CPU teaching example uses the one-call teaching API while its CPU
  benchmark uses staged factorization/solve corresponding to CUDA/OpenACC.
- cuSOLVER factorization and solve info values are recorded separately.
- Poisson CSR and dense-system helpers match their exact mathematical formulas.
- Warm-up restoration and per-trial restoration reset C, y,
  A/B/pivots/factorization-info/solve-info, random-generator state, and any
  other mutable input before each measured trial.
- Compute-scope tests allow successive operations inside one trial while proving
  that the next trial begins from canonical state; end-to-end tests restore host
  inputs and outputs before every repeat outside the timed interval.
- cuRAND tests distinguish compute-scope trial reset from successive blocks
  within a trial and verify that each end-to-end repeat creates/configures its
  generator from the same start state inside the measured scope.
- OpenACC managed arrays are allocated/moved by data directives only.
- Pointer-dependent descriptors are created only after data entry, and explicit
  allocation is confined to unmanaged library workspace.
- Each OpenACC CUDA-library probe includes the actual header, a CUDA Runtime
  call, and a real target-library call and uses its canonical imported targets.
- OpenACC Thrust has `-cuda` in compile and link options and links
  `CUDA::cudart` for its explicit Runtime synchronization.
- The OpenACC CUDAToolkit path matches the NVHPC-selected Toolkit and all Toolkit
  metadata is recorded.
- cuBLAS/cuSPARSE compute repeats carry state and verify every update, while
  end-to-end repeats restore canonical C/y before each timed pipeline.
- cuSOLVER rejects repeat other than 1 in configuration, runner, and executable
  layers for both scopes.
- cuRAND emits the required interval metadata and four metrics and applies the
  exact mean and second-central-moment formulas using N.
- cuRAND and Thrust serial production baselines record requested/effective
  threads separately and are labeled **Serial CPU baseline**.
- README and plot metadata state that cuRAND compares the same distribution and
  output type but different RNG algorithms.
- Reference backends never replace production backends silently.
- Dependency-free cuRAND CPU and Thrust serial CPU examples/benchmarks build and
  run locally through CMake and their documented direct compile commands.
- Available optimized CPU targets pass; unavailable targets have explicit
  reasons and are not reported as tested.
- All Phase 3 tests and `git diff --check` pass.

### Local validation

- Run CPU-only examples and tests for dependencies already present.
- Always build and run dependency-free cuRAND CPU and Thrust serial CPU targets.
- Unit-test CSR and dense-system generation independently of GPU libraries.
- Statically inspect OpenACC address-translation/data-region ordering.
- Unit-test every mutable-state restoration path, including solver info values
  and random-generator progression.
- Unit-test cuRAND bounds at deterministic synthetic pass/fail boundaries.
- Inspect generated target link interfaces and `-cuda` compile/link options.
- Record every unavailable GPU or optimized CPU test and its reason.
- Re-run all Phase 1 and Phase 2 required gates before opening Phase 4.

### Manual Pegasus validation

- Build each enabled backend in its correct build tree.
- Run examples, pilot benchmarks, verification, and sanitizer where applicable.
- Confirm library versions, effective CPU thread counts, and GPU UUID capture.

### Risks

- Optional CPU APIs and thread-control symbols vary by provider/version.
- Destructive solver operations can invalidate timing if restoration leaks into
  the measured region.
- OpenACC data lifetime errors can cause double allocation or stale addresses.
- NVHPC may select a Toolkit different from a separately discovered Toolkit;
  the build must fail the profile rather than mix them.

## Phase 4 — Suite runner, aggregation, validation, and plotting

### Deliverables

- `run_suite.py`.
- `aggregate.py`.
- `validate_results.py`.
- `plot.py` with optional matplotlib.
- Manifest build/merge and portable environment/provenance utilities.
- Exact pilot and benchmark calibration configurations.
- Python 3.9 compatibility checker and unit tests.

### Dependencies

- Complete benchmark executables and manifests from Phases 1–3.
- Protocol ordering/statistics and result schemas.

### Required gate

- Both canonical configurations contain exactly the approved initial problem
  candidates and scope-specific warm-up/repeat/trials values.
- Equality is enforced per benchmark + normalized problem + scope across CPU,
  CUDA, and OpenACC, without forcing compute and end-to-end repeats equal.
- Execution is interleaved by problem size; it never runs all CPU cases before
  all GPU cases.
- `run_suite.py` is the sole node raw-file writer: it invokes each executable
  through machine-readable stdout, validates records, writes one CSV header,
  and never permits a benchmark to append to a shared raw file.
- Direct path output and node raw output use exclusive creation and reject an
  existing file; CSV header duplication is impossible.
- Human-readable stdout contamination is rejected and preserved in diagnostics.
- Crash, signal, invalid output, and missing output create exactly one synthetic
  attempted failure at the first unresolved index and synthetic skipped rows at
  later unresolved indices, without replacing completed rows.
- `--dry-run` emits deterministic resolved config, manifest, order, and argv
  JSON while creating no directories, metadata, raw files, logs, or processes.
- Implementation permutation and size order use their independent specified
  formulas.
- Six-node/two-wave assignments cover both size orders for each implementation
  permutation; five/eight-node imbalance is recorded.
- Warm-up occurs immediately before each implementation, and canonical state is
  restored before every raw trial and every applicable end-to-end repeat.
- Aggregation proceeds block → wave → cross-wave; pooled exploratory output is
  never primary.
- Primary cross-wave summaries consume wave-level medians only, identify that
  input statistic, and never compute quartiles of quartiles, average IQRs, or
  repool node/block samples.
- Quantiles use `statistics.quantiles(..., method="inclusive")` with exact
  zero/one-sample behavior.
- Speedup uses only the configured production CPU backend and is omitted rather
  than substituted when unavailable.
- cuFFT primary speedup requires `cpu-fftw-threaded`; serial remains auxiliary.
- Each benchmark has exactly one production CPU backend in a primary campaign;
  additional CPU backends are separate campaigns or explicitly auxiliary
  series and never become an automatic speedup denominator.
- Success, failure, and skipped counts remain distinct, with visible origins;
  only valid successful trials enter performance statistics.
- Manifest merge rejects duplicate semantic/build keys, provider ambiguity,
  conflicting path/hash/metadata, and mismatched build profiles.
- Runner checks manifest/build metadata before launch and raw compiler/Git
  metadata after launch.
- Serial cuRAND/Thrust CPU baselines use the exact graph label and cuRAND plots
  carry the algorithm-comparison note in deterministic plot metadata.
- Every Python source passes AST parsing as Python 3.9, normal byte-compilation,
  standard-library API compatibility checks, and unit tests.
- All Phase 4 tests and `git diff --check` pass.

### Local validation

- Unit-test every permutation, size order, and multi-wave combination.
- Unit-test benchmark stdout validation, one-header append behavior, exclusive
  creation, stdout contamination, and every failure/skipped synthesis case.
- Verify dry-run filesystem and subprocess side effects with before/after
  directory snapshots and injected subprocess mocks.
- Unit-test all-trial and end-to-end-repeat state restoration through the suite
  execution loop.
- Unit-test hierarchy with 0, 1, 2, 5, 6, and 8 valid samples.
- Unit-test cross-wave summaries with unequal block counts to prove that the
  primary input is one median per wave rather than a flat pool.
- Validate malformed, duplicate, failed, nonfinite, and mixed-provenance input.
- Validate exact calibration configuration values and scope structure.
- Test manifest identity, merge collisions, build metadata comparison, and
  dirty source-snapshot hashing.
- Run `ast.parse(feature_version=(3, 9))`, compileall/py_compile with a temporary
  bytecode prefix, and the Python 3.9 API compatibility audit.
- Verify plotting's dependency-missing diagnostic without installing anything.
- Re-run all earlier required gates before opening Phase 5.

### Manual Pegasus validation

- Run a one/two-node pilot and inspect interleaving, assignments, raw rows, and
  aggregation before production.

### Risks

- Incorrect grouping can destroy paired comparisons.
- Partial waves can be mistaken for balanced data without assignment metadata.
- A mock-only dry-run test can miss filesystem writes; integration tests also
  compare an actual temporary directory tree before and after invocation.

## Phase 5 — Pegasus execution and collection

### Deliverables

- Pegasus README and JSON example.
- Job renderer and PBS template.
- `prepare_wave.py` for PBS job-master preflight and exclusive metadata
  creation.
- Separate CPU/CUDA and OpenACC build scripts.
- Runtime-environment collector and deterministic hash.
- Node runner, interval-aware telemetry, node status, and result collector.

### Dependencies

- Phase 4 runner/tools.
- Human-supplied Pegasus account, queue, module, and path settings.
- Prebuilt executable manifest.

### Required gate

- Pegasus configuration has separate `cpu_cuda_build_modules`,
  `openacc_build_modules`, and `benchmark_runtime_modules`, and each workflow
  loads only the appropriate group.
- The CPU/CUDA and OpenACC scripts enforce their exact build-profile settings
  and emit nonconflicting partial manifests.
- OpenACC build selection records and verifies CUDAToolkit/NVHPC CUDA path,
  versions, flags, and `NVHPC_CUDA_HOME` when present.
- The job master creates deterministic `runtime-environment.json` before
  preflight with every required module, path, driver/runtime/compiler/library,
  binary `ldd`, resolved-library, CPU-runtime, and NVHPC CUDA field.
- The exact runtime-environment hash is stored in campaign, wave, node, and raw
  provenance and is an immutable additional-wave match condition.
- Before the measurement launch, the PBS job master obtains a complete
  rank-host mapping with a short preflight launch and runs `prepare_wave.py`.
- The PBS job master alone validates campaign/wave identity, output collisions,
  mapping, and provenance and exclusively creates campaign/wave metadata; no
  benchmark-process collective coordinates metadata creation.
- The main measurement launch starts only after successful preflight, including
  for a single-node job, and each node process writes only its own node-local
  and shared node subtree.
- Existing wave/node output is never overwritten.
- Additional waves require matching configuration, manifest/binary, Git, and
  dirty-source provenance plus an identical runtime-environment hash.
- A runtime-environment mismatch explicitly requires a new run ID.
- Production rejects a dirty worktree before measurement.
- Rendered PBS, runtime environment, node metadata, and telemetry contain all
  eight CPU runtime variables with requested/effective thread distinction.
- Node-local scratch is recovered on normal and abnormal exit.
- Trial timestamps and telemetry timestamp metadata support interval-based UTC
  correlation, timezone/offset interpretation, and counted local-midnight
  rollovers in timestamped dmon output.
- Telemetry metadata has start/end UTC timestamps, sample interval, timezone,
  UTC offset, and `midnight_rollover_count`. Raw trial timestamps all have exact
  UTC millisecond format.
- Tests prove that a measurement UTC span may exceed `elapsed_total_sec` without
  making a valid end-to-end row inconsistent.
- GPU telemetry preserves raw columns and treats parse failure as non-fatal.
- CPU telemetry records required topology, binding, environment, module, and
  compiler information without privileged changes.
- Job submission remains a human action and benchmark jobs never compile.
- `run_node.sh` records benchmark, verification, and nonfatal tool failures in
  `node-status.json`, stops telemetry, and collects artifacts through its trap.
- A node runner that successfully writes status and collects artifacts exits 0
  after benchmark/tool failure so `mpirun` does not terminate other nodes.
- Scratch/status/shared-directory creation failure or artifact-collection
  failure is fatal infrastructure failure and may return nonzero.
- The PBS job master always runs `collect_results.py` after measurement
  `mpirun`. The collector checks every expected node status and causes the job
  master to exit nonzero for any missing node, benchmark failure, verification
  failure, or collection failure while preserving all artifacts.
- The collector lists and hashes collected artifacts but never aggregates,
  rewrites, deletes, or filters raw data.
- A simulated single-node benchmark failure cannot prevent other simulated
  nodes from collecting their artifacts.
- All Phase 5 tests, `bash -n`, applicable ShellCheck, and `git diff --check`
  pass.

### Local validation

- Render templates with synthetic, non-secret settings in temporary storage.
- Test missing-required-value failures and path/run-ID collision detection.
- Unit-test preflight mapping completeness, duplicate hostnames, metadata
  exclusive creation, provenance mismatch, and the single-node path.
- Unit-test runtime-environment hashing and rejection of a wave with changed
  runtime provenance.
- Unit-test normal, benchmark-failure, verification-failure, signal,
  missing-status, and collection-failure paths across multiple simulated nodes.
- Unit-test telemetry parsers against captured/synthetic headers without
  executing scheduler commands, including timezone, UTC offset, interval
  correlation, and midnight rollover.
- Verify all eight CPU runtime variables in rendered and collected artifacts.
- Verify node exit-code isolation and the collector/job-master final decision in
  separate tests.
- Run `bash -n` on every shell script and synthetic rendered PBS file.
- Run ShellCheck only when already available; otherwise record it as unexecuted
  and do not install it.
- Re-run all earlier required gates before opening Phase 6.

### Manual Pegasus validation

- Human checks queue/module/budget state.
- Build on an appropriate compute node.
- Submit a debug smoke job, then inspect trap recovery, metadata, hashes,
  telemetry, raw results, and aggregate eligibility.
- Proceed to five or preferably eight production nodes only after smoke success.

### Risks

- Scheduler/module changes can invalidate rendered assumptions.
- Abrupt termination can leave partial node collection requiring explicit
  status handling.
- Shared-filesystem timing or permissions can affect recovery.
- `ldd` and NVHPC selection probes are Pegasus/Linux validations; a local
  unavailable command is recorded rather than replaced by guessed data.

## Phase 6 — Documentation and final validation

### Deliverables

- Updated user-facing README and six complete library READMEs after
  implementation.
- `PORTABILITY.md`, `ADDING_A_NEW_VENDOR.md`, `ADDING_FORTRAN.md`, and the
  Japanese Pegasus manual-validation guide.
- A cross-document implementation/specification consistency audit. Phase 6 does
  not define schema, protocol, or Pegasus behavior for the first time.
- Complete local validation report.
- Manual Pegasus checklist and run report.

### Dependencies

- Phases 1–5 complete.
- Project owner review of teaching-material correspondence.

### Required gate

- User-facing documentation summarizes implemented behavior without replacing
  authoritative specifications.
- Every canonical example and benchmark is listed under the correct library and
  matches its canonical source/target/executable name.
- Direct compile commands for dependency-free CPU targets build and run; other
  commands state their dependencies without inventing local paths or versions.
- Scope-specific configuration, output ownership, primary/auxiliary CPU roles,
  failure/skipped semantics, provenance, and node failure isolation agree with
  the synchronized specifications and implemented tests.
- cuRAND documentation and plot metadata explain the distribution/type task
  comparison and different algorithms; cuRAND/Thrust serial CPU labels are
  exact.
- GPU and optimized CPU areas not executed locally remain visibly unverified.
- Every changed file, command, successful test, unexecuted test/reason, and
  required Pegasus check is reported.
- No GPU test is claimed successful unless actually run.
- No TODO-only implementation is reported complete.
- Canonical filenames contain no manually maintained version suffix.
- Full-text policy searches find no hard-coded architecture/site values,
  prohibited identifiers, scheduler execution in generated tools, or future
  placeholder implementations. `cublas_v2.h` is the documented upstream-name
  exception to version-suffix search.
- Every required gate from Phases 1–5 is rerun or its preserved evidence is
  audited, and the complete local test suite passes.
- `git diff --check` passes; `git diff --stat` and `git status --short` are
  captured for handoff.

### Local validation

- Configure and test the CPU-only profile in
  `/tmp/gpu-library-suite-local-build/cpu-only-validation`.
- Build and run dependency-free common, cuRAND CPU, and Thrust serial CPU
  targets plus all unit tests.
- Run schema, CLI, ordering, aggregation, provenance, Pegasus simulation,
  Python 3.9 compatibility, shell, and documentation consistency checks.
- Run the required canonical-version searches for `benchmark-v1`,
  `benchmark-v2`, `pilot-v1`, `run_suite_v1`, `-v1.json`, `-v2.json`,
  `-final.json`, and `-latest.json`.
- Inspect `git diff`, `git diff --check`, `git diff --stat`, and
  `git status --short`, preserving unrelated user changes.

### Manual Pegasus validation

- Complete smoke and production checklists with recorded provenance.
- Confirm result recovery and telemetry for normal and abnormal exit paths.

### Risks

- Documentation can drift if ownership boundaries are ignored.
- A locally unavailable dependency can leave a target manually unverified.

## External inputs and intentionally deferred scope

The implementation plan is decision-complete without guessing external site
state. The following remain human inputs or future scope rather than decisions
left to the implementer:

- Pegasus account, available queue, module names/versions, MPI version, CPU
  library installation path, NVHPC CUDA selection, and shared output path.
- Actual GPU/CPU performance, production-calibrated sizes/repeats after pilot,
  and anomaly/exclusion policy.
- Project license and publication policy.
- Future NVIDIA Fortran, AMD C/C++/Fortran, common Fortran, Miyabi, TSUBAME4.0,
  and Sirius implementation. Furo remains outside the approved job scope.

No empty directory, dummy target, or placeholder implementation is created for
deferred source areas. Configuration examples use explicit placeholders for
human-supplied site values.

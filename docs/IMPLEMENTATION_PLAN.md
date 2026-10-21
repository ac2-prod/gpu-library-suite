# GPU Library Suite Implementation Plan

## Authority and scope

This document owns implementation phases, deliverables, dependencies,
acceptance criteria, validation activities, risks, and unresolved decisions. It
does not redefine command-line behavior, mathematical problems, result schemas,
or system operating procedures. Those specifications are authoritative in:

- [`PROJECT_SPECIFICATION.md`](PROJECT_SPECIFICATION.md)
- [`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md)
- [`RESULT_SCHEMA.md`](RESULT_SCHEMA.md)
- [`PEGASUS_EXECUTION.md`](PEGASUS_EXECUTION.md)
- [`jobs/sirius/README.md`](../jobs/sirius/README.md)

The decision-complete details below are implementation requirements and inputs
to the Phase 1 specification-delta gate. Once synchronized, each owning
specification remains normative; this plan does not become a competing schema,
protocol, or operating manual.

The current [code-publication scope](../README.md#publication-scope) retains the
completed implementation and documentation. The phase criteria below describe
implementation and measurement validation; they do not require publication of
private measurement artifacts or additional general-purpose measurement
infrastructure before this code publication.

The original NVIDIA Phases 1–6 below retain their historical scope. The later
[AMD C/C++ and Sirius stages](#amd-cc-and-sirius-development-stages) have separate
approval gates. The recorded rocFFT build/basic-smoke milestone and the later
36-artifact six-family build are complete at their respective source snapshots.
The later focused jobs completed basic numerical smoke for the five remaining
families. NVIDIA local/fixture coexistence checks and the separately approved
Pegasus 36-artifact build/72-case smoke passed. Their source boundaries and
unresolved evidence are in [VALIDATION_REPORT](VALIDATION_REPORT.md#current-coexistence-status).
AMD C/C++ performance evaluation and the aligned AMD Fortran build, smoke and
two-wave evaluation are also complete at their recorded versions. NVIDIA
Fortran timing/build alignment has completed its own 18-benchmark rebuild,
72-case smoke and 6,480-row two-wave evaluation; see the
[current NVIDIA Fortran result](VALIDATION_REPORT.md#fortran-production-validation).
The [AMD Fortran completion status](#amd-fortran-source-implementation) and
validation report distinguish current results from earlier preparation milestones.
Further real execution requires separate approval.

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
host. Pegasus submission remains a human action. Sirius follows the latest local Shell
and separately approved noninteractive SSH/scheduler policy in its owning
operations document. Unavailable tests are reported rather than inferred successful.
The current AMD/Sirius implementation permits necessary source reads and
writes over the existing SSHFS mount, with local validation on a copied tree.
The completed rocFFT baseline is separate from future real execution, which needs
explicit approval under [Sirius operations](../jobs/sirius/README.md); neither
this plan nor a local test pass grants remote permission.

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

### Publication calibration configurations

Both canonical configurations encode `warmup`, `repeat`, and `trials` under each
scope. Implementations cannot override them. Equality is enforced across CPU,
CUDA, and OpenACC for each benchmark + normalized problem + scope; compute and
end-to-end scopes remain independent.

Both files retain `config_schema_version=1`; ordinary calibration changes do
not change that value.

Each case has the structural shape `problem` plus a `scopes` object containing
separate `compute` and `end-to-end` objects. Both canonical configurations use
JSON Lines as the node raw default, enable verification, and set
`continue_on_failure=true`; failures still make final campaign validation fail.
Both canonical publication calibration configurations request 48 CPU threads. A
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
      cpu_parallelism
      cpu_threads_effective
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

`configs/pilot.json` and `configs/benchmark.json` use the same three-case
publication workloads and compute-repeat candidates owned by
[`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md). Pilot uses one trial and
production uses five; both use warm-up 1 and end-to-end repeat 1. The expected
counts with Thrust are 108 pilot rows and 6,480 production rows. A human may
revise the canonical values once after the Pegasus pilot for a demonstrated
OOM, verification, walltime, or short-interval problem. No automatic repeat
search or competing canonical file is introduced.

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

Pegasus publication measurement uses `cpu-fftw-threaded` as the only cuFFT CPU
series. A separately invoked `cpu-fftw-serial` backend remains available for
teaching correspondence but is not present in either canonical publication
configuration and never replaces the threaded series.

cuRAND's `cpu-std-random-serial` and Thrust's `cpu-stl-serial` remain configured
production serial implementations. Publication plots label both exactly **CPU
serial reference**, never a parallel or algorithm-equivalent baseline, even
when the campaign requested 48 CPU threads.

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

This document is the stable software identity: module display layout, duplicate
search-path entries, `ldd` ASLR addresses/line order, job-master GPU UUID/name,
hostname, scheduler identity, timestamps, PIDs, and scratch paths are excluded.
Their unnormalized probe output is retained per wave in the separately hashed
`job-master/runtime-environment-evidence.json` sidecar. The sidecar links to the
canonical runtime and manifest hashes but may differ across otherwise compatible
waves.

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

### Pre-merge code-review remediation gate

The initial implementation remains pre-merge, so review-driven compatible
schema corrections are incorporated into schema version 1. No version-suffixed
schema or canonical filename is introduced. Before the implementation can be
reported review-ready, all of the following are required in addition to the
phase gates below:

- strict Linux C17 tests expose POSIX declarations on each test target and all
  C executables/tests that use mathematics link the platform math interface;
- every C argv fixture derives its own argument count, and sanitizer or exact
  count coverage prevents out-of-bounds parser access;
- fixture build and Python cache directories stay below the current
  `CMAKE_BINARY_DIR`, so source copies and consecutive build trees cannot share
  stale CMake caches;
- `.git`-less source produces valid deterministic metadata with explicit
  unavailable/null Git semantics, never inferred clean state, while production
  rejects unavailable Git provenance;
- CPU role/series/parallelism/effective-thread fields have one configuration
  source of truth, runner manifest lookup rejects duplicate benchmark keys, and
  C++ result initialization releases partially initialized state;
- raw/manifest flag metadata uses the scoped
  `global_configure_flags` name, and suite commands always carry every
  verification operand from effective configuration;
- all 18 benchmarks pass timestamp-placement, fatal-versus-verification
  continuation, setup/warm-up/API return checking, and GPU/runtime/library
  metadata source-policy tests; OpenACC cuSOLVER copies both info values back as
  part of end-to-end result retrieval;
- canonical examples remove redundant direct-CUDA synchronization, retain the
  required OpenACC completion boundary, use default public CPU headers, keep
  oneMKL Sparse as the canonical sparse lesson, check major API results, and
  handle a zero-byte cuSPARSE workspace;
- Pegasus preflight and measurement both preserve word splitting for
  `NQSV_MPIOPTS`, runtime compiler commands are optional by default, GPU identity
  is node-local, failure isolation remains intact, and PBS output parents are
  checked before human submission;
- CBLAS/LAPACKE discovery uses provider-specific cache variables and rejects
  mixed provider headers/libraries after a provider change;
- all six library READMEs contain the common required sections, the validation
  report distinguishes Mac evidence from Linux portability, and a minimal
  dependency-free Linux GCC/Clang CPU-only CI gate is present; and
- Mac CPU-only, clean Linux GCC, available Linux Clang, `.git`-less source,
  Python 3.9 syntax/bytecode/unit, shell, documentation/policy, and
  `git diff --check` gates pass. Actual CUDA, NVHPC, GPU, and Pegasus execution
  remains explicitly manual when unavailable.

### Second independent-review completion gate

Before merge readiness, the initial schema-version-1 implementation also
requires the following mutually consistent corrections and regressions:

- C and Python raw validators enforce identical success/failure/skipped,
  verification-origin, positive-size, non-negative-device, safe-hostname,
  exact-block-ID, and primary-metric membership invariants;
- `run_suite.py` compares every config-, manifest-, provenance-, and execution-
  context-owned raw field before its sole writer accepts the subprocess row,
  including library-specific verification threshold operands;
- common checked arithmetic covers unsigned-to-API conversions, dimension
  products/additions, and element byte counts, and all 18 benchmarks reject
  overflow, API-range truncation, zero/negative workspace mistakes, and C++
  length/allocation failures with schema-valid outcomes;
- compute warm-up uses and retains only its persistent compute context, while
  end-to-end warm-up executes complete temporary end-to-end pipelines and
  leaves no persistent compute resource before measurement; and
- all numerical verification checks finiteness before maxima/norm reductions,
  records nonfinite metrics as JSON null with overall verification failure, and
  treats JSON object/allocation/insertion failure as fatal result construction.

Required evidence includes C/C++ schema invariants, runner mismatch rejection,
overflow boundary tests, scope-specific warm-up lifetime tests, injected NaN
and both infinities, the complete Mac CPU/non-Git/Python/shell/documentation
gates, and `git diff --check`. The Linux GCC/Clang workflow keeps the same
dependency-free matrix and uses the approved current checkout action major.
Real CUDA, NVHPC, GPU, and Pegasus validation remains a separately reported
human gate when unavailable locally.

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
- `fft_cpu_bench` with distinct threaded and serial backend capability; the
  publication configuration selects only `cpu-fftw-threaded`.
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
- Pegasus publication configuration selects only `cpu-fftw-threaded`; a serial
  teaching-correspondence invocation is separate and never becomes primary.
- If threaded FFTW is unavailable, serial never becomes a publication series.
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
- cuRAND and Thrust serial production implementations record
  requested/effective threads separately and are labeled **CPU serial
  reference** in publication output.
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

- Both canonical configurations contain exactly the approved three-case
  publication candidates and scope-specific warm-up/repeat/trials values.
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
- cuFFT publication output contains only `cpu-fftw-threaded`; serial remains a
  separate teaching-correspondence capability.
- Each benchmark has exactly one production CPU backend in a primary campaign;
  additional CPU backends are separate campaigns or explicitly auxiliary
  series and never become an automatic speedup denominator.
- Success, failure, and skipped counts remain distinct, with visible origins;
  only valid successful trials enter performance statistics.
- Manifest merge rejects duplicate semantic/build keys, provider ambiguity,
  conflicting path/hash/metadata, and mismatched build profiles.
- Runner checks manifest/build metadata before launch and raw compiler/Git
  metadata after launch.
- Plotting emits exactly one two-panel millisecond elapsed-time figure per
  publication library from cross-wave summaries, with no speedup, throughput,
  reuse, amortized, or extra generic figure.
- cuRAND/Thrust are explicitly labeled single-threaded, with the exact approved
  legends owned by [Publication figures](BENCHMARK_PROTOCOL.md#publication-figures).
  cuRAND carries the algorithm-comparison note, and Thrust requires equal
  CUDA/OpenACC raw-result `library_version` evidence before any figure is written.
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
- The job master retains unnormalized runtime probe output in a per-wave raw
  evidence sidecar; wave metadata hashes it and the collector validates its
  canonical-runtime/manifest links without using its hash as a campaign gate.
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
- Submit a batch smoke job to the `gpu` queue, then inspect trap recovery,
  metadata, hashes, telemetry, raw results, and aggregate eligibility.
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
  comparison and different algorithms; cuRAND/Thrust use the exact **CPU serial
  reference** label.
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

## Approved Fortran implementation follow-on

The owner-approved 2026-09-17 Fortran extraction is now integrated separately
from the completed initial C/C++ phases: 18 teaching programs, 18 benchmarks,
helpers, the shared Thrust C++ function, optional-language CMake, language-aware
manifest/runner/plotting and English/Japanese usage. The implementation uses
Fortran 2008 features plus NVHPC CUDA/OpenACC extensions. It does not require
Fortran in the default C/C++ profile. See [integration requirements](ADDING_FORTRAN.md),
[usage](../nvidia/fortran/README.md) and
[local validation](VALIDATION_REPORT.md#fortran-local-validation).
The current [Pegasus validation and production record](VALIDATION_REPORT.md#fortran-production-validation)
covers the C/C++-aligned timing and build profiles at commit `6481f47`, an
18-benchmark rebuild, 72-case smoke and 6,480 verified rows across two waves
of six distinct hosts each. Aggregation and all 108 plotted points were checked
against those records. Earlier campaigns and failed attempts remain historical;
C/C++ and AMD results are unchanged. This evidence is separate from source
completion and CPU CI; its recorded limits and private-data boundary remain.

## AMD C/C++ and Sirius development stages

AMD C/C++ and Sirius are approved development targets. Stages 1 and 2 and the
first rocFFT build/basic-smoke milestone are complete for the recorded source
snapshot in [VALIDATION_REPORT](VALIDATION_REPORT.md#amd-rocfft-sirius-baseline).
The expansion to rocBLAS, rocSPARSE, rocSOLVER, rocRAND and rocThrust includes
CPU/HIP/OpenMP teaching sources, benchmarks, CMake integration, strict
provenance and local tests. The separately approved AOCL Sparse follow-up
verified all 36 real build artifacts at its recorded snapshot. Subsequent
focused jobs passed the remaining five families' basic numerical smoke; see
[the recorded results](VALIDATION_REPORT.md#amd-focused-numerical-smoke).
NVIDIA local/fixture coexistence checks and the Pegasus 36-artifact build/72-case
smoke passed at clean commit `eb9e594be3ea4f1dc4bef6fb562223961c60a43a`, transferred
using [the Git handoff procedure](PEGASUS_EXECUTION.md#git経由の検証版反映).
This closes the requested coexistence check within the
[recorded coverage](VALIDATION_REPORT.md#nvidia-coexistence-review-after-amd-addition).
The AMD real runs retain their earlier source snapshots; they are not relabelled
as tests of this commit. Performance evaluation needs separate resource/execution
approval. CPU worker-count and complete loaded-library evidence limitations
remain open, independently of numerical smoke.
The teaching inventory belongs to PROJECT_SPECIFICATION; Sirius operations
belong to jobs/sirius/README.md.

### Stage boundaries and approvals

| Stage | Deliverable | Permission and stopping gate |
| --- | --- | --- |
| 0 — Investigation | Repository facts, safe existing local-test evidence, and revised development plan. | Read-only investigation does not authorize implementation or remote access. |
| 1 — Local preparation | The documents, Sirius helper/probe, mock/fake tests, and fixed safe-validation entry listed below. | Completed historical local stage; its real SSH and scheduler counts remain zero. |
| 2 — Environment probe | One approved lightweight probe and a report of observed environment facts. | Completed via one approved owner-authenticated interactive SSH in the integrated terminal; Codex reviewed saved output. Historical failed transport records remain unchanged. |
| 3 — AMD implementation and validation | Complete all six CPU/HIP/OpenMP teaching/benchmark families; reconcile docs, CMake, result tooling and local checks; then final NVIDIA coexistence review. | Recorded AMD build/basic smoke and NVIDIA local/fixture plus Git-fixed Pegasus build/72-case smoke passed within their source boundaries. CPU worker-count and complete loaded-library evidence limitations remain. Further real execution requires separate approval. |
| 4 — Pilot | Verification, runtime/memory observations, and proposed production parameters. | Separately approve resources, sizes, submission count, and observation budget; analyze captured failures before another run. |
| 5 — Fixed version and production | A reviewed fixed source revision, clean rebuild, immutable provenance, and raw measurements. | Obtain measurement-version commit approval and separate execution approval. Do not modify the measured source/configuration or mix provenance. |
| 6 — Results and publication | Validated raw results, aggregation, six figures, and usage guidance. | Slide integration and push/PR/merge require their own authorization. Preserve private operational and measurement artifacts. |

No stage automatically grants permission for the next. The current task does
not authorize branch creation, commit or publication, or additional SSH,
scheduler/accounting, remote builds or GPU execution. Preserve Pegasus rules,
old runs and authentication state. A successful rocFFT baseline is not proof of
other libraries or of later source snapshots.

Expansion acceptance requires complete canonical source inventory, optional
per-library compile/link gates, CPU-only builds without ROCm, independent
CPU/HIP/OpenMP artifacts, real failure/nonfinite checks in fixtures, correct
restoration/timing, and campaign-wide provenance rejection. Explicitly report
unexecuted GPU/build checks. Do not start performance measurement until the
full AMD implementation and final coexistence review have been accepted.

The first six-family **allocated build-only** job requested 36 teaching/benchmark
artifacts in three independent CPU/HIP/OpenMP trees. It verified 24 HIP/OpenMP
artifacts; the CPU BLAS benchmark failed in an AOCL header and stopped the old
grouped CPU build. Two CPU Sparse targets lacked their production dependency.
The full evidence and incomplete CPU outcomes are recorded in
[VALIDATION_REPORT](VALIDATION_REPORT.md#amd-six-family-sirius-build).
The recovery aligns AMD CBLAS/LAPACKE feature-test preambles and
their configure probes, attempts independent targets separately, and records
unattempted targets and shared-dependency failures without retrying them.
Successful subsets require complete manifest/provenance checks; no binary is
accepted merely because a file exists. Profile and overall deadlines remain
shared, and fixture tests cover continuation and partial-result integrity.
The subsequent approved run verified 34 artifacts with no failed targets;
only the two selected oneMKL Sparse CPU targets lacked their dependency.
This confirms the recovery at its recorded source snapshot, not later edits.
The owner-approved local follow-up adds explicit AOCL Sparse selection,
teaching/benchmark API use, discovery/link validation, header/library and runtime
provenance, and fixture regressions. It retains oneMKL and no-fallback rules.
Sirius smoke inputs select the new provider explicitly and preserve all six
CPU/HIP/OpenMP families. The separately approved run
`sirius-amd-build-20261017T104351Z-a3697cae` (PBS `45731.pbs`) completed with
36 verified artifacts: 12 each for CPU, HIP and OpenMP. The AOCL Sparse LP64/API
compile/link check and both CPU Sparse artifacts passed. PBS and SSH exits
were zero. Its source snapshot was
`a3697cae2e446a51eec5a6107dedc39caf656fc8158b08e3591c59f0bc1148e6`;
see [the retained build evidence](VALIDATION_REPORT.md#amd-six-family-build-with-aocl-sparse).
Its operating contract belongs to
[Sirius operations](../jobs/sirius/README.md#prepared-six-family-allocated-build).
An unavailable CPU provider, disabled GPU target or failed compile/link is an
explicit incomplete result; a zero configure exit alone cannot close this
gate. The job gathers the remaining bounded CPU-provider inventory in the same
allocation and applies the explicitly reviewed AOCL candidate-selection rule
from the Sirius operating contract. CMake must verify the selected API link
inputs; missing or ambiguous candidates remain disabled. It does not run the
artifacts or replace a missing production CPU Sparse backend with a reference.
The successful build closes the compile/link gate for that snapshot; it ran
no generated programs. The later focused smoke closed the basic numerical
gate for the five new families without resolving complete runtime-library
bindings or effective worker counts. The subsequent NVIDIA coexistence check
also passed its bounded build/basic-smoke gate. Performance preparation follows
the minimal plan below; build preparation itself grants no execution permission.

The prebuilt six-family diagnostic smoke harness is implemented locally.
It verifies the entire original source snapshot and all 36 retained artifacts
before using the frozen runner/config/validator. Launcher identity is recorded
separately from binary source provenance. The intended acceptance is 18 teaching
example completions plus four distinct 72-row campaigns (serial/threaded CPU
requests × JSONL/CSV), with every numerical metric checked against the owning
protocol. Two preliminary JSONL campaigns supply runtime-library evidence and
are excluded from the 288 main rows. Artifact-specific link/loader checks must
not mistake headers or static inputs for DSOs. Loader diagnostics do not prove
worker counts or GPU code-object identity and cannot be used as performance
measurements. See the [Sirius diagnostic-smoke operating contract](../jobs/sirius/README.md#retained-six-family-diagnostic-smoke).
Local fake validation is preparation only. The historical diagnostic attempts
did not close their complete evidence gate; the later focused jobs established
basic numerical operation separately, as recorded in VALIDATION_REPORT.
The first approved attempt stopped before numerical execution on a missing
native-HIP `link.d`; see the
[retained failure](VALIDATION_REPORT.md#amd-six-family-smoke-link-evidence-failure).
Recovery uses an explicit HIP link-command proof from the successful target's
hash-verified verbose log and matching `link.txt`, preserving CPU/OpenMP depfile
checks and actual loader observations. Every artifact's evidence must pass
before any child command, including `ldd`. Local regression coverage includes
HIP without a depfile, wrong target/compiler, changed or ambiguous logs, and
failure on the last of 36 artifacts without launching an earlier one. Reusing
the binaries preserves their original source identity; it does not authorize
a new smoke job or close the runtime gate.

The second allocated attempt exposed an additional depfile assumption: the
OpenMP compiler's temporary image-wrapper object was treated as a persistent
dependency. Recovery records object prerequisites as historical link inputs
and continues to verify persistent libraries/archives/scripts and loaded DSO
identities. Local acceptance must exercise the production parser on missing
temporary objects, reject missing persistent dependencies, and cover the
complete 36-artifact preflight without launching a child. Replaying saved
depfile syntax with local fake files must be distinguished from checking live
Sirius dependencies. No previous failed run is rewritten; another allocated
smoke still requires separate approval.

The third allocated attempt passed all 36 link and `ldd` checks and ran the
first CPU BLAS example successfully, then exposed an invalid loader-parser
assumption: initialization messages were required to have matching DSO
finalization messages. The local recovery separates observed initialization
identity from destructor events and from the owned child's exit/timeout proof.
Acceptance includes replay of the saved trace without remote filesystem access,
sanitized trace fixtures with missing DSO finalizers, strict PID/program/event
validation, unchanged binary/provider/link binding and production-controller
integration with successful, failed and timed-out fake children. Missing selected
provider initialization remains a failure. Do not infer a complete trace or
effective worker count from accepted initialization events. Preserve all failed
run records. The later focused jobs verified basic numerical operation without
closing this diagnostic harness's complete runtime-evidence gate.

Local AOCL acceptance covers teaching and benchmark recurrence in both scopes,
warm-up restoration including zero warm-up, API/cleanup failure handling,
requested-versus-observed threads, version and environment recording, explicit
LP64 header validation and ILP64 rejection, ordered static link inputs, missing
dependency isolation, oneMKL/reference preservation, and campaign mismatch
rejection. Those local fixtures establish neither real numerical correctness,
loaded-DSO identity nor worker behavior. The later focused smoke supplies basic
numerical evidence; complete loaded-library identity and effective worker
counts remain unverified.

### Minimal performance evaluation plan

Status: **measurement closed; saved results audited, aggregated and plotted**. The
completed evidence and rendering status are recorded in
[VALIDATION_REPORT](VALIDATION_REPORT.md#amd-production-comparison).
The accepted FFT timing difference remains [D-076](DECISIONS.md); no timing
rewrite, NVIDIA remeasurement, further AMD wave or diagnostic campaign is
required. Sampling and statistical rules remain owned by
[BENCHMARK_PROTOCOL](BENCHMARK_PROTOCOL.md#matched-nvidiaamd-evaluation),
provenance by [RESULT_SCHEMA](RESULT_SCHEMA.md), and site execution by
[the Sirius document](../jobs/sirius/README.md#amd-production-preparation).

1. **Completed measurement gates.** The three-size pilot passed 108 rows.
   Approved clean builds supplied 18 benchmark artifacts; runtime collection
   and per-allocation CPU/NUMA/APU checks completed. Production used six
   exclusive physical hosts, one MPI worker/APU and 24 CPUs per host. The
   originally planned 6,480 rows must not be reported as all successful:
   wave 0 failed on a01; the owner accepted all five complete blocks, followed
   by a complete six-node wave 1 (5,940 accepted rows).
2. **Completed parallel-BLIS correction.** The original BLAS/Solver links were
   unsuitable for the intended parallel CPU baseline. The new clean build
   selects parallel BLIS without serial fallback; both families were measured
   again with CPU/HIP/OpenMP paired within each new node block, two six-host
   waves and 2,160 PASS rows. The older results and failure evidence remain.
3. **Final selection.** Use 3,960 retained FFT/Sparse/RAND/Thrust rows and
   2,160 corrected BLAS/Solver rows. Keep the two campaigns' source, binaries,
   config, runtime identities and aggregation separate. Retain all successful
   timings, including slow blocks. Both waves overlap in physical hosts; this
   is six distinct hosts overall, not twelve independent hosts. No missing
   trial or failed block is filled in.
4. **Reporting gate.** Replay the existing strict readers, hierarchical
   aggregation and NVIDIA figure renderer. Preserve library/end-to-end,
   per-operation milliseconds, three sizes and requested/serial CPU labels.
   Store exact inputs, selection, output hashes and rendering status in a new
   report directory; keep the measured datasets unchanged. No new benchmark
   framework or broad local test campaign is required.

The selected CPU providers are AOCL OpenMP FFTW, parallel BLIS, AOCL Sparse,
libFLAME/parallel BLIS, serial standard RNG and serial STL. Configured limits
are distinct from observed active workers. Complete actual loaded-DSO tracing
and BLAS/Sparse/Solver active worker counts remain unverified, as accepted by
the owner; do not silently mark them verified or add a campaign to obtain them.
The earlier a01 error's root cause is unresolved; its later successful wave
does not prove a hardware fix. Fortran/NVIDIA evidence remains unchanged.

No new build, measurement, SSH, scheduler action or Git publication is authorized
by this completed reporting plan. Any later change to measured code or providers
requires its own provenance and approved scope.

### Stage 1 deliverables

Document changes are limited to `AGENTS.md`, `PROJECT_SPECIFICATION.md`, this
plan, `DECISIONS.md`, and `ADDING_A_NEW_VENDOR.md`. They establish development
scope, ownership, the canonical `rocthrust_wrapper.hip` filename, the sole
rocThrust OpenMP two-translation-unit exception, and the approval gates.
Neither `RESULT_SCHEMA.md` nor `BENCHMARK_PROTOCOL.md` changes in this stage.

The Sirius files are `jobs/sirius/README.md`, `jobs/sirius/remote.py`, and
`jobs/sirius/probe.sh`, with `tests/python/test_sirius_remote.py`. The helper is
limited to an approved probe, not arbitrary remote commands. All helper tests
use mock/fake process boundaries. Their required coverage includes dry-run,
script hashing, process-wide serialization, persistent approval/execution
budgets, crashes/restarts, unknown execution outcomes, authentication/host-key/
connection-limit failures, log separation, and absence of unrelated connection
termination. Exact operating behavior is defined by the Sirius owner document.

Reproducible local validation uses the existing CTest/Python frameworks and
only these entry/profile files:

- `tests/python/run_safe_local_validation.py`: a fixed CPU-only configure/build,
  selected CTest, Python compatibility/unit, and suite-runner dry-run entry.
- `tests/python/run_profiled_unittests.py`: a safe independent profile and
  structured results, retaining existing profiles.
- `tests/python/test_safe_local_validation.py`: selection, output boundaries,
  classification, and missing/unclassified-test checks.

Run against a bounded local source copy under
`/tmp/gpu-library-suite-local-build`, with dedicated build, fixture, log, and
cache paths. Do not build on the mounted source workspace or reconstruct the
selection with ad hoc `python3 -c`. The fixed entry owns the exact selected test
IDs and exclusion reasons. Unknown or disappeared tests require review rather
than silent execution or omission. Reports distinguish `PASS`, `FAIL`, `SKIP`,
`POLICY_EXCLUDED`, and `NOT_RUN_DEPENDENCY`, with test IDs, reasons, commands,
tool versions, exit values, and stdout/stderr evidence. A prior test count is
not a substitute for complete selected-test accounting.

### Stage 1 acceptance and handoff

Every condition below is required before Stage 1 is reported complete:

- Actual SSH calls: zero. Actual scheduler/accounting operations: zero.
- Every required remote-helper mock/fake test has executed and passed; a skip
  or unexecuted required case does not satisfy the gate.
- Existing safe local tests have run through the fixed entry, with classified
  results and explicit reasons for every exclusion or unavailable dependency.
- Changed files, `git diff`, `git diff --check`, and `git status --short` are
  available for review, with unrelated edits preserved and no Git writes.
- The full proposed Stage 2 `probe.sh` and its actual SHA-256 are presented,
  together with target requirements, timeouts, connection budget, and the
  requirements for the persistent local state location. Never invent the hash
  before the final script exists.
- The handoff explicitly stops before Stage 2; neither state registration nor
  mock validation constitutes remote-execution approval.

The initial Stage 2 limits are proposed in the
[Sirius operating document](../jobs/sirius/README.md). Target identifiers,
compiler paths, ROCm versions, scheduler details, queues/accounts, and GPU
targets are not guessed during Stage 1. No actual probe or GPU validation is
claimed until the separately approved action has run and its evidence exists.

### First Stage 3 increment: rocFFT local acceptance

The approved six source paths and workload belong to the project specification.
Implement CPU FFTW serial/threaded, direct HIP/rocFFT, and OpenMP target/rocFFT
through existing common CLI/timing/verification/result facilities. Integrate
vendor-specific CMake profiles, real API compile/link detection, manifest,
runner, validators, aggregation and display without changing NVIDIA meanings.

Acceptance covers CPU-only build and positive/negative dependency fixtures,
full-batch FFT/nonfinite/overflow validation, timing boundaries, both raw
formats, runtime and artifact provenance, old NVIDIA compatibility, Python 3.9,
shell syntax and whitespace. Fixtures cannot certify HIP compilation, device
execution or OpenMP interoperation. Run related checks during development and
one final fixed local validation; apply only reviewed verified bytes to the
editing source after checking its baseline, then compare bytes without
repeating full tests. Preserve all prior user changes, state and logs.

Prepare `jobs/sirius/rocfft_build_smoke.sh` for a separately allocated compute
node: three build trees, compiler/header/library resolution evidence, mandatory
actual-device interoperation, small multiple-batch runs in both scopes, common
validation, and retained per-step logs/exit values. Scheduler/account/queue,
allocation command, selected GPU architecture, library roots, compiler choice,
CPU threads and device mappings remain owner/site inputs. Never run that body
on a login node or infer allocation settings from the login-node CPU count.
Stop after local implementation, self-review, source application and report.

### Stage 3 build and provenance gates

Manual teaching builds may use `hipcc`. Native CMake HIP uses a supported Clang
compiler, not the `hipcc` wrapper fixed as `CMAKE_HIP_COMPILER`. Determine
compiler paths, ROCm versions, GPU targets, and conditional CMake requirements
from the confirmed environment, retaining CPU-only configure without ROCm.
Use the same `rocthrust_wrapper.hip` name in teaching/distribution, CMake, tests,
and documentation; record both translation units' compile conditions and
explicit link inputs. No AMD source or CMake target is part of Stage 1.

Before implementing AMD result handling, update the owning
[`RESULT_SCHEMA.md`](RESULT_SCHEMA.md) and
[`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) contracts. Separating runtime
metadata from mathematical comparison keys must not allow different provenance
inside one campaign. Preserve the approved mapping of each artifact to its
compiler/build settings and binary, together with source, manifest, config,
ROCm, resolved libraries, and runtime-environment identity across waves.

Stage 3 regression tests must reject each of the following mismatches even
when mathematical comparison keys match:

- ROCm/runtime version or runtime-environment hash;
- resolved-library identity, version, or hash;
- compiler or either translation unit's build conditions;
- binary hash inconsistent with the manifest's artifact mapping;
- executable manifest, configuration, or source provenance.

Also retain existing NVIDIA config/result interpretation, hash calculation,
validation, and aggregation compatibility. Artifact-specific compiler/binary
differences prescribed by a single approved manifest are not cross-campaign
mixing; a result must match that artifact's fixed provenance. A production
measurement requires a clean fixed revision and fresh build. Permission to
commit that revision is separate from permission to push, create a PR, merge,
or publish private data.

## External inputs and intentionally deferred scope

The implementation plan is decision-complete without guessing external site
state. The following remain human inputs or future scope rather than decisions
left to the implementer:

- Pegasus account, available queue, module names/versions, MPI version, CPU
  library installation path, NVHPC CUDA selection, and shared output path.
- Actual GPU/CPU performance, production-calibrated sizes/repeats after pilot,
  and anomaly/exclusion policy.
- AMD teaching inputs and Sirius site evidence needed for the later approved
  development stages above; approval of Stage 1 does not supply these facts or
  authorize later execution.
- Future Miyabi and TSUBAME4.0 implementation. Furo remains outside
  the approved job scope.

The owner has approved MIT and the copyright-holder attribution described in
the [README](../README.md#license), together with publication operations for the
selected historical [code scope](../README.md#publication-scope). That approval
does not authorize AMD/Sirius Git writes or publication.

No empty directory, dummy target, or placeholder implementation is created for
deferred source areas. Configuration examples use explicit placeholders for
human-supplied site values.

## AMD Fortran source implementation

The six-library CPU/HIP/OpenMP family under `amd/fortran` supplies 18 teaching
programs, 18 benchmarks, explicit helper/link inputs, language-marked results
and a small numerical smoke config. The [usage guide](../amd/fortran/README.md)
is the entry point for application developers.

The initial Sirius build and numerical checks completed. Benchmark timing was
then aligned with AMD C/C++ at commit `b69847694c3e34fd220a3c44b7b1a98da22d6b63`.
That version passed a fresh 18-benchmark build, 72-row smoke and two full
six-host waves with 6,480 verified records. Aggregation and all six figures
were checked against those records. The [validation report](VALIDATION_REPORT.md#amd-fortran-production-validation)
owns run identities, coverage and limitations.

The earlier 5,400-row provisional comparison and failed waves remain
historical evidence, separate from the aligned campaign. Measurements after a
system update require a new environment identity and separate execution scope.
Effective external-library worker counts and complete loaded-library identity
remain unknown; they are disclosed rather than inferred from requested threads.
NVIDIA Fortran timing alignment is outside this completed AMD work.

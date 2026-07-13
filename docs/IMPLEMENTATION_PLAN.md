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

## Cross-phase implementation constraints

### Language standards

Repository C targets and the common C API target C17. C++ sources target C++17,
CUDA sources target CUDA C++17, and OpenACC C++ sources target C++17 with the
selected compiler's OpenACC mode. CMake must request these standards explicitly
without vendor-language extensions unless an approved dependency requires one.

### Build trees

CUDA and OpenACC use separate CMake configurations because a CUDA build normally
uses GCC/G++/nvcc while an OpenACC build uses `nvc++`. Do not attempt to mix
different C++ compilers in one configure.

Planned Pegasus trees:

```text
build/pegasus-cuda
build/pegasus-openacc
```

Local validation must avoid large SSHFS build artifacts and use:

```text
/tmp/gpu-library-suite-local-build/cpu
/tmp/gpu-library-suite-local-build/tests
```

### Planned CMake options

The implementation must expose:

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

CPU-only configuration must succeed without CUDA or NVHPC. CUDA architecture is
supplied through `CMAKE_CUDA_ARCHITECTURES`; the project must not hard-code
`sm_90`. The NVHPC target is also external input. Missing optional dependencies
disable only affected targets with a clear reason, not the entire configure.

Detection covers FFTW3, CBLAS implementations including oneMKL/OpenBLAS,
oneMKL Sparse BLAS, LAPACKE, the CUDA Toolkit, and NVHPC. Do not silently enable
architecture-specific fast math, TF32, Tensor Core modes, or altered numerical
semantics. Generated build metadata records Git commit/dirty provenance,
compiler ID/version/flags, library discovery, build type, and executable hashes.

### Python and configuration

Python tools target Python 3.9 or newer. Configuration is JSON rather than TOML
so aggregation, validation, and job rendering work with the standard library.
Plotting may use matplotlib as an optional dependency; absence produces a clear
diagnostic and never triggers installation.

## Phase 1 — Common benchmark infrastructure

### Deliverables

- Common C/C++ CLI parsing and validation.
- Shared monotonic clock and resolution recording.
- Machine-readable benchmark output support for CSV and JSON Lines.
- Metadata, hashing, and executable-manifest support.
- Verification helpers and state-restoration helpers.
- Python deterministic-serialization and strict JSON-loading utility.
- Unit tests for common components.

### Dependencies

- C/C++ standard libraries and POSIX monotonic clock.
- Python 3.9 standard library.
- The authoritative protocol and schema documents.

### Acceptance criteria

- Common options and invalid combinations conform to the benchmark protocol.
- Raw output and null/failure behavior conform to the result schema.
- Benchmark stdout/path behavior and exclusive-create rules conform to the
  protocol, including stderr-only diagnostics.
- JSON-loader tests reject duplicate keys, `NaN`, `Infinity`, `-Infinity`, and
  invalid UTF-8 rather than relying on permissive library defaults.
- One Python utility produces deterministic JSON bytes and hashes for all
  specified metadata documents, including runtime-environment metadata.
- Run-ID validation covers the accepted expression plus prohibited path/control
  cases.
- Verification output always uses object-shaped metrics and thresholds,
  including single-metric cases.
- Nonfinite verification produces null evidence and failure status without
  invalid JSON.
- Serialization/output tests reject nonfinite numeric output rather than
  emitting a non-standard JSON token.
- State-restoration helpers are unit-tested for restoration before every raw
  trial and before every end-to-end repeat, outside the timed interval.
- Unit tests cover the inclusive Q1/Q3 method, two-or-more samples, the one-sample
  null rule, and the zero-sample status-only record.
- Unit tests cover requested/effective CPU threads and serial/threaded/unknown
  classification.

### Local validation

- Configure and test CPU-only targets in `/tmp/gpu-library-suite-local-build`.
- Run Python unit tests with Python 3.9-compatible syntax and standard-library
  dependencies only.
- Validate sample records in memory or temporary storage, not as repository
  sample results.

### Manual Pegasus validation

- None required until executable implementations exist.

### Risks

- CSV/JSON representations may drift if one path bypasses common serialization.
- Platform-specific clocks or compilers may expose type/precision assumptions.
- Dirty-source hashing can be incomplete if untracked build inputs are ignored.

### Unresolved decisions

- Exact internal C/C++ JSON writer implementation, subject to schema tests.
- Whether dirty pilot provenance uses a complete Git binary diff or a source
  snapshot for each build workflow.

## Phase 2 — cuFFT examples and benchmarks

### Deliverables

- Three canonical cuFFT teaching examples.
- Three corresponding cuFFT benchmarks.
- FFTW serial and optional threaded CPU backends with distinct names.
- CMake integration for CPU, CUDA, and OpenACC build trees.
- Static and CPU-side tests.

### Dependencies

- Phase 1 infrastructure.
- Optional FFTW3 single-precision interface and threaded interface.
- Optional CUDA Toolkit and NVHPC for GPU targets.

### Acceptance criteria

- Filenames, teaching defaults, and plan choice match the project specification.
- Benchmark scope, warm-up restoration, timing, and DC/non-DC verification match
  the benchmark protocol.
- OpenACC data regions perform managed allocation/movement exactly once; no
  managed array is also explicitly allocated.
- CPU-only configure remains valid when GPU dependencies are absent.
- Executables appear in the manifest with content hashes.

### Local validation

- Compile and run CPU examples/tests only when an FFTW dependency is already
  available.
- Perform syntax/static checks for GPU sources without claiming GPU execution.
- Verify target-disable diagnostics in dependency-absent configurations.

### Manual Pegasus validation

- Build CPU/CUDA and OpenACC trees separately on an appropriate compute node.
- Run the teaching examples, a pilot benchmark, verification, and sanitizer.

### Risks

- FFTW threaded configuration may differ among installations.
- OpenACC/CUDA interoperability may depend on compiler-library compatibility.

### Unresolved decisions

- Actual Pegasus module versions and production FFTW backend availability.

## Phase 3 — cuBLAS, cuSPARSE, cuSOLVER, cuRAND, and Thrust

### Deliverables

- Canonical examples and benchmarks for the remaining five libraries.
- Production/reference CPU backends as dependencies permit.
- Library-specific verification and canonical-state restoration.
- CMake integration and tests.

### Dependencies

- Phases 1 and 2 patterns.
- Optional CBLAS, oneMKL/OpenBLAS, oneMKL Sparse BLAS, LAPACKE, CUDA Toolkit,
  NVHPC, and OpenMP support.

### Acceptance criteria

- All canonical filenames and teaching problems match the project specification.
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
- cuRAND and Thrust serial production baselines record effective thread count 1
  and are labeled as serial.
- Reference backends never replace production backends silently.

### Local validation

- Run CPU-only examples and tests for dependencies already present.
- Unit-test CSR and dense-system generation independently of GPU libraries.
- Statically inspect OpenACC address-translation/data-region ordering.
- Unit-test every mutable-state restoration path, including solver info values
  and random-generator progression.
- Record every unavailable GPU or optimized CPU test and its reason.

### Manual Pegasus validation

- Build each enabled backend in its correct build tree.
- Run examples, pilot benchmarks, verification, and sanitizer where applicable.
- Confirm library versions, effective CPU thread counts, and GPU UUID capture.

### Risks

- Optional CPU APIs and thread-control symbols vary by provider/version.
- Destructive solver operations can invalidate timing if restoration leaks into
  the measured region.
- OpenACC data lifetime errors can cause double allocation or stale addresses.

### Unresolved decisions

- Which optimized CPU backends are available and approved for production.
- Exact supported dependency versions after manual environment inspection.

## Phase 4 — Suite runner, aggregation, validation, and plotting

### Deliverables

- `run_suite.py`.
- `aggregate.py`.
- `validate_results.py`.
- `plot.py` with optional matplotlib.
- Python unit tests.

### Dependencies

- Complete benchmark executables and manifests from Phases 1–3.
- Protocol ordering/statistics and result schemas.

### Acceptance criteria

- Execution is interleaved by problem size; it never runs all CPU cases before
  all GPU cases.
- `run_suite.py` is the sole node raw-file writer: it invokes each executable
  through machine-readable stdout, validates records, writes one CSV header,
  and creates synthetic failure rows for crashes, signals, invalid output, and
  missing output without overwriting existing files.
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
- Each benchmark has exactly one production CPU backend in a primary campaign;
  additional CPU backends are separate campaigns or explicitly auxiliary
  series and never become an automatic speedup denominator.
- Verification/process failures remain raw and are excluded with visible counts.
- Serial CPU baselines have explicit graph labels.

### Local validation

- Unit-test every permutation, size order, and multi-wave combination.
- Unit-test benchmark stdout validation, one-header append behavior, exclusive
  creation, and each synthetic-failure-row case.
- Unit-test all-trial and end-to-end-repeat state restoration through the suite
  execution loop.
- Unit-test hierarchy with 0, 1, 2, 5, 6, and 8 valid samples.
- Unit-test cross-wave summaries with unequal block counts to prove that the
  primary input is one median per wave rather than a flat pool.
- Validate malformed, duplicate, failed, nonfinite, and mixed-provenance input.
- Verify plotting's dependency-missing diagnostic without installing anything.

### Manual Pegasus validation

- Run a one/two-node pilot and inspect interleaving, assignments, raw rows, and
  aggregation before production.

### Risks

- Incorrect grouping can destroy paired comparisons.
- Partial waves can be mistaken for balanced data without assignment metadata.

### Unresolved decisions

- Final plot presentation choices after representative real data exists.

## Phase 5 — Pegasus execution and collection

### Deliverables

- Pegasus README and JSON example.
- Job renderer and PBS template.
- `prepare_wave.py` for PBS job-master preflight and exclusive metadata
  creation.
- Separate CPU/CUDA and OpenACC build scripts.
- Node runner, telemetry, and result collector.

### Dependencies

- Phase 4 runner/tools.
- Human-supplied Pegasus account, queue, module, and path settings.
- Prebuilt executable manifest.

### Acceptance criteria

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
- Production rejects a dirty worktree before measurement.
- Build module groups remain separate from the human-validated compatible
  benchmark-runtime module group; benchmark jobs do not load every build module.
- Runtime-environment capture includes module/search paths, driver/runtime and
  dependency versions, binary dependency resolution, and thread/runtime
  variables and is deterministically serialized and hashed.
- Node-local scratch is recovered on normal and abnormal exit.
- Trial timestamps and telemetry timestamp metadata support interval-based UTC
  correlation, including local-midnight rollover in timestamped dmon output.
- GPU telemetry preserves raw columns and treats parse failure as non-fatal.
- CPU telemetry records required topology, binding, environment, module, and
  compiler information without privileged changes.
- Job submission remains a human action and benchmark jobs never compile.
- Benchmark, verification, and nonfatal tool failures remain recorded without
  prematurely aborting other node collection. After the measurement launch,
  the job master always runs the collector; missing/failed node status or
  collection makes the overall job fail while preserving artifacts.

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

### Unresolved decisions

- Account, available queue, module names/versions, MPI version, and shared path.

## Phase 6 — Documentation and final validation

### Deliverables

- Updated user-facing README and library documentation after implementation.
- Complete local validation report.
- Manual Pegasus checklist and run report.

### Dependencies

- Phases 1–5 complete.
- Project owner review of teaching-material correspondence.

### Acceptance criteria

- User-facing documentation summarizes implemented behavior without replacing
  authoritative specifications.
- Every changed file, command, successful test, unexecuted test/reason, and
  required Pegasus check is reported.
- No GPU test is claimed successful unless actually run.
- No TODO-only implementation is reported complete.
- Canonical filenames contain no manually maintained version suffix.

### Local validation

- Configure/test supported local CPU paths in `/tmp`.
- Run schema, CLI, ordering, aggregation, and documentation consistency checks.

### Manual Pegasus validation

- Complete smoke and production checklists with recorded provenance.
- Confirm result recovery and telemetry for normal and abnormal exit paths.

### Risks

- Documentation can drift if ownership boundaries are ignored.
- A locally unavailable dependency can leave a target manually unverified.

### Unresolved decisions

- Project license and publication policy remain owner decisions.
- Future Miyabi, TSUBAME4.0, and Sirius job support remains out of initial scope.

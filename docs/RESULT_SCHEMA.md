# GPU Library Suite Result and Configuration Schemas

## Authority and scope

This document owns the formats of raw and aggregate results, effective
configuration, executable manifests, and run/wave/node metadata. It defines
field names, types, serialization, hashes, paths, and write ownership. Timing,
verification algorithms, execution ordering, and statistical meaning are owned
by [`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md). Pegasus-specific population
and collection are owned by [`PEGASUS_EXECUTION.md`](PEGASUS_EXECUTION.md).

The initial values of `config_schema_version`, `result_schema_version`, and
`aggregate_schema_version` are the integer `1`. Increment one only for an
incompatible change to that machine-readable format. Do not place schema
versions in filenames.

## Deterministic JSON serialization profile

The project calls this the **project-defined deterministic JSON serialization
profile**. It is used for configuration and metadata reproducibility; no claim
is made that it is an external standard.

Standalone JSON documents use:

- UTF-8 encoding;
- keys sorted lexicographically (`sort_keys=true`);
- compact separators (`separators=(",", ":")`);
- Unicode characters preserved (`ensure_ascii=false`);
- nonfinite numbers rejected (`allow_nan=false`); and
- exactly one LF byte after the JSON value.

C/C++ result serialization fixes `LC_NUMERIC` to `C` before formatting. It uses
round-trip precision, a period decimal separator, lowercase normalized exponent
notation without redundant exponent sign/zero padding, and serializes every
zero including negative zero as `0`. It checks every floating value for
finiteness before formatting. NaN and infinities are represented by null plus
the applicable failure/nonfinite status and message, never by non-standard JSON
tokens.

Compute SHA-256 over the exact saved byte sequence, including the final LF.
Timestamps use UTC with exactly millisecond precision:
`YYYY-MM-DDTHH:MM:SS.sssZ`.

Every project Python JSON loader explicitly supplies both:

- `object_pairs_hook` (or an equivalent implementation) that rejects duplicate
  object keys rather than accepting the last value; and
- `parse_constant` that rejects `NaN`, `Infinity`, and `-Infinity`.

Do not rely on the permissive defaults of `json.load` or `json.loads`. Invalid
UTF-8 is also a load error. NaN, positive/negative infinity, and other nonfinite
values are never written as JSON numbers; represent them with null, a
failure/nonfinite status, and a diagnostic message.

The shared Python utility serializes and hashes `effective-config.json`,
`run-metadata.json`, `runtime-environment.json`, `wave-metadata.json`, and
`node-metadata.json`. C/C++ benchmarks must not independently serialize and hash
the effective configuration. Executable manifests use the same deterministic
profile.

For JSON Lines, each record is a deterministic-profile JSON object followed by
one LF. For an object embedded in a CSV cell, use the same sorted, compact,
UTF-8 JSON payload before the standalone-document LF terminator; CSV quoting
then preserves it as one cell and one logical row.

## Identifiers and execution context

### `run_id`

`run_id` identifies one measurement campaign that may contain multiple waves.
It must match:

```text
^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$
```

Additionally reject an empty value, a leading dot, the substring `..`, `/`,
backslash, any platform path separator, and every control character. Never
sanitize an invalid value silently.

### Wave, node, and block

- `wave` is an independent measurement wave within a campaign and is a
  non-negative integer.
- `node_index` is the non-negative MPI rank within the current job. It is not a
  physical-node identifier.
- `hostname` is the physical node name returned for that process.
- `block_id` is the scalar string:

  ```text
  <run_id>|<wave>|<hostname>
  ```

Store `run_id`, `wave`, and `hostname` as separate fields as well. A hostname
used as a directory component must be validated as a safe single component; do
not silently rewrite it. A duplicate hostname within one wave is an error under
the one-process-per-node model.

### Scheduler identity

`scheduler` and `scheduler_job_id` are nullable strings. Pegasus writes
`scheduler = "NQSV"` and obtains `scheduler_job_id` from `PBS_JOBID`. A local
run leaves both null.

## Raw result schema

One raw record represents one trial, including a failed trial. CSV contains one
logical row per trial with correct RFC-style quoting; JSON Lines contains one
object per LF-terminated record. A run may produce CSV, JSON Lines, or both as
selected by its effective configuration.

Within a node, `run_suite.py` is the only raw-file writer. It exclusively creates
the node raw file, writes the CSV header once when applicable, validates
benchmark stdout, and appends accepted rows. Benchmark executables emit
machine-readable stdout with `--output -` and never append to node raw files.
The runner preserves validated completed rows. For a crash, signal, invalid
output, or missing output, it synthesizes one attempted failure row for the
first unresolved trial and unattempted skipped rows for later trials that did
not start. It rejects duplicate or out-of-range trial indices and materializes
each expected index exactly once. Raw files are never overwritten.

### Fields

| Field | JSON type | Required | Meaning and null rule |
| --- | --- | --- | --- |
| `result_schema_version` | integer | yes | Initial value 1. |
| `run_id` | string | yes | Validated campaign ID. |
| `record_timestamp` | string | yes | UTC time when the raw row is generated. |
| `measurement_start_timestamp` | string/null | yes | UTC start of the first timed interval, or null if timing never started. |
| `measurement_end_timestamp` | string/null | yes | UTC end of the last timed interval, or null if timing never completed. |
| `system_label` | string | yes | User-provided portable label. |
| `wave` | integer | yes | Non-negative campaign wave. |
| `node_index` | integer | yes | Non-negative MPI rank. |
| `hostname` | string | yes | Physical node name. |
| `block_id` | string | yes | Exact scalar composition defined above. |
| `scheduler` | string/null | yes | Scheduler name or null. |
| `scheduler_job_id` | string/null | yes | Scheduler job identifier or null. |
| `implementation_order` | array | yes | Assigned three-implementation order. CSV stores deterministic JSON text. |
| `benchmark` | string | yes | `cufft`, `cublas`, `cusparse`, `cusolver`, `curand`, or `thrust`. |
| `implementation` | string | yes | `cpu`, `cuda`, or `openacc`. |
| `scope` | string | yes | `compute` or `end-to-end`. |
| `problem_size` | integer/null | yes | Primary x-axis value; null for non-square DGEMM. |
| `secondary_size` | integer/null | yes | Library-specific secondary size or null. |
| `parameters` | object | yes | Complete normalized library parameters. CSV stores deterministic JSON text. |
| `precision` | string | yes | For example `fp32` or `fp64`. |
| `cpu_backend` | string/null | yes | Stable backend name for CPU, otherwise null. |
| `cpu_backend_role` | string/null | yes | `production` or `reference` for CPU, otherwise null. |
| `series_role` | string | yes | `primary` or `auxiliary`. |
| `cpu_threads_requested` | integer/null | yes | Positive requested CPU count or null for GPU. |
| `cpu_threads_effective` | integer/null | yes | Positive known effective count, otherwise null. |
| `cpu_parallelism` | string/null | yes | `serial`, `threaded`, `unknown`, or null for GPU. |
| `warmup` | integer | yes | Non-negative warm-up count. |
| `repeat` | integer | yes | Positive count used in the trial. |
| `trial` | integer | yes | Non-negative trial index. |
| `attempted` | boolean | yes | True if this trial began execution; false only for an unstarted skipped trial. |
| `failure_origin` | string/null | yes | Null on success; otherwise the controlled origin defined below. |
| `elapsed_total_sec` | number/null | yes | Total measured time; null if unavailable/nonfinite. |
| `elapsed_sec` | number/null | yes | `elapsed_total_sec / repeat`; null if unavailable/nonfinite. |
| `clock_id` | string | yes | `CLOCK_MONOTONIC`. |
| `clock_resolution_sec` | number | yes | Recorded clock resolution in seconds. |
| `verification_metrics` | object | yes | Metric name to measured value; always an object. |
| `verification_thresholds` | object | yes | Metric name to complete condition; always an object. |
| `verification_primary_metric` | string/null | yes | Representative metric name or null. |
| `verification_status` | string | yes | `pass`, `failure`, `skipped`, or `nonfinite`. |
| `getrf_info` | integer/null | yes | cuSOLVER factorization info, otherwise null. |
| `getrs_info` | integer/null | yes | cuSOLVER solve info, otherwise null. |
| `device_id` | integer/null | yes | CUDA device index for GPU, otherwise null. |
| `gpu_name` | string/null | yes | Reported GPU name or null. |
| `gpu_uuid` | string/null | yes | Reported GPU UUID or null. |
| `cuda_driver_version` | string/null | yes | CUDA Driver API version returned by `cudaDriverGetVersion`, or null when not applicable/unavailable. |
| `compiler` | string | yes | Compiler identity. |
| `compiler_version` | string | yes | Compiler version. |
| `global_configure_flags` | string | yes | Language-level global CMake configure flags only; this does not claim target compile definitions/options, provider flags, or link options. |
| `library_name` | string | yes | CPU or GPU library/backend name. |
| `library_version` | string/null | yes | Version or null if unavailable. |
| `cuda_runtime_version` | string/null | yes | Runtime version for GPU or null. |
| `git_metadata_available` | boolean | yes | True only when both commit and worktree state were obtained from Git. |
| `git_commit` | string/null | yes | Source commit identifier, or null when Git metadata is unavailable. |
| `git_dirty` | boolean/null | yes | Source worktree state, or null when Git metadata is unavailable. |
| `git_diff_sha256` | string/null | yes | Dirty diff hash when that provenance mode is used. |
| `source_snapshot_sha256` | string/null | yes | Dirty source snapshot hash when required. |
| `config_sha256` | string | yes | Effective-config exact-byte hash. |
| `runtime_environment_sha256` | string | yes | Exact-byte hash of `runtime-environment.json`. |
| `binary_sha256` | string | yes | Exact executable-content hash. |
| `exit_code` | integer/null | yes | Process/result exit code, or null before one exists. |
| `status` | string | yes | `success`, `failure`, or `skipped`. |
| `message` | string | yes | Empty on normal success; diagnostic otherwise. |

CSV represents null as an empty cell. Boolean values are `true`/`false`.
Arrays and objects use deterministic-profile JSON text in a correctly escaped
cell. The schema defines no legacy unqualified thread-count field, scalar
verification-evidence field, or undifferentiated verification-tolerance field.

For CUDA and OpenACC rows, the benchmark obtains `gpu_name` and `gpu_uuid` from
`cudaGetDeviceProperties`, `cuda_driver_version` from
`cudaDriverGetVersion`, and `cuda_runtime_version` from
`cudaRuntimeGetVersion`. `library_version` comes from the selected library's
version API (or `THRUST_VERSION` for Thrust). Failure to obtain required runtime
metadata is a benchmark failure rather than permission to reuse job-master or
another node's values. Pegasus also independently captures the node-local
NVIDIA driver/package identity and loads node-local `libcudart` to query Driver
API and Runtime versions. Successful raw rows must agree with node metadata for
name, UUID, Driver API version, and Runtime version.

`git_metadata_available`, `git_commit`, and `git_dirty` have one coupled
state. When metadata is available, availability is true, commit is a nonempty
string, and dirty is a boolean. For a source archive or another `.git`-less
source, availability is false and commit and dirty are both null. Unknown Git
state is never represented as `git_dirty=false`; `"unknown"` and
`"unavailable"` are not commit identifiers. Production validation rejects
unavailable Git metadata. Archive/local validation may use the explicit
false/null state while still producing valid schema-version-1 JSON.

All three raw timestamps use `YYYY-MM-DDTHH:MM:SS.sssZ`. For end-to-end trials,
the span from `measurement_start_timestamp` to
`measurement_end_timestamp` can exceed `elapsed_total_sec` because the former
also spans untimed restoration and cleanup gaps between separately timed
repeats. `elapsed_total_sec` remains the sum of timed intervals only.

### Attempt and failure-origin rules

`failure_origin` is null or one of:

- `benchmark`: a started trial failed in setup, allocation, timing, or a library
  operation;
- `verification`: a started trial completed measurement but failed numerical or
  structural verification;
- `subprocess`: the runner observed a crash, signal, or abnormal process exit;
- `output-validation`: stdout was missing, malformed, contaminated, or failed
  schema validation;
- `prior-failure`: this trial did not start after a fatal predecessor; or
- `prerequisite`: this trial did not start because its executable or another
  required input was unavailable.

Successful rows require `attempted=true`, `status=success`, and null origin.
Started failure rows require `attempted=true` and `status=failure`. Skipped rows
require `attempted=false`, `status=skipped`, null measurement timestamps and
elapsed fields, `verification_status=skipped`, and null `exit_code`.

For a process/output failure, the first unresolved trial is an attempted
synthetic failure. Later unresolved trials are synthetic skipped rows with
`prior-failure`. A synthetic failure has null timing fields when measurement
never began and records an available subprocess exit code or signal-derived
code. Completed rows are never replaced.

### Verification objects

Even a single-metric benchmark uses objects. A threshold entry contains its
method and all operands required to reproduce the decision. An ordinary
absolute/relative entry contains `method`, `reference_scale`, `abs_tolerance`,
and `rel_tolerance`. Range or statistical entries contain explicitly named
bounds/expectations and tolerances.

The required distinct evidence includes:

- cuFFT: separate DC and non-DC metrics;
- cuSOLVER: `solution_relative_error` and `relative_residual`; and
- cuRAND: `observed_min`, `observed_max`, `sample_mean`, and
  `second_central_moment_about_half`, with interval contracts and the exact
  statistical operands specified in `BENCHMARK_PROTOCOL.md`.

A verification failure sets `verification_status = "failure"` and overall
`status = "failure"`. A nonfinite value is stored as null, sets
`verification_status = "nonfinite"` and overall failure, and includes a message.
The raw record is retained in both cases.

## Result directory and write ownership

```text
results/<run-id>/
  effective-config.json
  run-metadata.json
  runtime-environment.json
  waves/
    <wave>/
      wave-metadata.json
      nodes/
        <hostname>/
          node-metadata.json
          node-status.json
          raw-results.csv
          raw-results.jsonl
          logs/
          telemetry/
  aggregate/
  plots/
```

The **PBS job master** is the shell process coordinating the job outside the
measurement `mpirun`. It is the exclusive campaign/wave metadata owner; no MPI
process writes campaign metadata.

Before node output begins:

1. the job master runs a short preflight `mpirun` before the measurement
   `mpirun` and captures each `OMPI_COMM_WORLD_RANK`/hostname pair;
2. the job master runs `jobs/pegasus/prepare_wave.py` with that mapping;
3. `prepare_wave.py` validates run ID, wave, expected node count, complete
   rank-host mapping, hostname uniqueness, existing output, and
   config/binary/source/runtime provenance;
4. for a new campaign, the job master exclusively creates
   `effective-config.json`, `run-metadata.json`, and
   `runtime-environment.json`; for an existing campaign it validates their
   hashes and immutable provenance;
5. the job master exclusively creates the new `wave-metadata.json`; and
6. only after successful preflight does the job master start the measurement
   `mpirun` that launches `run_node.sh`.

Single-node execution uses the same job-master preflight sequence. No
measurement process sends coordination messages or participates in a
coordination collective. A preflight failure means the measurement `mpirun` is
not started.

Each node process writes only its own `nodes/<hostname>/` subtree. An existing
wave number, existing node output, duplicate hostname, unsafe component, or
incomplete mapping is an error; never overwrite or merge it silently.

During execution, each node writes raw results, logs, status, and telemetry to a
unique node-local scratch path. At shutdown it collects them into only its own
shared node directory. Collection occurs for normal and abnormal termination,
and raw results are not deleted.

Within that node, only `run_suite.py` creates and appends the node-level raw
file. It validates each benchmark's machine-readable stdout before appending.
Human-readable stdout is invalid. Subprocess crash, signal termination, invalid
output, and missing output produce synthetic failure rows rather than a silent
gap in the raw data.

`validate_results.py` first requires the complete configured invocation/trial
set, unique trial indices, and consistent configuration, binary, compiler, Git,
and runtime-environment provenance. A structurally complete campaign still has
`validation_status=failure` when any row is `failure` or `skipped`; the tool
writes the deterministic schema-version-1 validation summary and exits nonzero.
Only an all-success campaign has `validation_status=pass` and exit status zero.

## Effective configuration

`effective-config.json` is the final campaign configuration after merging:

1. built-in defaults;
2. the selected configuration file; and
3. CLI overrides that affect the measurement protocol.

The configuration has this required structural shape:

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

The same case-level scope values apply to CPU, direct CUDA, and OpenACC; an
implementation-local warm-up/repeat/trials override is invalid. Compute and
end-to-end values are independent. Unknown keys and missing required keys are
errors. Both canonical files begin with `config_schema_version = 1`, select
JSON Lines by default, enable verification, and set
`continue_on_failure = true`.

It includes `config_schema_version`, run mode (`smoke`, `pilot`, or
`production`), benchmarks, problem sizes in configuration order, scopes,
warm-up/repeat/trials, output formats, backend selection,
`default_speedup_cpu_backend` for every benchmark, every verification threshold,
and each configured series' `primary` or `auxiliary` role. A primary production
campaign selects exactly one production CPU backend per benchmark; alternative
production or reference backends are explicitly auxiliary or belong to a
separate campaign.

Execution context does not belong to the effective configuration. In
particular, exclude `run_id`, `wave`, `node_index`, hostname, scheduler job ID,
derived implementation order, and derived size order from its hash. Save the
document first, then hash its exact bytes.

Canonical configuration filenames are `configs/pilot.json` and
`configs/benchmark.json`; both contain `config_schema_version`. Normal changes
to sizes, repeat, trials, thresholds, or backend selection do not increment that
version. Do not rename the canonical files to encode a revision or date.

## Executables manifest and source provenance

Each build tree writes one deterministic `build-metadata.json`. It records the
profile, build type, Git metadata availability/commit/dirty state,
C/C++/CUDA compiler identities, versions and language-level global configure
flags, CUDA architectures and Toolkit root/version,
NVHPC CUDA home/GPU target, general OpenACC compile/link flags, and the extra
OpenACC Thrust `-cuda` compile/link interoperation flags. The CMake-generated
per-target manifest descriptor supplies the target name and backend variant;
the partial manifest binds every entry to the exact build-metadata hash.

The executable manifest contains an entry for every runnable binary with at
least:

| Field | Meaning |
| --- | --- |
| `artifact_id` | SHA-256 identity defined below. |
| `target_name` | Canonical CMake target and executable stem. |
| `build_profile` | `cpu-cuda` or `openacc`. |
| `backend_variant` | Compiled provider/capability variant. |
| `supported_cpu_backends` | Stable CPU backend names supported by this binary, or an empty array for non-CPU targets. |
| `executable_path` | Configured executable path. |
| `library` | Library identifier. |
| `implementation` | CPU, CUDA, or OpenACC. |
| `executable_role` | `example` or `benchmark`. |
| `build_type` | Recorded build configuration. |
| `binary_sha256` | SHA-256 of the exact executable bytes. |
| `build_metadata_sha256` | SHA-256 of deterministic generated build metadata. |
| `compiler` | Compiler identity. |
| `compiler_version` | Compiler version. |
| `global_configure_flags` | Language-level global configure flags associated with the selected compiler. |
| `git_metadata_available` | Whether commit and worktree state were obtained from Git. |
| `git_commit` | Source commit embedded by the build, or null when unavailable. |
| `git_dirty` | Source dirty state embedded by the build, or null when unavailable. |

`artifact_id` is the SHA-256 of the deterministic-profile JSON object containing
`library`, `implementation`, `executable_role`, `target_name`, `build_profile`,
`backend_variant`, and `binary_sha256`.

Within one complete manifest, both of these tuples are unique:

```text
(library, implementation, executable_role)
(target_name, build_profile, backend_variant)
```

A duplicate is an error even if the duplicate entries otherwise match. A merge
also rejects conflicting paths, binary hashes, build metadata hashes, compiler
metadata, Git state, build types, or backend providers. Different provider
builds require distinct campaigns or an explicitly separate auxiliary
manifest; they are never merged ambiguously.

`backend_variant` records the compiled capability/provider without renaming the
binary. The entry also records supported stable CPU backend names when one
binary supports more than one runtime-selectable series.

Serialize the complete manifest with the deterministic JSON profile and compute
`executables_manifest_sha256` over those exact bytes. The manifest hash thus
commits to every binary-content hash, not just its path.

Before launch, the runner compares the manifest with generated build metadata,
including Git commit/dirty state, compiler ID/version, build type, and profile.
The CMake-generated target descriptor is the source of the manifest's target
name and backend variant. After launch, the runner compares raw compiler/Git
fields with the selected manifest entry.

Production runs require available Git metadata and a clean worktree; they
reject `git_metadata_available = false`, null Git fields, and
`git_dirty = true` before measurement. Smoke or pilot runs may use dirty source
only when metadata stores
one of:

- `git_diff_sha256`, computed from an exact captured Git binary diff when that
  diff fully represents all changed build inputs; or
- `source_snapshot_sha256`, used when untracked or otherwise uncaptured source
  can affect the binaries.

When dirty, at least one complete dirty-source hash is required. Metadata states
which method is authoritative. A clean run stores both dirty-source fields as
null. Archive/local validation may store unavailable Git metadata, but it must
not infer clean state and cannot be used for a production campaign.

`tools/hash_source_snapshot.py REPOSITORY` computes the canonical source
snapshot hash without modifying Git state. It uses read-only `git ls-files` to
select the union of tracked and untracked, non-ignored files that currently
exist; a deleted tracked path is represented by its absence. It rejects
non-UTF-8 paths, traversal, duplicate normalized paths, symlinks, and non-files.
After sorting UTF-8 repository-relative POSIX paths, SHA-256 input is the domain
`gpu-library-suite-source-snapshot-v1` plus a NUL, the uint64 big-endian file
count, and for each file: uint64 path-byte length, path bytes, one executable-bit
byte, uint64 content length, and the 32 raw bytes of that file's SHA-256. A file
that changes during hashing is an error.

## Runtime software environment

`runtime-environment.json` records the software environment used to execute the
prebuilt CPU, CUDA, and OpenACC binaries together. It contains at least:

- complete `module list` output or a normalized module list;
- `PATH` and `LD_LIBRARY_PATH`;
- NVIDIA driver version and CUDA runtime/Toolkit version and Toolkit path;
- NVHPC compiler/runtime version;
- `NVHPC_CUDA_HOME` when set, or the actual CUDA Toolkit selected by NVHPC;
- detected FFTW, oneMKL, OpenBLAS, LAPACKE, and other selected CPU-library
  versions;
- `ldd` output for every executable in the manifest;
- resolved shared-library paths for every executable; and
- `OMP_NUM_THREADS`, `OMP_PROC_BIND`, `OMP_PLACES`, `OMP_DYNAMIC`,
  `MKL_NUM_THREADS`, `MKL_DYNAMIC`, `MKL_THREADING_LAYER`, and
  `OPENBLAS_NUM_THREADS`.

The job master creates this document with the project-defined deterministic JSON
profile after loading the benchmark runtime modules. Compute
`runtime_environment_sha256` over its exact saved bytes. The hash is stored in
run, wave, and node provenance and in raw rows.

An additional wave may share a run ID only when
`runtime_environment_sha256` matches. Results with different runtime
environment hashes must not be combined in one primary cross-wave aggregate.
They require a separate campaign or an explicitly non-primary comparison.

## Run metadata

`run-metadata.json` is immutable campaign metadata created once by the PBS job
master. It contains at least:

- `run_id`, campaign creation timestamp, `system_label`, and run mode;
- Git metadata availability, commit, and dirty state;
- authoritative dirty-source hash kind/value when applicable;
- effective-config SHA-256;
- executables-manifest SHA-256;
- runtime-environment SHA-256;
- launcher/tool version information; and
- submission host or local initiating hostname.

Because a campaign can gain waves, do not store one wave's scheduler job ID,
node count, or ordering as campaign-global facts.

An additional wave may use an existing run ID only when all of the following
match the immutable campaign metadata:

- effective-config SHA-256;
- executables-manifest SHA-256;
- runtime-environment SHA-256;
- Git commit;
- dirty state; and
- Git metadata availability; and
- when dirty, authoritative dirty-source hash kind and value.

Any mismatch requires a new run ID. In particular, waves built from different
dirty source cannot share a campaign.

## Wave metadata

The PBS job master exclusively creates `wave-metadata.json` after successful
preflight. It contains at least:

- `run_id`, non-negative `wave`, timestamp, `scheduler`, and
  `scheduler_job_id`;
- expected and observed node counts;
- the complete preflight MPI-rank-to-hostname mapping;
- `runtime_environment_sha256`;
- implementation permutation assignments and
  `permutation_assignment_counts` for indices 0 through 5;
- size-order assignments and `size_order_assignment_counts` for indices 0 and
  1; and
- validation outcome for campaign provenance and output collisions.

## Node metadata and status

Each node process writes its own `node-metadata.json` through the shared Python
serialization utility. It contains:

- `run_id`, `wave`, `node_index`, `hostname`, and scalar `block_id`;
- `permutation_index` and `implementation_order`;
- `size_order_index` and the effective per-benchmark `size_order` arrays;
- each series' `primary` or `auxiliary` classification;
- requested CPU threads and relevant thread environment; and
- executable/binary identifiers and `runtime_environment_sha256` used on that
  node;
- node-local `gpu_identity` with name, UUID, NVIDIA package-driver version,
  query status, and diagnostic;
- node-local `cuda_runtime_identity` with loaded `libcudart` path, CUDA Driver
  API version, CUDA Runtime version, query status, and diagnostic; and
- for cuRAND, the actual CPU engine or cuRAND generator, seed, offset, and
  order.

`node-status.json` records process completion, raw-result collection, log
collection, telemetry status, exit status, and messages. It does not replace
failure rows in raw results.

## Telemetry metadata

Each node's `telemetry/telemetry-metadata.json` contains at least:

- `telemetry_start_timestamp_utc`;
- `telemetry_end_timestamp_utc`;
- `sample_interval_sec`;
- `timezone`; and
- `utc_offset`; and
- `midnight_rollover_count`.

All UTC timestamps use `YYYY-MM-DDTHH:MM:SS.sssZ`. Raw
`nvidia-smi dmon -o T` output is preserved. Because its time column does not
contain a date or UTC offset, telemetry metadata and the parser supply the date,
timezone, UTC offset, and midnight-rollover handling needed to construct a
continuous UTC timeline. Associate telemetry samples with trials by each raw
row's `measurement_start_timestamp` and `measurement_end_timestamp`, not by
`record_timestamp`.

## Aggregate result schema

Aggregate output uses `aggregate_schema_version` and is separate from the raw
schema. Every aggregate record identifies:

- `summary_level`: `block`, `wave`, `cross-wave`, or
  `pooled-exploratory`;
- run, wave/hostname/block identifiers as applicable;
- benchmark, implementation or speedup comparison, scope, problem sizes, and
  parameter signature;
- selected production CPU backend for speedup;
- `runtime_environment_sha256`;
- `summary_input_statistic` describing the values summarized at that level;
- valid-success, attempted-failure, and unattempted-skipped sample counts plus
  exclusion counts grouped by `failure_origin`;
- median, Q1, Q3, IQR, minimum, and maximum where defined; and
- `aggregate_status`.

Campaign-level aggregate metadata includes:

- `unique_hostname_count`;
- `wave_count`;
- `block_count`;
- `blocks_per_wave`;
- `permutation_assignment_counts`; and
- `size_order_assignment_counts`.

Aggregation is hierarchical:

1. at block level, compute trial medians within each wave × hostname block;
2. compute paired speedups within that block;
3. at wave level, summarize block medians across hostnames and produce a
   wave-level median; and
4. at primary cross-wave level, use only each wave's wave-level median as an
   input and set `summary_input_statistic = "wave_median"`.

Recompute cross-wave median, Q1, Q3, IQR, minimum, and maximum from the vector of
wave medians. Do not use quartiles of quartiles, average wave IQRs, or re-pool
all blocks for a primary cross-wave record.

A pooled hostname × wave summary is optional, labeled
`pooled-exploratory`, stored as a separate record, and never used by a primary
graph.

For two or more valid samples, compute Q1 and Q3 with:

```python
statistics.quantiles(values, n=4, method="inclusive")
```

For exactly one valid sample, median/minimum/maximum equal that value, Q1/Q3/IQR
are null, and `aggregate_status` is `insufficient_sample_count`. For zero valid
samples, do not emit numerical summary fields; emit a status-only record with
`aggregate_status = "no_valid_samples"`.

Verification failure, nonfinite verification, process failure, and absence of
the configured production CPU backend exclude that record from performance
statistics. Preserve all raw rows and record counts/reasons. Never substitute a
reference backend for the speedup denominator.

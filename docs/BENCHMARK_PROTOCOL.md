# GPU Library Suite Benchmark Protocol

[English](BENCHMARK_PROTOCOL.md) | [日本語](BENCHMARK_PROTOCOL.ja.md)

## Authority and scope

This document owns benchmark command-line behavior, workload sweeps, timing
boundaries, warm-up and state restoration, verification, CPU-backend
classification, execution ordering, and statistical aggregation. Teaching
filenames, example defaults, and mathematical definitions are owned by
[`PROJECT_SPECIFICATION.md`](PROJECT_SPECIFICATION.md). Serialization and field
types are owned by [`RESULT_SCHEMA.md`](RESULT_SCHEMA.md). System launch and
telemetry details are owned by [`PEGASUS_EXECUTION.md`](PEGASUS_EXECUTION.md).

## General measurement principles

- CPU, direct CUDA, and OpenACC implementations solve the same mathematical
  problem with the same input, precision, problem size, scope, and repeat count.
- All implementations use the same wall-clock helper based on
  `clock_gettime(CLOCK_MONOTONIC, ...)`.
- Record `clock_getres(CLOCK_MONOTONIC, ...)` for every trial series.
- CUDA Events are not the primary comparison timer.
- Warm-up, verification, result serialization, and file I/O are outside timed
  regions.
- Store every raw trial. Benchmark executables do not perform cross-node
  aggregation or outlier removal.
- If one trial is too short for reliable timing, increase `repeat`; do not change
  the mathematical problem silently.
- CPU, CUDA, and OpenACC must use the same repeat for a benchmark/problem/scope.
  The suite runner rejects a mismatch.
- Do not choose problem sizes, algorithms, precision, or numerical modes based on
  detected hardware.
- Report insufficient memory with a clear diagnostic and a failure row.
- A benchmark executable is single-node and single-process. A launcher provides
  multi-node repetition with one process per node.

## Command-line interface

### Common options

Every benchmark supports:

| Option | Meaning |
| --- | --- |
| `--size` | Library-specific primary problem size, as defined below. |
| `--warmup` | Non-negative warm-up count. |
| `--repeat` | Positive operation or pipeline count within each trial. |
| `--trials` | Positive number of raw trials. |
| `--scope` | `compute` or `end-to-end`. |
| `--verify true\|false` | Enable or disable verification explicitly. |
| `--output <path>\|-` | Exclusive result path, or machine-readable stdout. |
| `--format csv\|jsonl` | Output serialization; the effective configuration or CLI must select it. |
| `--device` | Non-negative CUDA device index. |
| `--run-id` | Measurement campaign identifier. |
| `--system-label` | User-provided portable system label. |
| `--node-index` | MPI rank supplied by the runner. |
| `--wave` | Non-negative campaign wave supplied by the runner. |
| `--seed` | Explicit seed where the workload uses randomness. |
| `--cpu-threads` | Requested CPU thread count. |
| `--cpu-threads-effective` | Positive observed/configured effective CPU thread count; omit when unknown. |
| `--cpu-backend-role` | Config-owned CPU role: `production` or `reference`. |
| `--series-role` | Config-owned series role: `primary` or `auxiliary`. |
| `--cpu-parallelism` | Config-owned/observed CPU execution kind: `serial`, `threaded`, or `unknown`. |
| `--implementation-order <list>` | Runner-computed comma-separated list such as `cpu,cuda,openacc`. |
| `--abs-tolerance` | Non-negative absolute verification tolerance from the effective configuration. |
| `--rel-tolerance` | Non-negative relative verification tolerance from the effective configuration. |
| `--help` | Usage and option constraints. |

`--verify` accepts only lowercase `true` or `false`. `--format` accepts only
`csv` or `jsonl`. `--implementation-order` accepts one of the six exact
comma-separated CPU/CUDA/OpenACC permutations defined below.

`--node-index`, `--wave`, and `--implementation-order` are execution-context
options. During suite execution, only `run_suite.py` sets them; it rejects a
value that conflicts with the required ordering rules.

### Output contract and writer ownership

With `--output -`, a benchmark executable writes only its machine-readable
result to stdout. Every human-readable message, warning, progress line, and
diagnostic goes to stderr. Human-readable stdout contamination is an output
error.

With `--output <path>`, the executable creates a new file exclusively. It must
fail if the path already exists and must never overwrite or implicitly append.

During suite execution, `run_suite.py` always launches each benchmark with
`--output -`. It is the only writer of the node-level `raw-results.csv` or
`raw-results.jsonl`; benchmark executables never append directly to a node raw
file. The runner must:

- schema-validate subprocess stdout before accepting a row;
- write a CSV header exactly once;
- append each valid row to the new node raw file;
- reject any human-readable stdout content;
- preserve every validated completed trial;
- synthesize one schema-valid attempted failure row for the first unresolved
  trial after a subprocess crash, signal termination, invalid output, or
  missing output;
- synthesize schema-valid unattempted skipped rows for later trials that did not
  start because of that failure; and
- create the node raw file exclusively and never overwrite an existing file.

The runner rejects duplicate and out-of-range trial indices and materializes
each expected index exactly once. It never replaces a validated completed row
with a synthetic row.

`run_suite.py --dry-run` validates and prints the resolved configuration,
selected manifest entries, implementation/size order, and argv arrays as one
deterministic JSON document. A dry run creates no directory, metadata, result,
or log file and starts no benchmark subprocess.

### Library-specific options and `--size`

#### cuFFT

- Options: `--batch`, `--transform`, `--cpu-backend`.
- `--rel-tolerance` applies to the DC relative error and
  `--abs-tolerance` applies to the maximum non-DC absolute error.
- `--size` is `nfft`.
- `--batch` may be combined with `--size`.
- Initial accepted values are `--transform c2c-forward` and
  `--cpu-backend cpu-fftw-serial|cpu-fftw-threaded`.

#### cuBLAS

- Options: `--m`, `--n`, `--k`, `--alpha`, `--beta`, `--cpu-backend`.
- `--size N` means `m = n = k = N`.
- `--size` is mutually exclusive with `--m`, `--n`, and `--k`.
- If individual dimensions are used, all three are required.
- A square sweep records `problem_size = N`.
- A non-square case records `problem_size = null` and stores `m`, `n`, and `k`
  in `parameters`.

#### cuSPARSE

- Options: `--nx`, `--ny`, `--alpha`, `--beta`, `--cpu-backend`.
- `--size` is the total unknown count `N = nx * ny`.
- With `--size`, N must be a perfect square and
  `nx = ny = sqrt(N)`.
- `--size` is mutually exclusive with `--nx` and `--ny`.
- If dimensions are used, both are required.
- Record `problem_size = nx * ny` and `secondary_size = nnz`, using the exact
  nonzero formula in `PROJECT_SPECIFICATION.md`.

#### cuSOLVER

- Options: `--nrhs`, `--cpu-backend`.
- `--size` is the dense matrix order n.
- Record `secondary_size = nrhs`.

#### cuRAND

- Options: `--generator`, `--distribution`, `--offset`, `--order`.
- Statistical operands are passed explicitly as `--sigma-multiplier`,
  `--expected-mean`, and `--expected-second-central-moment` from the effective
  configuration.
- `--size` is the generated element count.
- Initial accepted values are `--generator pseudo-default`,
  `--distribution uniform-double`, and `--order default`.

#### Thrust

- Options: `--operation`, `--cpu-backend`.
- `--size` is the processed element count.
- The initial accepted operation is
  `--operation transform-reduce-square-sum`.

Ambiguous, incomplete, or contradictory dimension arguments are errors. No
option silently overrides another.

### Dynamic-size and overflow contract

Every runtime dimension is validated before allocation or conversion to a
classic library API integer. A benchmark uses checked conversion from the CLI's
unsigned representation to `int` and `size_t`, checked dimension products,
checked additions such as the CSR row-offset count, and checked
element-count-to-byte calculations. cuFFT length/batch, cuBLAS dimensions,
cuSPARSE `n`/`nnz` with its current 32-bit index mode, and cuSOLVER
`n`/`nrhs`/workspace counts must fit every API type they reach. cuRAND and
Thrust counts must fit both `size_t` and the selected host container's
`max_size()`.

No dimension, index count, or byte count may wrap or be silently truncated.
An out-of-range request produces an API-named diagnostic and a schema-valid
prerequisite or started benchmark failure row according to whether execution
had begun. Allocation and length-related C++ exceptions are handled explicitly;
they do not terminate the process without a result when a valid row can still
be emitted.

## Canonical publication configuration

Each benchmark configuration contains normalized cases. A case has one
`parameters` object and a `scopes` object with separate `compute` and
`end-to-end` entries. Each scope entry contains `warmup`, `repeat`, and
`trials`. Implementations cannot override those values.

CPU, direct CUDA, and OpenACC must have the same values for one
benchmark/normalized problem/scope. Compute and end-to-end settings are
independent and need not have the same repeat or warm-up. cuSOLVER requires
`repeat = 1` in every scope.

The canonical pilot and production configurations use the same publication
workloads, sizes, and repeats:

| Benchmark | Problem sizes | Compute repeats |
| --- | --- | --- |
| cuFFT | `nfft=[256,4096,16384]`, `batch=4096` | `[5197,328,83]` |
| cuBLAS | `size=[512,2048,4096]` | `[2184,133,18]` |
| cuSPARSE | `size=[65536,1048576,4194304]` | `[5776,1520,206]` |
| cuSOLVER | `size=[4096,8192,12288]`, `nrhs=16` | `[1,1,1]` |
| cuRAND | `size=[1048576,16777216,67108864]` | `[640,173,54]` |
| Thrust | `size=[1048576,16777216,67108864]` | `[1932,532,167]` |

Every scope uses `warmup=1`. Compute uses the table's per-case repeat;
end-to-end always uses `repeat=1`. `configs/pilot.json` uses one raw trial and
`configs/benchmark.json` uses five raw trials. Both request 48 CPU threads.
The cuFFT publication series contains threaded FFTW only.

With all six libraries, one pilot block contains 108 expected raw rows. The
six-node, two-wave production design contains 6,480 expected rows. If a pilot
finds different CUDA and OpenACC Thrust versions, exclude Thrust rather than
combining incomparable series; the corresponding counts are 90 and 5,400.
Such an explicitly reduced campaign is a five-library comparison, not a
successful reproduction of all six libraries. Preserve the mismatch evidence;
do not delete failed rows or silently omit an implementation from a figure.

The compute repeats are measurement amplification only: `elapsed_sec` remains
one library operation's elapsed time. A human may revise the canonical values
once after the Pegasus pilot only for OOM, verification failure, extreme
walltime, or a clearly inadequate timed interval. Do not add an automatic
repeat search or create another canonical configuration.

## Trial attempt, failure, and skip behavior

A trial that starts and then fails has `attempted=true` and `status=failure`.
An unstarted trial skipped because of a preceding fatal failure has
`attempted=false` and `status=skipped`. A missing executable or other
prerequisite also produces an unattempted skipped row. Exact fields and null
rules are owned by `RESULT_SCHEMA.md`.

Recoverable verification failures may be followed by later trials after full
canonical restoration. A fatal setup or execution failure stops that benchmark
invocation and marks only the remaining unstarted trials skipped. Failures and
skips remain visible and are never performance samples.

## CPU backends and parallelism

Every CPU backend has a stable name and a role:

- `production`: a principal performance-comparison series;
- `reference`: correctness, smoke testing, or operation without an optional
  optimized dependency.

A reference backend must never silently replace a production backend. Each
benchmark entry in the effective configuration names its
`default_speedup_cpu_backend`. Speedup is produced only against that named
production backend. If it is unavailable or invalid, omit the speedup and record
the reason; do not fall back to a reference series.

A primary production campaign selects exactly one production CPU backend for
each benchmark. Its CPU/CUDA/OpenACC three-way permutation must not contain
multiple CPU backends. An additional reference backend or alternative
production backend is measured in a separate campaign or as an explicitly
classified auxiliary series. It is never adopted automatically as the primary
speedup denominator. Effective configuration and node metadata record whether a
series is `primary` or `auxiliary`.

Raw results distinguish:

- `cpu_threads_requested`: the value requested by CLI/configuration;
- `cpu_threads_effective`: the thread count the backend actually uses, or null
  when it cannot be established; and
- `cpu_parallelism`: `serial`, `threaded`, or `unknown`.

The effective configuration is authoritative for `cpu_backend_role`,
`series_role`, and the expected parallelism/known effective-thread metadata of
each configured series. `run_suite.py` passes those values to the benchmark and
validates emitted rows against the same series object. Common result code must
not infer a role from a backend-name substring, special-case one backend name
to choose a series role, or classify every non-serial name as threaded. A
requested thread count is never copied into `cpu_threads_effective` merely
because no better observation exists. An unobservable effective count is null;
unestablished parallelism is `unknown`.

Backend-specific thread control is as follows:

- OpenMP runtime variables may control a linked library even when the calling
  source has no OpenMP parallel loop. Record `OMP_NUM_THREADS`, `OMP_PROC_BIND`,
  and `OMP_PLACES`; the presence or absence of the application's `-fopenmp`
  option does not identify the library's threading implementation.
- oneMKL uses `MKL_NUM_THREADS` or the supported local thread-control API and
  records the request separately from an established effective count. The
  saved Pegasus campaign used `MKL_NUM_THREADS=48` and
  `MKL_THREADING_LAYER=INTEL`; its oneMKL raw rows retain
  `cpu_threads_effective=null`, not a measured effective count of 48.
- OpenBLAS uses `OPENBLAS_NUM_THREADS` or a supported backend API. If a build
  cannot determine the effective count, record null and `unknown` rather than
  claiming the requested count.
- Threaded FFTW initializes its threaded interface and applies the requested
  count before plan creation. Serial FFTW always records effective count 1.
  The saved publication build used FFTW's `--enable-threads` pthread-based
  implementation and the FFTW threads API with 48 threads. The cuFFT CPU
  teaching example is serial; this threaded setting belongs to the benchmark.

FFTW serial and threaded series use distinct names, for example
`cpu-fftw-serial` and `cpu-fftw-threaded`. oneMKL and OpenBLAS series also use
implementation-specific names such as `cpu-onemkl` and `cpu-openblas`.
`cpu-reference-csr` is a reference backend.

On Pegasus, the canonical publication configuration contains only
`cpu-fftw-threaded` for cuFFT. A separately invoked `cpu-fftw-serial` executable
may still support teaching correspondence, but it is not a publication series
and never replaces the threaded backend.

The canonical cuRAND CPU benchmark is `cpu-std-random-serial`, role
`production`, using `std::mt19937_64` and
`std::uniform_real_distribution<double>`. Its effective thread count is 1. The
canonical Thrust CPU benchmark is `cpu-stl-serial`, role `production`, using
`std::transform_reduce` without an execution policy; its effective thread count
is also 1. The approved publication legends explicitly say **single thread**
for both, never parallel or algorithm-equivalent performance and never 48-core
CPU performance. Exact displayed labels are listed under
[Publication figures](#publication-figures).

## Fortran binding of this protocol

`nvidia/fortran` applies the same CLI, mathematics, FP32/FP64 precision,
verification thresholds, trial restoration, compute and one-shot E2E contracts.
Its Fortran workload callbacks use the existing C timer/CLI/result writer through
`ISO_C_BINDING`; the computational library calls remain Fortran. Array indexing
is one-based CSR where required, not a different Poisson problem. CPU solver
benchmarks use `dgetrf` plus `dgetrs`; teaching `dgesv` remains separate.

Fortran CPU production backends are `cpu-fftw-serial`/`cpu-fftw-threaded`,
`cpu-onemkl`, `cpu-fortran-random-serial`, and `cpu-fortran-sum-serial`.
The last two use compiler intrinsics without automatic parallelization or GPU
offload, record effective count 1 and compiler identity, and never identify
themselves as C++ MT19937/STL. FFTW initializes threads before planning; oneMKL
requests threads through its API but keeps an unobserved effective count null.
Fortran RNG records `cpu_engine="Fortran random_number"`; seed-vector elements
are filled with the configured seed (LP64 signed range), and offset consumes
that stream. CUDA uses the same cuRAND generator contract as the C/C++ path;
CPU/GPU stream identity is not required.

Fortran Release defaults to `-O3`. GNU target options are
`-fno-fast-math -ffp-contract=off`; NVHPC target options are
`-Kieee -Mnoflushz -Mnodaz`, with explicit separate GPU memory and OpenACC only
on GPU targets. The Fortran profile rejects unapproved global numerical,
default-kind-changing, automatic parallel/offload flags and unsafe bridge flags.
Unset `NVCOMPILER_FPU_STATE`; NVHPC benchmarks reject a nonempty override.
This does not change or retroactively approve the historical C/C++ `-fast`
condition. Verification is explicit and remains enabled when requested in Release.

The Fortran diagnostic configuration is `configs/fortran/pilot.json`; it is not
an approved production profile or the C/C++ publication workload. Language
identity and compatibility are specified in [`RESULT_SCHEMA.md`](RESULT_SCHEMA.md).
Use separate campaigns and the existing two-panel figure path; retain
Fortran-specific labels and never repurpose C/C++ measurements.

## Timing fields

For every raw trial:

```text
elapsed_total_sec = total time actually measured in the trial
elapsed_sec       = elapsed_total_sec / repeat
```

Both are in seconds and must be finite and non-negative for a successful trial.
The result schema separately records:

- `record_timestamp`: when the raw row is created;
- `measurement_start_timestamp`: the UTC start of the trial's first timed
  interval; and
- `measurement_end_timestamp`: the UTC end of the trial's last timed interval.

For end-to-end scope, untimed state restoration and cleanup may occur between
individually timed repeats. Therefore the UTC wall span from measurement start
to measurement end need not equal `elapsed_total_sec`, which is the sum of timed
intervals only.

### Compute scope

The generic GPU order is:

1. prepare allocation, transfers, plans, handles, descriptors, and workspace;
2. restore canonical trial-start state;
3. synchronize before timing;
4. read the start time;
5. execute the operation `repeat` times;
6. synchronize before the end timestamp; and
7. read the end time.

Allocation, host-device transfer, plan/handle/descriptor/workspace creation,
input restoration, verification, and cleanup are outside the timed region. CPU
implementations use the same timer and corresponding operation boundary.

cuFFT plans, cuBLAS handles, cuSPARSE descriptors/workspace, and cuRAND
generators are created before timing. Destructive input restoration remains
outside timing. cuSOLVER compute scope requires `repeat = 1`; a larger value is
an error. Use multiple trials instead.

### End-to-end scope

Host input allocation and initial-value generation are outside timing. Time each
repeat separately:

1. read its start time;
2. perform library-specific setup, device allocation, H2D, operation, required
   synchronization, and D2H/result retrieval;
3. read its end time; and
4. perform that repeat's cleanup after the end timestamp.

Cleanup, verification, and output are outside timing. Host input regeneration is
outside timing. Sum the individually measured repeat durations into
`elapsed_total_sec`.

### OpenACC compute scope

OpenACC-managed allocation and movement are defined by
`PROJECT_SPECIFICATION.md`. The preparation order is fixed:

1. create plans/handles that do not need managed-array device addresses;
2. enter the `acc data` region, where the OpenACC runtime allocates managed
   arrays and performs the requested copyin/create operation;
3. use `host_data use_device` to translate already-present managed data to
   device addresses;
4. create descriptors that need those device addresses;
5. query workspace size and explicitly allocate only library workspace not
   managed by an OpenACC data clause;
6. synchronize; and
7. read the start time.

Skip a step that a library does not use, but do not reorder applicable steps.
After the start timestamp, execute the CUDA library call `repeat` times,
synchronize before the end timestamp, and read the end time. OpenACC data exit,
copyout, verification, descriptor/plan destruction, and cleanup occur afterward
and are outside timing.

For cuSOLVER, whose calls need raw addresses rather than persistent data
descriptors, enter the data region before the address translations. Use the
minimal buffer-size-query translation before workspace allocation and the
minimal `getrf`/`getrs` translation during the operation. This does not change
the rule that data must already be present before address translation.

### OpenACC end-to-end scope

For each repeat, read the start time before plan/handle creation. The measured
pipeline includes:

- plan/handle creation;
- entry into the OpenACC data region, including runtime allocation and the
  requested copyin/create operation;
- device-address translation and pointer-dependent descriptor creation;
- library workspace size query and allocation;
- the CUDA library call;
- `cudaDeviceSynchronize`;
- data-region exit, including the required copyout; and
- host result availability.

Read the end timestamp only after data exit/copyout completes. Destroy plans,
handles, descriptors, and explicit workspace afterward as untimed cleanup.
OpenACC end-to-end measurements must not omit data entry/exit or copyin/copyout.

## Warm-up and canonical trial-start state

For each implementation and problem size, first perform setup, warm-up, and a
complete restoration of every input, output, and library state modified by
warm-up. Then apply this rule independently to every raw trial:

```text
restore canonical trial-start state
run one measured trial
```

The first trial is restored even though warm-up restoration just occurred. No
trial may inherit the final state of a preceding trial.

Warm-up exercises the same scope-specific pipeline selected for measurement.
For compute scope, create the persistent plan, handle, descriptor, generator,
device allocation, and workspace first; run the compute operation exactly
`warmup` times; restore canonical state; and measure trials with that same
persistent context. For end-to-end scope, do not create or retain a compute-only
persistent context. Instead run exactly `warmup` complete temporary pipelines,
each containing the scope's allocation, setup, transfer, operation, result
retrieval, and complete cleanup. No warm-up handle, descriptor, generator,
workspace, device allocation, or device container remains alive when the first
end-to-end trial starts.

In compute scope, operations within one trial's repeat loop may update C or y
successively. Verification accounts for all `repeat` updates, but C/y are reset
before the next raw trial. In end-to-end scope, restore host-side inputs and
outputs before every repeat, outside that repeat's timed interval, so every
pipeline starts from the same canonical initial values.

Required restoration is:

- cuBLAS: restore C before every trial and every end-to-end repeat;
- cuSPARSE: restore y before every trial and every end-to-end repeat;
- cuSOLVER: restore A, B, `ipiv`, `getrf_info`, and `getrs_info` before every
  trial; `repeat` remains exactly 1;
- cuRAND compute: before every trial restore seed, offset, and generator state;
  repeats within that trial generate successive blocks from that state;
- cuRAND end-to-end: create the generator and set seed/offset inside every
  timed repeat, giving each repeat the same start condition; and
- any future in-place cuFFT or Thrust operation: restore every destroyed input
  before the next trial or end-to-end repeat.

Warm-up for an implementation occurs immediately before measuring that
implementation. All required pre-trial and pre-repeat restoration is untimed.
If a helper can validate restored state without contaminating timing, it should
do so.

## Standard workloads and verification

Benchmark mathematics and default example inputs follow
`PROJECT_SPECIFICATION.md`; the sweep and verification rules below are
benchmark-specific.

### Verification schema and thresholds

All thresholds live in the effective configuration. Source code must not contain
unexplained tolerance magic numbers. Ordinary numerical error passes when:

```text
error <= abs_tolerance + rel_tolerance * reference_scale
```

Suite execution requires each benchmark's complete verification object in the
effective configuration and passes every operand explicitly on the benchmark
command line. The runner rejects missing operands. Standalone invocation keeps
documented smoke defaults: absolute `1e-12` and relative `1e-10`, except cuFFT
absolute `1e-4` and relative `1e-5`; cuRAND uses
`sigma_multiplier=6`, `expected_mean=0.5`, and
`expected_second_central_moment=1/12`. These built-in defaults are not
production-suite configuration and never override an effective configuration.

Every benchmark, including single-metric cases, emits objects:

- `verification_metrics`: metric name to measured value;
- `verification_thresholds`: metric name to its complete condition, including
  method and any reference scale, absolute/relative tolerance, or range bound;
- `verification_primary_metric`: a representative metric name or null; and
- `verification_status`: `pass`, `failure`, `skipped`, or `nonfinite`.

There is no scalar form of the metrics or thresholds. A nonfinite metric is
represented as null with `verification_status = nonfinite`, overall
`status = failure`, and a diagnostic message. Any verification failure remains
in raw results but is excluded from performance aggregation.

Verification checks every raw output, intermediate error/residual, and norm for
finiteness before passing it to `fmax`, `std::max`, or another reduction.
Neither `fmax` nor `std::max` is a NaN detector. Finite values within threshold
pass; finite values outside threshold fail verification; NaN or either infinity
always produces a nonfinite verification failure. Failure to allocate a JSON
verification object or insert a required metric/threshold is a fatal result-
construction/benchmark error, not a numerical verification failure and never a
partial success record.

### cuFFT

- Sweep batched 1D C2C forward FFT, FP32, primarily by FFT length N.
- Hold batch, transform, and precision fixed within one sweep.
- Use FFTW3's single-precision interface for CPU; threaded FFTW is a separately
  named optional production backend.
- Record separate metrics for the expected DC component and the maximum non-DC
  error, such as `dc_relative_error` and `non_dc_max_abs_error`.

### cuBLAS

- Sweep FP64 DGEMM, normally `m = n = k = N`.
- Use a named CBLAS-compatible production backend and record whether it is
  oneMKL, OpenBLAS, or another implementation.
- Verification uses the all-ones expectation and accounts for every C update
  caused by `repeat`.

### cuSPARSE

- Sweep the canonical FP64 Poisson CSR SpMV, normally with `nx = ny`, primarily
  by total unknown count `N = nx * ny`.
- Prefer a named optimized production backend such as oneMKL Sparse BLAS.
  `cpu-reference-csr` is a distinct reference series.
- Validate row offsets, column indices, values, and the numerical result.
- Verification accounts for every y update caused by `repeat`.

### cuSOLVER

- Sweep dense FP64 LU factorization plus solve primarily by matrix order N, with
  `nrhs` fixed within a sweep.
- The CPU benchmark uses `LAPACKE_dgetrf` followed by `LAPACKE_dgetrs`, not the
  `LAPACKE_dgesv` call used by the CPU teaching example.
- CUDA and OpenACC benchmarks use `getrf` followed by `getrs`.
- Store `getrf_info` and `getrs_info` separately.
- Store distinct `solution_relative_error` and `relative_residual` verification
  metrics.
- Require `repeat = 1` and use multiple trials.

### cuRAND

- Sweep uniform double generation primarily by element count N.
- Record the actual CPU engine, actual cuRAND generator, seed, offset, and order
  in metadata.
- Do not require element-wise identity between CPU and GPU streams.
- `std::mt19937_64` and `CURAND_RNG_PSEUDO_DEFAULT` are different random-number
  algorithms. Their elapsed times describe the same task, output type, and
  uniform distribution, not an algorithm-equivalent CPU/GPU comparison.
- CPU and GPU range sanity checks both permit `0.0 <= x <= 1.0`; record each
  backend's precise interval contract in `verification_thresholds` or
  `parameters`.
- Record the CPU standard-distribution contract as `[0,1)` and the cuRAND
  contract as `(0,1]` while applying the common inclusive sanity range.
- Configuration stores `sigma_multiplier`, `expected_mean`, and
  `expected_second_central_moment`; initial values are `6.0`, `0.5`, and
  `1/12` respectively.
- Required metrics are `observed_min`, `observed_max`, `sample_mean`, and
  `second_central_moment_about_half`, with:

  ```text
  second_central_moment_about_half = mean((x_i - 0.5)^2)
  ```

- Mean verification is:

  ```text
  abs(sample_mean - 0.5)
      <= sigma_multiplier * sqrt(1 / (12 * N))
  ```

- Second-central-moment verification is:

  ```text
  abs(second_central_moment_about_half - 1/12)
      <= sigma_multiplier * sqrt(1 / (180 * N))
  ```

- Verification uses the last retrieved N values and records
  `verification_sample_count=N`.
- Plots, plot metadata, aggregate metadata, and user-facing documentation state:
  **Same distribution and output type task; different RNG algorithms.**

### Thrust

- Sweep FP64 `transform_reduce` primarily by element count N.
- Record the CPU backend name and effective parallelism.
- Verify against the expected result N using a configured numerical threshold.

## Interleaved execution order

Running all CPU cases, then all CUDA cases, then all OpenACC cases is forbidden.
For each node:

```text
for each benchmark:
    determine this node's size order
    for each problem size in that order:
        run CPU, CUDA, and OpenACC in the assigned implementation permutation
```

The three implementations of one problem size must run close together in time.
Warm-up for each implementation occurs immediately before its measurement.

Implementation permutations are:

| Index | Order |
| --- | --- |
| 0 | CPU, CUDA, OpenACC |
| 1 | CPU, OpenACC, CUDA |
| 2 | CUDA, CPU, OpenACC |
| 3 | CUDA, OpenACC, CPU |
| 4 | OpenACC, CPU, CUDA |
| 5 | OpenACC, CUDA, CPU |

Use:

```text
permutation_index = (node_index + wave) mod 6
size_order_index  = node_index mod 2
```

`size_order_index = 0` uses configuration order; index 1 reverses that order.
Implementation and size order deliberately do not use the same expression.

With six nodes, two consecutive waves assign every implementation permutation
to both size orders. Five- and eight-node waves are permitted, but metadata must
record assignment counts and make imbalance explicit. Record
`implementation_order`, `permutation_index`, the effective per-benchmark
`size_order`, and `size_order_index` in node metadata.

## Blocks, waves, and aggregation

`node_index` is the MPI rank in the current job, not a persistent physical-node
identifier. `hostname` identifies the physical node. A block is the combination
of run ID, wave, and hostname defined in `RESULT_SCHEMA.md`.

Aggregation is hierarchical:

1. At block level, compute the median across valid trials for each
   benchmark/implementation/problem/scope.
2. Within that block, compute CPU/CUDA and CPU/OpenACC speedups using the
   configured production CPU median.
3. At wave level, aggregate the block medians across hostnames and produce the
   wave-level median and other within-wave statistics.
4. At primary cross-wave level, use only each wave's wave-level median as one
   input value. Set `summary_input_statistic = "wave_median"`.

Do not use quartiles of quartiles, average wave IQRs, or a re-pool of all blocks
as the primary cross-wave summary. Each primary cross-wave statistic is
recomputed from the vector of wave medians.

An optional pooled analysis is a separate record labeled
`pooled exploratory summary`; it is not a primary result or primary graph input.
Never compute speedup as the median of all CPU values divided by the median of
all GPU values.

For every group with at least two valid samples, use the Python 3.9 standard
library definition:

```python
statistics.quantiles(values, n=4, method="inclusive")
```

Record median, Q1, Q3, `IQR = Q3 - Q1`, minimum, and maximum. With one valid
sample:

- median, minimum, and maximum equal that value;
- Q1, Q3, and IQR are null; and
- `aggregate_status = "insufficient_sample_count"`.

With zero valid samples, emit no numerical summary; emit a status-only aggregate
record with `aggregate_status = "no_valid_samples"`.

Primary production measurement uses at least five nodes and preferably eight.
Additional waves may cover different time periods. Flag thermal, power, clock,
telemetry, or verification anomalies, but never remove a slow node merely
because of elapsed time. Assignment counts, exclusions caused by actual failure,
and valid sample counts remain visible in aggregate metadata.

## Publication figures

Publication plotting uses only primary `cross-wave` summary records. It emits
one two-panel elapsed-time figure per enabled publication library:

- `cufft-elapsed-time.png`;
- `cublas-elapsed-time.png`;
- `cusparse-elapsed-time.png`;
- `cusolver-elapsed-time.png`;
- `curand-elapsed-time.png`; and
- `thrust-elapsed-time.png`.

The left panel is titled **Library kernel execution time** and represents
data-resident compute. The right is titled **End-to-end execution time** and
represents the one-shot host-input-to-host-output pipeline. Both use the
library-specific problem size on the
x-axis and **Elapsed time [ms]** on the y-axis; lower is better. CPU, CUDA, and
OpenACC appear together. The plotted compute value is the existing
per-operation `elapsed_sec` converted from seconds to milliseconds; it is not
divided by repeat again. The one-shot value comes from `repeat=1` end-to-end
records. Do not produce a speedup, throughput, bandwidth, FLOPS, sample-rate,
element-rate, reuse-count, amortized, break-even, or additional generic elapsed
figure.

The approved shared legend is below the two panels. Its exact labels are:

| Series | Label |
| --- | --- |
| cuFFT CPU | `Intel Xeon Platinum 8468, FFTW (48 C)` |
| cuBLAS/cuSPARSE/cuSOLVER CPU | `Intel Xeon Platinum 8468, oneMKL (48 C)` |
| cuRAND CPU | `Intel Xeon Platinum 8468, std::mt19937_64 (single thread)` |
| Thrust CPU | `Intel Xeon Platinum 8468, STL (single thread)` |
| CUDA | `NVIDIA H100 PCIe, CUDA` |
| OpenACC | `NVIDIA H100 PCIe, OpenACC` |

The `(48 C)` labels describe the approved campaign configuration; they do not
change the oneMKL effective-thread evidence described above. cuRAND and Thrust
are serial references, not parallel or algorithm-equivalent denominators.
The logarithmic x-axis uses binary K/M notation (`1K=1024`, `1M=1048576`);
cuSOLVER retains `4K`, `8K`, `12K` for those input sizes and uses actual input
sizes for other sweeps. The table is the approved teaching configuration, not
a default for unknown hardware. Use `--display-config` with the explicit
`compact-requested` convention to reproduce it. Normal labels distinguish
requested and reported effective counts; missing values remain unknown.
`--node-metadata` can supply observed CPU identities and raw rows supply GPU
identities. The [display schema](RESULT_SCHEMA.md#plot-display-configuration)
defines provenance and validation; follow the
[own-measurement route](PORTABILITY.md#figures-from-your-own-measurements).

A Thrust figure requires raw-result evidence
that successful CUDA and OpenACC rows report one identical `library_version`.
Missing, mixed, or unequal evidence is an error before any figure is written;
never silently omit one implementation.

Each publication caption joins the plot's run ID and runtime-environment hash
to the existing immutable run and node metadata; no result-schema field is
added. It records the system label, CPU model, GPU model, precision, operation,
six nodes by two waves, five trials per node block, block median to wave median
to cross-wave median, both timing boundaries, compute repeat as measurement
amplification, end-to-end repeat 1, and that lower is better. It also records:

- cuBLAS: FP64 DGEMM with the cuBLAS default math mode;
- cuFFT: FP32 complex batched 1-D C2C forward transforms, `batch=4096`, and
  power-of-two lengths;
- cuSPARSE: FP64 CSR SpMV on the regular 2-D Poisson matrix, without an added
  preprocess stage;
- cuSOLVER: FP64 LU factorization plus solve with `nrhs=16`;
- cuRAND: uniform-double generation with the GPU pseudo-default generator and
  a non-algorithm-equivalent serial CPU generator; and
- Thrust: double `transform_reduce` with one common CCCL/Thrust version across
  CUDA and OpenACC.

The two panels are also the teaching model. Data-resident compute represents an
application that keeps data on the device for a library operation. The one-shot
pipeline includes setup, device allocation, H2D, one operation, completion, and
D2H until the host result is available. Input generation, verification,
serialization, file I/O, and protocol cleanup remain outside. Reusing
device-resident data can move observed application cost from the one-shot side
toward the compute side because transfer contributes relatively less. This is
an interpretation of the existing two scopes, not a third amortized scope. The
workloads are one representative operation per library and do not characterize
every algorithm in that library.

## Reading the figures and applying the results

The figure is a comparison of two boundaries, not a promise of whole-application
speedup. Read the x-axis as the configured problem size and the y-axis as
milliseconds per operation; lower is better. Compare CPU, CUDA, and OpenACC
only at the same size and within the same panel. Do not divide the plotted
compute value by `repeat` again.

| Work | Compute panel | One-shot E2E panel |
| --- | --- | --- |
| Host input allocation/generation and canonical restoration | Outside | Outside |
| GPU allocation, H2D, plan/handle/descriptor/workspace setup | Prepared before timing | Inside the per-repeat pipeline where the library requires it |
| Library operation and synchronization that establishes its completion | Inside | Inside |
| D2H/result retrieval | After compute timing | Inside, before the end timestamp |
| OpenACC data entry/exit and required copyin/copyout | Outside | Inside |
| Explicit post-result cleanup, numerical verification, serialization and file I/O | Outside | Outside |

CPU implementations use their corresponding operation/setup boundary without
inventing host-device copies. Library-specific resource details, especially
cuSOLVER restoration and FFT plans, remain in the scope and workload sections
above and each library README. E2E is **not** the entire program's wall time:
input generation, verification, output, and post-result cleanup are excluded.

Warm-up is untimed and follows the chosen scope. Each raw trial starts from
restored state. Compute repetition amplifies the measurable interval; E2E
publication repeat is 1, and cuSOLVER repeat is always 1. A block median is
formed from valid trials, then a wave median from blocks, then a cross-wave
median from wave medians. One block/wave is useful for checking the pipeline,
but does not establish cross-node/time variability. Inspect sample counts,
quartiles, failure counts, and provenance as well as the plotted median.

Before applying a result to your program, check whether its data stays on the
GPU, whether plans/handles can be reused, how often transfers and synchronization
are needed, and what fraction of application time is in this operation. A
shorter GPU compute time may coexist with a slower one-shot pipeline. Their
difference suggests costs to investigate; it is not a separately measured
transfer-only time or proof of a particular bottleneck. Host work, I/O, other
kernels, and contention can limit whole-application gains. No overlap or reuse
optimization is measured merely because it could be implemented.

Keep precision, operation, parameters, verification thresholds, CPU provider,
thread request/effective evidence, compiler flags, GPU/software versions, and timing scope
visible when comparing runs. The configured all-ones or analytical problems
are controlled examples, not a survey of real-world input distributions.
cuRAND compares different RNG algorithms; cuRAND/Thrust CPU series are
single-threaded. Never generalize their ratios to optimized parallel CPU
implementations. Failed, skipped, or nonfinite trials remain in the raw data
and are excluded from numerical aggregation, not erased as inconvenient data.

For the exact configuration edit locations and the executable command sequence,
use [the reader workflow](PORTABILITY.md#measuring-on-your-own-system).

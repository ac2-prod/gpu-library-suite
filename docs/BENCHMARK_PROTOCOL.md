# GPU Library Suite Benchmark Protocol

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
| `--implementation-order <list>` | Runner-computed comma-separated list such as `cpu,cuda,openacc`. |
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
- synthesize a schema-valid failure row for a subprocess crash, signal
  termination, invalid output, or missing output; and
- create the node raw file exclusively and never overwrite an existing file.

### Library-specific options and `--size`

#### cuFFT

- Options: `--batch`, `--transform`, `--cpu-backend`.
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

Backend-specific thread control is as follows:

- OpenMP code uses the requested value through `OMP_NUM_THREADS` and, where the
  implementation controls a region directly, the corresponding OpenMP runtime
  control. Record `OMP_PROC_BIND` and `OMP_PLACES`.
- oneMKL uses `MKL_NUM_THREADS` or the supported local thread-control API and
  records the effective setting.
- OpenBLAS uses `OPENBLAS_NUM_THREADS` or a supported backend API. If a build
  cannot determine the effective count, record null and `unknown` rather than
  claiming the requested count.
- Threaded FFTW initializes its threaded interface and applies the requested
  count before plan creation. Serial FFTW always records effective count 1.

FFTW serial and threaded series use distinct names, for example
`cpu-fftw-serial` and `cpu-fftw-threaded`. oneMKL and OpenBLAS series also use
implementation-specific names such as `cpu-onemkl` and `cpu-openblas`.
`cpu-reference-csr` is a reference backend.

The canonical cuRAND CPU benchmark is `cpu-std-random-serial`, role
`production`, using `std::mt19937_64` and
`std::uniform_real_distribution<double>`. Its effective thread count is 1. The
canonical Thrust CPU benchmark is `cpu-stl-serial`, role `production`, using
`std::transform_reduce` without an execution policy; its effective thread count
is also 1. Plots label both as **Serial CPU baseline**, never as 48-core CPU
performance.

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
  algorithms. This is a throughput comparison of the same task, output type,
  and uniform distribution, not an algorithm-equivalent comparison.
- CPU and GPU range sanity checks both permit `0.0 <= x <= 1.0`; record each
  backend's precise interval contract in `verification_thresholds` or
  `parameters`.
- Store separate range, mean, and variance evidence. At minimum, metrics retain
  observed minimum, maximum, mean, and variance, while threshold objects define
  the permitted uniform range and deviations from expected mean and variance.
- Plots and future user-facing documentation must state the equivalent of:
  **Same output distribution and type; different RNG algorithms.**

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

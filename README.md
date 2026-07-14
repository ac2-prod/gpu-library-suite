# GPU Library Suite

Portable NVIDIA C/C++ teaching examples, reproducible CPU/CUDA/OpenACC
benchmarks, strict result tooling, and Pegasus job support for cuFFT, cuBLAS,
cuSPARSE, cuSOLVER, cuRAND, and Thrust.

The implemented source scope is `nvidia/c-cpp`. AMD, Fortran, and additional
systems are intentionally deferred; the repository does not contain empty
placeholder implementations for them. The project is maintained under
`ac2-prod` and is not an official HAIRDESC repository.

## What is implemented

Each library has three direct, single-source teaching examples and three
measurement-oriented benchmarks. CMake target names and executable names equal
their source stems.

| Library | CPU | CUDA | OpenACC | CPU benchmark | CUDA benchmark | OpenACC benchmark |
| --- | --- | --- | --- | --- | --- | --- |
| cuFFT | `fft_cpu` | `fft_gpu` | `openacc_cufft` | `fft_cpu_bench` | `fft_gpu_bench` | `openacc_cufft_bench` |
| cuBLAS | `blas_cpu` | `blas_gpu` | `openacc_cublas` | `blas_cpu_bench` | `blas_gpu_bench` | `openacc_cublas_bench` |
| cuSPARSE | `sparse_cpu` | `sparse_gpu` | `openacc_cusparse` | `sparse_cpu_bench` | `sparse_gpu_bench` | `openacc_cusparse_bench` |
| cuSOLVER | `solver_cpu` | `solver_gpu` | `openacc_cusolver` | `solver_cpu_bench` | `solver_gpu_bench` | `openacc_cusolver_bench` |
| cuRAND | `rand_cpu` | `rand_gpu` | `openacc_curand` | `rand_cpu_bench` | `rand_gpu_bench` | `openacc_curand_bench` |
| Thrust | `reduce_cpu` | `reduce_gpu` | `openacc_thrust` | `reduce_cpu_bench` | `reduce_gpu_bench` | `openacc_thrust_bench` |

`examples/` is the teaching-material source of truth. Those programs have no
benchmark CLI or machine-readable output. `benchmarks/` adds two timing scopes,
warm-up, repeat/trial control, verification, UTC timestamps, and strict JSONL or
CSV records. The exact problems and filenames are in
[`docs/PROJECT_SPECIFICATION.md`](docs/PROJECT_SPECIFICATION.md); measurement
semantics are in [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md).

## Requirements and CPU-only build

The build requires CMake 3.20 or newer, a C17 compiler, and a C++17 compiler.
Python tools support Python 3.9 or newer. CUDA, NVHPC, FFTW, oneMKL, OpenBLAS,
and LAPACKE are optional: CMake compile-and-link probes disable only affected
targets and print the reason. No dependency is installed automatically.

A local CPU-only build does not probe or enable the CUDA language:

```bash
cmake -S . -B /tmp/gpu-library-suite-local-build/cpu-only-validation \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_SUITE_BUILD_CPU=ON \
  -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=OFF
cmake --build /tmp/gpu-library-suite-local-build/cpu-only-validation --parallel
ctest --test-dir /tmp/gpu-library-suite-local-build/cpu-only-validation \
  --output-on-failure
```

The dependency-free `rand_cpu`, `rand_cpu_bench`, `reduce_cpu`, and
`reduce_cpu_bench` targets are always available in this profile. Library
READMEs document direct teaching-example compile commands and optional provider
requirements.

Production artifacts use two non-overlapping build profiles:

- `cpu-cuda`: CPU ON, CUDA ON, OpenACC OFF;
- `openacc`: `nvc`/`nvc++`, CPU OFF, CUDA OFF, OpenACC ON.

CUDA architecture and NVHPC GPU target values are external build inputs. The
OpenACC tree requires its CMake CUDA Toolkit selection to match the Toolkit
selected by NVHPC. See [`docs/PORTABILITY.md`](docs/PORTABILITY.md) and the
[Pegasus job README](jobs/pegasus/README.md).

CMake build metadata records `global_configure_flags` for each compiler
language. That field is deliberately limited to global CMake configure flags;
it does not claim target compile definitions/options, provider flags, OpenACC
interoperation options, or link options. OpenACC and Thrust interoperation
flags are stored separately in build metadata.

A source copy without `.git` still produces valid metadata with
`git_metadata_available=false`, `git_commit=null`, and `git_dirty=null`.
Archive/local validation may build that source, but production execution
rejects unknown Git provenance instead of treating it as clean.

## Benchmark configuration and output

[`configs/pilot.json`](configs/pilot.json) and
[`configs/benchmark.json`](configs/benchmark.json) contain the approved initial
calibration candidates, with independent `compute` and `end-to-end` warm-up,
repeat, and trial settings. They are starting points, not final production
sizes. After calibration, update `configs/benchmark.json` in place; do not
create a dated or version-suffixed replacement.

For an individual benchmark:

- `--output -` emits only machine-readable records to stdout and diagnostics to
  stderr;
- `--output <path>` exclusively creates a new file and refuses overwrite or
  implicit append.

In a suite run, [`tools/run_suite.py`](tools/run_suite.py) starts every benchmark
with `--output -`, validates its records, synthesizes failure/skipped rows when
needed, and is the only writer of the node-level raw-result file. Its
`--dry-run` prints deterministic resolved JSON without creating files,
directories, metadata, or processes.

The normal data flow is:

```text
partial manifests -> merge_manifests.py -> run_suite.py -> validate_results.py
                                             |
                                             +-> aggregate.py -> plot.py
```

All output paths are exclusive-create. Raw rows distinguish a trial that
started and failed (`attempted=true`, `status=failure`) from a remaining trial
that never started (`attempted=false`, `status=skipped`). The schema and exact
provenance contract are in [`docs/RESULT_SCHEMA.md`](docs/RESULT_SCHEMA.md).

Every production suite invocation passes the complete verification object from
its effective configuration. Built-in CLI defaults exist only for standalone
teaching/smoke use and are not an alternative source of production thresholds.

## CPU baselines and verification notes

Pegasus cuFFT primary speedup uses `cpu-fftw-threaded`. The
`cpu-fftw-serial` series is auxiliary teaching correspondence and never becomes
an automatic fallback denominator. Without threaded FFTW, no primary cuFFT
speedup is produced.

cuRAND's `cpu-std-random-serial` and Thrust's `cpu-stl-serial` are production
serial baselines, displayed as **Serial CPU baseline** even when 48 CPU threads
were requested. cuRAND compares the same distribution and output type task
across CPU and GPU, but the RNG algorithms differ; element-by-element identity
is not required.

## Pegasus execution boundary

The files under [`jobs/pegasus`](jobs/pegasus) render builds and jobs, collect a
hashed runtime environment, prepare campaign/wave metadata, isolate node
failures, correlate telemetry intervals, and collect node artifacts. They do
not submit, query, cancel, or otherwise operate the scheduler. Account, queue,
module versions, Toolkit selection, architecture, and output paths remain
human-supplied.

Each node obtains its own GPU name, UUID, NVIDIA package-driver identity, and
CUDA Driver API/Runtime versions from node-local `libcudart`. GPU benchmarks
independently query CUDA device/runtime and library APIs; successful rows must
agree with that node metadata. One node's identity is never reused for all
nodes. For prebuilt binaries, `nvcc`, `nvc`, and `nvc++` are
optional provenance probes, while the driver, runtime, and resolved shared
libraries are runtime prerequisites.

A recovered node records benchmark or verification failure but normally exits
zero so another node can finish artifact recovery. After the measurement
launch, the job master collector checks every expected node and owns the final
nonzero job decision. The authoritative procedure is
[`docs/PEGASUS_EXECUTION.md`](docs/PEGASUS_EXECUTION.md); the human checklist is
[`docs/PEGASUS_MANUAL_VALIDATION.md`](docs/PEGASUS_MANUAL_VALIDATION.md).

## Validation status

Local validation covers strict serialization, schemas, all source inventories,
dependency probe fixtures, CPU benchmark fixtures, runner/aggregation/plot
logic, Python 3.9 compatibility, shell syntax, ShellCheck when available, job
rendering, telemetry parsing, and multi-node failure simulation. See
[`docs/VALIDATION_REPORT.md`](docs/VALIDATION_REPORT.md).

Clean Linux GCC and Clang CPU-only builds, including fake-provider fixtures and
the `.git`-less source-copy test, are mandatory pre-merge gates. The minimal
workflow is [`.github/workflows/cpu-linux.yml`](.github/workflows/cpu-linux.yml);
a workflow definition is not itself evidence that either job executed.

No real CUDA GPU, CUDA Toolkit, NVHPC compiler, Pegasus scheduler, or production
CPU-library installation was exercised by that local validation. Those checks
remain manual and must not be inferred from syntax-only or fake-provider tests.

## Documentation map

- [`AGENTS.md`](AGENTS.md): permanent repository rules and authority routing.
- [`docs/PROJECT_SPECIFICATION.md`](docs/PROJECT_SPECIFICATION.md): hierarchy,
  canonical sources, teaching problems, and OpenACC data rules.
- [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md): CLI, timing,
  verification, ordering, and statistics.
- [`docs/RESULT_SCHEMA.md`](docs/RESULT_SCHEMA.md): configuration, raw and
  aggregate records, manifests, metadata, and hashes.
- [`docs/PEGASUS_EXECUTION.md`](docs/PEGASUS_EXECUTION.md): Pegasus build/run,
  provenance, telemetry, and collection behavior.
- [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md): phased gates and
  acceptance criteria.
- [`docs/DECISIONS.md`](docs/DECISIONS.md): accepted decisions and rationale.
- [`docs/PORTABILITY.md`](docs/PORTABILITY.md): reuse and dependency boundaries.
- [`docs/ADDING_A_NEW_VENDOR.md`](docs/ADDING_A_NEW_VENDOR.md) and
  [`docs/ADDING_FORTRAN.md`](docs/ADDING_FORTRAN.md): deferred extension guides.

## License

A project license has not been selected. Do not infer permission to redistribute
the repository from its source availability.

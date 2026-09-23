# GPU Library Suite

[English](README.md) | [日本語](README.ja.md)

Portable NVIDIA C/C++ and Fortran teaching examples, reproducible CPU/CUDA/OpenACC
benchmarks, strict result tooling, and Pegasus job support for cuFFT, cuBLAS,
cuSPARSE, cuSOLVER, cuRAND, and Thrust.

The source tree includes `nvidia/c-cpp` and the new
[NVIDIA Fortran implementation](nvidia/fortran/README.md). Fortran GPU execution
is not yet validated; its local coverage is documented separately. AMD and additional
systems are intentionally deferred; the repository does not contain empty
placeholder implementations for them. The project is maintained under
`ac2-prod` and is not an official HAIRDESC repository.

This is the source repository for **Library Edition (NVIDIA GPU, C/C++)**
（ライブラリ編（NVIDIA GPU，C/C++））. You do not need Pegasus access or the
authors' conversation history to read or build the examples. Running a GPU
example does require a suitable NVIDIA GPU and the dependencies below.

## Publication scope

The selected publication scope is the NVIDIA C/C++ teaching and benchmark code,
its existing common code/tools, configurations and tests, and the documentation
for understanding, building, running and measuring it. Additional general-purpose
measurement infrastructure is not a prerequisite for this code publication.
The Fortran addition follows the same code-only boundary; this implementation
work does not itself publish a new version or provide Fortran measurements.

Saved measurements, raw results, execution logs, metadata and prepared local
distribution archives are **not included in this publication**. There is no
teaching-data download supplied with the source. The
[saved-data replay reference](docs/PORTABILITY.md#regenerating-the-teaching-figures-from-saved-data)
is only for readers who separately hold those inputs; the normal route below
uses the reader's own measurements.

## Start here

For Fortran, start with the [Fortran source map, build and measurement guide](nvidia/fortran/README.md)
([日本語](nvidia/fortran/README.ja.md)). The route and source table below describe
the existing C/C++ edition; compiler requirements are not interchangeable.

If you know CPU C/C++ but are new to CUDA, follow this route:

1. Read [the cuBLAS CPU example](nvidia/c-cpp/cublas/examples/blas_cpu.c) to
   recognize allocation, all-ones input, DGEMM, and result checking. Then compare
   [direct CUDA](nvidia/c-cpp/cublas/examples/blas_gpu.cu) with
   [OpenACC-managed data](nvidia/c-cpp/cublas/examples/openacc_cublas.cpp): both
   GPU programs call cuBLAS; they differ in who manages device storage/movement.
2. Follow [cuBLAS: first build and run](nvidia/c-cpp/cublas/README.md#first-build-and-run).
   Start with the CPU example, then run the available GPU variants and check
   their exit status and numerical output. A teaching example is not a timing
   experiment.
3. Read [how the example becomes a benchmark](nvidia/c-cpp/cublas/README.md#from-example-to-benchmark),
   then follow [measuring on your own system](docs/PORTABILITY.md#measuring-on-your-own-system)
   for a small check, configuration, measurement, validation, and aggregation.
4. Create figures from
   [your own measurements](docs/PORTABILITY.md#figures-from-your-own-measurements).
   Your own figures use observed or explicitly supplied machine identities,
   without assuming the teaching machine or requiring its saved data.
5. Use [reading the figures](docs/BENCHMARK_PROTOCOL.md#reading-the-figures-and-applying-the-results)
   to relate compute/E2E costs to your application. Other libraries have their
   own source, dependency, build, and verification notes in the table below.

| Read this area | What it helps you understand |
| --- | --- |
| Each library's `examples/` | One complete library-call flow, without benchmark infrastructure |
| Each library's `benchmarks/` | Runtime sizes, scope-specific resource lifetime, restoration, timing, and verification |
| [`common/c-cpp`](common/c-cpp) | Shared [CLI](common/c-cpp/src/cli.c), [monotonic clock](common/c-cpp/src/clock.c), [result writing](common/c-cpp/src/result.c), and [GPU provenance](common/c-cpp/include/gpu_suite/cuda_metadata.hpp); read these after an example, not first |
| [`configs/pilot.json`](configs/pilot.json), [`configs/benchmark.json`](configs/benchmark.json) | Workloads, CPU backends, verification thresholds, and per-scope repetitions/trials |
| [`tools/run_suite.py`](tools/run_suite.py) | Interleaved execution and the sole node raw-result writer |
| [`tools/validate_results.py`](tools/validate_results.py), [`tools/aggregate.py`](tools/aggregate.py), [`tools/plot.py`](tools/plot.py) | Check raw records, compute hierarchical summaries, and render the two-panel figures |
| [`jobs/pegasus`](jobs/pegasus) | Site-specific modules, PBS, telemetry, and recovery; this is not the general workstation entry point |

## What is implemented

Each C/C++ library has three direct, single-source teaching examples and three
measurement-oriented benchmarks. CMake target names and executable names equal
their source stems.

| Library | CPU | CUDA | OpenACC | CPU benchmark | CUDA benchmark | OpenACC benchmark |
| --- | --- | --- | --- | --- | --- | --- |
| [cuFFT](nvidia/c-cpp/cufft/README.md) | [fft_cpu](nvidia/c-cpp/cufft/examples/fft_cpu.c) | [fft_gpu](nvidia/c-cpp/cufft/examples/fft_gpu.cu) | [openacc_cufft](nvidia/c-cpp/cufft/examples/openacc_cufft.cpp) | [fft_cpu_bench](nvidia/c-cpp/cufft/benchmarks/fft_cpu_bench.c) | [fft_gpu_bench](nvidia/c-cpp/cufft/benchmarks/fft_gpu_bench.cu) | [openacc_cufft_bench](nvidia/c-cpp/cufft/benchmarks/openacc_cufft_bench.cpp) |
| [cuBLAS](nvidia/c-cpp/cublas/README.md) | [blas_cpu](nvidia/c-cpp/cublas/examples/blas_cpu.c) | [blas_gpu](nvidia/c-cpp/cublas/examples/blas_gpu.cu) | [openacc_cublas](nvidia/c-cpp/cublas/examples/openacc_cublas.cpp) | [blas_cpu_bench](nvidia/c-cpp/cublas/benchmarks/blas_cpu_bench.c) | [blas_gpu_bench](nvidia/c-cpp/cublas/benchmarks/blas_gpu_bench.cu) | [openacc_cublas_bench](nvidia/c-cpp/cublas/benchmarks/openacc_cublas_bench.cpp) |
| [cuSPARSE](nvidia/c-cpp/cusparse/README.md) | [sparse_cpu](nvidia/c-cpp/cusparse/examples/sparse_cpu.c) | [sparse_gpu](nvidia/c-cpp/cusparse/examples/sparse_gpu.cu) | [openacc_cusparse](nvidia/c-cpp/cusparse/examples/openacc_cusparse.cpp) | [sparse_cpu_bench](nvidia/c-cpp/cusparse/benchmarks/sparse_cpu_bench.c) | [sparse_gpu_bench](nvidia/c-cpp/cusparse/benchmarks/sparse_gpu_bench.cu) | [openacc_cusparse_bench](nvidia/c-cpp/cusparse/benchmarks/openacc_cusparse_bench.cpp) |
| [cuSOLVER](nvidia/c-cpp/cusolver/README.md) | [solver_cpu](nvidia/c-cpp/cusolver/examples/solver_cpu.c) | [solver_gpu](nvidia/c-cpp/cusolver/examples/solver_gpu.cu) | [openacc_cusolver](nvidia/c-cpp/cusolver/examples/openacc_cusolver.cpp) | [solver_cpu_bench](nvidia/c-cpp/cusolver/benchmarks/solver_cpu_bench.c) | [solver_gpu_bench](nvidia/c-cpp/cusolver/benchmarks/solver_gpu_bench.cu) | [openacc_cusolver_bench](nvidia/c-cpp/cusolver/benchmarks/openacc_cusolver_bench.cpp) |
| [cuRAND](nvidia/c-cpp/curand/README.md) | [rand_cpu](nvidia/c-cpp/curand/examples/rand_cpu.cpp) | [rand_gpu](nvidia/c-cpp/curand/examples/rand_gpu.cu) | [openacc_curand](nvidia/c-cpp/curand/examples/openacc_curand.cpp) | [rand_cpu_bench](nvidia/c-cpp/curand/benchmarks/rand_cpu_bench.cpp) | [rand_gpu_bench](nvidia/c-cpp/curand/benchmarks/rand_gpu_bench.cu) | [openacc_curand_bench](nvidia/c-cpp/curand/benchmarks/openacc_curand_bench.cpp) |
| [Thrust](nvidia/c-cpp/thrust/README.md) | [reduce_cpu](nvidia/c-cpp/thrust/examples/reduce_cpu.cpp) | [reduce_gpu](nvidia/c-cpp/thrust/examples/reduce_gpu.cu) | [openacc_thrust](nvidia/c-cpp/thrust/examples/openacc_thrust.cpp) | [reduce_cpu_bench](nvidia/c-cpp/thrust/benchmarks/reduce_cpu_bench.cpp) | [reduce_gpu_bench](nvidia/c-cpp/thrust/benchmarks/reduce_gpu_bench.cu) | [openacc_thrust_bench](nvidia/c-cpp/thrust/benchmarks/openacc_thrust_bench.cpp) |

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
[`configs/benchmark.json`](configs/benchmark.json) contain the same approved
three-size publication workloads and compute-repeat candidates. Pilot uses one
trial and production uses five; both retain independent `compute` and
`end-to-end` settings with end-to-end repeat 1. After the one-node Pegasus
pilot, a human may update these canonical files once for a demonstrated OOM,
verification, walltime, or short-interval issue; do not create a dated or
version-suffixed replacement.

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

Runtime dimensions, library API integer arguments, element-count products, and
allocation byte counts are checked before allocation or library entry. An
unsupported range is reported as a structured prerequisite/benchmark result;
it is never truncated into a smaller workload. Warm-up follows the selected
scope: compute reuses its persistent context, while end-to-end uses complete
temporary pipelines and leaves no warm-up resources alive. Verification treats
NaN or infinity in inputs, intermediates, or metrics as an explicit
`verification_status=nonfinite` failure, stores the primary metric as JSON
`null`, and never serializes non-standard `NaN`/`Infinity` tokens. Exact rules
are in [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md) and
[`docs/RESULT_SCHEMA.md`](docs/RESULT_SCHEMA.md).

Publication output is six library-specific elapsed-time figures. Each has a
data-resident compute panel and a one-shot host-input-to-host-output panel, uses
milliseconds, and places CPU, CUDA, and OpenACC together. Compute repeat only
amplifies the measured interval; the plotted value remains one operation's
elapsed time. No speedup, throughput, reuse-count, amortized, or extra generic
figure is produced. Plotting reads performance values only from cross-wave
summary records and uses raw results only to require identical CUDA/OpenACC
Thrust `library_version` evidence before writing a Thrust figure. Exact figure
and caption rules are in
[`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md).
Pass the aggregate JSONL as the positional input to `tools/plot.py` and the
campaign's node raw JSONL files through its required `--raw-results` option.

## CPU baselines and verification notes

The canonical Pegasus publication configuration contains only
`cpu-fftw-threaded` for cuFFT. The serial FFTW executable remains available for
separate teaching correspondence but is not a publication series and never
replaces threaded FFTW.

cuRAND's `cpu-std-random-serial` and Thrust's `cpu-stl-serial` are production
serial implementations, displayed with **single thread** in the approved
publication legends even when 48 CPU threads were requested. They are not
called parallel or algorithm-equivalent
baselines, and publication plotting computes no speedup from them. cuRAND
compares the same distribution and output type task across CPU and GPU, but the
RNG algorithms differ; element-by-element identity is not required.

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
must not be inferred from syntax-only or fake-provider tests. Separately saved
Pegasus records do contain real build, example, memcheck, and measurement
results, and Linux CI ran on the approved plotting commit. See
[saved execution evidence](docs/VALIDATION_REPORT.md#saved-execution-evidence-reviewed-for-publication)
for the code versions, coverage, and remaining limits. The initial reader-guide
checks were static; subsequent
[reader-tool tests and private offline replay](docs/VALIDATION_REPORT.md#reader-tools-and-offline-replay-validation)
were completed separately. Neither establishes GPU execution of the full reader
workflow in an arbitrary environment, and the private replay inputs are not
distributed with this code publication.

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

Original content in this repository is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Ryohei Kobayashi

Existing third-party copyright notices and license terms remain applicable.
This license does not relicense external libraries or separately distributed
teaching slides and videos.

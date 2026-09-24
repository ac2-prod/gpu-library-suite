# Validation report

## Scope and evidence boundary

This report records local implementation validation performed on 2026-07-14.
It is evidence for the local phase gates only. It is not evidence of Linux,
CUDA, NVHPC/OpenACC, GPU, or Pegasus execution. The authoritative acceptance
criteria remain in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

The original local-gate record below is retained as historical evidence. Later
saved Linux/Pegasus evidence and the status of the reader documentation are
reported separately under
[Saved execution evidence reviewed for publication](#saved-execution-evidence-reviewed-for-publication).
Do not interpret the original local-only exclusions as a claim that no later
real execution took place, or interpret the later records as tests rerun during
documentation editing.

The second-review remediation gate ran on arm64 macOS with Apple Clang 21.0.0,
CMake/CTest 4.4.0, the system Python 3.9.6, an additional Python 3.13.7, and
ShellCheck 0.11.0. No Linux container runtime was available. The host also did
not provide a real CUDA compiler/Toolkit, NVHPC, a GPU, a Pegasus allocation,
FFTW, or a selected production oneMKL/OpenBLAS/LAPACKE installation.
Controlled fixture providers exercise dependency detection, API-shaped source
paths, failure handling, and CPU algorithms only; they are not substitutes for
the production libraries or vendor compilers.

## Review-finding provenance

This gate is cumulative. Earlier review added explicit regressions for strict
C17 `libm` linking and valid unavailable-Git metadata. The current independent
review identified additional static-analysis gaps that a successful small local
run could not establish: unchecked runtime dimension/byte conversions, scope
warm-up retaining or omitting the wrong resource lifecycle, nonfinite values
passing through numerical reductions, and subprocess records being accepted
without exact comparison against runner-owned configuration and provenance.

The remediation checks every one of the 18 benchmark sources before allocation
or classic integer API entry, exercises all six CPU benchmark families with
oversized runtime inputs, injects NaN and both infinities into shared and
provider-backed verification paths, checks scope-specific source lifecycle
structure, and rejects unexpected or mismatched subprocess-owned fields,
including a one-ULP change to a configuration-owned verification operand.
These are regression tests for the reviewed defects; fake GPU headers remain
syntax evidence only.

## Successful checks in the original local gate

- A clean CPU-only configure and build succeeded with initial project languages
  C and C++, strict C17/C++17, explicit CUDA/OpenACC disable reasons, and the
  system Python 3.9 interpreter selected for tests.
- CTest passed 73 of 73 registered tests. This covers common C/C++ unit tests,
  compile/link probes, positive and negative FFTW/CBLAS/LAPACKE/oneMKL fixture
  providers, CPU benchmark fixtures, all 36 source/target policy checks,
  CUDA/OpenACC fixture-header syntax checks, CUDA runtime-metadata helper tests,
  Pegasus shell checks, Python tests, the Python 3.9 audit, and the `.git`-less
  source-build regression.
- The CTest Python suite ran under the actual system Python 3.9.6 and passed 140
  of 140 tests. The count includes exact runner field/verification contracts,
  all-family overflow subprocess checks, nonfinite provider injection, and the
  18-source arithmetic/scope policy audit.
- The `.git`-less regression copied the source without `.git`, configured and
  built it, ran all 72 applicable inner CTests successfully, and checked that
  unavailable Git provenance remains valid JSON with explicit availability and
  null-value semantics.
- All 54 Python source files passed `ast.parse(feature_version=(3, 9))`, the
  post-3.9 standard-library API audit, normal compilation, and `py_compile`.
  A separate system-Python-3.9 `compileall` pass also succeeded.
- `bash -n` passed for the three shell scripts and the PBS template. ShellCheck
  passed for all three directly analyzable shell scripts.
- The dependency-free cuRAND and Thrust CPU teaching examples compiled with
  `-std=c++17 -Wall -Wextra -Wpedantic` and ran successfully.
- Synthetic Pegasus tests cover deterministic rendering, build/runtime module
  separation, runtime-environment hashing, node-local CUDA driver/runtime
  identity collection, additional-wave mismatch, complete and duplicate rank
  mappings, timezone/UTC offset, local-midnight rollover, trial/telemetry
  correlation, node signal recovery, missing status, benchmark/verification/
  collection failure, and preservation of other-node artifacts after one node
  fails.
- Source-snapshot and manifest tests cover canonical path ordering, content and
  executable-mode changes, duplicate/traversal/non-file rejection, artifact
  identity, build-profile/backend conflicts, binary hashes, and versioned empty
  snapshot identity.
- [`.github/workflows/cpu-linux.yml`](../.github/workflows/cpu-linux.yml)
  defines mandatory clean Linux GCC and Clang CPU-only jobs and uses
  `actions/checkout@v7`. The workflow file is reviewable configuration, not
  execution evidence; neither job ran in this local task.

The clean second-review commands run from the repository root were:

```bash
PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  /usr/local/bin/cmake -S . \
  -B /tmp/gpu-library-suite-local-build/second-review-final \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_SUITE_BUILD_CPU=ON \
  -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=OFF \
  -DGPU_SUITE_BUILD_TESTS=ON \
  -DGPU_SUITE_BUILD_EXAMPLES=OFF \
  -DGPU_SUITE_BUILD_BENCHMARKS=ON \
  -DPython3_EXECUTABLE=/usr/bin/python3
/usr/local/bin/cmake --build \
  /tmp/gpu-library-suite-local-build/second-review-final --parallel 8
/usr/local/bin/ctest --test-dir \
  /tmp/gpu-library-suite-local-build/second-review-final \
  --output-on-failure --parallel 8
PYTHONPYCACHEPREFIX=/tmp/gpu-library-suite-local-build/second-review-final/standalone-pycache \
  /usr/bin/python3 tools/check_python39.py .
PYTHONPYCACHEPREFIX=/tmp/gpu-library-suite-local-build/second-review-final/compileall-pycache \
  /usr/bin/python3 -m compileall -q common tools jobs tests
bash -n jobs/pegasus/build_cpu_cuda.sh jobs/pegasus/build_openacc.sh \
  jobs/pegasus/run_node.sh jobs/pegasus/run_benchmarks.pbs.in
/usr/local/bin/shellcheck jobs/pegasus/build_cpu_cuda.sh \
  jobs/pegasus/build_openacc.sh jobs/pegasus/run_node.sh
/usr/bin/c++ -std=c++17 -Wall -Wextra -Wpedantic \
  nvidia/c-cpp/curand/examples/rand_cpu.cpp \
  -o /tmp/gpu-library-suite-local-build/second-review-final/manual/rand_cpu
/tmp/gpu-library-suite-local-build/second-review-final/manual/rand_cpu
/usr/bin/c++ -std=c++17 -Wall -Wextra -Wpedantic \
  nvidia/c-cpp/thrust/examples/reduce_cpu.cpp \
  -o /tmp/gpu-library-suite-local-build/second-review-final/manual/reduce_cpu
/tmp/gpu-library-suite-local-build/second-review-final/manual/reduce_cpu
git diff --check
```

The CUDA/OpenACC syntax tests above use controlled headers and a host compiler.
They verify project source integration and diagnostics but are not vendor
compile, link, runtime, numerical, or performance tests.

## Explicitly unexecuted checks in the original local gate

| Check | Status and reason |
| --- | --- |
| Clean Linux GCC CPU configure/build/test | Unexecuted locally: the host is macOS and no Linux container runtime was available. The mandatory CI job is defined but was not run here. |
| Clean Linux Clang CPU configure/build/test | Unexecuted locally for the same reason; Apple Clang is not Linux Clang evidence. |
| Real direct CUDA compile, link, and execution | Unexecuted: no local CUDA compiler, Toolkit, or GPU was available. |
| Real NVHPC/OpenACC compile, link, and execution | Unexecuted: no local NVHPC compiler/runtime was available. |
| Real FFTW threaded CPU execution | Unexecuted: no local FFTW installation was available; controlled compile/link/runtime fixtures passed. |
| Real oneMKL/OpenBLAS/LAPACKE execution | Unexecuted: no production provider was selected locally; controlled positive, negative, and provider-switch fixtures passed. |
| Configure/build under CMake 3.20 | Unexecuted: local CMake was 4.4.0. The source-policy tests reject known post-3.20 constructs, but a real 3.20 configure remains required. |
| GPU numerical verification, sanitizer, and performance measurement | Unexecuted: fixture-header syntax checks and CPU algorithms are not GPU tests. |
| Pegasus build, rendered-job submission, runtime collection, telemetry, and node recovery | Unexecuted: remote access and scheduler operations are outside local-agent authority. |
| Five/eight-node production campaign and cross-wave performance report | Unexecuted until human Pegasus smoke testing and calibration succeed. |

## Required human Pegasus work

Follow [`PEGASUS_MANUAL_VALIDATION.md`](PEGASUS_MANUAL_VALIDATION.md). In
particular, validate both build profiles and Toolkit identity, run normal and
failure smoke jobs, inspect raw CUDA library/driver/runtime identity against
node-local metadata, validate runtime/binary provenance and telemetry
correlation, then calibrate `configs/benchmark.json` before any production
campaign. No Pegasus run report exists because no Pegasus job was executed in
this local task.

## Saved execution evidence reviewed for publication

This section reports inspection of existing logs/data, **not new execution**.
The measurement implementation was
`916a8bda22f6188aa995d21954ddb1d6de2f3cf3`. The approved figure-layout commit is
`4d3c2333d2a58aa3fc2a8a82d43c403d2131343f`; the difference is limited to
`tools/gpu_suite/plotting.py` and `tests/python/test_plotting.py`. Thus the latter
does not identify a new GPU measurement or a change in the measured workloads.

| Saved record | Evidence observed | Scope/limit |
| --- | --- | --- |
| [CPU Linux portability, 2026-07-30](https://github.com/ac2-prod/gpu-library-suite/actions/runs/30516585324) | GCC and Clang jobs succeeded on the approved plotting commit; each reports 77 CTests passed and 62 Python 3.9-compatible files | Existing CI, not a rerun after the documentation edits; access follows repository visibility |
| Pegasus build-validation run `866605`, starting `2026-07-16T01:35:27Z` | CPU/CUDA and OpenACC configure/build succeeded; their CTest logs report 74/74 and 22/22; merged manifest has 36 entries | Real installed providers/compiler paths, distinct from fake-provider tests; build metadata records clean commit `916a8bd…` |
| The same run's teaching-example records | All 18 example exit-code files contain 0 | CPU, direct CUDA and OpenACC examples for all six libraries; this does not imply every possible input was tested |
| The same run's compute-sanitizer records | Six direct-CUDA teaching examples exit 0 and each reports `ERROR SUMMARY: 0 errors` | Memcheck of those six executables; not OpenACC memcheck or every benchmark size/tool mode |
| Campaign `publication-benchmark-20260716T025037Z` | 12 node blocks, six per wave, two waves, 6,480 rows; every inspected raw status is success and verification status is pass; all 12 node statuses are success | Saved production measurement, not a new run or proof of an arbitrary reader environment |

The saved build/runtime documents identify GNU 11.4.0, NVHPC 25.11 and external
CUDA Toolkit component version 13.0.88 (module release `cuda/13.0.2`). The
CPU/CUDA and OpenACC metadata select the same Toolkit root; NVHPC's
`cuda_home` agrees. GPU rows identify NVIDIA H100 PCIe and CUDA Driver
API/Runtime version 13.0.0. The runtime record reports oneMKL 2025.3, loaded
through the site `onemkl/2025.3.1` module. Module release, Toolkit component,
CUDA Runtime, and NVIDIA package-driver versions are distinct identities.

The saved NVHPC C/C++ Release flags are `-fast -O3 -DNDEBUG`. The later
[flag audit](#nvhpc-release-flag-audit) checks the individual targets and official
25.11 documentation. Passing verification is not policy approval. No saved
condition or result is changed; a different approved build needs new provenance.

The campaign uses `cpu-fftw-threaded` with requested/effective 48;
`cpu-onemkl` with request 48 and effective null; and serial cuRAND/Thrust with
effective 1. Runtime settings include `MKL_NUM_THREADS=48` and
`MKL_THREADING_LAYER=INTEL`. These do not prove observed 48-thread utilization
inside every oneMKL call.

The following saved file hashes were recomputed during read-only inspection and
match `run-metadata.json`:

| File | SHA-256 |
| --- | --- |
| `effective-config.json` | `8afd122b965f65400c94aaa3a3baa511e6d006908d6747bd6d21bdcebff461e9` |
| `executables-manifest.json` | `1d9a3a7d0fe1d88624b9cff73ee70b197c3a09c337df5802ef5a0dc9207b3fee` |
| `runtime-environment.json` | `82c65929273fc1d74b0509d2c7994e0b4fa8e1dfbf0ea871450f88473a68c057` |

These artifacts remain under ignored `manual-validation/pegasus/` storage and
are excluded from the selected code publication. The
[conditional saved-data reference](PORTABILITY.md#regenerating-the-teaching-figures-from-saved-data)
does not supply downloadable inputs or make their distribution a code-publication
requirement. This inspection does not certify a
two-node injected-failure smoke, every telemetry anomaly check, other GPU/CPU
systems, or complete PDF/PPTX correspondence; the material files were not
available for that comparison.

## Reader workflow validation status

This is the completed **earlier documentation-only stage**, not the status of
the subsequent tool work below. The README/library/portability/protocol additions were checked against existing
sources, target names, CLI parsers and output contracts, with static link/path/
argument checks and `git diff --check`. They were not followed by a new build,
test-suite run, GPU run, measurement, aggregation or PNG generation. Saved
execution evidence above must not be relabeled as execution of the new guide.

At that stage, documentation completion was separate from reader reproduction:

- examples and measurement commands describe existing capabilities, with
  explicit dependency and collector preconditions;
- module-free runtime collection, non-PBS metadata preparation and configurable
  machine/thread/provider labels and cuSOLVER ticks still needed tool changes;
- the recorded NVHPC Release flags still need the numerical-policy review
  described above; successful verification is not that review;
- source checkout alone cannot replay the teaching figures without the
  unpublished input bundle; and
- the proposed clean-environment walkthrough had not verified newly generated
  records, validation/aggregation outputs and six correctly annotated figures.

No license, rights-holder, support-policy or repository-publication decision was
made by that earlier stage. The current publication scope is recorded in the
[README](../README.md#publication-scope).

## NVHPC Release flag audit

This audit reads saved build-validation run `866605`, not a new build or GPU
measurement. Both caches identify CMake 3.22.1. The OpenACC build metadata
identifies NVHPC C/C++ 25.11.0; its C/C++ Release cache entries are
`-fast -O3 -DNDEBUG`. The CPU/CUDA tree instead uses GNU 11.4.0 / nvcc 13.0.88
with `-O3 -DNDEBUG`; the NVHPC flag must not be attributed to those targets.

Every relevant saved `flags.make`, `build.make` and `link.txt` was checked under
`build/openacc/nvidia/c-cpp/*/CMakeFiles/` (relative to that saved run):

| Targets | Compiler and generated compile flags | Generated executable link flags |
| --- | --- | --- |
| `openacc_cufft`, `openacc_cublas`, `openacc_cusparse`, `openacc_cusolver`, `openacc_curand`, and each corresponding `_bench` | `nvc++ -fast -O3 -DNDEBUG -gpu=cc90 -acc --c++17` | `-fast -O3 -DNDEBUG -gpu=cc90 -acc` |
| `openacc_thrust` and `openacc_thrust_bench` | `nvc++ -fast -O3 -DNDEBUG -cuda -gpu=cc90 -acc --c++17` | `-fast -O3 -DNDEBUG -cuda -gpu=cc90 -acc` |
| `gpu_suite_common` objects in the OpenACC tree | `nvc -fast -O3 -DNDEBUG -std=c17` | Static archive; linked into the six benchmark executables |

The generated compile recipes invoke the absolute NVHPC compiler with the
target definitions/includes, those flags, `-o <object> -c <source>`. Each saved
`link.txt` invokes `nvc++` with the flags above, the target object and external
CUDA 13.0.2 library paths; benchmarks also link the common archive and `-lm`.
The build log reports all 12 targets built. These are generated commands plus
completion evidence, **not** a verbose transcript of the compiler driver's
expanded internal commands. The saved files show no later `-Mnoflushz`,
`-Mnodaz`, `-Kieee` or GPU fast-math override. Flags on unrelated fixture/test
targets are not evidence of the production targets' settings.

The site build script selects Release but does not supply C/C++ Release flags.
The cache agrees with CMake 3.22.1's inherited compiler configuration:
[NVHPC.cmake](https://raw.githubusercontent.com/Kitware/CMake/v3.22.1/Modules/Compiler/NVHPC.cmake)
includes the PGI compiler module, whose
[Release initialization](https://raw.githubusercontent.com/Kitware/CMake/v3.22.1/Modules/Compiler/PGI.cmake)
appends `-fast -O3`. This supports the inherited-default origin; the original
site's installed CMake module contents and initial compiler environment were
not archived exhaustively. `-DNDEBUG` is independently present in the cache
and each generated command, regardless of that origin limit.

The [NVHPC 25.11 Reference Guide, -fast](https://docs.nvidia.com/hpc-sdk/archive/25.11/compilers/hpc-compilers-ref-guide/index.html#fast)
describes an aggregate enabling host SIMD/vectorization, cache alignment and
flush-to-zero, host-target selection, and C/C++ auto-inlining. It is not the
same switch as `-gpu=fastmath`; host flags do not recompile the prebuilt CUDA
libraries or by themselves prove TF32/Tensor Core use. Thrust device template
code, unlike the prebuilt libraries, is compiled in the application target.

The [25.11 User Guide, optimization and subnormal handling](https://docs.nvidia.com/hpc-sdk/archive/25.11/compilers/hpc-compilers-user-guide/index.html)
states that a trailing `-O3` replaces `-fast`'s optimization level, not its other
components. It also documents post-22.7 x86 defaults that treat subnormals as
zero and the runtime `NVCOMPILER_FPU_STATE` override. Thus merely removing
`-fast` would not establish strict subnormal handling. That variable was not
captured in the saved selected-environment records; absence from those records
does not establish that it was unset. Host-dependent expansion, site/user rc
files and the actual runtime FPU state remain unverified. Local `nvc`/`nvc++`
are unavailable, so compiler help/dryrun was not executed.

The governing text in [AGENTS.md](../AGENTS.md#portability-and-build-policy) is:
“Do not enable fast math, TF32, Tensor Core modes, or architecture-specific
numerical changes without an explicit approved option.” The inherited aggregate
has documented numerical implications, while an explicit approved numerical
profile is not established by the saved build evidence. This is an unresolved
policy-conformance issue, not grounds to erase or automatically invalidate the
measurements. This audit does not approve the inherited numerical profile.
A changed profile affects the 12 OpenACC programs and
their common host code; it needs separate builds, manifests and run provenance,
then verification before any new comparison. No effect on elapsed time has
been measured or inferred here. The selected code publication retains this
disclosure and does not include a new numerical profile or remeasurement.

Release does not compile out the suite's verification by `NDEBUG`: the six
OpenACC benchmark paths call explicit verification routines after successful
operations, controlled by `options.verify`/`--verify`, not `assert`. The cuFFT
routine is in `fft_bench_result.hpp`; the other families use their direct or
shared benchmark verification helpers. The saved effective config enables
verification, and saved rows record its metrics/status. This establishes that
verification was requested and reported, not that every floating-point behavior
or compiler transformation is policy-compliant. Neither flags, calculations,
verification criteria nor source numerical policy were changed in this audit.

## Reader tools and offline replay validation

The bounded tool changes add provenance-aware labels, input-derived cuSOLVER
ticks, an explicit module-free collection mode, and a one-host local metadata
preparation mode. Scheduler fields remain null for local execution; an inherited
`GPU_SUITE_SCHEDULER*` value cannot replace the runner's explicit launch context.
Missing required dependencies/GPU probes still fail. The existing PBS path
retains its separate scheduler/rank mapping requirements.

The local reader-tool gate used a dedicated copied Python 3.14.7 virtual
environment with matplotlib 3.11.2, NumPy 2.5.3 and Pillow 12.3.0; no global
Python environment was modified. The independent Python suite selected 177
tests: **173 passed, four skipped, zero failures/errors**. Skips were the
unconfigured CMake-generated metadata test, unconfigured non-Git CMake fixture,
the test that creates a symlink (prohibited by this task's workspace rules),
and the Linux-only real-`ldd` stability check. CPU-provider executable fixtures
were outside that independent profile and were not rebuilt. The Python 3.9
syntax/API audit passed all 64 Python files. `bash -n` passed the two build
scripts, node worker and PBS template.

New and existing tests cover approved teaching labels/layout, alternate
CPU/GPU/backend identities, requested/effective/unknown counts, conflicting or
partial identity evidence, alternate solver ticks, explicit module-free
collection with required dependency checks, local metadata and collisions,
scheduler-environment isolation, and the existing PBS preparation/rendering,
node failure collection, telemetry, schema, validation and aggregation paths.
Six additional synthetic display renders used a 160-character CPU identity,
an alternate backend and requested 8 / reported effective 4; all legends fit
the canvas. Those synthetic values are not GPU or CPU performance evidence.

The previously prepared local data candidate contains 43 byte-identical saved inputs plus an
explicit display config, README and checksum inventory (46 files total).
A companion source archive contains the accepted documentation edits and the
reader-tool changes, without `.git` or ignored private artifacts. The README
records measurement commit, renderer base commit, full working-tree source
snapshot hash, companion archive hash and the consumer commands. Exact archive
checksums belong to the supplied archive inventory; no public download URL,
Release or public repository is created by this work.
These archives remain unchanged private validation artifacts, not publication
deliverables; they do not incorporate the later publication-scope documentation
edits.

The candidate was extracted with its paired code to a new directory outside
the original workspace. The documented checksum, validation, aggregation and
plot commands succeeded with Python file-read audit guards blocking the
original workspace and `/work`, `/scr`, `/pmem`; a deliberately attempted
original-file read failed as the guard's negative control. The replay itself
made no blocked read. Original manifest executable paths were not dereferenced.
Validation passed all 6,480 rows, 12 blocks and two waves. Reaggregation with
`--no-pooled` produced the saved 2,700 rows byte-for-byte; the default's 180
additional exploratory pooled rows are a separate output option, not a change
to the plotted hierarchy or numerical values.

All six PNGs were generated at 1800 x 720. Comparison against the approved
layout's saved metadata found exact equality for 36 series / 108 points,
quartiles and labels; renderer-call comparison against the approved base found
the same axis titles/units/scales/ticks, legend and two-panel layout parameters.
All six images were also inspected. Raster SHA-256 values differ from the
approved matplotlib 3.6.2 images (about 5.6–6.9% of pixels differ with 3.11.2);
this is distinct from the exact data/display-condition comparison, not a claim
of byte-identical PNG reproduction. Original raw/aggregate/metadata/PNG files
were neither edited nor overwritten.

The local tests and offline replay described in this section use no GPU,
scheduler or remote host. Synthetic fixtures are not measurements. GPU execution
of the full local reader path and non-Pegasus telemetry/multi-node orchestration
remain [known limitations](PORTABILITY.md#known-limitations-and-out-of-scope-work),
not newly required work for this code publication. Local tool completion does
not approve a numerical profile, distribute the saved data or authorize repository
publication. No additional GPU diagnostic, general-purpose measurement framework
development or data release is required by the selected code-publication scope.

<a id="fortran-local-validation"></a>

## Fortran source addition: local validation

The following subsections retain the historical local handoff, smoke and
presurvey states. Current production completion is recorded under
[Fortran production validation](#fortran-production-validation); later evidence
does not change what was or was not executed at an earlier stage.

This 2026-09-23 implementation is based on
`94a8adbc310e158c048a08976a17471331e89d0a`, with uncommitted additions on
`work/nvidia-fortran`. It adds 18 teaching examples and 18 benchmark entry
points for six libraries, mathematical helpers, the shared Thrust C++ function,
optional Fortran CMake targets and language-aware use of the existing tools.
The [Fortran guide](../nvidia/fortran/README.md) records the extraction pages,
necessary distribution changes and exact user commands. The reference
extraction remains unchanged in ignored local inputs; the PPTX was not supplied,
edited or independently checked. C/C++ computation, existing CPU/GPU build
settings, stored measurements, teaching figures and LICENSE are unchanged.

The local host has GNU Fortran 16.2.0, Apple Clang 21, CMake 4.4 and Python 3.14.
It has no NVHPC/CUDA GPU environment or real FFTW3f/oneMKL providers available
for this check. All new local build/test outputs are under
`/tmp/gpu-library-suite-local-build`; no provider was installed.

### Executed CPU and tool checks

The CPU-only Fortran build used the following commands, with `gfortran`
resolving to the installed GNU compiler:

```bash
cmake -S . -B /tmp/gpu-library-suite-local-build/fortran-cpu \
  -DGPU_SUITE_SOURCE_LANGUAGE=fortran -DCMAKE_Fortran_COMPILER=gfortran \
  -DCMAKE_BUILD_TYPE=Release -DGPU_SUITE_BUILD_TESTS=ON \
  -DGPU_SUITE_BUILD_CUDA=OFF -DGPU_SUITE_BUILD_OPENACC=OFF
cmake --build /tmp/gpu-library-suite-local-build/fortran-cpu \
  --target gpu_suite_partial_manifest test_fortran_helpers --parallel 2
ctest --test-dir /tmp/gpu-library-suite-local-build/fortran-cpu --output-on-failure
```

Configure and build passed. The intrinsic CPU random/reduction examples and
benchmarks were enabled; FFTW/oneMKL-dependent targets were disabled with
provider/interface reasons. No GPU targets were requested. The generated GNU
Fortran compile flags include `-O3 -fno-fast-math -ffp-contract=off`.
Both teaching programs were executed at their unchanged 2^24-element defaults
and reported `verification PASS` with exit 0. Helper tests passed the dense
known-solution construction, one-based Poisson CSR and boundary cases.

The Fortran CTest profile passed **3/3** gates: helpers, Fortran tools and
controlled-provider checks. Its default-Python tool run skipped only the
matplotlib render test; that test was subsequently executed successfully in an
already existing local plotting environment, without installation. The complete
Fortran Python suite in that environment passed **11/11**, with no skips:

```bash
env PYTHONPATH=tools:tests/python \
  PYTHONPYCACHEPREFIX=/tmp/gpu-library-suite-local-build/fortran-cpu/pycache-reader \
  MPLCONFIGDIR=/tmp/gpu-library-suite-local-build/fortran-cpu/matplotlib \
  GPU_SUITE_FORTRAN_BUILD=/tmp/gpu-library-suite-local-build/fortran-cpu \
  /tmp/gpu-library-suite-local-build/reader-tools-env.iguhdV/bin/python \
  -m unittest test_fortran -v
```

The tests exercise intrinsic CPU benchmarks in compute/E2E, JSONL/CSV output,
warm-up/trial restoration, parameter/backend failures, language-specific
configuration/manifest/row checks, binary/build hashes, runner failure/skipped
classification and aggregation. Four external CPU call paths were separately
compiled and exercised through explicitly synthetic test providers. Their
matrix/FFT operations test normal results, NaN/+Inf/-Inf propagation, non-square
DGEMM, one-dimensional Poisson boundaries, solver getrf/getrs info separation,
FFTW serial/threaded selection and state restoration. Missing-symbol probes,
unsafe numerical flags and reuse of a different language's build tree are
negative controls. These fixtures are **not real FFTW/oneMKL ABI or performance
validation**, and cannot silently replace production backends.

Six synthetic two-panel figures were generated with matplotlib 3.11.2 and all
six inspected. The CPU/CUDA/OpenACC layout, Fortran labels, intrinsic backend
names and requested/effective/unknown thread labels were preserved. Deliberately
equal synthetic series overlap; no speed ranking is inferred. These figures do
not use saved C/C++ measurements and are not Fortran GPU measurement evidence.

### Existing C/C++ regression

A separate CPU-only default-language tree was configured with
`CMAKE_Fortran_COMPILER=/nonexistent/fortran-compiler`; configure and build passed
without enabling Fortran. Its 77 CTest gates initially had 75 passes and two
failures: a documentation assertion that still forbade all Fortran directories,
and a runner validation-order regression exposed by an existing duplicate-entry
test. Both were corrected; their targeted rerun (`python_unittest` and
`non_git_source_build`) passed 2/2. Thus all 77 gates have passing final results,
not one original 77/77 run.

The independent existing Python run selected 188 tests, with 182 passes and six
skips: five Fortran runtime tests require the separate configured Fortran tree
and were executed there, while one real-`ldd` check requires Linux. Existing
CPU-provider executable fixtures passed 12/12. The non-Git source-copy build's
76 internal CTest gates also passed. These checks include the existing strict
JSON/schema, runner, metadata, aggregation, plot validation and simulated
PBS/telemetry/failure-collection paths; simulated checks do not constitute site
execution. Documentation links, Python 3.9 compatibility, shell checks and
`git diff --check` were included in local validation.

### Not executed; first real-machine gate

This subsection records the local handoff state. The subsequent human-run
Pegasus evidence and the CPU-engine metadata correction are recorded below;
they do not turn the earlier static checks into GPU execution evidence.

The NVIDIA 25.11 [Fortran CUDA interfaces](https://docs.nvidia.com/hpc-sdk/archive/25.11/pdf/hpc2511cudaint.pdf),
[CUDA Fortran guide](https://docs.nvidia.com/hpc-sdk/archive/25.11/compilers/cuda-fortran-prog-guide/index.html)
and [compiler reference](https://docs.nvidia.com/hpc-sdk/archive/25.11/compilers/hpc-compilers-ref-guide/index.html)
were checked for library interfaces, typed device-array transfer counts,
separate-memory build options and numerical flags. This was static documentary
checking, not NVHPC compile/link or execution. The new Fortran flags do not
resolve the historical C/C++ FPU-state/numerical-policy limitation above.

Still unexecuted are real FFTW/oneMKL Fortran linkage, NVHPC CPU/CUDA/OpenACC
compilation, mixed NVCC/Fortran Thrust linkage, all GPU examples/benchmarks,
compute-sanitizer, rank-local GPU/library identity, MPI propagation, telemetry
and Pegasus collection for Fortran. The Linux real-`ldd` check also remains
unexecuted on this Mac. No Fortran performance result or production profile has
been approved.

The [human-only first validation procedure](PEGASUS_EXECUTION.md#fortran-first-validation)
uses existing confirmed site configuration and a legitimate GPU allocation:
new CPU/CUDA and OpenACC trees, 36-entry merged manifest, all 18 teaching
programs, small cuBLAS CPU/CUDA/OpenACC compute/E2E checks, then a one-node
six-row PBS smoke with verification and identity/hash/collection checks.
Only after those checks does the separate small 216-row Fortran pilot provide
input for the existing six figures. This implementation performs no Pegasus
connection, scheduler operation, GPU measurement, commit, push or publication.

### Fortran Pegasus evidence and CPU-engine metadata correction

The human-run evidence is retained, unchanged and Git-ignored, under
`manual-validation/pegasus/fortran.5AhatU/`. Final CPU/CUDA and OpenACC builds
exited 0; the explicit FFTW prefix resolved the missing CPU FFT targets. The
final merged manifest contains 36 entries (18 teaching and 18 benchmark).
All 18 teaching programs passed verification. The separate cuBLAS six-row
diagnostic passed, and its CUDA/OpenACC memcheck runs both exited 0 with
`ERROR SUMMARY: 0 errors`. These eight diagnostic rows are not smoke rows.

The one-node job `21154.nqsv` produced 216 Fortran smoke rows: six libraries,
three cases, two scopes, three implementations and two trials. Every row has
success status, verification pass and exit code 0. The node runner recorded
exit 0, and the collector recorded success with no failures. All 21 collected
artifact sizes and hashes match their saved manifest. The saved config,
manifest, build/runtime metadata, binary-hash references and source-snapshot
references agree. Binary files were not rehashed from the Mac. GPU identity and
CUDA Driver API/runtime records agree with the independent node probe; all GPU
library versions are recorded in raw rows and agree between CUDA and OpenACC,
but node metadata does not independently record every library version.

The existing classification was retained. Offline `validate_results.py` passed
with 216 rows and `aggregate.py` exited 0, saving its outputs in that session's
`analysis/` directory. The 720 aggregate records contain no failed/skipped
inputs or exclusions. Of these, 612 report `insufficient_sample_count` because
their input consists of one block, wave or paired ratio; this is not a
calculation failure or evidence of production-level statistical coverage.

The original node metadata incorrectly says
`curand.cpu_engine="std::mt19937_64"`. Its 12 CPU cuRAND raw rows correctly say
`Fortran random_number`, matching the actual Fortran intrinsic call. The cause
was a fixed C/C++ engine string in `jobs/pegasus/node_tools.py`, not replacement
of the Fortran computation by a C++ RNG. The earlier classification and
validation did not compare this field. Their recorded PASS results and all
original raw/metadata/artifact/analysis files remain intact, with this
provenance limitation explicitly attached to their interpretation.

The local correction derives the CPU engine from the validated configuration's
source language and CPU backend, preserving the C/C++ default and rejecting
inconsistent language/backend combinations. It does not infer the internal
algorithm of `random_number` or change the GPU generator. Classification with
node metadata now rejects CPU cuRAND engine mismatches and missing engine
identity on successful CPU rows. Failure/skipped rows without a measured
engine retain their existing classification.

Local regression command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools:tests/python \
  TMPDIR=/tmp/gpu-library-suite-local-build python3 -B -m unittest \
  test_pegasus_node_collection test_local_metadata test_pegasus_prepare_render \
  test_pegasus_runtime_telemetry test_validation test_config -v
```

Result: 63 tests, 62 passed, one skipped because real Linux `ldd` is unavailable
on the Mac. Seven added test methods cover C/C++ and Fortran generation and
matching, mismatch/missing detection, separation from GPU generator identity,
and preservation of failed/skipped records. An additional local check used
saved observations as injected inputs, with no hardware probe: the new check
rejects the unchanged original metadata; metadata generated into a temporary
directory differs only in the `curand` section and classifies all 216 saved raw
rows as success/verification pass. This is local producer/checker validation,
not corrected metadata generated by a new Pegasus job. No teaching, memcheck,
smoke, GPU build or measurement rerun was performed or required for this fix.
The documentation checks passed 9/9. A before/after SHA-256 inventory confirmed
that all 213 files in the saved session, including raw data, metadata, logs and
the existing validation/aggregation outputs, were unchanged.

Remaining evidence limits are unchanged: scheduler final exit status has not
been supplied as `scacctreq` evidence and is not inferred from runner or submit
exit codes; all 216 short trial intervals have zero in-interval telemetry
samples despite correct timestamp correlation with 45 captured GPU samples;
and `turbostat` reported that it was unavailable for the running kernel, so
detailed CPU frequency/power observation is missing. Observed GPU samples show
no thermal violation, not proof about unsampled trial interiors. The corrected
producer's actual-job execution remains unverified. Production conditions are
still a proposal, not an executed Fortran measurement campaign.

### Fortran full-size presurvey evidence

The later human-run job `21639.nqsv`, saved unchanged under the Git-ignored
`manual-validation/pegasus/fortran-presurvey.tPQsvR/`, completed all 108 rows
(six libraries, three sizes, CPU/CUDA/OpenACC, two scopes, one trial).
Every row records success, verification pass and exit code 0. The runner
recorded exit 0, node status and collector recorded success, and all 21
collected artifact sizes and hashes match. Config, manifest, runtime and
run/wave/node/raw identities and all 18 benchmark binary-hash references agree.
The six CPU cuRAND rows now match the actual-job node metadata's
`Fortran random_number`; this supersedes the corrected-producer execution
limit above without changing the earlier smoke originals. Requested CPU
threads are 48; effective threads remain 48 for FFTW, 1 for the Fortran
intrinsic baselines and null for unobserved oneMKL values.

The existing classification and collection were reused. Offline validation
and aggregation both exited 0 into that run's new `analysis/` directory.
All 720 aggregate records report `insufficient_sample_count`, with no failed
or skipped inputs: one trial/node/wave does not establish variability.
The sum of timed intervals is 30.285 seconds; scheduler timestamps span
138 seconds and its separately reported Elapse is 142 seconds. The final
scheduler exit code is not saved and is not inferred from submission/runner
exit codes. There are 129 GPU samples, but 94 of 108 trial intervals contain
no sample. Observed `tviol` is zero; two nonzero `pviol` samples lie outside
the recorded trial intervals. Turbostat is unavailable. Independent node
probes do not cover every library version. Original files were hash-checked
before and after analysis and preserved. These presurvey rows remain separate
from production data; five-trial, six-node, two-wave production has not run.

<a id="fortran-production-validation"></a>

### Fortran production validation and figures

The completed Pegasus production campaign measured commit
`9567d6e9fdd5e2bb9190fb523ca073ca2a68ee1c`. Its retained, non-distributed
`postprocessing-summary.json` records processing status `pass`, unchanged
source-snapshot identity and 305 unchanged original files. This summary is
under the Git-ignored production RUN_DIR, not an input supplied by a checkout.
The earlier local, teaching-example, memcheck, smoke and presurvey evidence
above remains separate from this campaign and from CPU CI.

- Each of two waves contains six node blocks and 3240 raw rows: **6480 rows**
  in total, all `success` and numerical verification `pass`. Each condition
  has five trials; the campaign contains 12 node/wave blocks on 11 distinct
  hosts. No smoke, presurvey or C/C++ rows were mixed in.
- Saved node status, collection, classification and identity/hash checks passed;
  252 collected artifacts were accounted for. Wave 0 validation was reused,
  wave 1 and combined validation passed, and aggregation and plotting exited 0.
- Aggregation produced 2880 records and 108 cross-wave elapsed-time points,
  without exclusions. The 864 `insufficient_sample_count` records are
  block-level paired-speedup summaries with one ratio of block medians, not
  failed timing trials or missing production measurements.
- Six library figures were generated and visually checked. Their left panel
  is compute, right panel E2E, and vertical axis elapsed time in milliseconds;
  values use `elapsed_sec * 1000`, without dividing by repeat again.

The requested CPU thread count was 48. FFTW recorded effective 48, the Fortran
intrinsic random/sum baselines effective 1, and oneMKL effective counts remain
null/unobserved. Intrinsic CPU RNG and cuRAND are not equivalent algorithms.
Independent node probes do not establish every library version; retained raw
rows record versions and CUDA/OpenACC versions agree within each library.

Both scheduler final exit codes remain unconfirmed; submission/runner status
is not a substitute. Of 6480 intervals, 5492 contain no telemetry sample.
There were 79 nonzero GPU `pviol` samples, overlapping 16 trial intervals;
no rows were removed. No nonzero `tviol` sample was observed, which does not
prove the state inside unsampled intervals. Turbostat was unavailable in all
12 blocks, and two waves do not establish broad cross-wave variability.

These are saved real-machine results, not a new GPU run performed for
publication. Raw data, metadata, logs, analysis outputs and figures are not
included in this source publication; users follow the existing own-measurement workflow
to generate their own inputs and figures.

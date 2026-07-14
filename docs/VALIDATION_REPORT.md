# Validation report

## Scope and evidence boundary

This report records local implementation validation performed on 2026-07-14.
It is evidence for the local phase gates only. It is not evidence of Linux,
CUDA, NVHPC/OpenACC, GPU, or Pegasus execution. The authoritative acceptance
criteria remain in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

The review-remediation gate ran on arm64 macOS with Apple Clang 21.0.0,
CMake/CTest 4.4.0, the system Python 3.9.6, an additional Python 3.13.7, and
ShellCheck 0.11.0. No Linux container runtime was available. The host also did
not provide a real CUDA compiler/Toolkit, NVHPC, a GPU, a Pegasus allocation,
FFTW, or a selected production oneMKL/OpenBLAS/LAPACKE installation.
Controlled fixture providers exercise dependency detection, API-shaped source
paths, failure handling, and CPU algorithms only; they are not substitutes for
the production libraries or vendor compilers.

## Review-finding provenance

Before this remediation, the initial local Apple-Clang gate passed 59 of 59
CTest tests and 97 Python tests. Subsequent review found two portability defects
that this host result could not establish: strict-C17 Linux linking needed an
explicit `libm` interface, and generated build metadata in a source tree without
`.git` could serialize unavailable Git fields as invalid JSON. These were code
review/static-analysis findings, not observed Linux or `.git`-less execution
failures. The current gates add explicit regressions for both conditions.

## Current successful local checks

- A clean CPU-only configure and build succeeded with initial project languages
  C and C++, strict C17/C++17, explicit CUDA/OpenACC disable reasons, and the
  system Python 3.9 interpreter selected for tests.
- CTest passed 72 of 72 registered tests. This covers common C/C++ unit tests,
  compile/link probes, positive and negative FFTW/CBLAS/LAPACKE/oneMKL fixture
  providers, CPU benchmark fixtures, all 36 source/target policy checks,
  CUDA/OpenACC fixture-header syntax checks, CUDA runtime-metadata helper tests,
  Pegasus shell checks, Python tests, the Python 3.9 audit, and the `.git`-less
  source-build regression.
- The CTest Python suite ran under the actual system Python 3.9.6 and passed 117
  of 117 tests.
- The `.git`-less regression copied the source without `.git`, configured and
  built it, ran all 71 applicable inner CTests successfully, and checked that
  unavailable Git provenance remains valid JSON with explicit availability and
  null-value semantics.
- All 53 Python source files passed `ast.parse(feature_version=(3, 9))`, the
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
  defines mandatory clean Linux GCC and Clang CPU-only jobs. The workflow file
  is reviewable configuration, not execution evidence; neither job ran in this
  local task.

The clean review-remediation commands run from the repository root were:

```bash
PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  /usr/local/bin/cmake -S . \
  -B /tmp/gpu-library-suite-local-build/review-final-gate-1 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_SUITE_BUILD_CPU=ON \
  -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=OFF \
  -DGPU_SUITE_BUILD_TESTS=ON \
  -DPython3_EXECUTABLE=/usr/bin/python3
/usr/local/bin/cmake --build \
  /tmp/gpu-library-suite-local-build/review-final-gate-1 --parallel 8
/usr/local/bin/ctest --test-dir \
  /tmp/gpu-library-suite-local-build/review-final-gate-1 \
  --output-on-failure --parallel 8
/usr/bin/python3 tools/check_python39.py .
PYTHONPYCACHEPREFIX=/tmp/gpu-library-suite-local-build/review-final-pycache \
  /usr/bin/python3 -m compileall -q common tools jobs tests
bash -n jobs/pegasus/build_cpu_cuda.sh
bash -n jobs/pegasus/build_openacc.sh
bash -n jobs/pegasus/run_node.sh
bash -n jobs/pegasus/run_benchmarks.pbs.in
/usr/local/bin/shellcheck jobs/pegasus/build_cpu_cuda.sh
/usr/local/bin/shellcheck jobs/pegasus/build_openacc.sh
/usr/local/bin/shellcheck jobs/pegasus/run_node.sh
/usr/bin/c++ -std=c++17 -Wall -Wextra -Wpedantic \
  nvidia/c-cpp/curand/examples/rand_cpu.cpp \
  -o /tmp/gpu-library-suite-local-build/review-final-gate-1/manual/rand_cpu
/tmp/gpu-library-suite-local-build/review-final-gate-1/manual/rand_cpu
/usr/bin/c++ -std=c++17 -Wall -Wextra -Wpedantic \
  nvidia/c-cpp/thrust/examples/reduce_cpu.cpp \
  -o /tmp/gpu-library-suite-local-build/review-final-gate-1/manual/reduce_cpu
/tmp/gpu-library-suite-local-build/review-final-gate-1/manual/reduce_cpu
git diff --check
```

The CUDA/OpenACC syntax tests above use controlled headers and a host compiler.
They verify project source integration and diagnostics but are not vendor
compile, link, runtime, numerical, or performance tests.

## Explicitly unexecuted checks

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

# Validation report

## Scope and status

This report records the local implementation validation performed on
2026-07-14. It is evidence for the local phase gates, not evidence of Pegasus or
GPU execution. The authoritative acceptance criteria remain in
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

The local host provided Apple Clang 21, CMake/CTest 4.4.0, Python 3.13, and
ShellCheck 0.11. It did not provide a real CUDA compiler/Toolkit, NVHPC,
Pegasus allocation, FFTW installation, or production oneMKL/OpenBLAS/LAPACKE
installation. Controlled fixture providers were used only to exercise positive
and negative dependency logic and CPU algorithms.

## Successful local checks

- CPU-only configure succeeded with initial project languages C and C++ and
  explicit CUDA/OpenACC disable reasons.
- Separate CUDA-requested and OpenACC-requested negative configures remained
  successful without their toolchains and reported `no CUDA compiler` and
  `OpenACC CXX support was not found` respectively.
- Strict C17/C++17 common code, dependency-free CPU examples/benchmarks, C/C++
  unit tests, and controlled CPU-provider fixtures built successfully.
- CTest passed 59 of 59 registered tests. This includes six common tests,
  compile/link positive and negative fixtures, CPU benchmark fixtures, all 36
  source/target policy checks, CUDA/OpenACC fixture-header syntax checks,
  Pegasus shell checks, Python unittest discovery, and the Python 3.9 audit.
- CTest-injected Python unittest passed 97 of 97 tests. Standalone discovery
  found the same 97 tests: 88 passed and nine integration cases were explicitly
  skipped because CMake fixture executable paths were not injected; those nine
  cases ran in the CTest invocation.
- All 51 Python sources passed
  `ast.parse(feature_version=(3, 9))`, normal compilation, `py_compile`, and
  the project post-3.9 standard-library API audit.
- `compileall`, `bash -n` for all three shell scripts and the PBS template, and
  ShellCheck for all directly analyzable shell scripts succeeded.
- Synthetic Pegasus tests covered deterministic rendering, runtime-environment
  hashing, additional-wave mismatch, complete/duplicate rank mapping,
  timezone/UTC offset, local-midnight rollover, interval telemetry correlation,
  node signal recovery, missing status, verification failure, collection
  failure, and preservation of other-node artifacts after one node fails.
- Source-snapshot tests covered canonical path ordering, content and executable
  mode changes, duplicate/traversal/non-file rejection, and a versioned empty
  snapshot identity.
- The actual local partial manifest drove one `configs/pilot.json` suite run in
  `/tmp`: dependency-free cuRAND/Thrust produced four successful rows, the 34
  unavailable optional series produced explicit prerequisite skips, all 38
  trial indices were retained, and `validate_results.py` wrote
  `validation_status=failure` and returned nonzero as required.

Commands used from the repository root were:

```bash
cmake -S . -B /tmp/gpu-library-suite-local-build/cpu-only-validation-final \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_SUITE_BUILD_CPU=ON \
  -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=OFF \
  -DGPU_SUITE_BUILD_TESTS=ON
cmake --build /tmp/gpu-library-suite-local-build/cpu-only-validation-final \
  --parallel 4
ctest --test-dir /tmp/gpu-library-suite-local-build/cpu-only-validation-final \
  --output-on-failure --parallel 8
cmake -S . \
  -B /tmp/gpu-library-suite-local-build/cuda-requested-no-toolchain-final \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_SUITE_BUILD_CPU=ON \
  -DGPU_SUITE_BUILD_CUDA=ON \
  -DGPU_SUITE_BUILD_OPENACC=OFF \
  -DGPU_SUITE_BUILD_TESTS=OFF
cmake -S . \
  -B /tmp/gpu-library-suite-local-build/openacc-requested-no-toolchain-final \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_SUITE_BUILD_CPU=OFF \
  -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=ON \
  -DGPU_SUITE_BUILD_TESTS=OFF \
  -DGPU_SUITE_NVHPC_GPU_TARGET=test-target
PYTHONPATH=tools:tests/python \
  PYTHONPYCACHEPREFIX=/tmp/gpu-library-suite-local-build/pycache \
  python3 -m unittest discover -s tests/python -v
PYTHONPYCACHEPREFIX=/tmp/gpu-library-suite-local-build/compileall \
  python3 -m compileall -q -f tools jobs tests
python3 tools/check_python39.py .
python3 tools/hash_source_snapshot.py .
bash -n jobs/pegasus/build_cpu_cuda.sh
bash -n jobs/pegasus/build_openacc.sh
bash -n jobs/pegasus/run_node.sh
bash -n jobs/pegasus/run_benchmarks.pbs.in
shellcheck jobs/pegasus/build_cpu_cuda.sh \
  jobs/pegasus/build_openacc.sh jobs/pegasus/run_node.sh
c++ -std=c++17 -O2 nvidia/c-cpp/curand/examples/rand_cpu.cpp \
  -o /tmp/gpu-library-suite-local-build/rand_cpu-direct
/tmp/gpu-library-suite-local-build/rand_cpu-direct
c++ -std=c++17 -O2 nvidia/c-cpp/thrust/examples/reduce_cpu.cpp \
  -o /tmp/gpu-library-suite-local-build/reduce_cpu-direct
/tmp/gpu-library-suite-local-build/reduce_cpu-direct
git diff --check
```

The CUDA-requested configure completed successfully and reported that CUDA
targets were disabled because no CUDA compiler was found; it was not a CUDA
compile or execution test. Final documentation tests and full-text policy
searches also passed.

## Explicitly unexecuted checks

| Check | Status and reason |
| --- | --- |
| Real direct CUDA compile, link, and execution | Unexecuted: no local CUDA compiler, Toolkit, or GPU was available. |
| Real NVHPC/OpenACC compile, link, and execution | Unexecuted: no local NVHPC compiler/runtime was available. |
| Real FFTW threaded CPU execution | Unexecuted: no local FFTW installation was available; a controlled link/runtime fixture passed. |
| Real oneMKL/OpenBLAS/LAPACKE execution | Unexecuted: no selected production installation was available; controlled positive/negative provider fixtures passed. |
| Unit tests under an actual Python 3.9 interpreter | Unexecuted: the local interpreter was Python 3.13; AST 3.9 parsing, API audit, and byte-compilation passed, while the real Python 3.9 run remains a Pegasus manual check. |
| Configure/build under CMake 3.20 | Unexecuted: the local CMake was 4.4.0; only 3.20-compatible features are used and the real-site 3.20 check remains manual. |
| GPU numerical verification, sanitizer, and performance measurement | Unexecuted: fixture-header syntax checks are not GPU tests. |
| Pegasus build, rendered-job submission, runtime collection, telemetry, and node recovery | Unexecuted: remote access and scheduler operations are outside local-agent authority. |
| Five/eight-node production campaign and cross-wave performance report | Unexecuted until human Pegasus smoke and calibration succeed. |

## Required human Pegasus work

Follow [`PEGASUS_MANUAL_VALIDATION.md`](PEGASUS_MANUAL_VALIDATION.md). In
particular, validate the two build profiles and Toolkit match, run normal and
failure smoke jobs, inspect runtime/binary provenance and telemetry correlation,
then calibrate `configs/benchmark.json` before any production campaign. No
Pegasus run report exists yet because no Pegasus job was executed in this local
task.

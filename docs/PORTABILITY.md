# Portability guide

## Purpose

This guide summarizes how to reuse the implemented NVIDIA C/C++ suite without
turning site assumptions into portable source. It is not a second build or
benchmark specification. Permanent policy is in [`AGENTS.md`](../AGENTS.md),
build requirements are in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md),
and timing and schema behavior are in
[`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) and
[`RESULT_SCHEMA.md`](RESULT_SCHEMA.md).

## Portable layers

- `nvidia/c-cpp/*/examples` contains direct teaching programs with no site
  paths, scheduler options, or architecture flags.
- `nvidia/c-cpp/*/benchmarks` contains single-process benchmark programs that
  receive every workload and execution control through the common CLI.
- `common/c-cpp` supplies strict C17 measurement, checked arithmetic, CLI,
  serialization, and metadata support usable from C++17.
- `configs`, `tools`, and `tests` are site-neutral and require only Python 3.9
  standard-library behavior, except that plotting optionally uses matplotlib.
- `jobs/<system>` is the only layer allowed to know a scheduler, module system,
  node-local filesystem, or site launch convention.

## Build boundaries

The top-level CMake project enables only C and C++ at initial configure. Direct
CUDA is checked and enabled only when requested. Missing FFTW, CBLAS, oneMKL
Sparse, LAPACKE, CUDA Toolkit, OpenACC, or Thrust interoperation disables the
affected target with a configure-summary reason; a detected header or library
name alone is insufficient.

Use separate production build trees:

| Profile | Contents |
| --- | --- |
| `cpu-cuda` | CPU and direct CUDA targets; no OpenACC targets |
| `openacc` | NVHPC OpenACC targets; no CPU or direct CUDA targets |

Do not merge two provider builds under an ambiguous manifest identity. Each
artifact records its target, build profile, backend variant, compiler/build
metadata, and binary hash. CUDA architecture and NVHPC GPU target are external
inputs and are never inferred from a device at runtime.

## Numerical portability

The suite does not enable fast math, TF32, Tensor Core modes, or a
hardware-specific numerical path by default. CPU, CUDA, and OpenACC series use
the same configured problem, precision, scope, repeat count, and verification
contract. An out-of-memory condition is a visible failed trial; it does not
silently reduce the problem.

Production CPU providers are explicit. A reference or auxiliary implementation
cannot replace a missing production denominator. In particular, serial FFTW
does not replace threaded FFTW for primary cuFFT speedup.

## Runtime portability

A campaign's effective configuration, source state, executable manifest,
binary hashes, build metadata, and runtime-environment hash travel with its
results. Results from different runtime hashes require a new run ID rather than
an implicit cross-wave merge.

The eight CPU runtime variables are preserved even when a serial backend uses
one effective thread. On Linux, runtime collection records `ldd`; an unavailable
local command is reported as unavailable and never replaced with guessed
output. Site-specific collectors may extend telemetry, but cannot remove the
required raw records or UTC timestamp correlation.

## Porting checklist

Before using the suite on another system:

1. configure a CPU-only tree and inspect every enable/disable reason;
2. build CUDA and OpenACC in separate trees with externally selected
   architecture and Toolkit values;
3. merge manifests and reject every provider/profile collision;
4. run teaching examples and a small verified pilot before calibration;
5. capture the complete runtime environment and binary dependencies;
6. validate raw results before aggregation; and
7. document every unexecuted optional backend instead of reporting it passed.

Adding a new vendor or language is a source-scope change, not merely a new
build flag. Follow [`ADDING_A_NEW_VENDOR.md`](ADDING_A_NEW_VENDOR.md) or
[`ADDING_FORTRAN.md`](ADDING_FORTRAN.md) only after the project owner approves
that expansion.

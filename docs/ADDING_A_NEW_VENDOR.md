# Adding a new GPU vendor

## Scope gate

This is an extension guide, not open-ended authorization to create vendor trees.
The implemented source scope includes NVIDIA C/C++, the approved NVIDIA Fortran
addition, and AMD C/C++ CPU/HIP/OpenMP forms for rocFFT, rocBLAS, rocSPARSE,
rocSOLVER, rocRAND and rocThrust. AMD basic numerical coverage and the bounded
NVIDIA coexistence review are recorded in
[VALIDATION_REPORT](VALIDATION_REPORT.md#current-coexistence-status).
Saved NVIDIA and AMD performance campaigns are also recorded there; current
rocSPARSE v2/csr_rowsplit coverage is in its [README](../amd/c-cpp/rocsparse/README.md).
These records do not authorize new execution. Use the
[implementation plan](IMPLEMENTATION_PLAN.md) when extending the project.
AMD Fortran remains outside this increment.

Implementation is not the same as real-device validation. The retained Sirius
six-family build verified all 36 artifacts with explicit AOCL Sparse; rocFFT
and the focused remaining-family runs provide 576 passing numerical records
at their distinct snapshots. Pegasus subsequently verified 36 NVIDIA C/C++/
Fortran benchmarks and 72 numerical cases at its recorded clean commit.
These results do not establish performance, effective CPU worker counts or
complete loaded-library identity. New execution still needs separate approval.
See the [stage gates](IMPLEMENTATION_PLAN.md#amd-cc-and-sirius-development-stages).
For any additional scope, obtain applicable owner approval and update authority
documents before writing vendor source. Do not create an empty vendor directory,
dummy target or TODO-only example.

The hierarchy and canonical filenames belong to
[`PROJECT_SPECIFICATION.md`](PROJECT_SPECIFICATION.md). CLI/timing behavior,
machine-readable formats, and system operations belong respectively to
[`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md),
[`RESULT_SCHEMA.md`](RESULT_SCHEMA.md), and the registered system owner document:
[`PEGASUS_EXECUTION.md`](PEGASUS_EXECUTION.md) for Pegasus or
[`jobs/sirius/README.md`](../jobs/sirius/README.md) for Sirius. Do not duplicate an
operating contract in another document.

## Required design work

1. Identify the authoritative teaching material, supported programming
   language, initial library set, exact source filenames, teaching problems,
   precision, and data-management model.
2. Record accepted scope and backend decisions in `DECISIONS.md`, and add the
   real hierarchy to `PROJECT_SPECIFICATION.md` without duplicating rules in a
   README.
3. Map each new GPU library to the existing benchmark concepts: `cpu`, vendor
   GPU, and directive/offload implementation; `compute` and `end-to-end`
   scopes; canonical restoration; verification; and one production CPU
   denominator.
4. Decide whether the existing raw schema can represent the vendor. An
   incompatible field or semantic change requires a schema-version decision;
   a compatible new value does not justify a versioned filename.
5. Define a build profile and manifest `build_profile`/`backend_variant` values
   that cannot collide with NVIDIA artifacts. Compiler and runtime detection
   must use real compile-and-link probes.
6. Add system support only under `jobs/<system>` and only for a system actually
   in scope. Follow its registered execution/approval owner; Pegasus retains
   human submission, while current Sirius execution follows its separate owner
   document and the latest explicit user approval.

## Implementation rules

- Add complete examples and benchmarks together; every teaching example must
  compile independently with its explicit inputs and expose the vendor-library
  call flow directly. The sole C/C++ teaching exception to single-source form
  is rocThrust OpenMP's caller plus `rocthrust_wrapper.hip`, as specified in
  [`PROJECT_SPECIFICATION.md`](PROJECT_SPECIFICATION.md#approved-amd-filename-and-teaching-exception).
- Keep source-stem, CMake target, executable, and documented name identical.
- Never select a smaller workload, alternate algorithm, lower precision, or
  numerical mode after inspecting hardware.
- Keep architecture flags out of source and portable configuration.
- Use the common result writer and strict loader instead of a vendor-specific
  raw format.
- Make missing optional dependencies disable only affected targets with a
  specific reason.
- Preserve all completed trials and distinguish attempted failure from
  unattempted skip.

## AMD/Sirius preparation history and current implementation

Stage 1 established the scope/decision/plan documents, bounded Sirius probe
helper and mock/fake tests, and reproducible CTest/Python validation entry. That
historical approval did not include AMD source or remote execution; later
explicit approvals expanded the scope without deleting the original stop gates
or failed authentication evidence.

Stage 2 completed with owner-authenticated interactive SSH. The first Stage 3
increment implemented rocFFT CPU/HIP/OpenMP and common integration. Subsequent
separately approved Sirius runs established compiler/module dependencies,
serial/threaded AOCL FFTW, HIP/OpenMP builds and basic numerical correctness.
The final allocated rocFFT smoke rebuilt all three profiles from one source
snapshot and passed all 96 serial/threaded CPU, HIP and OpenMP records. This
is neither a performance campaign nor evidence for another AMD library.

The local expansion implements the remaining five CPU/HIP/OpenMP
families, including complete teaching and benchmark sources, optional API/link
discovery, manifest/config/result integration and failure/verification fixtures.
The mathematical tasks follow the corresponding NVIDIA families under the
[specification owner](PROJECT_SPECIFICATION.md); rocRAND explicitly compares
different CPU/GPU RNG algorithms. The subsequent local/fixture and minimal
Pegasus build/smoke coexistence checks passed within the recorded coverage.
Future extensions must continue to preserve saved NVIDIA formats and results;
that acceptance does not certify later edits or performance.

The owner-approved AOCL Sparse increment adds an independent, explicit CPU
production choice for AMD rocSPARSE while retaining oneMKL and reference
behavior. The backend, threading and ordered-link contract belongs to
[BENCHMARK_PROTOCOL](BENCHMARK_PROTOCOL.md#cpu-backends-and-parallelism), and
header/library/runtime identity belongs to [RESULT_SCHEMA](RESULT_SCHEMA.md).
Configured input hashes do not certify which shared object was actually loaded;
the remaining runtime-provenance gate is separate from numerical smoke success.

`rocthrust_wrapper.hip` is the owner-selected canonical wrapper name across
teaching source, distribution, CMake, tests and documentation. Its HIP object
and the OpenMP caller are explicit separate compile/link inputs, with both
translation units' conditions captured in provenance. Manual teaching builds
may use `hipcc`; native CMake HIP uses supported Clang rather than fixing
`hipcc` as `CMAKE_HIP_COMPILER`. Compiler paths, ROCm versions and GPU targets
must come from confirmed environment evidence.

Source access through the existing SSHFS mount does not authorize an SSH,
scheduler or GPU call. Future execution uses local Shell tools under the
current approval, not Computer Use of an integrated terminal. System-specific
operation limits, scheduler requirements, logs and stopping rules belong only
to [Sirius operations](../jobs/sirius/README.md). Preserve historical
authentication state and unused transports; do not resume transport development.

Separate runtime evidence from mathematical comparison keys only while keeping
strict campaign provenance. Bind every library/implementation to its approved
artifact, compiler/build conditions and binary, require coherent shared HIP and
per-library dependencies, and reject later source/config/runtime/library changes.
Header-only dependencies and the rocThrust HIP wrapper still have content/build
identity; neither may disappear from provenance because no standalone DSO exists.
The [result schema](RESULT_SCHEMA.md#amd-additive-representation-and-campaign-identity)
owns those representations and the implementation plan owns their acceptance gates.

## Acceptance before declaring support

- The new canonical inventory has no empty or pseudocode files.
- CPU-only configure remains successful without the new compiler or runtime.
- Positive and negative symbol probes prove that headers alone cannot enable a
  target.
- All common CLI, schema, output-ownership, ordering, failure-isolation, and
  aggregation tests cover the new implementation value.
- GPU verification and timing are run on real supported hardware; syntax-only
  checks are reported separately.
- User documentation lists exact dependencies and all unexecuted systems.
- Full source search finds no embedded site path, scheduler directive, device
  model, architecture, credential, or manually versioned canonical filename.

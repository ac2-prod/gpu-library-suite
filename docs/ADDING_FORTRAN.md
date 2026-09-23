# Adding Fortran support

## Scope gate

The project owner approved NVIDIA Fortran examples and benchmarks using the
2026-09-17 teaching extraction. This guide records integration requirements;
see [NVIDIA Fortran](../nvidia/fortran/README.md) for usage and validation limits.

The source integration is implemented with `GPU_SUITE_SOURCE_LANGUAGE=fortran`
in separate trees, Fortran workload callbacks and a narrow C ABI bridge. GNU
CPU/local fixtures and GPU/NVHPC real-machine checks are distinct gates; see
[executed local validation](VALIDATION_REPORT.md#fortran-local-validation).

Update [`PROJECT_SPECIFICATION.md`](PROJECT_SPECIFICATION.md) first with the
authoritative teaching edition, exact filenames, language/library hierarchy,
teaching problems, precision, and data-management rules. Record decisions in
[`DECISIONS.md`](DECISIONS.md). Do not translate the C/C++ files mechanically
and assume that names, APIs, allocation rules, or verification are canonical.

## Language and build design

1. Add Fortran to a dedicated build profile only when requested, following the
   same CPU-only configure principle used for conditional CUDA language
   enablement.
2. Select and document the minimum Fortran standard and compiler behavior using
   features available to CMake 3.20.
3. Give each source a target and executable with the same stem, and add an
   unambiguous manifest `compiler_language`, `build_profile`, and
   `backend_variant`.
4. Probe real linkable symbols and module usability. Finding a `.mod` file or
   library filename is not sufficient.
5. Decide how the Fortran implementation calls common timing, timestamp,
   serialization, and checked-size functionality. It must emit exactly the
   existing result schema or an explicitly approved incompatible successor.
6. Keep NVIDIA, AMD, CPU-provider, and OpenACC compiler requirements isolated
   by target and build profile.

## Source and benchmark requirements

- Teaching examples do not call the benchmark wrapper. The explicitly
  documented mathematical helpers and Thrust interop object are separate link
  inputs, as approved in `PROJECT_SPECIFICATION.md`.
- Benchmarks implement the same problem, precision, scope, repeat-state,
  restoration, verification, and output-ownership contracts as corresponding
  C/C++ benchmarks.
- Column-major or language-indexing differences must be reflected in the
  mathematical helper tests, not hidden by changing the problem.
- GPU/directive managed data must not be double allocated, and pointer/address
  interoperation must occur only after data entry.
- A Fortran CPU reference cannot silently replace the selected production CPU
  backend.

## Required validation

- Configure without a Fortran compiler and prove existing targets are
  unaffected.
- Test positive and negative module/symbol probes with real or controlled
  fixtures.
- Compile every teaching source independently.
- Run numerical and state-restoration tests for every available CPU backend.
- Run the GPU implementations on supported hardware and keep unexecuted
  compiler/GPU combinations visible.
- Validate that Fortran rows round-trip through the strict Python loader and
  aggregate with corresponding CPU/GPU series without schema exceptions.
- Re-run manifest collision, output ownership, crash synthesis, telemetry
  correlation, and documentation consistency tests.

System scripts may be added only for an approved system and must retain the
same human-submission, runtime-provenance, node-isolation, and collector-owned
final-status boundaries.

# Adding a new GPU vendor

## Scope gate

This is an extension guide, not authorization to create another vendor tree.
The implemented scope remains NVIDIA C/C++. Obtain explicit project-owner
approval before adding a vendor, then update the authority documents before
writing source. Do not create an empty vendor directory, dummy target, or
TODO-only example.

The hierarchy and canonical filenames belong to
[`PROJECT_SPECIFICATION.md`](PROJECT_SPECIFICATION.md). CLI/timing behavior,
machine-readable formats, and system operations belong respectively to
[`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md),
[`RESULT_SCHEMA.md`](RESULT_SCHEMA.md), and a system document parallel to
[`PEGASUS_EXECUTION.md`](PEGASUS_EXECUTION.md).

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
   in scope. Submission remains a human action.

## Implementation rules

- Add complete examples and benchmarks together; every teaching example must
  compile independently and expose the vendor-library call flow directly.
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

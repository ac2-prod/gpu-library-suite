# GPU Library Suite Agent Rules

## Purpose and authority

This file owns permanent repository rules, prohibitions, and pointers to
authoritative specifications. It intentionally does not duplicate detailed CLI,
schema, mathematical, or system-operating rules.

For teaching code, canonical filenames, and default teaching problems, the
authority order is:

1. current user instructions;
2. the 2026-07-13 C/C++ edition and the approved 2026-09-17 Fortran
   extraction of *Library Edition (NVIDIA GPU)*, for their respective languages; and
3. other material, including editions dated 2026-07-12 or earlier, as historical
   reference only.

## Specification ownership

| Document | Owns |
| --- | --- |
| `AGENTS.md` | Permanent rules, prohibitions, authority order, and document routing. |
| `docs/PROJECT_SPECIFICATION.md` | Hierarchy, canonical filenames, teaching examples, mathematics, and example data management. |
| `docs/BENCHMARK_PROTOCOL.md` | CLI, workloads, timing, warm-up, verification, CPU backends, ordering, and statistics. |
| `docs/RESULT_SCHEMA.md` | Raw/aggregate formats, configuration, executable manifests, and run/wave/node metadata. |
| `docs/PEGASUS_EXECUTION.md` | Pegasus-specific PBS, scratch, telemetry, build, and run operations. |
| `jobs/sirius/README.md` | Sirius-specific approval, probe, connection budgets, execution records, and stopping rules. |
| `docs/IMPLEMENTATION_PLAN.md` | Phases, dependencies, acceptance criteria, validation, risks, and unresolved decisions. |
| `docs/DECISIONS.md` | Decision status, concise rationale, and links to authoritative specifications. |
| `README.md` | Non-normative user-facing summary. |

Do not redefine an owned specification elsewhere. Link to its owner.

## Workspace and safety boundary

- Read and write only inside the current repository workspace, except for small
  local validation artifacts under `/tmp/gpu-library-suite-local-build`.
- Use the local Shell tool for commands. Do not operate a terminal through
  Computer Use. Remote operations require explicit approval for the host and
  scope and use noninteractive SSH commands; never collect authentication input.
- On login nodes perform only file/lightweight environment checks, approved job
  submission/queue observation, and log collection. Computational builds,
  tests, benchmarks, and GPU execution must use the system scheduler.
- Local syntax and fixture checks do not establish remote launch readiness.
  Before presenting an operation for approval, check its exact noninteractive
  invocation against target-specific evidence. Report unresolved command-path
  or environment prerequisites explicitly; follow the system operating document.
- Report the SSH argv, stdout, stderr, exit code and, after submission, job ID.
  Deletion, overwriting existing run artifacts, job cancellation, and bulk
  resubmission require prior approval. Preserve historical runs and auth state.
- The current development task authorizes editing through the existing SSHFS
  mount and Mac-local validation only; it does not authorize a new SSH or
  scheduler operation. Pegasus-specific restrictions remain in force.
- Do not access `/work`, `/scr`, or `/pmem` from a local development host.
- Do not request network access or install packages with Homebrew, pip, conda,
  apt, or similar tools.
- Do not assume CUDA, NVHPC, oneMKL, FFTW, OpenBLAS, LAPACKE, or any GPU is
  installed locally.
- Do not create symbolic links, delete existing files without explicit
  authorization, or create large build/temp trees on the mounted workspace.
- Do not edit `.git` or the Git index. Do not run commit, push, pull, fetch,
  checkout, switch, reset, clean, or rebase. Read-only commands such as status,
  diff, log, and rev-parse are allowed.
- Keep credentials, tokens, private keys, account values, and other secrets out
  of the repository. Treat the repository as private but not as secret storage.
- Never guess a supercomputer account, queue, module version, or absolute user
  path.

## Project identity and scope

- The repository contains portable GPU-library teaching examples, reproducible
  benchmarks, common measurement/validation/reporting tools, and system-specific
  job support.
- It is maintained under `ac2-prod` and is not an official HAIRDESC GitHub
  repository.
- Do not use HAIRDESC in repository identifiers, namespaces, executable names,
  package names, or library names.
- Do not add a license until the project owner explicitly selects one.

The navigation hierarchy is vendor → programming language → GPU library →
purpose (`examples/` or `benchmarks/`). Principal top-level areas are
`nvidia/`, `amd/`, `common/`, `configs/`, `tools/`, `jobs/`, `docs/`, and
`tests/`.

The NVIDIA C/C++ and approved NVIDIA Fortran implementations coexist with the
AMD C/C++ development family: rocFFT, rocBLAS, rocSPARSE, rocSOLVER, rocRAND,
and rocThrust. rocFFT's recorded Sirius build and 96-case basic smoke form the
baseline for the remaining five libraries. AMD basic numerical coverage and the
subsequent NVIDIA local/fixture plus Pegasus build/smoke coexistence review are
recorded in `docs/VALIDATION_REPORT.md`, with their distinct source snapshots.
Performance preparation follows `docs/IMPLEMENTATION_PLAN.md` and requires
separate execution approval; a passed baseline does not validate later edits.
The owner-approved AMD Fortran source family follows the 2026-10-20 teaching
material under `amd/fortran`, with CPU, HIP/hipFORT and OpenMP target examples
and benchmarks. This source addition is not evidence of an AMD Fortran real
build, GPU smoke or performance measurement. Miyabi and TSUBAME4.0 remain
future scope. Do not create empty
future directories or dummy implementations. System support belongs under
`jobs/pegasus` and `jobs/sirius`; do not create Furo job directories.
Preserve historical authentication state and unused transport source; do not
resume SSH transport development.

## Teaching examples

- `examples/` is the source of truth for code printed in the teaching material.
- Filenames must exactly match the canonical material.
- Each C/C++ example is a complete, independently compilable single-source
  program, except that rocThrust OpenMP interoperation uses two translation
  units: its OpenMP source and the canonical `rocthrust_wrapper.hip`.
  Both are explicit compile/link inputs. This is the only C/C++ teaching
  exception. Fortran helpers and the Thrust interop object remain explicit
  link inputs as specified in `docs/PROJECT_SPECIFICATION.md`.
- Do not use ellipses, pseudocode, hidden helper source, benchmark CLI, repeated
  trials, machine-readable result output, or environment metadata.
- Preserve teaching clarity; avoid abstraction that hides the library-call flow.
- Use concise CUDA Runtime/library error checks without obscuring that flow.
- Do not embed architecture-specific compiler options in source.
- Do not force examples through common benchmark wrappers.

See `docs/PROJECT_SPECIFICATION.md` for filenames, mathematics, algorithm
defaults, and OpenACC data-management rules.

## Benchmarks

- `benchmarks/` is performance-measurement code, separate from teaching examples.
- CPU, CUDA, and OpenACC variants use the same problem, input, precision, size,
  scope, and repeat count.
- Runtime controls include problem size, warm-up, repeat, trials, scope,
  verification, output, format, device, campaign identity, and CPU backend/thread
  settings as specified in `docs/BENCHMARK_PROTOCOL.md`.
- Portable source contains no site path, module, queue, scheduler directive, GPU
  model, or compute capability.
- Never change size, algorithm, precision, or numerical mode silently after
  hardware detection. Report out-of-memory failure clearly.
- Store raw trials. Do not aggregate nodes or remove outliers in benchmark
  executables.
- Benchmark executables emit machine-readable records; the suite runner is the
  only writer of each node-level raw-result file. Never overwrite or implicitly
  append to an existing output.
- Use the shared monotonic wall clock for CPU/CUDA/OpenACC; CUDA Events are not
  the primary comparison timer.
- Do not add unnecessary `cudaDeviceSynchronize` calls inside repeat loops.
- Benchmark executables remain single-node and single-process. Multi-node
  repetition belongs to the launcher.
- Pair CPU/CUDA/OpenACC within each node block. Do not exclude a node merely
  because it is slow.
- Restore all state modified by warm-up before measurement, restore canonical
  state before every raw trial, and perform required end-to-end per-repeat
  restoration outside the timed interval.
- Reference CPU backends must never silently replace production backends.
- A primary campaign has exactly one production CPU backend per benchmark;
  additional CPU backends are separate campaigns or explicitly auxiliary.

## Canonical names and reproducibility

- Do not append manually maintained version numbers, dates, “final”, or “latest”
  labels to repository-owned canonical filenames.
- Canonical configurations are `configs/benchmark.json` and
  `configs/pilot.json`; the executable example manifest is
  `configs/executables.json.example`.
- Use Git history, tags, and releases for artifact history.
- Store format versions inside machine-readable documents and increment them only
  for incompatible format changes.
- Preserve upstream-mandated names such as `cublas_v2.h`.
- Every campaign stores its effective configuration, immutable run metadata,
  Git commit/dirty provenance, executable-manifest hash, binary hashes, and a
  deterministic runtime-environment hash as specified in
  `docs/RESULT_SCHEMA.md`.
- Production runs require a clean worktree. Dirty smoke/pilot runs require a
  complete Git-diff or source-snapshot hash. Never combine waves with different
  source, binary, configuration, or runtime-environment provenance under one run
  ID.
- Project JSON input rejects duplicate keys, non-standard numeric constants, and
  invalid UTF-8. Do not rely on permissive language-library defaults.

## Portability and build policy

- Put supercomputer-specific information only under `jobs/<system>/`.
- Supply GPU architecture from CMake or system configuration. Do not hard-code
  `sm_90`, `cc90`, H100, or Pegasus values in portable source.
- The build must support CPU-only configure.
- Configure CUDA and OpenACC in separate build trees.
- AMD teaching instructions may use `hipcc` for manual builds. Native CMake HIP
  uses a supported Clang compiler, not the `hipcc` wrapper fixed as
  `CMAKE_HIP_COMPILER`. Do not choose Sirius compiler paths, ROCm versions, or
  GPU targets before environment evidence and the corresponding approval.
- Do not enable fast math, TF32, Tensor Core modes, or architecture-specific
  numerical changes without an explicit approved option.
- Missing optional dependencies disable only affected targets with a clear
  reason; do not install packages automatically.
- Portable source must remain reusable on Pegasus, Miyabi, TSUBAME4.0, Sirius,
  and local GPU servers.

## Pegasus boundary

Only `docs/PEGASUS_EXECUTION.md` owns Pegasus operating details. Job submission
is always a human action. Generated tools must not run scheduler, accounting, or
interactive-login commands. Build and benchmark jobs remain separate. Before
the measurement launch, the PBS job master obtains and validates a complete
rank-host mapping, runs the planned `prepare_wave.py`, and exclusively creates
campaign/wave metadata. Each node process then writes node-local scratch and
only its own result subtree. No benchmark-process collective owns metadata
creation.

## Validation and reporting

- Never report an unexecuted GPU test as passed.
- Report every unexecuted test and its reason.
- A TODO-only or empty placeholder is not complete.
- Keep unverified areas visible in user documentation and the final report.
- At the end of every task, report changed files, commands run, passed tests,
  unexecuted tests, and manual Pegasus checks still required.
- Do not commit or push changes without explicit owner authorization.
- Approval to commit a fixed measurement version is separate from approval to
  push, create a PR, or merge. The current AMD development approval authorizes
  none of these publication operations or additional real build/smoke runs.
- For an explicitly approved publication, completion includes the PR, required
  CI, main integration, current public usage guidance, and local synchronization.
  Do not leave completed, approved implementation work only on a work branch;
  this completion rule does not itself authorize Git writes or publication.
- Before handoff, inspect `git diff`, `git diff --check`, and `git status --short`;
  preserve unrelated user changes.

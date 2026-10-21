# Validation report

## Current coexistence status

This report distinguishes source compatibility checks, real numerical execution
and performance measurements. The current summary includes the AMD Fortran
evaluation completed on 2026-10-21. Each result applies to its recorded source
and environment; historical sections retain the status at the time of that run.

| Evidence | Confirmed result | Source boundary |
| --- | --- | --- |
| AMD C/C++ on Sirius | Six-family build: 36 teaching/benchmark artifacts; focused numerical coverage: 18 teaching examples and 576 records, all passing | rocFFT and the other five families use the distinct snapshots recorded below; these are not one campaign or a test of every later edit. |
| NVIDIA local/fixture compatibility after AMD addition | Existing CMake, CPU-provider, CLI/schema/manifest, runner/aggregation/plot and Fortran checks passed | Synthetic providers and local compilers are not GPU or performance evidence. |
| NVIDIA C/C++ and Fortran on Pegasus | 36 benchmark artifacts built; 72/72 numerical smoke cases passed, including Fortran Thrust CUDA/OpenACC interoperation | Clean commit `eb9e594be3ea4f1dc4bef6fb562223961c60a43a`; details and private evidence identities are recorded below. |
| AMD three-size pilot on Sirius | 108/108 numerical records passed; PBS `45781.pbs`, exit 0, walltime 00:12:59 | One APU, one trial per condition; retained dirty source snapshot `a3697cae…`. No physical-node or multi-trial dispersion evidence. |
| AMD C/C++ performance on Sirius | Completed comparisons, with replacement campaigns for corrected BLIS links and rocSPARSE v2 | See the [C/C++ comparison record](#amd-production-comparison) and [rocSPARSE evidence](../amd/c-cpp/rocsparse/README.md#validation-and-limits); keep their distinct source identities. |
| AMD Fortran on Sirius | 36 initial artifacts and 18 teaching examples verified; after timing alignment, 18 rebuilt benchmarks, 72 numerical smoke rows and 6,480 production rows passed | Clean measurement commit `b69847694c3e34fd220a3c44b7b1a98da22d6b63`; [details below](#amd-fortran-production-validation). |

The NVIDIA coexistence result does not validate a different AMD measurement
snapshot. Effective external-library CPU worker counts and complete
loaded-library provenance remain unverified. Historical failures and intermediate
pending states below retain their original meaning; later successful runs do
not rewrite them. Saved raw data, job logs and teaching slides are not distributed
in this repository. Public configs and source allow users to make their own
measurements with the [build guide](BUILDING.md) and [measurement protocol](BENCHMARK_PROTOCOL.md).

## AMD Fortran production validation

The six AMD Fortran families have real CPU, HIP and OpenMP target execution
evidence. The initial build produced 18 examples and 18 benchmarks; the examples
and 72 numerical smoke records passed, including rocThrust interoperation.
After aligning benchmark timing with AMD C/C++, the changed benchmarks were
rebuilt and checked again before measuring. The current comparison uses only
those aligned binaries and their 6,480 records, not C/C++ data or the earlier
5,400-row provisional Fortran comparison.

| Stage | Saved run | Result |
| --- | --- | --- |
| Initial examples and benchmarks | `sirius-amd-fortran-build-20261021T044526Z-402336eb` | 36 artifacts built; subsequent smoke ran all 18 examples and passed 72 benchmark records |
| Aligned benchmark build | `sirius-amd-fortran-clean18-20261021T123440Z-a02098a3` | 18/18 built; CPU/APU checks and runtime-input collection completed |
| Aligned numerical smoke | `sirius-amd-fortran-smoke-20261021T123440Z-a02098a3` | 72/72 passed, including both rocThrust GPU paths |
| Aligned production | `sirius-amd-fortran-production-20261021T133446Z-a02098a3` | Wave 0: 3,240/3,240; wave 1: 3,240/3,240; no failed or excluded rows |

All four aligned build/smoke/wave jobs returned PBS exit 0. The production
matrix is six libraries × three sizes × three implementations × two scopes ×
five trials × six hosts × two waves. Both waves used the same six physical
hosts, one process with a matched 24-CPU/APU allocation on each host. HIP and
OpenMP device identity was checked. All recorded times were finite and positive,
with consistent repeat normalization. See the [Sirius procedure](../jobs/sirius/README.md#amd-fortran-current-workflow)
for allocation and runtime setup.

The measured build used GNU Fortran 14.2.1, AMD LLVM Flang 22.0.0, ROCm 7.2.0
with matching hipFORT, CMake 3.31.8 and AOCL 5.1.0. BLAS calls Fortran `dgemm`
and links multithreaded `libblis-mt.so.5.1.0`. Solver calls Fortran `dgetrf` /
`dgetrs` through `libflame.so` with the same multithreaded BLIS dependency.
FFTW uses its single-precision OpenMP thread library. AOCL Sparse LP64,
Fortran `random_number` and intrinsic `sum` complete the CPU providers.
The latter two are serial. Shared backend identifiers containing `cblas` or
`lapacke` do not describe the Fortran API calls.

| Measurement identity | Value |
| --- | --- |
| Source commit | `b69847694c3e34fd220a3c44b7b1a98da22d6b63` |
| Source snapshot SHA-256 | `a02098a3bb99c509ea5c6e6185235d2237be8156e55442108d98d9c25ccaaf78` |
| Effective config SHA-256 | `d5b4b6589799188fe3066ab3fe087f7509708ea233e4a97b8c7e8e508fd0456f` |
| Production manifest SHA-256 | `730ea2a21a7d9ebbff2aa8dbab95b799be9643ea85088c6cac214399d44f3eb2` |
| Production runtime-input SHA-256 | `70f62d55e50ab13d1a3ba5c9208248524baa1d8262ecf9a227ae0b08fed4b4f4` |

The retained manifest contains all 18 binary hashes. Its values are tied to
this build; a later documentation commit does not change the measured source
identity. The private completion report, per-wave validation and aggregation
review record 6,480 numerical passes, zero failures, 12 complete node/wave
blocks, and 108 elapsed-time points checked against the raw data. Six figures
use the same renderer and library/end-to-end format as C/C++. Each node/wave
block contributes its trial median; each wave contributes the median across
its six nodes; the final value is the median of the two equally weighted wave
values. Trials are not pooled and slow nodes are not removed.

Interpretation limits remain:

- The 24-thread requests for FFT/BLAS/Sparse/Solver are not observed worker
  counts; RAND and reduction remain serial baselines.
- Configured links and resolved library identities were verified, but complete
  actual loaded-library identity was not measured.
- Timing boundaries, synchronization and repetitions follow AMD C/C++;
  compiler, language binding and RNG differences still prevent attributing a
  language-to-language time difference solely to the language.
- The two waves sample repeated use of six hosts, not twelve different hosts.
- Earlier failed waves and provisional data are retained separately. A later
  successful run does not prove the cause of an earlier device failure or
  establish that a system defect has been fixed.
- These results describe the recorded software environment before the planned
  system update. They do not validate the updated installation or NVIDIA
  Fortran timing alignment.

## AMD three-size pilot and timing review

This section records the C/C++ pilot and preparation status before the later
[completed comparisons](#amd-production-comparison).

The retained run `sirius-amd-pilot-20261017T183416Z-a3697cae` completed as
PBS `45781.pbs`, state `F`, exit 0; SSH exit was 0. All 108 actual rows passed
numerical verification: six libraries, three sizes, CPU/HIP/OpenMP and both
scopes, with one trial. PBS walltime was 779 seconds; the node runner took
731.531582 seconds. Timed intervals totalled 202.306040168 seconds. These
quantities have different boundaries and must not be substituted for one another.
Per-condition dispersion is unavailable with one trial, not zero.

Evidence under `runs/<RUN_ID>/` is `pilot/node0/raw.jsonl`,
`pilot/validation.json`, `pilot/timing-summary.json`, `dispatch-result.json`
and `review/review.json`. Important identities are:

| Item | SHA-256 |
| --- | --- |
| Source snapshot, originally dirty at `c2b6e30b264496f624178057bdfe4db121857291` | `a3697cae2e446a51eec5a6107dedc39caf656fc8158b08e3591c59f0bc1148e6` |
| Effective pilot config | `bb528ca0e3d4ec0b027728e6023085e8483fb9fa9a516ade0b9df6c814dc6fb1` |
| Selected 18-benchmark manifest | `85781c82bc0ece717d15d0b091728127d7012283013e1288109bbc1b971ecfd3` |
| Raw 108-row JSONL | `e9442f7e4182bab450d40b0d8f38a72e4c14881d922f5cf47e2e653da4930762` |

The four-module environment and one-rank MPI preflight/worker route completed.
This does not validate six-host placement. `loaded_libraries_verified=false`
and unobserved effective BLAS/Sparse/Solver worker counts remain unresolved.
RAND/Thrust CPU series remain serial. Existing dirty artifacts are not clean
production binaries; their source identity must not be relabelled.

A read-only comparison used this artifact source and the saved NVIDIA
publication source at `916a8bda22f6188aa995d21954ddb1d6de2f3cf3`.
The common clock implementation was byte-identical. All 108 AMD pilot rows,
108 NVIDIA pilot rows and 6,480 NVIDIA production rows recorded
`CLOCK_MONOTONIC` and a reported resolution of `1e-9` seconds, with consistent
`elapsed_sec = elapsed_total_sec / repeat`. Reported resolution is not an
accuracy measurement. FFT compute interval grouping differed as described in
the [protocol](BENCHMARK_PROTOCOL.md#rocfft-cpuhipopenmp-contract).
The owner accepted preserving both methods and all existing results. No
benchmark implementation or historical measurement was changed by this review.

Production preparation uses `jobs/sirius/amd-benchmark.json` and the
[Sirius production plan](../jobs/sirius/README.md#amd-production-preparation).
The config SHA-256 is
`7bb6636f3d58180900a289aabd2f0fbed2f9c20b2f06cf2c18e40ca45bd69c8e`.
Nine focused offline checks passed: strict production config, historical
NVIDIA workload equality, bounded pilot-to-production change, reference
inventory, 12-block schedule/counts, balanced orders, dirty-build rejection,
preserved source/config/result bytes and saved pilot evidence. The dirty-build
check mocked filesystem lookups; an initial missing dependency-hash mock was
corrected and the checks rerun, without changing production code or old logs.
These are preparation checks, not GPU or multi-host results. Clean
production artifacts, actual six-host/APU placement and 6,480 production rows
have not been executed or validated.

## AMD rocFFT Sirius baseline

The saved Sirius run `sirius-rocfft-smoke-20261017T062713Z-b29643fc`
completed as PBS job `45718.pbs`, terminal state `F`, exit status `0`.
This is an inspection of retained execution evidence, not a new remote run.
The source snapshot was
`e3d209963b11f940db580a400aee6cafb2adf6a66f0faabaf4ab7bb4fff000b6`.
Later edits require their own validation; this evidence does not certify the
other five AMD libraries or performance at production problem sizes.

All three fresh CPU/HIP/OpenMP builds and teaching examples passed. Four
separate campaigns (serial and threaded CPU, each in JSONL and CSV) contain
24 successful verified records each: 96 total, 0 failed or skipped. Across
formats and campaigns the counts are CPU serial 16, CPU two-thread 16, HIP 32
and OpenMP 32. Cases used FFT length/batch 32/2 and 256/3, both timing scopes,
warm-up 1, repeat 2 and trials 2. Maximum reported verification error was zero.

The run used GCC 14.2.1, AOCL 5.1.0 with its OpenMP FFTW provider, ROCm 7.2.0,
AMD Clang 22 and an allocated MI300A target `gfx942:sramecc+:xnack-`.
Compiler and module observations belong to that run; portable source does not
hard-code them. Its recorded operation counts are one SSH, one job submission
and five job-status queries. No additional remote operation was needed to
review the mounted logs.

Private local evidence is retained under `runs/<RUN_ID>/` (not published):
`dispatch-result.json`, `compute/job-summary.json`,
`compute/source-identity.json`, `compute/profile-status.json`,
`compute/artifact-sha256.json`, `compute/executables.json`, and each of
`compute/serial/` and `compute/threaded/` with `raw.jsonl`, `raw.csv`, effective
configs and validation reports. Historical failed runs remain intact.

The local expansion covers all six AMD teaching and benchmark families.
Host fixtures check numerical, failure and recording behavior; they are not
evidence of real ROCm linkage, target offload, or performance. The subsequent
[complete six-family build](#amd-six-family-build-with-aocl-sparse) establishes
compile/link evidence at its recorded snapshot. The later
[focused smoke](#amd-focused-numerical-smoke) provides basic numerical evidence
for the remaining families. NVIDIA real-machine coexistence and performance
remain separate gates.

## AMD focused numerical smoke

Retained BLAS run `sirius-blas-smoke-20261017T135452Z-a3697cae`
(PBS `45771.pbs`, exit 0) passed all three teaching examples and 96 numerical
records, with zero absolute error. Retained run
`sirius-amd4-smoke-20261017T141232Z-a3697cae` (PBS `45772.pbs`, state `F`,
exit 0) passed 12 teaching examples and 384 numerical records: 96 each for
rocSPARSE, rocSOLVER, rocRAND and rocThrust. CPU, HIP and OpenMP target are
covered. All recorded teaching/benchmark child exits were zero.

Both jobs used unchanged artifacts from the successful six-family build,
source snapshot
`a3697cae2e446a51eec5a6107dedc39caf656fc8158b08e3591c59f0bc1148e6`.
Sparse and Thrust errors were zero; Solver's maximum relative residual was
`4.440892098500626e-16` and maximum relative solution error was
`1.7763568394002505e-15`; RAND passed range and moment checks.
The private per-run evidence remains under `runs/<RUN_ID>/`, including
`smoke/summary.json`, child logs, scheduler records and local SSH evidence.
It is not part of the source commit.

Together with the separate rocFFT baseline, all six AMD families have build
and basic numerical smoke evidence: 18 teaching examples and 576 numerical
records across the retained runs. This count combines coverage, not campaigns
or source provenance. It does not validate later edits or production sizes.
RAND/Thrust CPU backends are sequential; BLAS/Sparse/Solver effective worker
counts and complete actual loaded-library identity remain unverified.
No profiling, additional remote operation or performance measurement is
implied by this review of saved logs. The subsequent
[NVIDIA coexistence review](#nvidia-coexistence-review-after-amd-addition)
completed its local/fixture and Pegasus build/basic-smoke gates independently.

## AMD six-family Sirius build

Retained run `sirius-amd-build-20261017T080308Z-b3d866be` used PBS job
`45728.pbs`, which finished in state `F` with exit status `1`. Its source
snapshot was
`b3d866beae6ae6b1b0e6bf9a9a28d00cb0c2ce5b3c60a0260783203b3a85cc97`.
The source identity matched at the initial check, before each profile and after
the build. This was a build-only allocation: no produced teaching or benchmark
program was run.
Its recorded operations were one SSH, one submission and 21 status queries.
Reviewing the retained mounted logs required no additional remote operation.

| Profile | Recorded outcome | Evidence limit |
| --- | --- | --- |
| HIP | 12 teaching/benchmark artifacts built and hash-verified | All six families; no execution or numerical verification |
| OpenMP target | 12 teaching/benchmark artifacts built and hash-verified | Includes the explicit rocThrust HIP wrapper; no offload execution |
| CPU | Configure succeeded; the BLAS benchmark compile failed | Grouped build stopped before later CPU targets; no CPU partial manifest |
| CPU Sparse | Two targets disabled for missing production provider | No replacement by a reference provider |

The historical controller summary is retained unchanged: `BUILT=24`,
`FAILED=10`, `NOT_BUILT_DEPENDENCY=2`. The ten CPU failure labels are coarse
profile-level results, not ten observed compiler failures. Logs show that
`blas_cpu` linked, `blas_cpu_bench` failed, and eight subsequent configured CPU
targets were not reached. The linked teaching binary was not certified by a
CPU manifest and is not added to the 24 verified artifacts.

Both CPU BLAS targets used strict C17. The benchmark included standard/common
headers before AOCL's `cblas.h` and failed on `pthread_barrier_t` and
`pthread_barrierattr_t`; the teaching source and configure probe included the
provider header first. The project sources and recorded compiler argv supplied
no early POSIX feature-test define. The installed AOCL header internals
were not available in the local review.
AOCL CBLAS/BLIS, LAPACKE/libFLAME and OpenMP FFTW passed their configure API
link probes. That establishes discovery/link evidence, not ABI, numerical or
threading correctness. HIP and OpenMP used the recorded GCC 14.2.1/AOCL 5.1.0/
ROCm 7.2.0 environment, AMD Clang 22 and `gfx942:sramecc+:xnack-`.

Private evidence remains under the run directory: `dispatch-result.json`,
`amd-build/summary.json`, `amd-build/logs/`, per-profile build metadata and
manifests, `job.stdout`, `job.stderr`, and `owner-ssh/`. All 24 verified binary
hashes and all 63 recorded command stdout/stderr hash pairs were checked.
Local recovery edits made after this run are not covered by that real-build
evidence. The separately approved follow-up builds are recorded below;
new-family runtime smoke, final NVIDIA coexistence and performance validation
remain open.

## AMD CPU recovery and AOCL Sparse follow-up

Retained run `sirius-amd-build-20261017T095316Z-249f96a5` used PBS `45730.pbs`
and source snapshot
`249f96a519e58aa8f9d0be8c0d06bfbe91dd67c256a2bb41a99cb3ec7aedcf90`.
The job finished `F`, exit 1, with **34 BUILT, 2 NOT_BUILT_DEPENDENCY, 0 FAILED**:
CPU 10, HIP 12 and OpenMP 12. The absent pair was oneMKL Sparse CPU teaching and
benchmark. CPU BLAS/LAPACKE compiled with the corrected POSIX preambles; later
independent targets were attempted and separately recorded. The job was build
only: no generated teaching/benchmark executable or GPU test was run.

All 34 binary hashes, metadata/manifests, 74 controller stdout/stderr pairs and
25 dispatcher pairs were checked from retained evidence. Source identity and
Git HEAD were unchanged. The operation counts were one approved SSH, one
submission and 19 status queries, without retry; SSH exit 2 denotes the
incomplete 34/36 result, not an authentication failure. The completion marker
matched this run and remote exit 1. Earlier run logs remain unchanged.
`amd-build/summary.json` has SHA-256
`4edd20dbdefcef87f05897245bc77f1eb95e836e10cfb92e8d4ebcd57019bdb4`;
`dispatch-result.json` has SHA-256
`510e11a9a68eec37eea69d2d099f87adbe78e70ae81da80282e5e475558da9c3`.

The bounded provider inventory records an AOCL Sparse header and shared/static
libraries. That supply motivated the subsequent owner-approved **local-only**
AOCL Sparse backend addition; this inventory alone does not establish that the
new code compiled or ran on Sirius. Public 5.1 API declarations were reviewed,
while host fixtures exercise selection, numerical recurrence, lifecycle errors, metadata and
provenance rejection. Missing dependencies never select a fallback. Explicit
oneMKL and reference paths remain available. The subsequent build below closes
the real compile/link gate for its snapshot. Runtime smoke, actual loaded DSO
identity, worker behavior, final NVIDIA coexistence review and performance
validation remain open gates.

## AMD six-family build with AOCL Sparse

Retained run `sirius-amd-build-20261017T104351Z-a3697cae` completed as PBS job
`45731.pbs`, terminal state `F`, exit status `0`, with dispatcher result
`BUILD_COMPLETE` and SSH exit `0`. Its source snapshot was
`a3697cae2e446a51eec5a6107dedc39caf656fc8158b08e3591c59f0bc1148e6`.
The initial source check, each profile's pre-build check and the final check
all recorded this same hash. This evidence applies to those source bytes;
later documentation or implementation edits do not change the retained run.

| Profile | Verified artifacts | Coverage |
| --- | --- | --- |
| CPU | 12 | Teaching and benchmark artifacts for all six families, including explicit AOCL Sparse |
| HIP | 12 | Teaching and benchmark artifacts for all six families |
| OpenMP target | 12 | Teaching and benchmark artifacts for all six families, including the explicit rocThrust HIP wrapper |

All 36 requested slots are `BUILT`, with no failed, unavailable or unattempted
targets. Saved evidence was checked against all 36 binary hashes, build
metadata/manifests, 76 controller command stdout/stderr hash pairs and 27
dispatcher command pairs. The execution used one approved noninteractive SSH,
one submission and 21 `qstat` calls: one queue query and 20 observations of the
same job. PBS recorded one execution with reruns disabled. The sole completion
marker matched this run and exit `0`; the job-body exit was also `0`.

The CPU configuration explicitly selected AOCL Sparse. Its LP64 header/type
assertions and descriptor/create/hint/optimize/SpMV/destroy/version API
compile/link probe passed, and both `sparse_cpu` and `sparse_cpu_bench` compiled
and linked. Their manifests identify `cpu-aocl-sparse` and the `aocl-sparse`
variant. The configured header and shared-library identities match the earlier
bounded inventory and are recorded in build metadata. This establishes the
tested API and LP64 compile/link contract, not loaded-DSO identity or runtime
ABI behavior. The same allocation used GCC 14.2.1, AOCL 5.1.0, ROCm 7.2.0,
AMD Clang 22 and the explicit `gfx942:sramecc+:xnack-` target.

Private evidence remains under `runs/<RUN_ID>/`, including
`amd-build/summary.json`, per-profile manifests and metadata,
`scheduler-logs/`, `job.stdout`, `job.stderr` and `local-ssh-evidence/review.json`.
The saved files have these SHA-256 values:

| Evidence | SHA-256 |
| --- | --- |
| `amd-build/summary.json` | `efb1c112a1b59856f207d6fba0e6bfb13ed4b0a5069dc4cd5401e6fc35eb3883` |
| `dispatch-result.json` | `c69ef3918fe023ddf39f926858f5e1625a859cecbf9c0ae3b27da4ab975bf646` |
| `amd-build/build-cpu/build-metadata.json` | `2a1e0f9eb84c799fd6c45218f4750fc5ec537bfa61f4ca289c96a2098256b01f` |
| `local-ssh-evidence/review.json` | `47008894ab1ac6b6b86c7f84b61ce496b027de5728ec68ae428db74980c6b1d2` |

This was build-only: produced-program executions and GPU queries were both
zero. No numerical smoke, loaded-DSO check or CPU worker observation was run.
The five new families' runtime correctness, current-snapshot all-family smoke,
final NVIDIA coexistence review and performance evaluation remain unverified.
The earlier rocFFT 96-case smoke covers its own historical snapshot only.
Old failed and 34-artifact runs remain intact. This review performed no
additional SSH, scheduler operation or test execution.

## AMD six-family smoke link evidence failure

Retained run `sirius-amd-smoke-20261017T112314Z-a3697cae` used the preceding
36 successful build artifacts in PBS job `45734.pbs`. PBS finished in state
`F`, exit `1`; the dispatcher reported `SMOKE_FAILED` and SSH exit `2`.
The body exit and both completion markers agree. One noninteractive SSH,
one submission and four `qstat` calls were used, with no retry.

The controller stopped with `link dependency file must be a regular non-symlink`
after 13 successful `ldd` commands: the twelve CPU artifacts and the first HIP
artifact, `blas_gpu_bench`. The native-HIP build retained `link.txt` and its
successful verbose link command but did not generate `link.d`; the helper
incorrectly required that depfile for every profile. All twelve HIP targets
have this evidence shape. CPU and OpenMP depfiles are present. The local fake
tests had always supplied a depfile and missed this difference.

No teaching or benchmark artifact executed: teaching **0/18**, diagnostic
rows **0/144**, main smoke rows **0/288**. No GPU program or rebuild ran.
This is a smoke-harness failure, not an observed numerical failure or an
authentication error. The preceding compile/link success remains valid at its
original source identity; runtime correctness remains unverified.

The read-only review checked 49 existing evidence files, command-stream hashes,
and terminal proof consistency. Original bytes were preserved. Private evidence
is under the run's `smoke/summary.json`, `dispatch-result.json`,
`scheduler-logs/` and `local-ssh-evidence/`. The summary SHA-256 is
`b6ac5603e3052d43f294ac0fefccbf4805ace239a7fb23bd8e8396de63ceb662`;
the review SHA-256 is
`f96927d72a226113c03c31e48b9af72d8e377f17ab7fcc8c6abfd72a100162b7`.
Subsequent local repair must retain this failed run and require a new approval
before another allocated smoke attempt.

The local repair adds explicit native-HIP command evidence, retaining CPU and
OpenMP depfile checks and per-child library identity checks. A fresh read-only
source proof verified all 36 retained artifacts and 265 evidence files:
24 CPU/OpenMP depfiles and 12 HIP commands bound to their successful target logs.
This check uses the existing mounted files and the original source snapshot;
it does not execute a binary or inspect Sirius's live external library files.
The selected library paths and hashes are rechecked in the allocated job before
any child command. No new real smoke result is claimed by the local repair.

### AMD OpenMP temporary-object evidence failure

The next approved attempt, `sirius-amd-smoke-20261017T115800Z-a3697cae`,
submitted job `45744.pbs`. PBS confirmed `F`, `Exit_status=1`, one execution
and no rerun. The dispatcher returned `SMOKE_FAILED` (SSH exit 2), with one
SSH process, one submission and five `qstat` calls. The expected completion
marker uniquely recorded exit 1. No retry occurred.

This attempt stopped in the all-artifact link preflight with `commands=[]`:
teaching **0/18**, diagnostic **0/144**, main rows **0/288**, no `ldd`, no
benchmark/GPU artifact execution and no rebuild. OpenMP's successful build
depfile lists a temporary image-wrapper `.o` that no longer exists. The parser
incorrectly required every absolute prerequisite to remain present. The saved
build log still proves successful linking; the failure does not demonstrate an
AMD numerical defect. The source-only local proof had checked depfile bytes
and hashes, not the live prerequisites, and did not catch this assumption.

The read-only review verified 32 payload/evidence/log identities. Original
records remain under that run's `dispatch-result.json`, `smoke/summary.json`,
`scheduler-logs/` and `local-ssh-evidence/`. The summary SHA-256 is
`a9c4e9517317238bfd4344aaec520a140a5e2906845fe3eef8f4c271edb33e41`;
the review SHA-256 is
`1a2eeb0676a65f2a0acda84a314ea58d072adbb81d40a245e6e3a45b9fbd92d8`.

The local correction separates historical object tokens from current
library/archive/script evidence, retaining strict configured-library and
loaded-library hash checks. Local fake files and replayed depfile syntax do
not establish live external dependencies or actual loader-trace compatibility.
All six-family numerical results remain pending a new approved allocated run.

### AMD loader-finalization evidence failure

Run `sirius-amd-smoke-20261017T125733Z-a3697cae`, PBS job `45759.pbs`,
finished in state `F` with exit 1, one execution and reruns disabled. The
dispatcher reported `SMOKE_FAILED`; SSH exited 2. One SSH process, one
submission and six `qstat` calls were used without retry. Both completion
markers agreed with the body and scheduler exit.

All 36 retained artifacts passed link-evidence preflight and `ldd` resolution.
The first teaching example, CPU BLAS, exited zero and printed maximum error
zero. Its subsequent trace check failed with
`incomplete loader lifecycle or object finalization`. The trace contains five
DSO initialization events but no corresponding finalization events for the
loader and libc; program initialization, control transfer and main finalization
are present. Requiring every initialized DSO to emit finalization was an
incorrect harness assumption. This is not evidence of numerical failure or
proof that the log was truncated. Upstream glibc has implementations which
emit finalization messages only when destructor entries exist; the saved log
does not itself establish Sirius's patched libc source or ELF destructor tags.

Teaching execution was **1/18**, with that example's loader-evidence validation
incomplete. The remaining 17 examples, diagnostic **0/144** and main **0/288**
rows were unexecuted; no benchmark or GPU artifact ran and no rebuild occurred.
The read-only review verified 155 evidence identities and preserved the
preceding run's 32 recorded identities. Private evidence remains in the run's
`smoke/`, `scheduler-logs/` and `local-ssh-evidence/` directories. The summary
SHA-256 is
`977df6e31292256860a14b62ffb150a895b4d931814c8d17cf57bc5e2694ffa0`;
the review SHA-256 is
`4e36edb45d393e3a94fff9e38d1faec0ce0ec456ad313b17fba5341616d6db0b`.

The local correction records unmatched initialization events without inventing
finalizers or excluding named system libraries. It retains the separate owned
child exit/timeout gate, numerical gate and exact binary/provider/link checks.
Local trace replay and fake files cannot establish six-family GPU correctness,
all future trace grammar or CPU worker counts. No failed run is rewritten and
the runtime gate remains open.

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

The historical campaign below predates the current NVIDIA Fortran timing
alignment. Its numerical passes remain valid for its recorded binaries; its
timings do not validate the revised measurement path. The revised source requires
a new Pegasus build of all 18 benchmarks, numerical smoke and performance
measurement before replacement figures can be published. No historical raw
record or figure is rewritten by this source change.

Local checks cover the shared driver's callback/timer sequence, GPU-branch
preprocessing and CPU calls against controlled FFTW/oneMKL fixtures, plus the
intrinsic RNG/reduction paths. These checks do not establish NVHPC compilation,
CUDA/OpenACC execution or GPU performance. The revised Fortran build settings
now follow the published C/C++ profiles, including the explicit NVIDIA OpenACC
`-fast -O3` exception and nvc++ Thrust wrapper. They still require Pegasus
build/smoke and new Fortran measurements; see
[the Fortran timing contract](BENCHMARK_PROTOCOL.md#fortran-binding-of-this-protocol).
The C/C++ and AMD sources, build policies and saved results are unchanged.

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
  is library, right panel end-to-end, and vertical axis elapsed time in milliseconds;
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

## NVIDIA coexistence review after AMD addition

### Local and fixture evidence

The local review compared the AMD changes with the existing NVIDIA C/C++ and
Fortran paths: CMake vendor/language selection, common CLI/result handling,
shared FFT code, configuration, manifests, scheduling, validation, aggregation
and plot labels. NVIDIA's default profile and metadata format are retained;
the existing benchmark/implementation enum values used by the Fortran bridge
are unchanged. The shared FFT retains NVIDIA's original compute timing branch.
No regression requiring a code fix was found within this review's scope.

Validation was deliberately limited to six existing Python tests: optional and
strict language configuration, language-separated aggregation, Fortran intrinsic
RNG identity, six-family Fortran plot labels, deterministic runner dry-run, and
rejection of mismatched effective CPU thread counts. All six passed. Two fresh
Mac-local CPU build trees then built only NVIDIA C/C++ `fft_cpu_bench` using
the existing synthetic FFTW provider and NVIDIA Fortran `rand_cpu_bench` using
GNU Fortran. FFT serial/compute, FFT threaded/end-to-end and Fortran RNG each
produced one schema-valid, numerically passing record. Both metadata documents
retained the NVIDIA schema, and neither compile inventory selected AMD/HIP
sources. The synthetic FFTW check is not evidence about a real FFTW installation.

No full local suite, CUDA/NVHPC build, GPU execution, SSH, scheduler operation
or Git publication was performed. CUDA/OpenACC C/C++ and Fortran execution on
Pegasus was still pending at this local-review stage; this local review
does not revalidate those GPU binaries. Existing AMD/Sirius and NVIDIA/Pegasus
run records were preserved unchanged.

The focused follow-up reused existing regression tests instead of rerunning
the full suite. Ten Python tests passed for the legacy example manifest,
JSONL/CSV records and CSV header, exact pilot/benchmark record matrices,
wave-level aggregation, configured primary CPU pairing, six-family plot series,
seconds-to-milliseconds rendering, and Thrust version mismatch rejection.
CTest's selected inventory was checked to contain exactly eight tests before
execution; all eight passed with no skips: CLI and C++ compatibility, plus the
six-test oneMKL fixture configure/build/manifest/BLAS/Sparse/Solver chain.

The existing GNU Fortran controlled-provider check also passed. It built the
six CPU benchmark targets, exercised FFT/BLAS/Sparse/Solver through the shared
C++ bridge, and retained the existing thread, nonfinite/error and dependency
checks. These providers are synthetic, not real oneMKL/FFTW performance or
thread-count evidence. The C/C++ fixture manifest contained all ten expected
targets (FFTW was deliberately disabled in that fixture); the Fortran manifest
contained all six expected benchmarks. Both fresh configurations used ordinary
local compilers without ROCm/AOCL environment hints. Their NVIDIA/Fortran
production compile commands contained no AMD compiler/library paths or
AOCL-specific POSIX definition. No compiler was changed within an existing tree.

The unchanged Fortran Thrust sources and CMake path were explicitly reviewed:
the canonical `thrust_wrapper.cu` remains an input to both GPU teaching and
benchmark targets, with matching `bind(C)`/`extern "C"` entry points and explicit
C++ runtime linking. This is source-level evidence only; the NVHPC/NVCC mixed
link and CUDA/OpenACC GPU smoke were pending at this source-review stage and
were subsequently checked below. No new regression tests or implementation
changes were needed for the local review.

The coexistence checks do not resolve the AMD CPU parallelism or loaded-library
provenance limitations: RAND/Thrust CPU backends remain sequential;
BLAS/Sparse/Solver effective thread counts and complete actual loaded-library
identity remain unverified. No diagnostic campaign or performance rerun was
added, and no additional remote operation was performed.

### Pegasus build and numerical smoke after AMD addition

Run `pegasus-nvidia-coexistence-20261017T165103Z-eb9e594` used the Git-transferred,
clean commit `eb9e594be3ea4f1dc4bef6fb562223961c60a43a`. Build job `135494.nqsv`
and the separately approved smoke job `135525.nqsv` both completed with exit 0.
Each submission used one local-Shell noninteractive SSH and one `qsub`; neither
performed a queue query or automatic retry. Source identity was checked before
and after execution. No source copy replaced the Git checkout.

The four independent build trees were C/C++ CPU+CUDA, C/C++ OpenACC,
Fortran CPU+CUDA, and Fortran OpenACC. All configure/build steps passed;
the manifests contain 18 C/C++ and 18 Fortran benchmarks, with six libraries
per CPU/CUDA/OpenACC implementation. Real FFTW pthread and oneMKL CPU providers
were used. Fortran CUDA `reduce_gpu_bench` and OpenACC `openacc_thrust_bench`
both explicitly compiled `thrust_wrapper.cu` and linked its object with the
Fortran objects. CMake unused-variable, NVHPC unreachable-statement and
`pgacclnk` ignored-`-lcudart_static` warnings were retained; no link failed.

Smoke reused those 36 binaries without rebuilding. Each table entry covers
CPU/CUDA/OpenACC in both compute and end-to-end scopes, with warm-up/repeat/trials
of 1/1/1. The result is numerical verification at small sizes, not a performance
campaign or a repetition of the historical NVIDIA measurements.

| Library | C/C++ cases | Fortran cases |
| --- | --- | --- |
| FFT | 6/6 PASS | 6/6 PASS |
| BLAS | 6/6 PASS | 6/6 PASS |
| Sparse | 6/6 PASS | 6/6 PASS |
| Solver | 6/6 PASS | 6/6 PASS |
| RAND | 6/6 PASS | 6/6 PASS |
| Thrust | 6/6 PASS | 6/6 PASS |
| Total | 36/36 PASS | 36/36 PASS |

All 72 child exits were zero, case stderr files were empty, numerical metrics
passed the configured tolerances, and case identity/schema checks passed.
No case failed or remained unexecuted. Each case's binary hash matched the
successful build receipt; recorded stdout/stderr hashes also matched the files.

Before submission, a variable-name collision in the one-off smoke helper was
fixed in new `smoke-launch/check_smoke.py` and `smoke-launch/smoke.pbs` copies.
The verification-option loop had overwritten the case/log name. The narrow
local check confirmed 72 unique names and unchanged executable arguments;
shell syntax passed. Original scripts, build artifacts and older logs were
preserved. This launcher correction did not change the tested source commit.

Private evidence is on the **Pegasus** checkout under `runs/<RUN_ID>/`:
`build/verified.json`, per-profile manifests/build metadata and Fortran compile/
link commands; `smoke/summary.json` and 72 case JSON/stdout/stderr sets;
`smoke-job/exit-code.txt`, scheduler output, `submission/`, `smoke-submission/`
and `smoke-launch/launcher-fix.patch`. The Sirius checkout is a different
directory even when its remote path has the same spelling. Key SHA-256 values:

| Evidence | SHA-256 |
| --- | --- |
| `build/verified.json` | `0d6fdfccf03a50b3fa092065a722225d2dcf22017f8579a53bb1cd4449cf1ea5` |
| `smoke/summary.json` | `f3a3598726e18630b7642c13bf295e358814725ed07c06cf5be08db191224c68` |
| `smoke-launch/check_smoke.py` | `352f7da9766f4ad41bc6ba8935c0a6bbba64472a455abf35af67a2824e9d984c` |
| `smoke-launch/smoke.pbs` | `83e557e9e81b82a7d51cbd63eb4c67ef3d1f0afeaa03aea6a4d94041b2c72390` |

### Limits retained after coexistence acceptance

- AMD coverage still refers to its recorded snapshots, not to a new Sirius
  execution of the Pegasus-tested commit. Later source changes need their own
  scoped evidence; different snapshots/campaigns are never pooled.
- This Pegasus increment built and ran benchmarks. NVIDIA teaching examples
  were not rebuilt or rerun here; their older evidence remains historical.
- CPU RAND/Thrust backends are serial, including Fortran intrinsic RNG/sum.
  AOCL/oneMKL BLAS/Sparse/Solver effective worker counts remain unobserved;
  requested thread counts do not establish parallel scaling.
- Configured/resolved dependency evidence and binary hashes do not establish
  complete actually loaded-library identity. The earlier AMD diagnostic
  campaign remains incomplete; numerical success does not relabel it PASS.
- Production-size behavior, performance calibration, complete production
  campaign provenance and new performance figures were not validated by these
  small smoke jobs. Existing NVIDIA production data retain their own provenance.

No profiling or additional diagnostic campaign is required merely to record
this coexistence result. The next bounded work is planned in the
[implementation plan](IMPLEMENTATION_PLAN.md#minimal-performance-evaluation-plan);
new real execution and Git publication still require separate approval.

<a id="amd-production-comparison"></a>

## AMD本評価：BLIS修正後の比較と測定終了

2026-10-18の再監査では、採用データについて追加の再測定を必要とする
重大な不整合は見つかりませんでした。これは保存済みコード・生成済みbuild条件・
ログ・raw結果の照合であり、新しいGPU実行や完全な障害原因の証明ではありません。
測定を終了し、下記の6,120行をAMD CPU/HIP/OpenMP targetの比較に採用します。

### 採用元と保持する履歴

| 採用分野 | campaign | 採用行数・標本 |
| --- | --- | --- |
| FFT / Sparse / RAND / Thrust | `sirius-amd-production-20261018T085456Z-e00328df` | 3,960行。wave 0の5ノード＋wave 1の6ノード、各条件5 trial |
| BLAS / Solver | `sirius-amd-blis-20261018T124826Z-1798941b` | 2,160行。各wave 6ノード×2 wave、各条件5 trial |

前者の成果物commitは `e00328df028b8ff75b9275593a14bbf62f868ecd`、後者は
`036387542e6e2db95c1193eb38efaa577ee3bb5f` です。この間のAMD benchmark・common・
CMake依存探索・共通readerのソース差分はありません。後者ではサイトのbuild入力を
修正し、CPU BLAS/SolverをAOCLの並列 `libblis-mt.so.5.1.0` にリンクし直しました。
CPUだけを旧GPU時間と組み合わせず、HIP/OpenMPも同じ新campaignで再測定しています。

修正後のbuildはPBS `45912.pbs`（18成果物、runtime収集、00:05:37）、
wave 0/1は `45913.pbs`（00:24:01）/ `45914.pbs`（00:23:44）です。
後者の2 waveは各1,080行すべて数値PASS。両campaignの物理ホストは
a24/a01/a19/a23/a14/a15で重複しており、全体で6物理ホストです。
各ホストの1 APUと対応する24 CPUを使用しています。

旧datasetの5,940行と旧BLAS/Solver結果は削除・改変していません。旧wave 0とa01の
`FAILED`、5ノード採用記録、a01の部分成功5行・未実施分もそのまま保持しています。
a01の不完全block全体を除外し、欠落を補完していません。後のwave 1では同じa01が
成功しましたが、原因の確定やハードウェア修復の証明とは扱いません。

### 再監査した範囲

- 旧5,940行＋修正後2,160行、計8,100行を既存 `validate_results.py` で再検証。
  全行success/PASS、重複・不足・数学条件・manifestとの対応の検査を通過しました。
  採用対象はそのうち6,120行です。
- config/manifest/runtime、採用18 binaryとbuild metadataのhashを再照合。
  23個の採用node/wave blockの完了・CPU affinity・NUMA/APU証拠・runtime一致も確認。
  旧Sparseは並列BLISのみ、旧FFTはOpenMP FFTWを解決しており、今回の誤リンクの
  横展開は見つかりませんでした。RAND/ThrustのCPUは意図した逐次backendです。
- 18 targetの生成済みcompile flagsを確認し、追加のfast-math指定はありません。
  CPUはGNU 14.2.1、HIP/OpenMPはClang 22.0.0、保存runtimeはROCm 7.2.0・AOCL 5.1.0。
- 六分野の計測区間・GPU完了同期・warm-up後とtrial間の入力復元・検証処理を確認。
  主時計は共通 `CLOCK_MONOTONIC`、時間は正の有限値で、総時間/repeatと1操作時間が一致。
  FFTの復元を除く区間積算、Solverのrepeat=1、RANDの再初期化と逐次CPUの区別も維持。
- NVIDIAの保存configと一致する現行 `configs/benchmark.json`（SHA-256
  `8afd122b965f65400c94aaa3a3baa511e6d006908d6747bd6d21bdcebff461e9`）に対し、
  六分野のサイズ・精度・warm-up/repeat/trials・検証閾値が一致。
  CPU要求数はAMD 24 / NVIDIA 48であり、同一CPU条件とは称しません。
- `aggregate.py --no-pooled` をcampaignごとに一度ずつ再実行。
  以前の集計とバイト単位で一致し、rawから独立に算出した階層中央値とも一致しました。
  CPU/GPUの速度比は同一node/wave block内で対応付けた値から集計します。

集計はtrial中央値 → wave内ノード中央値 → 2 wave間中央値です。旧4分野は
5/6 blockの不均等を明示したうえで各waveを同じ重みで扱います。2つのcampaignの
rawを単一campaignへ混ぜず、4分野と2分野を別々に集計・描画します。
ノードが遅いことを理由に除外していません。例えば新Solver/OpenMPの最大library条件では、
wave 0/a15のtrial中央値219.119 msも採用しています。
node/waveのQ1/Q3/IQR/min/maxは集計JSONLに保存し、2 wave間のばらつきを
12独立ノードの分散とは解釈しません。

### 比較値

最大サイズでの1操作当たり時間（階層中央値、ms）を示します。速度比は
対応付きblockの比を階層集計した値で、表示時間同士の単純な割算ではありません。
全3サイズ・両scopeの数値と四分位点は下記の成果物に保存しています。
表・図の表示名はlibrary／end-to-endです。内部scope値の`compute`／`end-to-end`との
対応は[表示用語の規定](BENCHMARK_PROTOCOL.md#publication-figures)に従い、保存データの値は変更しません。

| 分野・最大サイズ | 表示区間 | CPU ms | HIP ms | OpenMP target ms | CPU/HIP | CPU/OpenMP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FFT・16,384 × batch 4,096 | library | 11.777 | 0.641 | 0.676 | 18.38× | 17.84× |
| FFT・16,384 × batch 4,096 | end-to-end | 12.333 | 86.984 | 79.352 | 0.14× | 0.16× |
| BLAS・4,096 | library | 110.398 | 1.480 | 1.472 | 74.64× | 74.99× |
| BLAS・4,096 | end-to-end | 110.674 | 29.932 | 26.900 | 3.70× | 4.12× |
| Sparse・4,194,304 | library | 10.670 | 0.113 | 0.113 | 94.71× | 94.75× |
| Sparse・4,194,304 | end-to-end | 139.950 | 42.632 | 46.574 | 3.28× | 3.00× |
| Solver・12,288 | library | 2013.045 | 157.883 | 157.821 | 12.75× | 12.71× |
| Solver・12,288 | end-to-end | 2000.595 | 232.428 | 213.845 | 8.62× | 9.38× |
| RAND・67,108,864 | library | 487.224 | 0.176 | 0.177 | 2770.96× | 2753.75× |
| RAND・67,108,864 | end-to-end | 477.423 | 30.538 | 21.430 | 15.65× | 22.31× |
| Thrust・67,108,864 | library | 22.499 | 0.148 | 0.150 | 151.62× | 150.39× |
| Thrust・67,108,864 | end-to-end | 22.326 | 26.748 | 18.566 | 0.83× | 1.20× |

libraryではGPUが速くても、allocation・データ移動・結果回収を含むend-to-endでは
逆転し得ます。RANDは異なる乱数生成器同士の同じ分布生成タスクであり、
RAND/Thrustの逐次CPUとの速度比を24-worker CPUとの比較と読み替えないでください。

### 成果物と図の状態

今回の新しい報告ディレクトリは
`runs/amd-cpu-gpu-comparison-20261018T142831Z/` です。
このディレクトリと下記の図は非配布の保存済み成果物で、Git checkoutには含まれません。
`selection.json` が採用campaignと分野を、`audit.json` が検査結果と実行コマンドを記録します。
`retained/` と `corrected/` に検証結果、集計JSONL、表示条件、生成済みplot metadataを
分けて保存しました。`selected-summary.json` は出典を保持した表示用の一覧であり、
新しい測定campaignではありません。

2026-10-19、所有者承認によりMacローカルの独立したPython 3.12.15環境へ
matplotlib 3.11.2を導入し、NVIDIAと同じ `tools/plot.py` / `gpu_suite.plotting` を
変更せず使用して、6図を生成しました。システムPythonは変更していません。
1800×720、library／end-to-endの2パネル、横軸log2・K/M表記、縦軸ms、丸印・線・
3列の共通凡例・配色・余白を維持しています。既存の描画呼出しfixture 1件のPASSに
加え、実PNG全6枚の形式・寸法を確認し、全図を目視して文字切れや重なりがないことを
確認しました。再測定・再集計は行っていません。

| 分野 | 比較図（上記ディレクトリからの相対パス） |
| --- | --- |
| FFT | `figures/rocfft-elapsed-time.png` |
| BLAS | `figures/rocblas-elapsed-time.png` |
| Sparse | `figures/rocsparse-elapsed-time.png` |
| Solver | `figures/rocsolver-elapsed-time.png` |
| RAND | `figures/rocrand-elapsed-time.png` |
| Thrust | `figures/rocthrust-elapsed-time.png` |

`rendering.json` に描画コマンド・終了値・PNGのSHA-256を、
`render-completion.json` に目視確認・依存パッケージ版・入力保持の確認を記録しました。
以前の依存不足の記録は履歴として保持し、現在の描画完了状態は後者を参照します。

CPU凡例の `24 C` は要求したCPU数で、実稼働worker数ではありません。
`AMD MI300A CPU` はAPU構成を示す表示名であり、未収集のCPUID名称を観測値として
補ったものではありません。RAND/Thrustは `single thread` と表記します。

### 残す限界と終了範囲

FFTのAMD/NVIDIA計測方式の差は[既承認の方針](DECISIONS.md)どおり維持します。
FFTWの設定thread数、BLAS/Sparse/Solverの要求値は実稼働worker数の証明ではなく、
link/ldd/hash証拠も実行中の全動的ロードを追跡した証明ではありません。
`observed_cpu_worker_count=null` / `loaded_libraries_verified=false` を維持し、
これらを解決済みとせず、追加のworker監視・profiling・DSO追跡を要求しません。

今回の処理は既存ログの読取り、集計、ローカル描画環境の導入、描画、文書更新です。追加測定・実SSH・
scheduler操作・再ビルド・commit・pushは0回。NVIDIA/Pegasusの新しい実機確認も
実施しておらず、この文書・後処理変更のために追加する必要はありません。

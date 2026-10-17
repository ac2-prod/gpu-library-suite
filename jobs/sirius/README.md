# Sirius: approved execution route and AMD validation

This document owns Sirius operating rules. AMD teaching and mathematical rules
belong to [PROJECT_SPECIFICATION](../../docs/PROJECT_SPECIFICATION.md); staged
acceptance belongs to [IMPLEMENTATION_PLAN](../../docs/IMPLEMENTATION_PLAN.md).

## Current status and execution route

The rocFFT baseline completed normally in the recorded run
`sirius-rocfft-smoke-20261017T062713Z-b29643fc`: all three fresh builds,
teaching examples, and 96 serial/threaded CPU/HIP/OpenMP smoke records passed.
See [the validation evidence](../../docs/VALIDATION_REPORT.md#amd-rocfft-sirius-baseline).
Earlier build-only runs and authentication failures remain historical evidence;
their artifacts must not be combined with the successful snapshot.

The later six-family build
`sirius-amd-build-20261017T104351Z-a3697cae` (PBS `45731.pbs`) completed with
all 36 artifacts verified: CPU 12, HIP 12 and OpenMP 12, including the explicit
AOCL Sparse CPU pair. PBS and SSH exits were zero. This is compile/link
evidence only; no produced program or numerical smoke ran in that build job.
Subsequent focused smoke jobs below verified all five remaining families.
Complete loaded-DSO identity, parallel worker behavior, final NVIDIA coexistence
review and performance remain open. See
[the complete build evidence](../../docs/VALIDATION_REPORT.md#amd-six-family-build-with-aocl-sparse)
for its exact source snapshot, hashes and operation counts.

The owner's current execution policy supersedes earlier terminal/login-build
instructions retained in the history below:

- Use the local Shell tool, never Computer Use terminal interaction.
- Remote commands use noninteractive SSH to the explicitly approved host.
  Authentication failure stops the operation; do not request or capture secrets.
- Login-node operations are limited to file/lightweight environment checks,
  approved submission/queue observation, and log collection.
- Computational builds, tests, benchmarks and GPU execution run through PBS.
  The historical login-server build-only workflow is no longer the current route.
- Report SSH argv, stdout, stderr, exit code and the job ID after submission.
- Deletion, overwriting run artifacts, cancellation and bulk resubmission require
  prior approval. Each new execution uses a new `runs/<RUN_ID>`; retain old runs.
- Current work is local AMD implementation and validation. No additional SSH,
  scheduler execution or performance campaign is authorized by that work.

### Prepared six-family allocated build

[amd_build_job.sh](amd_build_job.sh) and [amd_build.py](amd_build.py) support the
allocated build-only check of rocFFT, rocBLAS, rocSPARSE, rocSOLVER, rocRAND and
rocThrust. They require an explicit build approval flag and a PBS job identity;
they do not submit a job or open SSH. The job uses a fresh, canonical
`SOURCE_ROOT/runs/<RUN_ID>` and a reviewed source-snapshot hash. Its saved
controller and module helper must match the hashes in the prepared job inputs.
The unchanged [rocfft_compute_modules.sh](rocfft_compute_modules.sh) supplies
the previously successful module setup inside the allocation, including the
GCC runtime search-order correction. Its historical `STAGE31_*` inputs are
mapped explicitly by the new job body; this does not invoke the rocFFT smoke.

CPU, native HIP and OpenMP use three separate build trees. The architecture is
an explicit reviewed input, including the HIP wrapper used by OpenMP
rocThrust; no GPU architecture discovery is performed. Native HIP uses the
AMD Clang invocation path, while the CPU profile retains its GCC invocation
paths. The requested inventory is 36 artifacts: six libraries, three execution
models, and one teaching plus one benchmark artifact for each combination.
Build parallelism is one. Command timeouts also share a bounded overall
deadline. Each artifact target is attempted separately, at most once; an
ordinary failure does not suppress later independent targets. Shared benchmark
support and the OpenMP rocThrust wrapper each have one explicit build attempt
per profile before their consumers; a failed shared dependency blocks its
consumers without retrying it.
The build timeout is shared by the profile, not reset for every target.

The controller records source hashes before and after, compiler identities and
versions, effective search paths, configure/build argv and stdout/stderr/exit,
CMake cache and compile commands, build metadata, manifests and artifact
hashes. Per-target records distinguish dependency exclusion, an unattempted
target, compile/link failure, and a compiled artifact whose provenance could
not be verified. A successful compile alone is not `BUILT`: only the successful
subset is passed to the manifest generator, with metadata, descriptor, binary
and source checks. Manifest finalization does not invoke another CMake build or
retry failed targets. None of the incomplete outcomes constitutes a completed
36-artifact build. It never executes a produced
program, CTest, a benchmark, `ldd`, or a GPU query. Review the build results
before proposing a separately approved smoke job.

The first allocated six-family build verified all 24 HIP/OpenMP artifacts.
CPU configure found AOCL BLIS/CBLAS, libFLAME/LAPACKE and OpenMP FFTW and passed
their API link probes. The CPU BLAS benchmark then failed on POSIX thread types
in the AOCL header; the old grouped build did not reach later CPU targets.
The selected oneMKL Sparse provider was not found in the configured search
environment. See the
[recorded build evidence](../../docs/VALIDATION_REPORT.md#amd-six-family-sirius-build).
The local recovery establishes POSIX.1-2008 before all includes in the AMD
CBLAS/LAPACKE teaching and benchmark sources and their CMake probes; higher
caller feature levels are retained and lower levels are rejected. The subsequent
build below confirms compilation; numerical smoke remains a separate gate.

That recovery was subsequently built in retained run
`sirius-amd-build-20261017T095316Z-249f96a5` (PBS `45730.pbs`). It verified
34 artifacts: CPU 10, HIP 12 and OpenMP 12; the two oneMKL Sparse CPU targets
remained `NOT_BUILT_DEPENDENCY`. There were no failed compile/link targets and
no generated-program executions. The job exit was 1 and controller SSH exit 2
because 34/36 is incomplete. The old failure record is retained unchanged;
see [validation evidence](../../docs/VALIDATION_REPORT.md#amd-cpu-recovery-and-aocl-sparse-follow-up).

The saved inventory contains
`/system/apps/rhel/9.6-202602/aocl/5.1.0/gcc/include_LP64/aoclsparse.h`
(SHA-256 `fa381654c3671dc3cba29e36488dd4de10eee254d540664b4f25c50b79a2337a`)
and `lib_LP64/libaoclsparse.so.5.1.0` under the same prefix
(SHA-256 `a02def7a39a5ad3d1873fc2e4c6f48e60a363b505636190882c847cc0638425b`).
Their presence is supply evidence. The subsequent 36-artifact run verified
the matching header/library identities and the AOCL Sparse LP64/API
compile/link contract. The focused sparse smoke below subsequently verified its
basic numerical operation; effective worker count remains unverified.
The successful build required no further supply search or private installation.

The allocated build records a bounded inventory of relevant files under the
loaded AOCL prefix and any explicit oneMKL prefix. The prepared job explicitly requests
`GENERIC_CBLAS`, `GENERIC_LAPACKE` and `--aocl-cpu-candidates`. This opt-in rule
only considers `cblas.h`/`lapacke.h` in `include`/`include_LP64`, `libblis.so`
for CBLAS, and `libflame.so` or `liblapacke.so` followed by the selected BLIS
library for LAPACKE, in `lib`/`lib_LP64` under the loaded `AOCL_ROOT`.
These remain selection rules, not a guarantee of supply in a future module environment.
Aliases of the same resolved file count once. Distinct candidates, ILP64
paths, escaping paths or missing inputs disable the affected CPU pair with a
recorded reason; they do not trigger another provider or reference fallback.
The controller saves candidate paths, resolved identities, ordered link inputs
and selection reasons. The existing CMake API compile/link checks must still
pass before targets are enabled. An unmarked path is not proof of LP64 ABI.
API/ABI behavior, numerical correctness, runtime bindings and threading remain
smoke requirements even when compile/link succeeds.
There is no recursive system search or execution of candidate libraries.
Manual explicit generic CBLAS/LAPACKE selection and ordered link inputs follow
[BENCHMARK_PROTOCOL](../../docs/BENCHMARK_PROTOCOL.md); their identities follow
[RESULT_SCHEMA](../../docs/RESULT_SCHEMA.md). AOCL Sparse is not treated as
oneMKL-compatible, and a missing production sparse backend is not replaced by
the reference backend.

The AOCL Sparse follow-up is opt-in: use `AMD_BUILD_CPU_SPARSE_BACKEND=AOCLSPARSE`
in a newly reviewed job packet, forwarded as `--cpu-sparse-backend AOCLSPARSE`.
ONEMKL remains the default for existing packets. With `--aocl-cpu-candidates`,
the controller additionally selects `aoclsparse.h` and `libaoclsparse.so` in the
same bounded include/library locations only for this explicit backend. Missing,
distinct, escaping or ILP64 candidates fail that pair without choosing another
provider. A manual selection uses the AOCL cache inputs defined in
[BENCHMARK_PROTOCOL](../../docs/BENCHMARK_PROTOCOL.md); do not combine it with
candidate mode. Header identity and the complete ordered library interface must
match build metadata before accepting an artifact. New source hashes and a new
run ID are required; the historical 34-artifact build does not validate this
subsequent implementation.

For later smoke preparation, [amd-smoke.json](amd-smoke.json) and
[amd-threaded-smoke.json](amd-threaded-smoke.json) explicitly choose generic
CBLAS, generic LAPACKE and AOCL Sparse, with one and two requested CPU threads.
Their FFT series are respectively serial and threaded and must use matching
artifacts and distinct campaigns. RAND/STL remain serial. All six CPU/HIP/OpenMP
series are retained; a missing CPU Sparse artifact cannot silently remove a
primary comparison. These configs are input preparation, not submitted jobs or
runtime evidence. Loaded-DSO bindings, all-family numerical smoke and actual
thread behavior still need a separately approved allocated run. Do not claim
configured library hashes alone prove loaded DSO identity.

These files and local mock results do not authorize remote execution. Before
use, review the complete job packet, source and helper hashes, new run ID,
allocation and walltime, explicit CPU provider choices, one-submission limit,
SSH command and observation budget. The completed build used its own explicit
approval: one SSH, one submission and 21 `qstat` calls, with no retry. That
approval does not authorize smoke, another build or another remote operation.

### Focused numerical smoke

BLAS completed on Sirius in `sirius-blas-smoke-20261017T135452Z-a3697cae`
(PBS `45771.pbs`, exit 0): three teaching examples and 96 numerical rows passed,
all with zero absolute error. This establishes basic numerical operation;
effective CBLAS worker count and performance remain unmeasured.

The remaining four families completed in
`sirius-amd4-smoke-20261017T141232Z-a3697cae` (PBS `45772.pbs`, state `F`,
exit 0) on Sirius `a10` / AMD Instinct MI300A. All 12 teaching examples and
384 numerical records passed: 96 each for rocSPARSE, rocSOLVER, rocRAND and
rocThrust, covering CPU, HIP and OpenMP target. Sparse and Thrust errors were
zero; Solver's maximum relative residual was `4.440892098500626e-16` and
maximum relative solution error was `1.7763568394002505e-15`; RAND passed its
range and moment checks. These are basic numerical checks, not performance results.

The 24 artifacts came unchanged from the successful six-family build, source
snapshot `a3697cae2e446a51eec5a6107dedc39caf656fc8158b08e3591c59f0bc1148e6`.
This evidence does not validate later source changes. One noninteractive SSH,
one submission and 15 bounded queue observations were used, without retries,
rebuilds or profiling. SSH exit was 0 and its stderr was empty. The job's saved
stderr retains module listings, shell job-control warnings and OpenMP ordinal
messages; every teaching/benchmark child exited 0. The complete result and
separate command logs are retained privately in the run's
`local-ssh-evidence/execution-review.json` and surrounding directory; see the
[source-visible evidence summary](../../docs/VALIDATION_REPORT.md#amd-focused-numerical-smoke).
Together with the recorded FFT and BLAS
results, all six families now have Sirius build and basic numerical smoke evidence.

The same numerical path can select an explicit `libraries` array. The remaining
set is rocSPARSE, rocSOLVER, rocRAND and rocThrust: 24 prebuilt artifacts,
12 teaching examples and 384 numerical rows in one allocated job. Each family
has two small cases, both scopes and two trials in the four JSONL/CSV campaigns.
The second sizes are respectively 100, 32, 8192 and 8192. Sparse uses the explicit
AOCL Sparse backend, Solver the built generic LAPACKE provider; their effective
CPU worker counts remain unknown. The standard-library CPU RAND and Thrust
backends remain serial, including when the campaign requests two threads.
No FFT or BLAS executable is selected in this remaining-family run.
Local preparation checks only changed syntax and the selected campaign matrix;
the authoritative execution check is the approved Sirius job. No full local
regression suite is required for this selection-only change.

The successful BLAS run used the six CPU/HIP/OpenMP teaching
and benchmark artifacts from `sirius-amd-build-20261017T104351Z-a3697cae`.
An explicitly approved packet selects `library: "rocblas"` and
`purpose: "numerical-smoke"`. It runs three teaching examples and four separate
campaigns: CPU thread requests 1/2 crossed with JSONL/CSV. Each campaign contains
CPU, HIP and OpenMP target; DGEMM sizes 16 and 31;
compute/end-to-end; warm-up 1, repeat 2, trials 2. That is **24 rows per campaign,
96 total**, the same scale as the completed FFT smoke. The size-31 case uses
alpha=1.25 and beta=0.5. No other library's executable is run; no build is done.

This path uses ordinary child execution, without `LD_DEBUG`, profiler tools or
extra diagnostic campaigns. Exit status, all expected numerical records and
their thresholds determine numerical success. Missing, skipped or failed rows
cannot pass. stdout/stderr remain separate and unchanged. The old six-family
diagnostic path below is retained for its saved records and is not selected.

Existing source/manifest/compiler/binary/config/environment checks and the six
BLAS artifacts' `ldd` resolution records remain. They are not proof of every DSO
loaded dynamically: `loaded_libraries_verified` remains false; no loader trace
is fabricated. CPU uses the built AOCL CBLAS provider with `BLIS_NUM_THREADS`
and `OMP_NUM_THREADS` set to the requested count. Effective CPU worker count
remains unknown (`cpu_threads_effective: null`); numerical success alone does
not prove parallel execution or speedup. See the owning
[result schema](../../docs/RESULT_SCHEMA.md).

A new run ID and separate approval are required before SSH or PBS submission.
All outputs stay under that run's `runs/<RUN_ID>/smoke`; old runs are preserved.

### Retained six-family diagnostic smoke

`amd_smoke_job.sh` and `amd_smoke.py` prepare a separately approved, single-PBS-job
diagnostic smoke. They neither connect nor submit jobs and perform no build.
The packet controller requires an explicit smoke approval, a batch allocation,
a new `runs/<RUN_ID>` attempt, and the approved packet and helper hashes.
Only the disposable allocated shell loads the reviewed modules through the
existing `rocfft_compute_modules.sh`; the login node never executes a benchmark.

`amd_smoke_source.py` checks the complete saved source which produced the old
36 binaries, source-before/profile-before/source-after records, finished PBS
proof, manifests, compiler metadata, logs and binary hashes. The old source
snapshot is copied into the new packet as `artifact-source/` with unchanged
relative paths, bytes and executable bits. No old run is repaired or rewritten.
The frozen snapshot supplies the runner, result validator, numerical definitions
and canonical smoke configs. Newly edited launcher code has a separate hash;
it cannot relabel an old binary with the current worktree's source identity.
See [result provenance](../../docs/RESULT_SCHEMA.md) for the owning specification.

The planned coverage is 18 unchanged teaching examples, two JSONL diagnostic
campaigns (144 raw rows), then four main smoke campaigns: serial/threaded CPU
requests crossed with JSONL/CSV, 72 rows each, **288 main rows**. Each campaign
contains all six families, CPU/HIP/OpenMP target, both scopes and two trials.
Diagnostic rows remain separate. `amd_smoke_campaign.py` requires the exact
matrix, actual successful attempts, source/config/runtime/backend identity and
each numerical metric within its canonical threshold. A successful process or
the expected number of rows alone cannot pass validation.

`amd_smoke_evidence.py` separates artifact-specific link inputs from runtime
evidence. Headers and static archives are not treated as loaded DSOs; GNU linker
scripts are tied to the actual target's `link.d` for CPU and OpenMP builds.
Native HIP uses an explicit recorded-link-command mode: the target's `link.txt`
must match the actual compiler argv in its hash-verified, successful verbose
build log, including the compiler, output target and ordered inputs. The source
proof binds that log to the target in the completed build summary. Missing,
changed or ambiguous evidence stops the smoke; a missing depfile does not
silently select an alternative. All 36 artifacts' link evidence is checked
before the first child command. The chosen evidence and hashes are saved in
`smoke/link-evidence.json` and retained with the runtime records.
CPU/OpenMP depfiles also contain compiler-generated object prerequisites,
including temporary OpenMP image wrappers that need not survive a successful
link. Their original tokens are retained as historical object inputs, without
claiming current existence or content verification. Persistent library,
archive and linker-script inputs still require current file evidence; missing
libraries are not exempted. Relative library inputs use the artifact's link
directory. This distinction does not weaken the retained binary/source hashes
or the configured-versus-loaded library checks. The source-only preflight
checks saved evidence, not live external library availability.
`ldd` records resolution only.
Each actual child additionally receives `LD_DEBUG=libs`; its PID, raw stdout,
raw stderr, exit and initialized-library identities are retained. The two
diagnostic campaigns establish the observed identities before the four main
campaigns' runtime hashes are fixed. A main invocation with a changed identity
is rejected. Unknown syntax, PID/program mismatches, missing required program
events and inconsistent observed initialization/finalization events fail.
An initialized DSO need not emit a matching `calling fini`: for example,
[glibc 2.34's finalizer](https://github.com/bminor/glibc/blob/glibc-2.34/elf/dl-fini.c)
emits that message conditionally on destructor entries. No library-name
exception or invented finalization event is used. The trace records observed
finalizations and initialization events without a matching observed finalization;
neither is a statement about final memory mappings or successful process exit.
The owned child must separately finish with exit zero and no timeout before
trace validation, and numerical validation remains a separate gate. Program
initialization, transfer and main-finalization events are required for this
smoke contract, but do not prove the whole trace or process has completed.
This records glibc initialization events, not all memory mappings, symbol
ownership, GPU code-object identity or observed CPU worker counts. Requested
thread counts and verified allocation limits are distinct from worker evidence.
These instrumented timings are **not performance measurements**.

Per-child and overall timeouts stop the attempt without retry; only the owned
child process group is signalled. Each log and result is created exclusively.
The controller stops on a process error, evidence mismatch or failed numerical
campaign, preserving partial output. It never cancels/requeues a PBS job. A new
actual smoke run still needs a concrete packet, resource/budget review and
separate owner approval. Local fake tests do not establish Sirius loader/runtime
compatibility or GPU numerical correctness; the focused real-device results above
supply the numerical evidence for the remaining five libraries.

The first approved attempt stopped before numerical execution because the
original helper required a `link.d` which native HIP had not generated. See
[the retained failure and recovery evidence](../../docs/VALIDATION_REPORT.md#amd-six-family-smoke-link-evidence-failure).
That attempt remains failed; local repair does not relabel it as successful.
The second attempt also stopped before numerical execution: its depfile parser
required an OpenMP temporary object to remain present. Both failures and their
local corrections are recorded at the same link. Neither attempt establishes
runtime correctness for the five new families.

The successful module inventory used an interactive login Bash and found
`gcc/14.2.1`, `aocl/5.1.0/gcc14.2.1`, and `rocm/7.2.0`. The earlier noninteractive
inventory did not initialize the site's module paths. Use the observed module
environment rather than repeating an unconfigured filesystem search.

[rocfft_module_build.sh](rocfft_module_build.sh) loads the CPU compiler and AOCL
in one disposable child shell, and ROCm separately for each GPU profile. Each
child purges only its own module environment. CPU `gcc`/`g++` must identify as
14.2.1; their absolute invocation paths are passed explicitly. GPU profiles use
AMD Clang from the loaded ROCm environment, with the previously observed
`gfx942:sramecc+:xnack-` target and no automatic OpenMPI addition. Module
definitions, loaded modules and compiler versions are retained in the run log.

The saved module definitions show that AOCL prepends system library directories
ahead of the GCC toolset runtime directories. In the CPU child shell, the script
records `LD_LIBRARY_PATH` before and after loading GCC and identifies the exact
directories GCC added. After loading AOCL, it puts those directories first in
their module-provided order, then retains the remaining AOCL/other entries in
order, removing only exact duplicates. It does not guess a toolset prefix,
discard AOCL paths, change `LIBRARY_PATH`/include hints, or alter GPU/parent
environments. Empty components, relative paths, missing GCC-added directories,
or an unidentifiable GCC contribution stop the build. An unset/empty initial
variable represents no entries and is allowed. Before/after values, the GCC
contribution and the final value are logged; `toolchain.json` records the final
environment used by configure. CMake's existing compiler check verifies linking
without running the produced executable. The saved successful build verified
this correction and the serial FFTW link, but did not run those artifacts.

FFTW's single-precision API uses `fftw3.h`, with the `fftwf_*` symbols and
`libfftw3f` libraries; do not replace the header search with `fftw3f.h`. Use the
loaded AOCL include/library hints instead of scanning unrelated installation
trees. The approved compute job reproduces the reviewed compiler/provider
selection and records its combined module environment before runtime checks.

[rocfft_build_only.py](rocfft_build_only.py) builds one profile with parallelism
one, separate build/log/cache directories under `runs/<RUN_ID>`, and no produced
binary execution. It records source/compiler/dependency/artifact provenance
and rejects a missing target even if CMake configure returns zero. A `BUILT`
result confirms compilation/linking only, not runtime or numerical correctness.
Build summaries must be reviewed before a compute job is prepared/submitted.
The existing allocated smoke helper retains its allocation gate; its optional
`CPU_CC`/`CPU_CXX` overrides are absolute executable paths and never silently
fall back when invalid. Without overrides it prefers `gcc`/`g++` to system
`cc`/`c++` aliases.

### FFTW threading discovery and a separate threaded smoke

The saved AOCL CPU build selected `include/fftw3.h` and `lib/libfftw3f.so`
under the observed AOCL GCC prefix; their resolved paths are in `include_LP64`
and `lib_LP64`. The saved pkg-config hint says `3.3.10`; the filename suffix
`libfftw3f.so.3.6.10` is not a package release declaration. Neither this evidence
nor a missing `fftw3f_threads` proves that no threaded provider is installed.

The portable selection and API requirements are owned by
[BENCHMARK_PROTOCOL](../../docs/BENCHMARK_PROTOCOL.md#rocfft-cpuhipopenmp-contract)
and provenance by
[RESULT_SCHEMA](../../docs/RESULT_SCHEMA.md#rocfft-additive-representation-and-campaign-identity).
The build helpers accept the explicitly selected environment value
`FFTW3f_THREADING=AUTO|PTHREAD|OPENMP|COMBINED|NONE` and pass it to CMake.
Separate threading libraries are searched only alongside the chosen base's
logical and resolved directory pair. An override pointing at another provider
is rejected for threading; arbitrary cross-prefix combinations require further
review rather than silent mixing. Every fresh run records the actual selected
backend and link inputs, not just the requested mode. A changed selection
requires a fresh build/run and must not be combined with saved campaign data.

`rocfft-smoke.json` remains the serial smoke. The new
[rocfft-threaded-smoke.json](rocfft-threaded-smoke.json) is a separate
two-thread starting configuration with exactly one production CPU series,
`cpu-fftw-threaded`, plus HIP and OpenMP target. The allocated compute helper's
`--cpu-threads N` selects the threaded configuration for `N > 1` and records the
effective count. The job body accepts `STAGE31_CPU_THREADS` (default one,
maximum 24 for the confirmed one-vnode allocation). The reviewed dispatcher
explicitly exports that value into its job. The helper bounds it by the confirmed allocation and, where available,
process CPU affinity, sets `OMP_NUM_THREADS=N` and `OMP_DYNAMIC=FALSE`, and
rejects an inherited `OMP_THREAD_LIMIT` that would reduce that count. It
requires both manifest capability and threading provenance. Missing support
stops the threaded smoke; it never becomes a serial success. This path is the
allocated compute helper/job body, not the older generic build-smoke shell.

The approved full smoke sets `STAGE31_CPU_THREADS=2` and
`STAGE31_INCLUDE_SERIAL=yes`. The job body forwards the latter as
`--include-serial-campaign`. Serial and threaded campaigns use separate run IDs,
effective configurations, raw outputs and runtime identities for JSONL and CSV.
Both retain every case, scope and trial in their canonical smoke configuration.
All CPU/HIP/OpenMP target builds, dependency checks and teaching examples must
pass before the campaigns start; missing profiles are not a successful smoke.

[rocfft_compute_modules.sh](rocfft_compute_modules.sh) prepares the explicitly
approved allocated job environment. It requires PBS allocation and canonical
`STAGE31_*` paths plus `GPU_SUITE_SIRIUS_COMPUTE_APPROVED=yes`. It loads the
confirmed GCC, AOCL and ROCm modules in the job's disposable shell, restores
GCC library precedence after all module loads, and records the resulting
environment. This combined environment is a new runtime identity; it is not
asserted to equal either saved build-only environment. The compute helper must
verify actual loaded dependencies against the newly built metadata before
running the comparison. A dependency disagreement remains a failure.

For GCC OpenMP runtime linker scripts, the configured script hash remains a
build-input identity. The bounded INPUT/GROUP parser also checks every referenced
shared library against the actual loaded path and hash. A `-l` static archive
requires the compiler-generated `link.d` evidence naming both the script and
that unique archive. Unsupported scripts stop the run; arbitrary `.so` inputs
are never exempted. These bindings are part of each campaign's runtime record.

The recorded allocated rocFFT baseline verified both timing scopes, all batches,
actual loaded FFTW/OpenMP runtime identities and thread settings. A build-only
result alone does not establish those facts. Any subsequent real execution
requires approval of the selected provider, thread count, CPU/GPU module
environments and GCC runtime ordering.

### Prepared read-only threading inventory

[fftw_thread_inventory.py](fftw_thread_inventory.py) was executed in the approved
read-only run `sirius-fftw-threads-20261017T061353Z-8f4da780`. It found AOCL's
OpenMP shared library exporting the three required thread APIs. Static archives
were not inspected, so its overall inventory status was `INCOMPLETE`; this did
not invalidate the shared-library evidence. On a separately approved connection, run it
with the previously observed prefix (no module load is needed just to read
these known files):

```sh
python3 -I -B jobs/sirius/fftw_thread_inventory.py \
  --aocl-root /system/apps/rhel/9.6-202602/aocl/5.1.0/gcc
```

The future launcher must save stdout JSON, stderr, exit status and the script's
SHA-256 in a new exclusive `runs/<RUN_ID>/` directory and preserve old logs.
No SSH command or authorization is implied by this example. The tool checks
only bounded `lib`, `lib64`, `lib_LP64`, include and pkg-config locations under
that prefix. It records real paths/content hashes, public header tokens, static
pkg-config text, defined thread/FFT exports versus undefined references, and
ELF `NEEDED` entries using `readelf` with optional `nm` fallback. It never
evaluates a module, loads a library, runs `ldd`, compiles/links, runs a transform,
queries devices, or invokes a scheduler. Missing tools, output/time limits and
uninspectable files remain explicit unknown/incomplete evidence. A complete
static inventory establishes only that collection finished, not thread support.

Remaining real-machine checks are the installed provider layout/exports,
successful compile/link with the selected host compiler/runtime, and subsequent
allocated numerical smoke. The latter two require separate build/job approval;
they are not part of the read-only inventory. Do not infer API removal or create
a private FFTW installation from a missing filename alone.

Stage 2 completed with an explicitly approved, single interactive SSH probe in
this chat's integrated terminal. The owner entered authentication directly;
Codex analyzed the saved completion marker and outputs. SSH and remote exit
were zero with one matching completion marker. The login-node observation
found CMake 3.31.8, Python 3.9.25, GCC 11.5.0, Clang 21.1.8 and AMD Clang
22.0.0git (`roc-7.2.0` build text), with HIP 7.2.26015-fc0010cf6a.
`ROCM_PATH`/`HIP_PATH` were unset and `module` unavailable in that noninteractive
shell. This does not mean ROCm is absent, establish library paths/versions,
verify a GPU, or determine a compute-node thread count.

The historical Sirius route at that stage was **separately approved interactive
SSH in the integrated terminal**. It is superseded by the noninteractive local
Shell route in [Current status and execution route](#current-status-and-execution-route).
The owner alone supplied keyboard-interactive inputs;
Codex must not request, read, record, transcribe or automate them. After
completion Codex reviews saved command output. Do not use `remote.py` direct or
control-master transports, create a ControlMaster, or resume transport design.
The transport/state documentation below is retained historical reference,
**not a current instruction to execute its examples**. Preserve failed v1/v2
records and all prior state/logs without repair, conversion or deletion.

The earlier Stage 3 implementation approval covered local rocFFT CPU/HIP/OpenMP
code, common integration, local testing, self-review, and application through
existing SSHFS. It did not itself authorize SSH, scheduler/accounting, remote
build, GPU execution, Git writes or publication. The previous build-only attempt
had a separate owner approval; its local repair did not authorize a retry.
The other five AMD libraries were deferred at that stage; their subsequent
implementation and 36-artifact build are recorded above. Pegasus rules are
unchanged.

## Stages 1 and 1.1 boundary

Stage 1 creates these documents, the probe-only transport, and local fake tests.
It makes **zero real SSH calls and zero scheduler calls**. Production state
initialization, approval registration, and probe execution have not been carried
out as part of stages 1 or 1.1; initialization and execution in tests use
disposable, isolated state and fake children only.

Stage 1.1 hardens the SSH invocation and persistent-state path checks, adds
branch and untracked-file reporting to the probe, and reruns local verification.
It does not authorize the first real connection or production state registration.

AMD library implementations, remote builds, GPU runs, branches, commits, pushes,
PRs, merges, and scheduler operations are outside stages 1 and 1.1. Nothing here
changes [Pegasus operating rules](../../docs/PEGASUS_EXECUTION.md).

## Stages 2.2 and 2.3 local boundary

Stage 2.2 makes the transport's public-key authentication policy explicit and
checks it with local mock/fake tests. It does not authorize another connection,
`ssh-add`, access to a real agent socket, or registration of a new production
approval/execution. Existing state, coordination, terminal execution records,
and output evidence remain unchanged. The probe script and its reviewed hash
are unchanged.

Stage 2.3 adds a separate `control-master` transport for a future, explicitly
approved logical session over an owner-authenticated master. It is local-only:
no real SSH client, ControlMaster creation, control-socket access, agent access,
network access, scheduler operation, or production-state mutation is permitted.
Tests use fake socket metadata and fake children. Existing v1/v2 approvals,
terminal executions, and their stdout/stderr are retained without conversion.

## Historical authentication investigation and superseded master proposal

The reviewed direct public-key probe reached server acceptance of the public
key, reported public-key authentication with partial success, and then required
`keyboard-interactive`. Its terminal result was `CONNECTION_STOP`. The earlier
direct probe also ended in `CONNECTION_STOP`. Preserve both as terminal evidence;
do not erase, repair, or reuse their spent approvals/executions.

This observed Sirius route requires public-key plus keyboard-interactive
authentication. The evidence does not identify whether the interactive step is
a password, OTP, or another mechanism. Further publickey-only automatic new
connections are not the selected Sirius route. The existing `direct` transport
remains readable and backward compatible for legacy records; that compatibility
does not authorize another direct attempt.

The superseded Stage 2.3 proposal had the owner authenticate in a Mac terminal
and create one bounded-time ControlMaster, with separately approved logical
sessions. It was not selected for the successful Stage 2 execution and is not
the current route.
In that historical proposal the owner alone started/stopped the master.
Never provide keyboard-interactive input,
passwords, OTPs, passphrases, private/public key bodies, or agent identity bodies
to Codex or `remote.py`. The master's interactive exchange stays in the owner's
terminal. Any additional rules in the internal Sirius user guide take precedence
over the operational templates here.

## Historical helper review packet (not the current route)

The historical helper proposal required this packet; it does not authorize
current execution or state registration:

| Item | Required value |
| --- | --- |
| Target | Explicit SSH config alias and canonical remote Git root; currently unspecified |
| Script | Full contents of [probe.sh](probe.sh), with the actual SHA-256 calculated after review |
| Operation | One lightweight environment probe only |
| Transport | `control-master` was proposed; neither helper transport is the current route |
| Master | Owner-approved absolute private local control-socket path, supplied separately |
| Budget | One logical multiplexed session per approval ID and execution ID |
| Concurrency | One session globally for this local user, across approvals/processes |
| Retry | None; `ConnectionAttempts=1` |
| Timeouts | Proposed connection timeout 20 seconds; total timeout 120 seconds |
| State | Existing owner-approved persistent, private Mac-local state; no reset or reinitialization |
| Result | Review the recorded outputs and stop; no automatic stage 3 |

Calculate the hash with `shasum -a 256 jobs/sirius/probe.sh`. A hash is not an
approval. Do not reuse an approval after changing the script. The transport
reads the script once, compares those bytes to the approved hash, and sends
exactly those bytes without rereading the path.

The stage 1.1 probe SHA-256 is
`ad03e6df37ded557779e974df5557b524d738fea1c11f3a9b3c2a027200f21ea`.
The superseded stage 1 hash
`86239c2c6cce538a4e23248b21f0ab99e8e9750e1f5b7afe199c190d2e03e221`
is invalid for stage 2. Review the current full script and hash together before
authorizing a connection.

The script first verifies that both the physical working directory and Git's
top-level directory equal the approved remote root. It reports UTC time, host
identity, Git HEAD, branch, tracked changes and untracked files, OS/CPU count,
available module list, compiler/CMake/Python locations and version text, and
limited configured ROCm version/header hints. It does not read SSH configuration
or credentials, dump the environment, initialize a GPU, discover GPU devices,
compile or link, recursively scan installations, or invoke scheduler/accounting
commands.
Unavailable tools or configuration are reported as unavailable. The script
does not infer a queue, account, compiler path, GPU target, or ROCm version.
Git status uses `--short --branch --untracked-files=all`; it does not enumerate
ignored files or recursively hash file contents.

The probe does not validate library linking, GPU visibility, OpenMP host
fallback, HIP/OpenMP device-pointer compatibility, or performance. Those need
separate approved work. One owner-created network connection and a logical
session over that connection are different resources. `remote.py` does not
create the master's underlying connection. Its counters do not certify the
master's prior connection history, SSHFS reconnects, or unrelated processes.

## Local state and command interface

`remote.py` offers only `initialize`, `approve`, and `probe`. It has no arbitrary
command, alternate script, alternate SSH executable, or scheduler interface.
Approval records are local records of an owner's prior authorization; the
software cannot itself grant approval.

All examples below are syntax templates requiring explicit owner-supplied
values. They are not instructions to connect or register production state during
local-only stages. Existing state must not be initialized again. The initialize
subcommand is only for separately approved first-time state setup.

```sh
python3 jobs/sirius/remote.py --state-directory "$APPROVED_STATE_DIRECTORY" approve \
  --approval-id "$APPROVAL_ID" --alias "$APPROVED_ALIAS" \
  --remote-root "$APPROVED_REMOTE_ROOT" --script-sha256 "$APPROVED_PROBE_SHA256" \
  --transport control-master --control-path "$APPROVED_CONTROL_SOCKET" \
  --connect-timeout 20 --timeout 120
python3 jobs/sirius/remote.py --state-directory "$APPROVED_STATE_DIRECTORY" probe \
  --approval-id "$APPROVAL_ID" --execution-id "$EXECUTION_ID" --dry-run
```

The control-master approval fixes a logical-session budget of one. Its transport
type and exact socket path are part of the approval alongside the alias, remote
root, script hash, state root, and timeouts. A different socket needs separate
human approval, not an automatic retry. No production approval is registered by
stage 2.3. Omitting `--transport` retains the legacy `direct` CLI behavior for
backward compatibility. Neither transport is selected for future Sirius
execution; retain this interface only as historical implementation reference.

Only after separate stage 2 authorization may the final command be issued
without `--dry-run`. Dry-run takes the nonblocking lock and reads existing state,
but starts no process, writes no ledger/log, consumes no budget, and does not
recover or mark interrupted executions. For `control-master`, it also starts no
alive check and no logical session. Initialize/approve are separate local
mutations, never implicit parts of dry-run or probe.

The persistent state path must not exist before initial registration. Its parent
must already exist on the same Mac-local filesystem as `/tmp`; FUSE/SSHFS and
other mounted volumes are rejected. The production CLI rejects `/tmp`,
`/private/tmp`, the platform temporary directory, their resolved locations,
and every descendant of those roots.
The state must therefore be on the same Mac-local filesystem as `/tmp`, but
outside temporary-directory trees. Choose a durable directory outside any Git
worktree. State directories use mode 0700 and files use mode 0600. Final state
roots and state files cannot be symlinks. No credentials belong in state.
Tests use an internal substitute for the temporary-root check and isolated
temporary fixtures. No CLI option or environment override relaxes this rule.

Every CLI invocation uses the same fixed per-user coordination directory:
`/tmp/gpu-library-suite-local-build/sirius-control-<uid>`. Its parent must already
exist locally. There is no CLI/environment override. Explicit initialization
creates this directory exclusively, binds it to the persistent state root with
a matching identity token, and creates the shared `fcntl.flock` lock. The lock
is held from state/budget inspection through output capture and durable result
recording. A contender fails immediately; it does not wait and later connect.

Existing coordination or state is never overwritten by initialization. Missing,
malformed, mismatched, or incomplete state stops execution. If temporary
coordination is lost after reboot/cleanup, retain the durable state and stop
for human reconciliation; do not initialize a different root to reset history.
There is deliberately no automatic repair/reset/replay command. These records
protect normal use and process crashes, not intentional administrator deletion
or modification of all evidence.

## Durable accounting and results

The state contains `identity.json`, `ledger.json`, and
`executions/<execution-id>/{stdout,stderr}.bin`. The coordinator contains a
matching `binding.json` and a lock file which is never replaced during use.
Project strict JSON parsing rejects duplicate keys, nonfinite numbers, and
invalid UTF-8. Structural checks reject missing counters, inconsistent exits,
wrong identities, missing output, or changed output hashes.

The ledger binds an approval ID to alias, remote root, script SHA-256, state
directory, timeouts, and budgets; control-master approvals additionally bind the
transport and exact socket path. Legacy approvals without transport fields retain
their original direct interpretation and bytes. Each execution records that approval, its own
ID, argv, start/end timestamps, reservation, observed SSH call count, local SSH
exit, confirmed remote exit, completion evidence, output hashes, and outcome.
The binary logs keep stdout and stderr separate. Stdout is preserved byte for
byte. For new executions, SSH stderr is received in memory and SHA-256/MD5
fingerprints are replaced with `[REDACTED_FINGERPRINT]` before any stderr is
saved. Authentication-stage messages and the completion marker remain. The
ledger's stderr hash describes those saved, redacted bytes; raw stderr and
fingerprints are not written to a separate log. Existing output records are
not rewritten or rehashed. If a process crash prevents stderr collection, the
record may have no recovered stderr; missing evidence cannot establish success.

The direct connection or control-master logical session is reserved and the
ledger is atomically replaced and `fsync`ed **before** process creation.
A launch error consumes that reservation;
it is not silently refunded. The actual observed call count remains a separate
field. Between reservation and durable spawn observation, a crash means that
child's call count is unknown (`null`), never falsely zero. A known successful
spawn is recorded as one. A known process-creation failure is zero, with a consumed
reservation. Creating a new execution ID does not replenish approval budget.

An execution-specific random completion token is passed as an argument and
returned in an EXIT marker over the authenticated SSH channel. A matching
single marker and consistent SSH exit are required to confirm remote completion.
SSH exit zero without proof, duplicate/wrong proof, exit 255, or timeout cannot
be successful completion. Authentication/host-key/connection-limit diagnostics
override apparent completion evidence.

| State | Interpretation |
| --- | --- |
| `RESERVED` / `RUNNING` | Incomplete durable record; a later mutating invocation records `UNKNOWN` and stops |
| `COMPLETE` | Matching single completion proof and consistent exit zero |
| `REMOTE_FAILED` | Matching proof and consistent nonzero remote command exit |
| `MASTER_UNAVAILABLE` | Existing-master alive check fails, or its socket is missing/stale/unresponsive before a session |
| `MASTER_LOST` | Master disappears after the alive check or reports a lost multiplexed session; outcome needs human reconciliation |
| `LOCAL_STOP` | Socket attributes, approval, hash, ledger, or other local validation fails; no repair |
| `CONNECTION_STOP` | Legacy direct authentication refusal, host-key anomaly, or connection-limit signal |
| `LAUNCH_FAILED` | Local process creation failed; reservation remains spent |
| `UNKNOWN` | Timeout, disconnect, interruption, absent/inconsistent proof, or post-launch local failure |

A recorded `CONNECTION_STOP` is a terminal outcome, not an unresolved `UNKNOWN`.
Existing records remain readable without conversion, and their consumed budgets
and output evidence must be retained. A later, separately human-approved
approval can be registered without rewriting that terminal execution. This does
not authorize reuse of its exhausted approval or automatic retries.

Any unresolved `UNKNOWN` or `MASTER_LOST` blocks new probes and approvals, even with another
approval/execution ID. `RESERVED` and `RUNNING` likewise block those operations;
interruption recovery records `UNKNOWN` and stops. Parent-process death can
leave its child alive; recovering the stale record therefore never launches
another process. Human review must
reconcile the previous operation before a later separately designed recovery.
There are no automatic retries in any state.

The SSH executable is fixed to `/usr/bin/ssh`, without PATH lookup or an
alternate-executable CLI option. Before reserving a connection, a filesystem-only
check requires that executable to be a regular, executable file. The check starts
no subprocess. Dry-run, a script-hash mismatch, and exhausted budgets start no
subprocess, including for executable checks.

The legacy direct SSH invocation explicitly uses the following options:

- `-T`, `BatchMode=yes`, `StrictHostKeyChecking=yes`, `ConnectionAttempts=1`,
  and `ConnectTimeout=<approved value>`.
- `ControlMaster=no`, `ControlPath=none`, and `ControlPersist=no`.
- `ClearAllForwardings=yes`, `ForwardAgent=no`, `ForwardX11=no`,
  `PermitLocalCommand=no`, `ForkAfterAuthentication=no`, and `StdinNull=no`.
- `AddKeysToAgent=no`, `UpdateHostKeys=no`, and `CheckHostIP=no`.
- `PreferredAuthentications=publickey`, `PubkeyAuthentication=yes`,
  `KbdInteractiveAuthentication=no`, `PasswordAuthentication=no`,
  `GSSAPIAuthentication=no`, and `HostbasedAuthentication=no`.
- `IdentitiesOnly=yes`, `IdentityAgent=SSH_AUTH_SOCK`, and `LogLevel=DEBUG1`.

The approved alias may require `ProxyJump` or `ProxyCommand`; the legacy direct transport does
not disable or override either setting. It does not edit SSH settings, terminate
existing SSHFS or ControlMaster connections, or send existing-connection control
commands such as `-O`. Timeout handling kills only the exact child process this
invocation created; killing it does not prove the remote command stopped.

## Owner-authenticated ControlMaster transport

### Socket approval and local checks

The owner supplies the socket path later. No user-specific socket path is
embedded in the code or this document. Approval requires all of the following:

- An absolute path whose normalized spelling and resolved path exactly match
  the approved value; no symlink in the socket path or its ancestors, percent
  expansion, dollar expansion, whitespace, quotes, backslashes, control
  characters, or double-leading slash. These conservative syntax restrictions
  keep the exact approved path literal in OpenSSH options.
- An existing Unix-domain socket owned by the current local user, with no
  group/other permission bits.
- A parent directory owned by that user, with no group/other permission bits.
- A durable Mac-local location on the same filesystem as `/tmp`, outside FUSE,
  SSHFS, `/tmp`, `/private/tmp`, and platform temporary trees and their aliases.
- A location outside any Git worktree and separate from the persistent-state
  directory, including its descendants.

The path is private operational information and is retained only in private
approval/execution records. The helper never creates, removes, repairs, chmods,
or chowns the socket or its directory. A missing or stale master requires owner
review, not recreation by the helper. Approval binds the observed device,
inode, owner, and mode as well as the path; replacing a socket does not preserve
that approval. The owner must attest that this socket belongs to the master
authenticated for the approved Sirius alias and remote target, not an unrelated
master. A local socket check cannot prove that target identity.

### Alive check and guarded session

Before a session, the helper checks only the approved existing master. The
essential alive-check argv is the following syntax template; actual argv is
recorded with the execution:

```sh
/usr/bin/ssh -F /dev/null -S "$APPROVED_CONTROL_SOCKET" -O check \
  -o ControlPath="$APPROVED_CONTROL_SOCKET" -o ControlMaster=no \
  -o ControlPersist=no -o CanonicalizeHostname=no -o IdentityAgent=none \
  -o ProxyCommand=/usr/bin/false -- "$APPROVED_ALIAS"
```

The check must exit zero and report the expected `Master running (pid=...)`
response. This proves a responding local master, not its remote target identity.
Socket identity, owner, permissions, and inode must remain unchanged across the
check and before the session starts. Check failure, missing/stale socket, or an attribute mismatch
stops execution. The check is a local control-socket request, not a new network
connection or authentication attempt. The helper never issues `-O exit`,
`stop`, forwarding-management operations, or process termination for the master.

The logical-session argv uses `/usr/bin/ssh`, `-T`, `-S` and `ControlPath` set
to the exact approved socket, `ControlMaster=no`, `ControlPersist=no`,
`BatchMode=yes`, `ClearAllForwardings=yes`, `ForwardAgent=no`, `ForwardX11=no`,
`PermitLocalCommand=no`, `AddKeysToAgent=no`, `UpdateHostKeys=no`, and
`CheckHostIP=no`. It explicitly includes `ProxyCommand=/usr/bin/false`.
Both master-client invocations use `-F /dev/null`, `CanonicalizeHostname=no`,
`IdentityAgent=none`, and `IdentityFile=none`, with every fallback authentication
method disabled. Even the default public identity files are not loaded.
This avoids executing configured `Match exec`, introducing a configured
ProxyJump/ProxyCommand, canonicalization, or contacting an agent. It is scoped
to the master-client transport: the owner already used the appropriate alias
configuration when authenticating the master, and legacy direct behavior is
unchanged.
If the existing master cannot be used, that fallback guard fails locally;
the helper must not establish a new TCP connection or SSH authentication.
It does not add a ProxyJump route or public-key, keyboard-interactive, password,
GSSAPI, or hostbased fallback. It does not weaken host-key or signature-algorithm
policies. No automatic transition to `direct`, another socket, another approval,
or another execution ID is allowed after failure.

`ForwardAgent=no` disables remote agent forwarding; it is distinct from local
agent use during the owner's prior authentication. `AddKeysToAgent=no` disables
key registration; it does not delete an existing agent identity. The helper
neither queries nor changes the agent for this transport. Owner authentication
secrets and key/identity bodies are never collected or persisted. Private logs
keep stdout/stderr separate and retain completion evidence; fingerprint
redaction remains in effect before stderr is saved.

The global nonblocking lock covers validation, budget reservation, alive check,
session execution, and durable recording. A competing process stops immediately.
The alive check cannot authorize a new session or replenish a spent budget.
The logical-session reservation survives restarts and is never automatically
refunded, including when the alive check fails before any session starts.
A missing completion marker or timeout after a session starts is
`UNKNOWN`, even if the local child has stopped. A specific lost-master result
does not establish that the remote command never ran.

If final output capture or ledger persistence fails after a session may have
started, the helper reports `UNKNOWN` without repeating the final write or
repairing state. A write may already have succeeded before a later `fsync`
failure. Inspect the durable evidence separately; the reported uncertainty does
not authorize rewriting a completed record or starting another session.

Record these quantities separately:

| Quantity | Meaning |
| --- | --- |
| `underlying_connections_created` | Zero; the owner created the master beforehand |
| `local_check_calls` | Observed local SSH client processes requesting an alive check |
| `logical_session_calls` | Observed session-client processes; separate from `reserved_logical_sessions` |
| `logical_sessions_confirmed` | One only when completion proof confirms a session; otherwise unknown after a possible launch, zero before launch |
| `ssh_calls` | Total observed local SSH client processes, not network connections |
| `new_tcp_attempts` | Zero under the fallback guard; not a measure of the owner's master history |
| `new_ssh_auth_attempts` | Zero; no interactive or automatic authentication fallback |
| `remote_command_attempts` | Durable intent to submit the probe through a session-client launch; does not prove receipt by the master |
| `remote_commands_confirmed` | One only with completion proof; unknown if it may have run without proof, zero if no session was started |

The local SSH process count must not be reported as a new network-connection
count. Crash windows and unconfirmed remote outcomes must be reported explicitly,
not turned into known-zero counters.

### Owner-only creation and shutdown template

This is an unexecuted syntax template for the owner's terminal. It requires a
separate decision to create the master, the internal Sirius rules, a suitable
private directory, and a finite agreed operating window. Do not paste secrets
into Codex. An existing socket must not be overwritten or repaired automatically.

```sh
CONTROL_DIR="<OWNER_APPROVED_PRIVATE_LOCAL_DIRECTORY>"
CONTROL_SOCKET="$CONTROL_DIR/sirius-master.sock"

mkdir -p "$CONTROL_DIR"
chmod 700 "$CONTROL_DIR"

/usr/bin/ssh \
  -M \
  -S "$CONTROL_SOCKET" \
  -o ControlMaster=yes \
  -o ControlPersist=no \
  -o ClearAllForwardings=yes \
  -o ForwardAgent=no \
  -o ForwardX11=no \
  -o PermitLocalCommand=no \
  -o ExitOnForwardFailure=yes \
  -N \
  sirius
```

The owner completes public-key plus keyboard-interactive authentication there
and keeps that foreground master only for the agreed finite window. The owner
ends it from that terminal when the window expires or work finishes; the helper
does not perform shutdown. Creating one master does not authorize any logical
session: each probe still needs its explicit packet and budget. Stage 2.3 does
not execute this template or proceed to manual master creation.

## Local verification and handoff

Run the fixed validation entry on a Git-free local source copy and a new work
directory under `/tmp/gpu-library-suite-local-build`:

```sh
python3 tests/python/run_safe_local_validation.py \
  --source-copy "$LOCAL_SOURCE_COPY" --work-dir "$NEW_LOCAL_VALIDATION_DIRECTORY"
```

The existing CTest/Python infrastructure records PASS, FAIL, SKIP, policy
exclusions, and dependencies preventing execution separately. The dedicated
remote tests use injected fake children, forked lock contenders, and abrupt
local test-process exits, fake socket metadata, and fake master responses. They
never launch SSH or run the probe on a remote host, query a real agent, access a
real agent/control socket, or create a real ControlMaster, and they create no
symlinks. Test success is not evidence of ROCm, GPU, OpenMP offload, remote
networking, or scheduler success.

Stages 1 and 1.1 end with all required remote fake tests passing, the existing safe
local tests rerun, reviewable changed files/diff, the actual probe hash and full
script, and the proposed stage 2 packet. Stop there. Do not auto-connect or
proceed to stage 2.

Stage 2.2 ends after the local validation results, unchanged probe hash, complete
stage-specific diff, and proposed future approval/execution IDs are reported.
Do not register those IDs, retry stage 2, or proceed to stage 3.

Stage 2.3 ends after all required remote mock/fake tests and legacy-state checks
pass, the fixed safe validation reports zero failures, the unchanged probe bytes
and hash are verified, and the full stage-specific diff is reviewable. Real SSH,
network, master creation, socket/agent access, scheduler operations, and
production-state changes all remain zero. Do not create the owner's master,
register new production approvals, retry stage 2, or proceed to stage 3.

For later AMD teaching builds, manual builds may use `hipcc`; CMake native HIP
uses a supported Clang, without fixing the `hipcc` wrapper as
`CMAKE_HIP_COMPILER`. The only approved teaching exception is the explicit
OpenMP rocThrust translation unit plus `rocthrust_wrapper.hip`. Both compilation
conditions and link inputs must be recorded. Filenames and mathematical
contracts remain owned by the project specification; campaign provenance
remains owned by the result schema and benchmark protocol.

## Historical rocFFT build and smoke preparation: compute node only

The first allocated build stopped before examples or benchmarks ran. The
later baseline described above resolved those failures. The following notes
retain that historical diagnosis; they are not a new execution request. The
subsequent approved rerun built HIP and OpenMP targets and ran both teaching
examples with maximum error zero; OpenMP device interoperation was also
reported. This confirms the compiler invocation-name and capability-probe
repairs for that run. It does not certify the benchmarks: CPU FFTW discovery
remained blocked, and both GPU benchmarks failed during result initialization.

The local repair preserves the runtime object created by common result
initialization and adds HIP version fields to it, rather than reinserting its
key. A missing CPU executable retains unanimous campaign source provenance,
while its compiler and binary remain unavailable and its status remains
unexecuted. Mixed source provenance is still rejected. Local fake-provider
regressions cover these repairs; a new real benchmark run remains separately
approved work. Existing failed results are never rewritten.

The next run uses a fresh `SOURCE_ROOT/runs/<RUN_ID>` directory, visible through
the existing repository SSHFS mount and excluded from Git by `/runs/`. Build
trees, command logs, runtime caches and results stay inside that run. The run
directory must be a direct child of `runs/`, match its run ID, and use canonical
paths without a symlinked `runs/` directory. Never reuse a moved CMake cache or
overwrite historical logs. Source snapshots exclude `runs/` and `.fuse_hidden*`
path components before hashing file contents.

[rocfft_job_body.sh](rocfft_job_body.sh) and
[rocfft_compute_smoke.py](rocfft_compute_smoke.py) are the maintained versions
of the allocated-job payload. The separately prepared, private submission
packet copies those exact files into the new run directory. It retains the
site-confirmed absolute PBS tool paths, one submission, and no retry or polling.
No submission packet is executed by this local repair task. The body records
user, hostname, inherited HOME and PATH; HOME is never derived from the project.

Compiler commands retain their discovered invocation paths, including the C++
driver names. Resolved paths and binary hashes are separate provenance fields
in `compute/toolchain.json`; they never replace the command's executable.
The HIP and OpenMP probes require the C++ allocation runtime as well as real
HIP/rocFFT API linkage. A header-only or C-only successful link is insufficient.

FFTW discovery accepts `FFTW_ROOT`, `FFTW3_ROOT`, `FFTW3F_ROOT` and `FFTW3f_ROOT`
as CMake or inherited environment hints, normal CMake prefix/include/library
search paths, and optional `pkg-config fftw3f` hints. The private submission
packet preserves these hints and `PKG_CONFIG_PATH`/`PKG_CONFIG_LIBDIR`; the
compute record saves the actual search environment. No site library directory
is guessed, package installed, or module loaded. A missing header/library and
a failed API link have different diagnostics. If neither the configured search
paths nor installed pkg-config metadata locate FFTW, its actual location or
required module remains an owner/site input before a complete CPU comparison.

### FFTW supply investigation and noninteractive module discovery

The owner supplied Sirius guide section 6.3: FFTW is provided within AOCL,
with modules `aocl/5.1.0/gcc14.2.1` and `aocl/5.1.0/aocc5.0.0`.
These are documented candidates, not proof of their installed paths or compiler
requirements. The first completed inventory observed versioned FP32 FFTW,
threads and OpenMP runtime entries in the system library cache. Its limited
standard-prefix checks did not locate the development header, pkg-config file,
or unversioned link libraries. The noninteractive shell had no `module` command
and no inherited `MODULEPATH`, so it did not inspect the AOCL module definitions.
System runtime files alone do not identify AOCL as their supplier, and their
filename suffixes are not a verified FFTW release number.
No self-installation is planned. The previous CPU build used GCC 11.5; inspect
the selected AOCL module's compiler prerequisites and actual FP32/threaded
library layout before choosing the next CPU toolchain. The GPU compiler is a
separate profile and must not be silently changed by CPU environment setup.

[fftw_probe.sh](fftw_probe.sh) and its adjacent
[fftw_module_inventory.py](fftw_module_inventory.py) are a two-file read-only
inventory payload for a separately approved Sirius login session. They start
no connection and submit no job. Python 3.9 or newer is required; a missing
reader or interpreter stops the shell payload before any query. The revised
payload has only local fake validation until its next separately approved run.
The proposed scope is one interactive SSH, no retry, connection timeout 20
seconds and an overall inventory limit of 120 seconds. Authentication remains
the owner's direct terminal action. Neither this document nor the payload
authorizes that connection. Do not reuse the historical environment-probe
approval or alter `probe.sh` or persistent authentication state.

The payload records user, hostname, UTC time, inherited HOME/PATH and selected
FFTW/search/module environment variables, including `MODULEPATH`, `MODULESHOME`
and `LMOD_CMD`. It queries `pkg-config` version/flags/directories for `fftw3f`
and `ldconfig -p` (retaining FFTW lines), when already available. It does not
call `module`, `modulecmd`, a login shell or an initialization script, even if
one exists. It neither sources profiles nor evaluates their generated shell
code. This avoids automatic default-module loads and modulefile side effects.

The Python reader instead performs bounded static inspection of standard
module-system configuration candidates, relevant module/Lmod files in
`/etc/profile.d`, inherited module roots, and literal references found in those
files. A standard configuration candidate is a place to check, not an assertion
that Sirius uses it. Paths, declarations and source line numbers establish
where each candidate came from; no Sirius installation root is guessed.
The reader looks only for the two named AOCL modules under the discovered
roots, including their Lua forms. It records compiler prerequisites,
dependencies, conflicts and load declarations as data; it does not perform
those operations. It follows supported literal Tcl/Lua path declarations and
simple variables/path joins without interpreting either language.

Conditional or dynamic declarations, missing files, unsupported expressions
and inspection limits are reported explicitly. Candidate paths are not claimed
to be the active environment, even when the referenced files exist. If the
site uses an unsupported bootstrap or module hierarchy, the report identifies
the unresolved input instead of evaluating code or searching the filesystem
recursively. The reader also accepts repeated `--module-root` and `--config`
arguments for absolute paths obtained from site evidence; these are manual
diagnostic inputs, not inferred installation locations.

Candidate files are inspected under the exact include/library/prefix paths
declared by the module definitions and their shallow standard layouts, plus
standard `/usr` and `/usr/local` prefixes, existing FFTW roots, a successful
pkg-config prefix, and optional observed absolute prefixes. The report separates
`fftw3.h`, `fftw3f.pc`, base FP32, pthreads and OpenMP libraries, including
versioned runtimes versus unversioned/shared or static link inputs. It neither
loads nor hashes candidate libraries. There is no recursive filesystem search.
An AOCL description without an exposed prefix or clear single-precision/thread
layout requires further site guidance; do not guess its path. No-match from the
currently unloaded environment does not contradict the guide's AOCL provision.

After execution approval, the payload command within that same Sirius session
is the following (optional arguments are observed absolute prefixes):

```sh
GPU_SUITE_FFTW_PROBE_APPROVED=yes bash jobs/sirius/fftw_probe.sh
```

Prepare a fresh `SOURCE_ROOT/runs/<RUN_ID>` for separate stdout/stderr, SSH exit,
both payload SHA-256 values and start/end times before a future approved
invocation. Copy both files together and verify both source and saved bytes
before running the saved shell payload. The old single-file payload hash or
an earlier invocation does not authorize this revised bundle. Preserve all
earlier logs. No run directory or remote execution is created by this local
preparation. A missing
query or no match does not establish that FFTW is uninstalled. An inventory
completion marker means collection ended, not that FFTW passed compile/link.
The static reader emits structured records including unresolved discoveries;
its nonzero process exit is retained by the shell payload. A zero collection
exit does not turn incomplete module discovery into confirmed availability.

The decision needed is the actual provider of `fftw3.h`, `libfftw3f`, and (for
threaded benchmarks) `libfftw3f_threads`, with its observed version/prefix and
required module/search environment. Preserve that environment explicitly in
the next compute job; a login-node discovery alone does not prove availability
inside PBS. The next separately approved compute build must still perform the
existing API/link and dependency-resolution checks. Keep CPU unavailable if
those checks fail; never substitute a different backend silently.

The NVIDIA implementation already shares `FindFFTW3f.cmake` and the CPU FFT
benchmark core. The [saved Pegasus validation](../../docs/VALIDATION_REPORT.md#fortran-pegasus-evidence-and-cpu-engine-metadata-correction)
records resolving missing CPU FFT targets by supplying an explicit FFTW
prefix. That is the approach to reproduce using Sirius evidence, not a reason
to copy Pegasus paths or infer Sirius module names.

[rocfft_build_smoke.sh](rocfft_build_smoke.sh) is a scheduler-independent body;
it does not request an allocation or open a connection. Do not run it on a
login node. First obtain a separately approved allocation using the current
Sirius site guide, then supply the selected compute-node toolchain, architecture,
FFTW/ROCm roots, build parallelism and distinct HIP/OpenMP device selectors.
No queue/account, allocation command, library location, architecture, module
load sequence or compute-node CPU count is inferred from the login-node probe.
The script requires explicit allocation confirmation; that flag is an owner
assertion, not scheduler verification.

The body uses new CPU/HIP/OpenMP build trees, real API/link checks, records
compiler versions, caches/compile commands/build metadata and actual dynamic
library resolution. [rocfft_smoke_evidence.py](rocfft_smoke_evidence.py) rejects
loaded/configured dependency path/hash disagreement and retains raw ldd logs
while omitting ASLR addresses from software identity. The body builds the three
teaching examples, then runs the small
multiple-batch rocFFT configuration through `run_suite.py` for both timing
scopes and validates results with `validate_results.py`. OpenMP runs use
`OMP_TARGET_OFFLOAD=MANDATORY`; the benchmark also checks real offload and HIP
compatibility of the mapped device pointers outside timing. No integer-index
identity assumption is sufficient. Every command has separate stdout, stderr
and exit logs; existing output paths are rejected and any failure stops.

Supply these environment variables in the **already allocated compute-node
shell**, then invoke only the body:

```sh
# Every value below is site/owner selected; no login-node build is implied.
# SOURCE_ROOT: approved repository; RUN_ROOT: new SOURCE_ROOT/runs/<RUN_ID>.
# CMAKE, PYTHON, CPU_CC, CPU_CXX, HIP_CLANG, OPENMP_CXX: executable paths.
# ROCM_ROOT, FFTW_ROOT: confirmed dependency roots.
# GPU_ARCH: explicit supported target (never native); BUILD_JOBS: approved count.
# HIP_DEVICE and OMP_DEVICE: independently selected visible runtime indices.
export GPU_SUITE_ALLOCATED_COMPUTE_NODE=yes
bash "$SOURCE_ROOT/jobs/sirius/rocfft_build_smoke.sh"
```

The default [rocfft-smoke.json](rocfft-smoke.json) is a smoke workload, not a
production measurement configuration. `SMOKE_CONFIG` may select a separately
reviewed rocFFT-only smoke configuration, including a threaded FFTW series with
an explicit allocation-appropriate count. The configuration must stop on first
failure (`continue_on_failure=false`); no serial substitution is permitted.
The script records its actual configuration and all provenance. Further site
inputs required before execution are the allocation mechanism/queue/account,
compute-node filesystem/output policy, available resources and GPU target,
compiler/library roots and runtime search path, and permitted CPU/thread/build
counts. Missing dependencies or device interoperation remain real-machine
checks, not local fixture passes.

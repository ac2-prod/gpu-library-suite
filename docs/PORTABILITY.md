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
- Configuration, execution, validation, and aggregation code under `configs`,
  `tools`, and `tests` uses Python 3.9+ standard-library behavior. Plotting also
  needs matplotlib to produce PNGs. Plot labels come from recorded identities
  or an explicit display configuration; unknown hardware is never replaced
  with the teaching machine. cuSOLVER ticks follow the actual problem sizes.
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

Project policy forbids unapproved fast math, TF32, Tensor Core modes, and
hardware-specific numerical changes. That policy is not proof that inherited
compiler defaults are numerically strict: the saved NVHPC OpenACC Release
metadata contains `-fast -O3 -DNDEBUG`, and the current CMake layer records but
does not reject `-fast`. The target-specific flags and documented floating-point
implications are reviewed in the
[saved-build audit](VALIDATION_REPORT.md#nvhpc-release-flag-audit);
policy approval and the original runtime FPU state remain unresolved. In particular,
the trailing `-O3` does not cancel the other components of `-fast`;
do not silently change the saved condition or describe it as audited strict math.

CPU, CUDA, and OpenACC series use the same configured problem, precision,
scope, repeat count, and verification contract. An out-of-memory condition is
a visible failed trial; it does not silently reduce the problem.

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

## Measuring on your own system

This is a reader recipe, not a replacement for the
[benchmark protocol](BENCHMARK_PROTOCOL.md) or [result schema](RESULT_SCHEMA.md).
All shell blocks below use **Bash, from the repository root, in one session**.
Commands are source/CLI-checked instructions, not a claim that this workflow was
executed on your machine. Keep the generated files in each fresh run directory.

Start with [cuBLAS's first build and run](../nvidia/c-cpp/cublas/README.md#first-build-and-run).
That CPU-only route needs neither a GPU nor Pegasus. Then compare its CUDA and
OpenACC examples if the dependencies are available. The six library READMEs
linked from the [root table](../README.md#what-is-implemented) give the remaining
example commands and checks.

### 1. Prepare dependencies and separate builds

| Work you want to run | Required environment |
| --- | --- |
| cuRAND/Thrust CPU examples | C++17 compiler; no CUDA or optimized CPU library |
| Other CPU examples/benchmarks | C17/C++17 plus FFTW3f, a selected CBLAS/LAPACKE provider, and oneMKL Sparse where requested; threaded FFTW additionally needs `fftw3f_threads` |
| Direct CUDA | Compatible NVIDIA GPU/driver, CUDA Toolkit with the relevant libraries and Thrust/CCCL, compatible host compiler |
| OpenACC library calls | The same GPU and external CUDA Toolkit, plus NVHPC `nvc`/`nvc++`; OpenACC calls the CUDA libraries using managed device data |
| Complete canonical CPU/CUDA/OpenACC comparison | All of the above, FFTW threaded and oneMKL CPU backends, Python 3.9+, CMake 3.20+, and a single-config build tool |
| PNG output from existing data | Python 3.9+ and matplotlib; no GPU/compiler/Pegasus access is needed |

Have your installed toolchain/provider environment ready first; these commands
do not install dependencies or choose modules. For the full comparison, set
`CUDA_TOOLKIT_ROOT` to the actual external Toolkit root, `CUDA_ARCHITECTURES` to
the CMake architecture value(s) for your GPU, and `NVHPC_GPU_TARGET` to the
corresponding NVHPC target. Export `MKLROOT` and, for a non-default FFTW installation,
`FFTW_ROOT` with their real installed prefixes. Do not copy Pegasus paths or
architecture values onto another machine.

The local suite path below targets Linux, with or without a module system.
It uses the existing dependency/GPU probes and single-host preparation tools;
it does not create a scheduler allocation. A missing module system is recorded
explicitly, while missing required binaries, dependencies or GPU probes still
fail. A multi-GPU machine needs an explicitly selected GPU UUID as described
below; no machine/queue/module/version is guessed.

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your installed external CUDA Toolkit root}"
: "${CUDA_ARCHITECTURES:?Set your GPU's CMake CUDA architectures}"
: "${NVHPC_GPU_TARGET:?Set your GPU's NVHPC target}"
export NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
export CUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT"
export CUDA_HOME="$CUDA_TOOLKIT_ROOT"
export CUDA_PATH="$CUDA_TOOLKIT_ROOT"
export PATH="$CUDA_TOOLKIT_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p build results
BUILD_ROOT="$(mktemp -d "$PWD/build/reader.XXXXXX")"
RUN_DIR="$(mktemp -d "$PWD/results/local-pilot.XXXXXX")"
export CPU_CUDA_BUILD="$BUILD_ROOT/cpu-cuda"
export OPENACC_BUILD="$BUILD_ROOT/openacc"
set -o noclobber

cmake -S . -B "$CPU_CUDA_BUILD" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DGPU_SUITE_BUILD_TESTS=OFF \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_CUDA=ON \
  -DGPU_SUITE_BUILD_OPENACC=OFF \
  -DCMAKE_CUDA_COMPILER="$CUDA_TOOLKIT_ROOT/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
  -DGPU_SUITE_CPU_BLAS_BACKEND=ONEMKL \
  -DGPU_SUITE_CPU_SPARSE_BACKEND=ONEMKL \
  -DGPU_SUITE_CPU_LAPACK_BACKEND=ONEMKL
cmake --build "$CPU_CUDA_BUILD" --target gpu_suite_partial_manifest --parallel 2

cmake -S . -B "$OPENACC_BUILD" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DGPU_SUITE_BUILD_TESTS=OFF \
  -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=ON \
  -DCUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="$NVHPC_GPU_TARGET"
cmake --build "$OPENACC_BUILD" --target gpu_suite_partial_manifest --parallel 2
```

Inspect each configure summary and the compiler flags in `build-metadata.json`;
resolve the [numerical-policy caveat](#numerical-portability) before treating a
new measurement as an approved comparison. Stop on a failed command. The manifest
target builds all enabled examples/benchmarks and writes `partial-manifest.json`
in each build tree; configure writes `build-metadata.json` there. A successful
configure does not mean all optional targets were enabled. For all six
three-way comparisons, all 18 benchmark targets and their CPU backends must be
available. A missing NVHPC/FFTW/oneMKL/backend is an incomplete comparison, not
permission to replace or silently remove that series.

Run the enabled teaching examples using each library README's **Run the
examples** section (cuBLAS: **First build and run**). Those sections honor the
two exported build directory variables above. Record nonzero exits and inspect
the numerical output before proceeding to benchmarks.

### 2. Make a small measurement check, then choose the suite settings

After the cuBLAS example succeeds, this modest CPU benchmark checks the CLI,
verification and result format without the publication workload:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE \
"$CPU_CUDA_BUILD/nvidia/c-cpp/cublas/blas_cpu_bench" \
  --size 128 --alpha 1 --beta 1 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --abs-tolerance 1e-12 \
  --rel-tolerance 1e-10 --cpu-backend cpu-onemkl --cpu-threads 1 \
  --cpu-backend-role production --series-role primary --cpu-parallelism threaded \
  --output "$RUN_DIR/standalone-cublas.jsonl" --format jsonl \
  2> "$RUN_DIR/standalone-cublas.stderr"
```

Check exit status, `status="success"`, `verification_status="pass"`, and the
metrics/thresholds in that new JSONL file. This standalone record is a diagnostic
check, not a complete suite/campaign input; do **not** mix it with the runner's
raw file below. For GPU smoke checks, the matching benchmark executable accepts
the same size/scope/verification options without CPU-backend options. Follow the
library README and use a different output filename for each invocation.

Now prepare the actual suite inputs:

```bash
python3 tools/merge_manifests.py --output "$RUN_DIR/executables.json" \
  "$CPU_CUDA_BUILD/partial-manifest.json" \
  "$OPENACC_BUILD/partial-manifest.json"
cp configs/pilot.json "$RUN_DIR/config-input.json"
CONFIG_INPUT="$RUN_DIR/config-input.json"
MANIFEST="$RUN_DIR/executables.json"
```

`pilot.json` uses one trial but **the same large sizes and compute repeats as
the publication campaign**; it is not a low-memory smoke preset. Before running
it, edit the newly created config-input copy in a text editor if your own
experiment needs smaller cases. Keep the repository's canonical files and
previous runs unchanged. This saved copy becomes immutable once measurement
starts; a later change requires a fresh run directory/ID.

For a concrete first **three-way suite diagnostic**, set `run_mode="smoke"`,
`cpu_threads=1`, and only `benchmarks.cublas.enabled=true` (set the other five
`enabled` values to false, but keep their objects). Retain only the first cuBLAS
case, change its `size` to 128, and set `warmup=1`, `repeat=1`, `trials=1` in
both scopes. Keep `alpha=beta=1`, the three primary series, oneMKL's effective
count null, and the existing tolerances. This produces six expected rows:
one case x two scopes x three implementations x one trial. It can be validated
and aggregated below, but **cannot generate the six publication figures**.
For a later full sweep, start a fresh run directory/configuration; unchanged
build trees can be reused rather than rebuilt solely to change problem sizes.

| JSON location in the input copy (then serialized as effective config) | Meaning and safe use |
| --- | --- |
| `run_mode` | Use `smoke` for a reduced diagnostic or `pilot` for a preliminary sweep; `production` requires clean, known Git provenance and Release binaries |
| `benchmarks.<library>.cases[i].parameters` | Set the workload size explicitly; e.g. cuBLAS `size=128`, cuFFT `nfft=256,batch=8`, cuSPARSE a perfect-square `size=4096`, cuSOLVER `size=128,nrhs=16`, cuRAND/Thrust `size=1048576` are diagnostic candidates, not measured performance recommendations |
| `cases[i].scopes.compute` and `.end-to-end` | `warmup`, `repeat`, `trials`; a small check can use `1,1,1`. cuSOLVER repeat must stay 1; publication E2E repeat stays 1 |
| `cpu_threads` | Requested count within your allocation; also controls the environment set in step 3 |
| `benchmarks.<library>.series` | Explicit backend/role/thread metadata; retain primary CPU/CUDA/OpenACC entries. FFTW effective count follows its configured thread API; oneMKL stays null when unobserved; cuRAND/Thrust stay effective 1 |
| `output_format` | Keep `jsonl` for this route: validation/aggregation also accept CSV, but the current plot raw-input loader requires JSONL |
| `verification`, `precision`, algorithm parameters | Keep the approved mathematical/precision/verification contract. A precision string is not a switch that recompiles the algorithm; never relax tolerances merely to obtain a fast successful result |

Configuration requires all six library objects and both scopes. You may mark a
whole library `enabled=false` for an explicitly partial diagnostic, but cannot
remove only a required primary implementation and call it a complete comparison.
The plotter has stricter requirements: three distinct cases per library, both
scopes, and all three implementations. A one-case or single-library smoke is
not a figure input. CPU backend names come from the data, and cuSOLVER ticks
now follow the cases rather than assuming publication sizes.

After editing, create the immutable effective copy using the canonical JSON
serializer. The input copy may be pretty-printed; the effective copy must use
the deterministic representation required by campaign preparation:

```bash
CONFIG="$RUN_DIR/effective-config.json"
PYTHONPATH=tools python3 - "$CONFIG_INPUT" "$CONFIG" <<'PY'
import sys
from pathlib import Path
from gpu_suite.config import load_config
from gpu_suite.strict_json import dump_bytes
with Path(sys.argv[2]).open("xb") as output:
    output.write(dump_bytes(load_config(sys.argv[1])))
PY
```

### 3. Capture the real runtime environment and launch one local block

Stay in the same configured toolchain shell. These exports set requests, not
measurements of actual CPU utilization. No application OpenMP loop is added.

```bash
CPU_THREADS="$(PYTHONPATH=tools python3 -c \
  'import sys; from gpu_suite.config import load_config; print(load_config(sys.argv[1])["cpu_threads"])' \
  "$CONFIG")"
export OMP_NUM_THREADS="$CPU_THREADS" OMP_PROC_BIND=spread OMP_PLACES=cores
export OMP_DYNAMIC=FALSE MKL_NUM_THREADS="$CPU_THREADS" MKL_DYNAMIC=FALSE
export MKL_THREADING_LAYER=INTEL OPENBLAS_NUM_THREADS="$CPU_THREADS"
CUDA_TOOLKIT_VERSION="$(PYTHONPATH=tools python3 -c \
  'import sys; from gpu_suite.strict_json import load; print(load(sys.argv[1])["cuda"]["toolkit_version"])' \
  "$CPU_CUDA_BUILD/build-metadata.json")"
MODULE_OPTIONS=(--no-module-system)
# If your shell actually uses modules, replace that declaration with:
# module list > "$RUN_DIR/module-list.txt" 2>&1
# MODULE_OPTIONS=(--module-list "$RUN_DIR/module-list.txt")
GPU_SELECTION_ARGS=()
RUNTIME_HASH="$(python3 jobs/pegasus/collect_runtime_environment.py \
  --manifest "$MANIFEST" \
  --build-metadata "$CPU_CUDA_BUILD/build-metadata.json" \
  --build-metadata "$OPENACC_BUILD/build-metadata.json" \
  "${MODULE_OPTIONS[@]}" \
  --cuda-toolkit-root "$CUDA_TOOLKIT_ROOT" \
  --cuda-toolkit-version "$CUDA_TOOLKIT_VERSION" \
  --nvhpc-cuda-home "$NVHPC_CUDA_HOME" \
  --require-ldd --require-gpu-tools \
  --output "$RUN_DIR/runtime-environment.json" \
  --evidence-output "$RUN_DIR/runtime-environment-evidence.json")"
```

Although located under `jobs/pegasus`, this collection command does not submit a
job or require an account/queue. `--no-module-system` records an explicit
user declaration, an empty identity list, and no fabricated module-command
output/hash. When using modules, supply their real accepted list instead;
a failed/empty module command is not automatically reclassified as module-free.
All eight CPU variables, consistent Toolkit roots/version, canonical absolute
search paths, and resolved binary dependencies are still required. If the
collector exits nonzero, **stop here**. The two
`--require-*` flags make dependency and GPU-runtime checks mandatory; optional
compiler/package-version probes may instead be recorded as unavailable. Preserve
that distinction rather than claiming an unavailable version was observed.
Do not invent a module identity, use a dummy hash, or hash an arbitrary `env`
dump in place of the specified runtime document.

Before collection on a multi-GPU machine, select the UUID observed by
`nvidia-smi`, export that exact UUID as `CUDA_VISIBLE_DEVICES`, and set
`GPU_SELECTION_ARGS=(--gpu-uuid "$CUDA_VISIBLE_DEVICES")` after the empty-array
initialization above. The local worker uses logical device 0. Local preparation
rejects ambiguous/mismatched visibility instead of guessing the GPU.

The saved runtime document embeds build metadata and binary/dependency hashes;
the command prints its exact SHA-256 as `RUNTIME_HASH`. Keep both build trees
unchanged because the manifest's absolute executable paths and hashes are used
at launch. Prepare the actual local run/wave/node provenance next. This uses
the observed hostname, `lscpu` CPU identity when available, and independent GPU
queries. Scheduler fields stay null; no PBS identifier or job state is invented.
`PROVENANCE_ARGS` is empty only for the clean-build case assumed by this recipe:

```bash
RUN_ID="$(basename "$RUN_DIR")"
SYSTEM_LABEL=local-nvidia
PROVENANCE_ARGS=()
mkdir "$RUN_DIR/campaigns"
NODE_DIR="$(python3 jobs/pegasus/prepare_wave.py --local \
  --result-root "$RUN_DIR/campaigns" --run-id "$RUN_ID" --wave 0 \
  --config "$CONFIG" --manifest "$MANIFEST" \
  --build-metadata "$CPU_CUDA_BUILD/build-metadata.json" \
  --build-metadata "$OPENACC_BUILD/build-metadata.json" \
  --runtime-environment "$RUN_DIR/runtime-environment.json" \
  --runtime-environment-evidence "$RUN_DIR/runtime-environment-evidence.json" \
  --system-label "$SYSTEM_LABEL" "${GPU_SELECTION_ARGS[@]}" "${PROVENANCE_ARGS[@]}")"
python3 tools/run_suite.py --config "$CONFIG" --manifest "$MANIFEST" \
  --build-metadata "$CPU_CUDA_BUILD/build-metadata.json" \
  --build-metadata "$OPENACC_BUILD/build-metadata.json" \
  --run-id "$RUN_ID" --system-label "$SYSTEM_LABEL" --wave 0 --node-index 0 \
  --runtime-environment-sha256 "$RUNTIME_HASH" "${PROVENANCE_ARGS[@]}" --dry-run \
  > "$RUN_DIR/resolved-plan.json"
```

Inspect the plan's selected binaries, parameters, CPU backends, ordering and
`prerequisite_failure` fields before execution. This command checks artifacts
but does not run them. A dirty build needs one complete source/diff hash as
specified in [source provenance](RESULT_SCHEMA.md#executables-manifest-and-source-provenance); do not
describe an archive without Git metadata as a clean production build. The recipe
assumes clean build metadata; if that is not your state, set `PROVENANCE_ARGS`
to the appropriate `--git-diff-sha256` or `--source-snapshot-sha256` and the
authoritative build-time hash **before preparation**, and pass it unchanged to
both runner commands. Do not substitute a made-up value or hash changed sources
after the binary was built.

```bash
python3 tools/run_suite.py --config "$CONFIG" --manifest "$MANIFEST" \
  --build-metadata "$CPU_CUDA_BUILD/build-metadata.json" \
  --build-metadata "$OPENACC_BUILD/build-metadata.json" \
  --run-id "$RUN_ID" --system-label "$SYSTEM_LABEL" --wave 0 --node-index 0 \
  --runtime-environment-sha256 "$RUNTIME_HASH" "${PROVENANCE_ARGS[@]}" \
  --output "$NODE_DIR/raw-results.jsonl" \
  2> "$NODE_DIR/runner.stderr"
RUNNER_RC=$?
printf 'runner exit status: %s\n' "$RUNNER_RC"
RAW_FILES=("$NODE_DIR/raw-results.jsonl")
PLOT_OPTIONS=(--node-metadata "$NODE_DIR/node-metadata.json")
AGGREGATE_OPTIONS=()
ANALYSIS_DIR="$RUN_DIR/analysis"
mkdir "$ANALYSIS_DIR"
python3 jobs/pegasus/node_tools.py classify --raw "${RAW_FILES[0]}" \
  --node-metadata "$NODE_DIR/node-metadata.json" \
  --output "$ANALYSIS_DIR/node-classification.json"
```

The runner interleaves implementations and exclusively creates the raw file.
Run/wave/node metadata were created by the preparation command, not the runner.
The local path does not invent job completion/collection status and does not
implement telemetry or a multi-node allocation. This is a **single-node pilot analysis**, not
the six-node/two-wave teaching result. On failure retain raw rows and stderr;
do not retry into the same output path. Use the validation step next to inspect
failure records, and do not advance to performance figures as though they passed.
For the full Pegasus launcher, metadata and recovery workflow use
[the dedicated procedure](PEGASUS_EXECUTION.md) instead; other systems need a
corresponding launcher, not invented PBS identifiers.

## Validate and aggregate a completed run

Inputs are `CONFIG`, `MANIFEST`, and the Bash `RAW_FILES` array set by the local
route above or the saved-data route below. `ANALYSIS_DIR` must be a fresh,
already-created output directory. Do not glob in standalone diagnostic files or
mix configurations, binaries, runtime hashes, or run IDs.

```bash
python3 tools/validate_results.py --config "$CONFIG" --manifest "$MANIFEST" \
  --report-output "$ANALYSIS_DIR/validation-report.json" "${RAW_FILES[@]}"
python3 -m json.tool "$ANALYSIS_DIR/validation-report.json"
```

Require exit 0, `validation_status="pass"`, and only successful statuses for a
complete comparison. Inspect raw `verification_status`, metrics and thresholds;
they must show the requested numerical verification passed. The validator also
checks configured cases/trials, ordering and manifest/config provenance. Its
offline manifest check does not require the original executable paths to exist.
It is not a substitute for node-local runtime/telemetry checks in a production
launcher. If validation fails, keep the report and raw data, diagnose the reason,
and stop the successful-comparison route.

```bash
python3 tools/aggregate.py --config "$CONFIG" "${AGGREGATE_OPTIONS[@]}" \
  --output "$ANALYSIS_DIR/aggregate-results.jsonl" \
  --metadata-output "$ANALYSIS_DIR/aggregate-metadata.json" "${RAW_FILES[@]}"
AGGREGATE_INPUT="$ANALYSIS_DIR/aggregate-results.jsonl"
```

The output contains block, wave, cross-wave and separately labeled exploratory
records. Plotting selects primary cross-wave elapsed-time records, not speedup
records or a re-pooling of every trial. Inspect aggregate sample/failure counts
and `aggregate_status`; one wave/block can have a median but insufficient data
for quartiles and for claims about variability. The aggregation implementation
retains failure counts while excluding failed/skipped/nonfinite rows from
numeric samples. Do not remove slow successful nodes or repair raw records.

## Figures from your own measurements

This route aims at the **same method and two-panel structure**, not the same
elapsed times or ranking as the teaching machine. The current plotter requires:

- primary cross-wave records from one run/runtime provenance;
- cuFFT, cuBLAS, cuSPARSE, cuSOLVER and cuRAND; Thrust is optional to the tool,
  but must be included for this material's complete six-library result;
- exactly three matched cases and CPU/CUDA/OpenACC series in both scopes;
- the actual CPU backend identity for each primary CPU series; and
- successful Thrust CUDA/OpenACC raw rows with one identical library version.

GPU names come from successful raw rows. `--node-metadata` supplies observed
CPU names for the local route; absent identities are labeled unknown. Requested
and reported effective CPU counts remain separate, and known backend IDs have
readable names (other IDs are displayed verbatim). Mixed machine models are
rejected instead of hidden behind one label. cuSOLVER ticks use the three actual
sizes, retaining `4K/8K/12K` for the teaching cases.
Normal legends are stacked below the same two panels, with long labels wrapped;
the explicit teaching-display mode retains the approved three-column legend.

If an identity cannot be observed, supply the exact
[display configuration](RESULT_SCHEMA.md#plot-display-configuration) through
`--display-config`. A supplied value is marked `user-specified` in plot metadata;
it cannot contradict an observed identity or the run/runtime provenance.
For this route use `thread_label_mode="requested-and-effective"`. Set
`PLOT_OPTIONS=(--display-config "$RUN_DIR/plot-display.json" --node-metadata "$NODE_DIR/node-metadata.json")`
only after creating that configuration with your actual values. Do not write
an identity merely to make a legend look complete. `--system-label` remains a
campaign identifier, not a hardware override.

After these checks, continue to [Render and inspect figures](#render-and-inspect-figures).
The saved-data reference below is not required for your own measurements.

## Regenerating the teaching figures from saved data

**Not included in this code publication:** the measured data and prepared local
distribution archives remain private, outside tracked source. There is no
Release/data URL to follow. The commands below are a conditional reference for
readers who separately hold the required saved inputs; source checkout does not
supply them. The normal reader route is
[figures from your own measurements](#figures-from-your-own-measurements).

The saved campaign is `publication-benchmark-20260716T025037Z`, measured with
commit `916a8bda22f6188aa995d21954ddb1d6de2f3cf3`. Its approved renderer is commit
`4d3c2333d2a58aa3fc2a8a82d43c403d2131343f`; only plotting and its tests changed
between those revisions. The updated reader tools preserve that display when
the bundle's explicit `plot-display.json` is supplied. Use the code revision
and working-tree/source hash recorded in the bundle README, with the **saved**
effective configuration, not a newly
edited config. The [saved evidence](VALIDATION_REPORT.md#saved-execution-evidence-reviewed-for-publication)
and [publication conditions](BENCHMARK_PROTOCOL.md#canonical-publication-configuration)
describe six nodes per wave, two waves, five trials, three cases, both scopes,
and all six libraries (6,480 rows). The six output filenames map directly to
the six library names in [Publication figures](BENCHMARK_PROTOCOL.md#publication-figures).

| Purpose | Required data; what the current tools actually read |
| --- | --- |
| Render the six PNGs from existing summaries | `publication-output/aggregate-results.jsonl`, JSONL raw evidence from the same campaign, and `plot-display.json` for the approved labels. The CLI always requires raw input; Thrust checks successful CUDA/OpenACC `library_version` and matching run/runtime identity |
| Validate and rebuild the summaries as well | All 12 original node `raw-results.jsonl` files, byte-identical `effective-config.json`, and `executables-manifest.json`. Validation reads the manifest as metadata; no original GPU binaries are needed for this offline path |
| Explain and audit the measured conditions | `run-metadata.json`, `runtime-environment.json`, wave/node metadata, each node's CPU hardware evidence (`telemetry/cpu-telemetry.txt` in the saved campaign), and a checksum inventory; these are provenance/caption evidence, not positional inputs to `plot.py`. Node metadata alone does not contain the CPU model |
| Useful references, not required plot inputs | Saved validation/aggregate/plot metadata and approved PNGs. PNGs alone are not regeneration inputs |

The completed private replay used all 12 original raw files, not just two
Thrust rows sufficient for the renderer's version check, together with the
aggregate, effective config, manifest and provenance/CPU hardware evidence.
Full build logs, continuous GPU telemetry, PBS logs and binary distributions
are not required merely to render PNGs. Original artifacts remain preserved;
the prepared archives are private validation artifacts, not deliverables of
this code publication.

For the conditional commands below, use Bash from the repository root and set
`DATA_DIR` to the root of separately supplied saved data with the relative layout
shown below. Its absolute original executable paths are historical manifest
values, not paths a replaying user must possess. The replay route does not
depend on shell variables from the measurement route:

```bash
: "${DATA_DIR:?Separately supplied saved data is required; not included with source}"
(cd "$DATA_DIR" && shasum -a 256 -c SHA256SUMS)
CONFIG="$DATA_DIR/effective-config.json"
MANIFEST="$DATA_DIR/executables-manifest.json"
RAW_FILES=("$DATA_DIR"/waves/*/nodes/*/raw-results.jsonl)
AGGREGATE_INPUT="$DATA_DIR/publication-output/aggregate-results.jsonl"
PLOT_OPTIONS=(--display-config "$DATA_DIR/plot-display.json")
AGGREGATE_OPTIONS=(--no-pooled)
test -f "$CONFIG" && test -f "$MANIFEST" && test -f "$AGGREGATE_INPUT"
test "${#RAW_FILES[@]}" -eq 12
mkdir -p results
ANALYSIS_DIR="$(mktemp -d "$PWD/results/replay.XXXXXX")"
```

Stop if files/checksums are missing or these checks fail. Use
[validation](#validate-and-aggregate-a-completed-run) to inspect the supplied
raw campaign. To render the supplied aggregate directly, keep `AGGREGATE_INPUT`
as set above and proceed to rendering; rerunning aggregation is optional. If
you choose to rebuild summaries, the aggregation command writes a new file in
`ANALYSIS_DIR` and resets `AGGREGATE_INPUT` to that new file. Never overwrite
the bundle. `--no-pooled` matches the saved 2,700-row aggregate: the default
additionally emits 180 labeled exploratory pooled rows, without changing the
block/wave/cross-wave values used by the figures. No GPU or Pegasus login is
needed for either offline path.

## Render and inspect figures

For your own measurements, run this after the annotation/series requirements
above are satisfied. The conditional saved-data route also uses this command,
but only when its separate input prerequisites are met. Inputs are the existing
`AGGREGATE_INPUT` and original JSONL `RAW_FILES`; the two outputs below must
not already exist.

```bash
mkdir "$ANALYSIS_DIR/plots"
python3 tools/plot.py "$AGGREGATE_INPUT" \
  --output-directory "$ANALYSIS_DIR/plots" \
  --metadata-output "$ANALYSIS_DIR/plot-metadata.json" \
  --raw-results "${RAW_FILES[@]}" "${PLOT_OPTIONS[@]}"
python3 -m json.tool "$ANALYSIS_DIR/plot-metadata.json"
ls "$ANALYSIS_DIR/plots"
```

Check `matplotlib.status="rendered"`, all six expected `generated_files`, and
the actual PNGs. Exit 0 alone is insufficient: when matplotlib is unavailable,
the tool records `unexecuted` and can exit 0 without PNGs. Open the images in an
image viewer and check the two titles, shared bottom legend, hardware/backend/
thread labels, three points per implementation, elapsed-time units and binary
ticks. Follow [the interpretation guide](BENCHMARK_PROTOCOL.md#reading-the-figures-and-applying-the-results).
Different matplotlib/font/platform versions can change raster appearance;
using the same data/layout does not promise byte-identical PNG files. Record
the renderer revision and the matplotlib version saved in plot metadata.

## Known limitations and out-of-scope work

Machine labels, variable cuSOLVER ticks, explicit module-free collection and
single-host non-PBS run/wave/node preparation are implemented. The following
limits remain visible; additional infrastructure, GPU diagnostics and saved-data
distribution are not prerequisites for the selected
[code-publication scope](../README.md#publication-scope).

| Area | Current limit |
| --- | --- |
| GPU execution of the local reader path | A full GPU reader workflow on a Linux NVIDIA/NVHPC installation without modules/PBS has not been executed. CPU-only fixtures and saved-data replay do not certify that environment |
| Multi-node orchestration and telemetry outside Pegasus | Not implemented by the single-host recipe and outside this publication's development scope. No fake scheduler status is produced |
| Numerical-policy review of inherited NVHPC Release flags | The recorded `-fast` condition and unknown original runtime FPU state remain disclosed, not declared policy-compliant. This publication neither introduces a numerical profile nor remeasures the campaign; saved artifacts remain unchanged |
| Teaching-data replay | Saved inputs and prepared archives are excluded from this publication. The reference recipe requires separately supplied data; the normal reader workflow measures its own data |

The [validation report](VALIDATION_REPORT.md#reader-tools-and-offline-replay-validation)
separates local tool tests and offline replay from GPU execution and publication.

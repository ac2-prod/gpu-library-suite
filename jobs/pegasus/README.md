# Pegasus job support

This directory renders and supports Pegasus jobs; it never submits, queries, or
cancels a job. Account, queue, module versions, MPI version, CUDA Toolkit,
architecture, GPU target, and shared paths are human-supplied values. The
normative operating rules are in
[`docs/PEGASUS_EXECUTION.md`](../../docs/PEGASUS_EXECUTION.md).

## Configuration and separate builds

Copy `pegasus.json.example` to a private, repository-local or otherwise
appropriate configuration path and replace every placeholder after manually
checking the current Pegasus environment. The three module arrays have separate
ownership:

- `cpu_cuda_build_modules` is loaded only by `build_cpu_cuda.sh`;
- `openacc_build_modules` is loaded only by `build_openacc.sh`; and
- `benchmark_runtime_modules` is loaded only by the rendered measurement job.

The build scripts do not invoke a scheduler. Run them manually on an appropriate
build node with distinct absolute build directories:

```bash
bash jobs/pegasus/build_cpu_cuda.sh \
  --config /path/to/pegasus.json \
  --source /path/to/repository \
  --build /path/to/build/cpu-cuda

bash jobs/pegasus/build_openacc.sh \
  --config /path/to/pegasus.json \
  --source /path/to/repository \
  --build /path/to/build/openacc
```

The first profile fixes CPU ON, CUDA ON, and OpenACC OFF. The second fixes
`nvc`/`nvc++`, CPU OFF, CUDA OFF, and OpenACC ON. Both build
`gpu_suite_partial_manifest`; merge the two outputs with
`tools/merge_manifests.py`. A provider/profile collision is an error.

## Rendering and human submission

Render only after the configuration, merged manifest, both build-metadata
documents, and canonical benchmark configuration are final:

```bash
python3 jobs/pegasus/render_job.py \
  --config /path/to/pegasus.json \
  --repository-root /path/to/repository \
  --benchmark-config /path/to/repository/configs/pilot.json \
  --manifest /path/to/executables.json \
  --build-metadata /path/to/build/cpu-cuda/build-metadata.json \
  --build-metadata /path/to/build/openacc/build-metadata.json \
  --run-id HUMAN_SELECTED_RUN_ID \
  --wave 0 \
  --system-label HUMAN_SELECTED_LABEL \
  --output /path/to/rendered-run.pbs
```

For a dirty smoke/pilot build, add exactly one complete
`--git-diff-sha256` or `--source-snapshot-sha256`. Production rejects dirty
builds. When untracked, non-ignored inputs are present, generate the latter from
the exact built source tree without changing Git state:

```bash
python3 tools/hash_source_snapshot.py /path/to/repository
```

Inspect the rendered file and run `bash -n` before a human submits it. The
renderer itself never runs `qsub` or any Pegasus inspection command.
Both configured PBS stdout and stderr parent directories must already exist
before rendering/submission. The renderer validates those parents and never
creates them implicitly.

The benchmark runtime preflight requires the NVIDIA driver/runtime and every
resolved shared library needed by the prebuilt binaries. `nvcc`, `nvc`, and
`nvc++` are collected as optional metadata and their absence does not fail the
standard runtime preflight. A site that requires runtime compiler commands must
set `require_runtime_compilers=true` and document that separate policy; the
canonical example defaults it to false.

## Runtime flow and failure ownership

The rendered job loads only benchmark runtime modules and the selected OpenMPI,
exports all eight configured CPU runtime variables, and then performs this
sequence:

1. collect and hash canonical `runtime-environment.json` plus the linked
   per-wave `runtime-environment-evidence.json` raw-probe sidecar;
2. run a short rank/hostname preflight;
3. let `prepare_wave.py` exclusively validate/create campaign and wave metadata;
4. launch one `run_node.sh` process per node; and
5. always run `collect_results.py` after the measurement launch; it also checks
   the sidecar hash and its canonical-runtime/manifest links.

Each node writes measurement artifacts to its own `/scr` directory. Its EXIT
trap stops telemetry and copies artifacts only into that hostname's shared node
subtree. A benchmark, verification, or nonfatal telemetry/tool failure is
recorded in `node-status.json`; after successful recovery the node process exits
zero so one failed benchmark does not make `mpirun` kill other nodes. Scratch,
shared-directory, status-write, or artifact-copy failure is fatal and may exit
nonzero.

Before node metadata is written, each node independently queries device 0 for
GPU name, UUID, and NVIDIA package-driver version. It also loads node-local
`libcudart` and queries CUDA Driver API and Runtime versions without requiring a
compiler. Successful CUDA/OpenACC raw rows are checked against that node's name,
UUID, Driver API, and Runtime values. An unavailable query is stored as null
plus a diagnostic; no job-master or neighboring-node identity is substituted.

The collector, not an individual node, owns final job success. Missing status,
benchmark failure, verification failure, or collection failure makes the
collector and job fail after all available artifacts have been retained and
hashed. It never aggregates, filters, deletes, or rewrites raw results.

GPU and NVHPC execution remains a manual Pegasus validation requirement. Local
tests exercise rendering, provenance, ordering, telemetry parsing, midnight
rollover, trap recovery, and multi-node failure isolation with synthetic data;
they do not claim that a Pegasus job or GPU workload was executed.

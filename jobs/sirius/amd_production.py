#!/usr/bin/env python3
"""Six independent AMD workers in an existing PBS allocation; never submit.

The Pegasus launcher supplies the ownership model: MPI observation, master-only
metadata, independent node writers, then collection. Site placement is an
explicit input, not inferred from select=6 or from an MPI local rank.
"""

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_suite import runner
from gpu_suite.config import load_config
from gpu_suite.families import AMD_BENCHMARKS
from gpu_suite.hashing import sha256_file
from gpu_suite.ordering import implementation_order, permutation_index, size_order_index
from gpu_suite.results_io import load_raw_results
from gpu_suite.strict_json import dump_bytes, load
from gpu_suite.validation import validate_campaign
from amd_smoke import Processes, StopRequested, request_stop, smoke_environment, preflight_link_evidence
from amd_smoke_campaign import _numerical_pass
import amd_smoke_evidence


NODES = 6
INVOCATIONS = 108
ROWS = 540
VISIBILITY = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
              "GPU_DEVICE_ORDINAL", "OMP_DEFAULT_DEVICE")
# Job IDs, hostnames, affinity, GPU identity and node-local cache paths belong
# to observations. These software/thread controls remain campaign identity.
ENVIRONMENT = (
    "PATH", "LD_LIBRARY_PATH", "ROCM_PATH", "HIP_PATH", "AOCL_ROOT", "LOADEDMODULES",
    "HSA_XNACK", "OMP_TARGET_OFFLOAD", "OMP_NUM_THREADS", "OMP_DYNAMIC",
    "OMP_THREAD_LIMIT", "OMP_PROC_BIND", "OMP_PLACES", "AOCLSPARSE_NUM_THREADS",
    "AOCL_ENABLE_INSTRUCTIONS", "BLIS_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "MKL_DYNAMIC", "MKL_THREADING_LAYER", "ROCBLAS_LAYER",
    "ROCBLAS_TENSILE_LIBPATH", "ROCBLAS_USE_HIPBLASLT", "HIPBLASLT_TENSILE_LIBPATH",
)
HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]{0,253}\Z")
# This bounded tool-only revision cannot change any benchmark/build input.
# Both the artifact commit and the later clean launcher commit remain recorded.
TOOL_REVISION_PATHS = frozenset({
    "jobs/sirius/README.md", "jobs/sirius/amd_prepare.py", "jobs/sirius/amd_wave.py",
    "jobs/sirius/amd_production.py", "jobs/sirius/amd_smoke_evidence.py",
    "tests/python/run_profiled_unittests.py", "tests/python/test_sirius_amd_prepare.py",
    "tests/python/test_sirius_amd_production.py", "tests/python/test_sirius_amd_smoke_evidence.py",
    "docs/RESULT_SCHEMA.md",
})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(document):
    return hashlib.sha256(dump_bytes(document)).hexdigest()


def save(path, document):
    with Path(path).open("xb") as stream:
        stream.write(dump_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())


def checked_file(identity):
    require(isinstance(identity, dict) and set(identity) == {"path", "sha256"},
            "Expected a path/SHA-256 identity")
    path = Path(identity["path"])
    runner.validate_sha256(identity["sha256"], "input SHA-256")
    require(path.is_absolute() and path.is_file() and sha256_file(path) == identity["sha256"],
            "Missing or changed input: " + str(path))
    return path


def checked_json(identity):
    return load(checked_file(identity))


def validate_placement(placement, job_id):
    require(isinstance(placement, dict) and placement.get("status") == "site-verified",
            "Physical-host/APU placement is unresolved")
    require(placement.get("scheduler_job_id") == job_id and placement.get("method"),
            "Placement belongs to another allocation or has no confirmed method")
    checked_file(placement["evidence"])
    records = placement.get("rank_bindings", [])
    require(len(records) == NODES and {r["rank"] for r in records} == set(range(NODES)),
            "Placement requires exactly ranks 0..5")
    hosts = []
    for row in records:
        host = row["hostname"]
        require(isinstance(host, str) and HOST.fullmatch(host), "Unsafe hostname")
        hosts.append(host.split(".")[0].lower())
        cpus = row["affinity_cpus"]
        require(len(cpus) == 24 and cpus == sorted(set(cpus))
                and all(type(c) is int and c >= 0 for c in cpus), "Expected 24 bound CPUs")
        require(row.get("local_rank") == 0 and row.get("gpu_uuid"), "Missing physical APU mapping evidence")
        visibility = row["visibility"]
        require(set(visibility) == set(VISIBILITY)
                and all(v is None or isinstance(v, str) for v in visibility.values()),
                "Incomplete GPU visibility evidence")
        mask = visibility["ROCR_VISIBLE_DEVICES"]
        if placement.get("method") == "full-host-device0-numa":
            proof = checked_json(row["proof"])
            require(proof["status"] == "COMPLETE" and proof["world_size"] == NODES
                    and proof["scheduler_job_id"] == job_id and proof["rank"] == row["rank"]
                    and proof["local_rank"] == 0 and proof["allocation"]["hostname"] == host
                    and proof["allocation"]["slots"] == 4
                    and proof["allocation"]["nodefile"] == placement["evidence"]
                    and set(cpus) <= set(proof["allocation"]["inherited_cpus"])
                    and len(proof["allocation"]["inherited_cpus"]) >= 96
                    and proof["binding"]["affinity_cpus"] == cpus
                    and proof["visibility"] == visibility and proof["gpu_mask_modified"] is False
                    and proof["allocated_vnode"] is None and row.get("allocated_vnode") is None,
                    "Incomplete exclusive-host CPU/APU proof")
            observations = proof["observations"]
            require(len(observations) == 2 and all(
                o["hip_device"] == 0 and o["visible_hip_devices"] == 4
                and o["gpu_uuid"] == row["gpu_uuid"]
                and o["pci_bus_id"].lower() == proof["binding"]["pci_bus_id"]
                for o in observations) and observations[1]["hip_openmp_same_device"] is True
                and visibility["OMP_DEFAULT_DEVICE"] == "0", "Unverified HIP/OpenMP physical APU")
        else:
            require(row.get("allocated_vnode") and isinstance(mask, str) and mask.strip()
                    and "," not in mask and visibility["OMP_DEFAULT_DEVICE"] == "0", "Expected one visible APU")
    require(len(set(hosts)) == NODES, "Six distinct physical hosts are required")
    return sorted(records, key=lambda row: row["rank"])


def artifact_inputs(plan):
    config = load_config(checked_file(plan["config"]))
    require(config["run_mode"] == "production" and config.get("vendor") == "amd"
            and config["cpu_threads"] == 24 and config["output_format"] == "jsonl"
            and config["continue_on_failure"] is False, "Expected AMD production configuration")
    manifest, _ = runner.load_manifest(checked_file(plan["manifest"]))
    expected = {(family, variant) for family in AMD_BENCHMARKS for variant in ("cpu", "hip", "openmp")}
    entries = manifest["entries"]
    require(len(entries) == 18 and {(e["library"], e["implementation"]) for e in entries} == expected
            and all(e["executable_role"] == "benchmark" for e in entries),
            "Expected exactly 18 AMD benchmarks")
    metadata = runner.load_build_metadata([checked_file(item) for item in plan["build_metadata"]])
    runner.validate_manifest_artifacts(manifest, metadata, "production", False)
    commits = {e["git_commit"] for e in entries}
    require(len(commits) == 1, "Mixed source commits")
    for field, document in (("config", config), ("manifest", manifest)):
        require(checked_file(plan[field]).read_bytes() == dump_bytes(document),
                "Production inputs must use deterministic JSON: " + field)
    return config, manifest, metadata


def environment_controls(environment):
    controls = {key: environment.get(key) for key in ENVIRONMENT}
    # Only discard repeated identical search entries. Preserve lookup order and
    # symlink spelling; never strip a changed library directory from identity.
    for key in ("PATH", "LD_LIBRARY_PATH"):
        if controls[key] is not None:
            parts = controls[key].split(":")
            require(all(part.startswith("/") for part in parts),
                    key + " contains an empty or relative directory")
            controls[key] = ":".join(dict.fromkeys(parts))
    return controls


def check_runtime_environment(environment, expected, diagnostic_path=None):
    """Keep exact equality and persist differences before MPI can stop peers."""
    observed = environment_controls(environment)
    if observed == expected:
        return
    differences = {
        key: dict(expected=expected.get(key), observed=observed.get(key),
                  expected_present=key in expected, observed_present=key in observed)
        for key in sorted(set(expected) | set(observed))
        if expected.get(key) != observed.get(key) or (key in expected) != (key in observed)
    }
    record = dict(status="MISMATCH", timestamp=runner.utc_timestamp(),
                  hostname=socket.gethostname(), scheduler_job_id=environment.get("PBS_JOBID"),
                  rank=environment.get("OMPI_COMM_WORLD_RANK"),
                  differences=differences)
    if diagnostic_path is not None:
        save(diagnostic_path, record)
    # Only compared runtime controls are included, never the full environment.
    raise ValueError("Runtime environment changed: " + dump_bytes(record).decode("utf-8").strip())


def inputs(plan):
    config, manifest, metadata = artifact_inputs(plan)
    runtime = checked_json(plan["runtime"])
    require(checked_file(plan["runtime"]).read_bytes() == dump_bytes(runtime),
            "Production inputs must use deterministic JSON: runtime")
    require(runtime.get("runtime_environment_schema_version") == 1
            and runtime.get("executables_manifest_sha256") == plan["manifest"]["sha256"]
            and {r["sha256"] for r in runtime.get("build_metadata", [])} == set(metadata),
            "Runtime identity does not match the artifacts")
    check = runtime.get("sirius_execution_checks", {})
    require(set(check.get("environment", {})) == set(ENVIRONMENT)
            and check.get("files") and runtime.get("rocm_version"),
            "Missing normalized runtime controls/files/ROCm version")
    require(not ({"hostname", "job_id", "scheduler_job_id", "wave", "rank", "timestamp",
                  "affinity_cpus", "gpu_uuid"} & set(runtime)),
            "Node/job observations must not be campaign runtime identity")
    require(set(runtime.get("resolved_libraries", {})) == {e["artifact_id"] for e in manifest["entries"]},
            "Missing per-artifact resolved-library identity")
    recorded_files = {f["path"] for f in check["files"]}
    compiler_paths = {m[language]["compiler_path"] for m in metadata.values()
                      for language in ("c", "cxx", "hip")
                      if isinstance(m.get(language), dict) and m[language].get("compiler_path")}
    require(compiler_paths and compiler_paths <= recorded_files
            and {plan["mpirun"]["path"], sys.executable} <= recorded_files,
            "Runtime file checks must include compilers, MPI and Python")
    return config, manifest, metadata, runtime


def collect_runtime(request, output, *, processes=None, environment=None):
    """Allocated-job preparation only: 18 ldd resolutions, no program/GPU run.

    Use the existing pilot link/resolution validator. Actual link evidence comes
    from the new clean build, never from a relabelled historical pilot.
    """
    require(request.get("sirius_runtime_request_schema_version") == 1, "Unknown runtime request")
    _, manifest, metadata = artifact_inputs(request)
    links = checked_json(request["link_evidence"])
    preflight_link_evidence(dict(entries=manifest["entries"], metadata_by_hash=metadata,
                                link_evidence=links), amd_smoke_evidence, 18)
    for field in ("mpirun", "ldd", "rocm_version_file"):
        checked_file(request[field])
    require(all(os.access(request[field]["path"], os.X_OK) for field in ("mpirun", "ldd")),
            "MPI/ldd input is not executable")
    version_bytes = checked_file(request["rocm_version_file"]).read_bytes()
    require(len(version_bytes) <= 256, "Oversized ROCm version evidence")
    version = version_bytes.decode("utf-8", errors="strict").strip()
    require(re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.+_-]+)?", version),
            "Unrecognized ROCm version evidence")
    environment = worker_environment(os.environ if environment is None else environment, output)
    controls = environment_controls(environment)
    paths = {request[field]["path"] for field in ("mpirun", "ldd", "rocm_version_file")}
    paths.add(sys.executable)
    paths.update(m[language]["compiler_path"] for m in metadata.values()
                 for language in ("c", "cxx", "hip")
                 if isinstance(m.get(language), dict) and m[language].get("compiler_path"))
    files = [{"path": path, "sha256": sha256_file(path)} for path in sorted(paths)]
    output.mkdir()
    for directory in ("tmp", "cache"):
        (output / directory).mkdir()
    processes = processes if processes is not None else Processes(output / "commands", 600, 30)
    evidence = dict(status="FAILED", hostname=socket.gethostname(),
                    scheduler_job_id=environment["PBS_JOBID"], request=request,
                    started=runner.utc_timestamp(), resolution={})
    try:
        # The binaries retain their verified clean build provenance. Collection
        # tooling may be repaired independently; never relabel the artifacts.
        evidence["collector_source"] = collector_source(processes)
        for entry in manifest["entries"]:
            result = processes.run([request["ldd"]["path"], entry["executable_path"]],
                                   environment, "ldd-" + entry["target_name"])
            evidence["resolution"][entry["artifact_id"]] = amd_smoke_evidence.validate_artifact_resolution(
                entry, metadata[entry["build_metadata_sha256"]], result.stdout,
                link_dependencies=links[entry["artifact_id"]])
        # Detect changes during collection before freezing the document.
        artifact_inputs(request)
        checked_file(request["link_evidence"])
        for item in files:
            checked_file(item)
        require(collector_source(processes) == evidence["collector_source"],
                "Runtime collection source changed during collection")
        resolution = evidence["resolution"]
        document = dict(runtime_environment_schema_version=1,
            executables_manifest_sha256=request["manifest"]["sha256"],
            build_metadata=sorted(request["build_metadata"], key=lambda item: item["sha256"]),
            rocm_version=version, rocm_version_file=request["rocm_version_file"],
            sirius_execution_checks=dict(environment=controls, files=files),
            resolved_libraries={key: [{"path": path, "sha256": sha} for path, sha in
                                      sorted(value["resolved_objects"].items())]
                                for key, value in resolution.items()},
            library_resolution={key: value["normalized_lines"] for key, value in resolution.items()},
            loaded_libraries_verified=False, observed_cpu_worker_count=None)
        for libraries in document["resolved_libraries"].values():
            for library in libraries:
                checked_file(library)
        save(output / "runtime-environment.json", document)
        evidence.update(status="COMPLETE", runtime_environment_sha256=digest(document))
        return document
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        evidence["error"] = str(error)
        raise
    finally:
        evidence.update(ended=runner.utc_timestamp(), commands=processes.records)
        save(output / "runtime-environment-evidence.json", evidence)


def allocation_hosts(identity):
    lines = checked_file(identity).read_text(encoding="utf-8", errors="strict").splitlines()
    require(lines and all(HOST.fullmatch(line) for line in lines), "Invalid PBS nodefile")
    hosts = {line.split(".")[0].lower() for line in lines}
    require(len(hosts) == NODES, "PBS must allocate six physical hosts; -npernode cannot create hosts")
    return hosts


def validate_plan(plan, job_id):
    require(plan.get("sirius_launch_schema_version") == 1, "Unknown launch plan version")
    runner.validate_execution_context(plan["run_id"], "Sirius", "master", plan["wave"], 0, 0)
    require(plan["wave"] in (0, 1), "Only the two planned waves are supported")
    placement = checked_json(plan["placement"]) if plan.get("placement") else None
    bindings = validate_placement(placement, job_id)
    require(allocation_hosts(plan["allocation_nodefile"]) ==
            {row["hostname"].split(".")[0].lower() for row in bindings},
            "PBS allocation and placement hosts differ")
    checked_file(plan["mpirun"])
    require(os.access(plan["mpirun"]["path"], os.X_OK), "MPI launcher is not executable")
    options = plan["mpi_options"]
    require(isinstance(options, list) and all(isinstance(v, str) and v for v in options),
            "MPI options must be an explicit argument list")
    require(len(options) == len(set(options)) and
            set(options) <= {"--display-map", "--report-bindings"},
            "Only MPI observation options are accepted; rank/host/binding policy is fixed")
    require(plan["controller_timeout_seconds"] == 5100
            and plan["invocation_timeout_seconds"] == 600, "Unexpected timeout budget")
    require(plan["launcher_sha256"] == sha256_file(__file__), "Launcher bytes changed")
    return bindings


def worker_environment(base, node):
    for key in ("LD_PRELOAD", "LD_AUDIT", "LD_DEBUG", "LD_DEBUG_OUTPUT"):
        require(not base.get(key), "Unsupported loader override: " + key)
    return smoke_environment(base, node, 24)


def bind_worker(plan):
    """Apply a confirmed 24-CPU subset, without inventing a GPU mask.

    MPI's --bind-to none avoids its default single-core binding. The explicit
    subset is still constrained by this process's scheduler-provided affinity.
    """
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "-1"))
    require(rank in range(NODES) and os.environ.get("OMPI_COMM_WORLD_SIZE") == str(NODES)
            and os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK") == "0", "Invalid worker rank")
    binding = validate_placement(checked_json(plan["placement"]), os.environ["PBS_JOBID"])[rank]
    require(socket.gethostname() == binding["hostname"], "Wrong host for CPU/APU binding")
    effective = worker_environment(os.environ, Path("/unused-node-path"))
    require({key: effective.get(key) for key in VISIBILITY} == binding["visibility"],
            "Inherited GPU visibility differs; do not replace the scheduler mask")
    cpus = set(binding["affinity_cpus"])
    require(cpus <= set(os.sched_getaffinity(0)), "Requested CPUs exceed scheduler affinity")
    os.sched_setaffinity(0, cpus)
    require(set(os.sched_getaffinity(0)) == cpus, "Could not bind the requested 24 CPUs")


def observe(plan, runtime, environment=None, diagnostic_path=None):
    env = os.environ if environment is None else environment
    rank, size, local = (int(env.get(key, "-1")) for key in
                         ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_RANK"))
    require(size == NODES and rank in range(NODES) and local == 0, "MPI rank/size/local rank mismatch")
    effective = worker_environment(env, Path("/unused-node-path"))
    check_runtime_environment(effective, runtime["sirius_execution_checks"]["environment"],
                              diagnostic_path)
    for item in runtime["sirius_execution_checks"]["files"]:
        checked_file(item)
    unique_libraries = {}
    for libraries in runtime["resolved_libraries"].values():
        require(isinstance(libraries, list) and libraries, "Empty resolved-library inventory")
        for library in libraries:
            require(unique_libraries.get(library["path"], library["sha256"]) == library["sha256"],
                    "Conflicting resolved-library identity")
            unique_libraries[library["path"]] = library["sha256"]
    for path, sha in unique_libraries.items():
        checked_file({"path": path, "sha256": sha})
    record = dict(rank=rank, size=size, local_rank=local, hostname=socket.gethostname(),
                  scheduler_job_id=env["PBS_JOBID"], affinity_cpus=sorted(os.sched_getaffinity(0)),
                  visibility={key: effective.get(key) for key in VISIBILITY},
                  runtime_environment_sha256=plan["runtime"]["sha256"])
    binding = validate_placement(checked_json(plan["placement"]), env["PBS_JOBID"])[rank]
    for key in ("rank", "local_rank", "hostname", "affinity_cpus", "visibility"):
        require(record[key] == binding[key], "Observed placement differs: " + key)
    return record


def context_for(config, plan, binding, runtime):
    rank = binding["rank"]
    return dict(config=config, run_id=plan["run_id"], wave=plan["wave"], node_index=rank,
                hostname=binding["hostname"], system_label="Sirius", scheduler="PBS",
                scheduler_job_id=checked_json(plan["placement"])["scheduler_job_id"],
                cpu_threads=24, device=0, implementation_order=list(implementation_order(rank, plan["wave"], "amd")),
                aocl_sparse_environment={key: runtime["sirius_execution_checks"]["environment"].get(key)
                                        for key in runner.AOCL_SPARSE_ENVIRONMENT_KEYS})


def schedule_for(config, manifest, context):
    schedule = runner.build_schedule(config, manifest, context)
    require(len(schedule) == INVOCATIONS and sum(i["scope_settings"]["trials"] for i in schedule) == ROWS
            and all(i["entry"] and not i["prerequisite_failure"] for i in schedule),
            "Incomplete 108-invocation/540-row node schedule")
    return schedule


def prepare_wave(run_root, plan, bindings, observations):
    """Master only; called after six successful MPI observations."""
    config, manifest, metadata, runtime = inputs(plan)
    require(len(observations) == NODES and {o["rank"] for o in observations} == set(range(NODES)),
            "Incomplete or duplicate preflight ranks")
    for observation in observations:
        expected = bindings[observation["rank"]]
        for key in ("rank", "hostname", "local_rank", "affinity_cpus", "visibility"):
            require(observation[key] == expected[key], "Preflight mapping mismatch: " + key)
        require(observation["runtime_environment_sha256"] == plan["runtime"]["sha256"]
                and observation["size"] == NODES
                and observation["scheduler_job_id"] == checked_json(plan["placement"])["scheduler_job_id"],
                "Preflight runtime/allocation mismatch")
    identity = dict(run_metadata_schema_version=1, run_id=plan["run_id"], run_mode="production",
                    system_label="Sirius", git_metadata_available=True, git_dirty=False,
                    git_commit=manifest["entries"][0]["git_commit"], dirty_source_provenance=None,
                    effective_config_sha256=plan["config"]["sha256"],
                    executables_manifest_sha256=plan["manifest"]["sha256"],
                    runtime_environment_sha256=plan["runtime"]["sha256"],
                    launcher={"name": "amd_production.py", "sha256": plan["launcher_sha256"],
                              "mpirun": plan["mpirun"], "ranks_per_host": 1,
                              "source_revision": plan.get("source_revision")})
    frozen = {"effective-config.json": "config", "executables-manifest.json": "manifest",
              "runtime-environment.json": "runtime"}
    if plan["wave"] == 0:
        for name, field in frozen.items():
            with (run_root / name).open("xb") as stream:
                stream.write(checked_file(plan[field]).read_bytes())
        save(run_root / "run-metadata.json", dict(identity, campaign_creation_timestamp=runner.utc_timestamp(),
                                                  submission_host=socket.gethostname()))
        (run_root / "waves").mkdir()
    else:
        existing = load(run_root / "run-metadata.json")
        require(all(existing.get(k) == v for k, v in identity.items()), "Campaign provenance changed between waves")
        for name, field in frozen.items():
            require(sha256_file(run_root / name) == plan[field]["sha256"], "Frozen campaign input changed")
        require(load(run_root / "waves/0/artifact-manifest.json")["status"] == "COMPLETE",
                "Wave 0 must be reviewed as complete before wave 1")
        require(load(run_root / "launch/0/status.json")["status"] == "COMPLETE", "Wave 0 master did not complete")
    wave_root = run_root / "waves" / str(plan["wave"])
    wave_root.mkdir()
    (wave_root / "nodes").mkdir()
    (wave_root / "job-master").mkdir()
    evidence = dict(runtime_environment_evidence_schema_version=1,
                    runtime_environment_sha256=plan["runtime"]["sha256"],
                    executables_manifest_sha256=plan["manifest"]["sha256"],
                    placement=checked_json(plan["placement"]),
                    preflight_observations=sorted(observations, key=lambda o: o["rank"]),
                    command_logs_relative_to_run_root="launch/{0}/commands".format(plan["wave"]),
                    loaded_libraries_verified=False, observed_cpu_worker_count=None)
    save(wave_root / "job-master/runtime-environment-evidence.json", evidence)
    wave = dict(wave_metadata_schema_version=1, run_id=plan["run_id"], wave=plan["wave"],
                scheduler="PBS", scheduler_job_id=observations[0]["scheduler_job_id"],
                expected_node_count=NODES, observed_node_count=NODES, expected_rows=NODES * ROWS,
                effective_config_sha256=plan["config"]["sha256"],
                executables_manifest_sha256=plan["manifest"]["sha256"],
                runtime_environment_sha256=plan["runtime"]["sha256"],
                runtime_environment_evidence_sha256=digest(evidence),
                launch_plan_sha256=digest(plan), rank_host_mapping=bindings,
                preflight_observations=sorted(observations, key=lambda o: o["rank"]),
                timestamp=runner.utc_timestamp(), submission_host=socket.gethostname(), assignments=[])
    for binding in bindings:
        context = context_for(config, plan, binding, runtime)
        schedule_for(config, manifest, context)
        wave["assignments"].append(dict(rank=binding["rank"], hostname=binding["hostname"],
            implementation_order=list(context["implementation_order"]),
            permutation_index=permutation_index(binding["rank"], plan["wave"]),
            size_order_index=size_order_index(binding["rank"])))
    save(wave_root / "wave-metadata.json", wave)
    return wave


class LoggedRunner:
    """Adapt the existing runner without tracing, profiling or retrying."""

    def __init__(self, processes, schedule):
        self.processes, self.schedule, self.calls = processes, schedule, 0

    def run(self, argv, **kwargs):
        require(self.calls < len(self.schedule) and argv == self.schedule[self.calls]["argv"],
                "Unexpected benchmark invocation")
        self.calls += 1
        return self.processes.run(argv, kwargs["env"], "benchmark")


def validate_rows(rows, plan, binding, config, manifest, runtime):
    context = context_for(config, plan, binding, runtime)
    schedule = schedule_for(config, manifest, context)
    require(len(rows) == ROWS, "Missing or extra node rows")
    report = validate_campaign(rows, config, plan["config"]["sha256"], manifest)
    require(report["validation_status"] == "pass" and report["block_count"] == 1,
            "Node campaign did not pass")
    offset = 0
    for item in schedule:
        if runner.aocl_sparse_environment_overrides(item, context):
            item = dict(item, aocl_sparse_environment=context["aocl_sparse_environment"])
        for trial in range(item["scope_settings"]["trials"]):
            row = rows[offset]
            offset += 1
            require(row["trial"] == trial and row["attempted"] and row["status"] == "success"
                    and row["verification_status"] == "pass", "Non-success or out-of-order trial")
            runner.validate_subprocess_record(row, item, context, plan["config"]["sha256"],
                                               plan["runtime"]["sha256"], None, None)
            _numerical_pass(row, item, context)
            require(all(isinstance(row[k], (int, float)) and math.isfinite(row[k]) and row[k] > 0
                        for k in ("elapsed_sec", "elapsed_total_sec")), "Invalid timing")
            if row["implementation"] != "cpu" and row["gpu_uuid"] is not None:
                require(row["gpu_uuid"] == binding["gpu_uuid"], "Measured physical GPU differs from placement")
    report["gpu_uuid_observed_in_all_gpu_rows"] = all(
        r["gpu_uuid"] is not None for r in rows if r["implementation"] != "cpu")
    return report


def run_worker(run_root, plan, wave_hash):
    wave_root = run_root / "waves" / str(plan["wave"])
    require(sha256_file(wave_root / "wave-metadata.json") == wave_hash, "Wave metadata changed")
    wave = load(wave_root / "wave-metadata.json")
    require(wave["launch_plan_sha256"] == digest(plan), "Wave/launch plan mismatch")
    config, manifest, metadata, runtime = inputs(plan)
    diagnostic = (run_root / "launch" / str(plan["wave"]) / "observations" /
                  (os.environ.get("OMPI_COMM_WORLD_RANK", "-1") + "-worker-runtime-mismatch.json"))
    observation = observe(plan, runtime, diagnostic_path=diagnostic)
    rank = observation["rank"]
    require(observation == wave["preflight_observations"][rank], "Worker changed since preflight")
    binding = wave["rank_host_mapping"][rank]
    node = wave_root / "nodes" / binding["hostname"]
    node.mkdir()
    for name in ("tmp", "cache"):
        (node / name).mkdir()
    environment = worker_environment(os.environ, node)
    context = context_for(config, plan, binding, runtime)
    schedule = schedule_for(config, manifest, context)
    save(node / "node-metadata.json", dict(node_metadata_schema_version=1,
        **{k: v for k, v in context.items() if k != "config"},
        block_id="{0}|{1}|{2}".format(plan["run_id"], plan["wave"], binding["hostname"]),
        runtime_environment_sha256=plan["runtime"]["sha256"], observation=observation,
        permutation_index=permutation_index(rank, plan["wave"]), size_order_index=size_order_index(rank),
        series={family: definition["series"] for family, definition in config["benchmarks"].items()},
        size_order={family: list(reversed(definition["cases"])) if size_order_index(rank) else definition["cases"]
                    for family, definition in config["benchmarks"].items()},
        placement=binding, loaded_libraries_verified=False, observed_cpu_worker_count=None))
    processes = Processes(node / "commands", plan["controller_timeout_seconds"], plan["invocation_timeout_seconds"])
    adapter = LoggedRunner(processes, schedule)
    previous_env, previous_subprocess = dict(os.environ), runner.subprocess
    result = dict(status="FAILED", started=runner.utc_timestamp(), wave_metadata_sha256=wave_hash)
    code = 1
    try:
        os.environ.clear()
        os.environ.update(environment)
        runner.subprocess = adapter
        code = runner.execute_schedule(schedule, context, node / "raw.jsonl", plan["config"]["sha256"],
                                       plan["runtime"]["sha256"], metadata, None, None)
        require(code == 0 and adapter.calls == INVOCATIONS, "Incomplete benchmark invocation count")
        rows = load_raw_results(node / "raw.jsonl")
        result["validation"] = validate_rows(rows, plan, binding, config, manifest, runtime)
        result["status"] = "COMPLETE"
    except (OSError, ValueError, RuntimeError) as error:
        result["error"] = str(error)
        code = 1
        if isinstance(error, StopRequested) or any(r["timed_out"] for r in processes.records):
            result["status"] = "UNKNOWN"
    finally:
        runner.subprocess = previous_subprocess
        os.environ.clear()
        os.environ.update(previous_env)
        result.update(ended=runner.utc_timestamp(), benchmark_calls=adapter.calls, exit_code=code,
                      commands=processes.records)
        save(node / "node-status.json", result)
    return code


def collect_wave(run_root, plan, mpi_exit):
    """Retain incomplete nodes and inventory their evidence; never fill missing rows."""
    wave_root = run_root / "waves" / str(plan["wave"])
    wave = load(wave_root / "wave-metadata.json")
    report = dict(status="FAILED", mpi_exit=mpi_exit, expected_rows=NODES * ROWS,
                  observed_rows=0, errors=[], nodes=[], files=[], loaded_libraries_verified=False,
                  observed_cpu_worker_count=None, wave_metadata_sha256=sha256_file(wave_root / "wave-metadata.json"))
    all_rows = []
    try:
        config, manifest, _, runtime = inputs(plan)
        require(wave["launch_plan_sha256"] == digest(plan), "Collection plan changed")
        require(sha256_file(wave_root / "job-master/runtime-environment-evidence.json")
                == wave["runtime_environment_evidence_sha256"], "Missing or changed preflight evidence")
        expected_hosts = {b["hostname"] for b in wave["rank_host_mapping"]}
        require({p.name for p in (wave_root / "nodes").iterdir()} <= expected_hosts, "Unexpected node output")
        for binding in wave["rank_host_mapping"]:
            node = wave_root / "nodes" / binding["hostname"]
            try:
                status = load(node / "node-status.json")
                require(status["status"] == "COMPLETE" and status["exit_code"] == 0
                        and status["benchmark_calls"] == INVOCATIONS
                        and status["wave_metadata_sha256"] == report["wave_metadata_sha256"],
                        "Missing successful node completion")
                node_metadata = load(node / "node-metadata.json")
                expected_context = context_for(config, plan, binding, runtime)
                require(all(node_metadata.get(k) == v for k, v in expected_context.items() if k != "config")
                        and node_metadata.get("observation") == wave["preflight_observations"][binding["rank"]]
                        and node_metadata.get("runtime_environment_sha256") == plan["runtime"]["sha256"],
                        "Node metadata/observation mismatch")
                commands = status["commands"]
                schedule = schedule_for(config, manifest, expected_context)
                require(len(commands) == INVOCATIONS, "Incomplete command evidence")
                for command, item in zip(commands, schedule):
                    require(command["argv"] == item["argv"] and command["exit"] == 0
                            and command["timed_out"] is False, "Failed or unexpected command evidence")
                    for channel in ("stdout", "stderr"):
                        path = checked_file({"path": command[channel], "sha256": command[channel + "_sha256"]})
                        require(path.parent == node / "commands", "Command log is outside its node directory")
                rows = load_raw_results(node / "raw.jsonl")
                validate_rows(rows, plan, binding, config, manifest, runtime)
                all_rows.extend(rows)
                report["nodes"].append({"hostname": binding["hostname"], "rows": len(rows), "status": "PASS"})
            except (OSError, ValueError, KeyError, TypeError) as error:
                report["errors"].append(binding["hostname"] + ": " + str(error))
        report["observed_rows"] = len(all_rows)
        require(mpi_exit == 0 and not report["errors"] and len(all_rows) == NODES * ROWS,
                "MPI failure or incomplete six-host wave")
        report["validation"] = validate_campaign(all_rows, config, plan["config"]["sha256"], manifest)
        require(report["validation"]["block_count"] == NODES
                and report["validation"]["wave_count"] == 1
                and report["validation"]["validation_status"] == "pass", "Invalid wave inventory")
        report["status"] = "COMPLETE"
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["errors"].append(str(error))
    for path in sorted((wave_root / "nodes").rglob("*")):
        if path.is_file() and not any(part in {"cache", "tmp"} for part in path.relative_to(wave_root).parts):
            report["files"].append({"path": str(path.relative_to(wave_root)), "sha256": sha256_file(path)})
    save(wave_root / "artifact-manifest.json", report)
    return report


def mpi_command(plan, frozen_plan, action, wave_hash=None):
    argv = [plan["mpirun"]["path"], *plan["mpi_options"],
            "--hostfile", plan["allocation_nodefile"]["path"],
            "-np", str(NODES), "-npernode", "1", "--bind-to", "none",
            sys.executable, "-B", str(Path(__file__).resolve()),
            action, "--plan", str(frozen_plan), "--plan-sha256", digest(plan)]
    if wave_hash is not None:
        argv += ["--wave-sha256", wave_hash]
    return argv


def collector_source(processes):
    """Record collection-tool provenance separately from the clean binaries."""
    git = shutil.which("git")
    require(git is not None, "Git unavailable")
    prefix = [git, "--no-optional-locks", "-C", str(ROOT), "-c", "core.fsmonitor=false"]
    commands = {"head": ["rev-parse", "HEAD"],
                "status": ["status", "--porcelain", "--untracked-files=all"],
                "diff": ["diff", "--no-ext-diff", "--binary", "HEAD"]}
    state = {key: processes.run(prefix + argv, os.environ, "collector-" + key).stdout
             for key, argv in commands.items()}
    paths = {Path(module.__file__).resolve() for module in list(sys.modules.values())
             if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")}
    return dict(git_commit=state["head"].strip(), git_status=state["status"],
                git_diff_sha256=hashlib.sha256(state["diff"].encode("utf-8")).hexdigest(),
                files=[dict(path=str(path), sha256=sha256_file(path))
                       for path in sorted(paths) if ROOT in path.parents])


def clean_source(processes, manifest, revision=None):
    git = shutil.which("git")
    require(git is not None, "Git unavailable")
    head = processes.run([git, "--no-optional-locks", "-C", str(ROOT), "rev-parse", "HEAD"], os.environ, "git-head")
    status = processes.run([git, "--no-optional-locks", "-C", str(ROOT), "-c", "core.fsmonitor=false",
                            "status", "--porcelain", "--untracked-files=all"], os.environ, "git-status")
    require(not status.stdout.strip(), "Production source must be clean")
    commits = {e["git_commit"] for e in manifest["entries"]}
    if revision is None:
        require({head.stdout.strip()} == commits, "Production source must match the binaries")
        return
    require(set(revision) == {"artifact_commit", "launcher_commit", "changed_paths"}
            and all(isinstance(revision[k], str) and re.fullmatch(r"[0-9a-f]{40}", revision[k])
                    for k in ("artifact_commit", "launcher_commit"))
            and commits == {revision["artifact_commit"]}
            and head.stdout.strip() == revision["launcher_commit"], "Unsealed or mismatched source revision")
    paths = revision["changed_paths"]
    require(isinstance(paths, list) and all(isinstance(p, str) for p in paths)
            and paths == sorted(set(paths)) and set(paths) <= TOOL_REVISION_PATHS,
            "Source revision contains a benchmark/build input")
    changed = processes.run([git, "--no-optional-locks", "-C", str(ROOT), "diff", "--no-ext-diff",
                             "--no-renames", "--name-only", revision["artifact_commit"],
                             revision["launcher_commit"], "--"], os.environ, "git-tool-revision")
    require(sorted(changed.stdout.splitlines()) == paths, "Unapproved source revision paths")


def run_master(run_root, plan):
    bindings = validate_plan(plan, os.environ["PBS_JOBID"])
    _, manifest, _, _ = inputs(plan)
    if plan["wave"] == 0:
        run_root.mkdir()
        (run_root / "launch").mkdir()
    else:
        require(load(run_root / "waves/0/artifact-manifest.json")["status"] == "COMPLETE", "Wave 0 incomplete")
        require(load(run_root / "launch/0/status.json")["status"] == "COMPLETE", "Wave 0 master incomplete")
    launch = run_root / "launch" / str(plan["wave"])
    launch.mkdir()
    (launch / "observations").mkdir()
    frozen_plan = launch / "plan.json"
    save(frozen_plan, plan)
    processes = Processes(launch / "commands", 5100, 5100)
    result = dict(status="FAILED", started=runner.utc_timestamp(), wave=plan["wave"])
    code = 1
    try:
        clean_source(processes, manifest, plan.get("source_revision"))
        processes.run(mpi_command(plan, frozen_plan, "observe"), os.environ, "mpi-preflight")
        observations = [load(launch / "observations" / (str(rank) + ".json")) for rank in range(NODES)]
        wave = prepare_wave(run_root, plan, bindings, observations)
        mpi_exit = None
        try:
            measured = processes.run(mpi_command(plan, frozen_plan, "worker", digest(wave)), os.environ, "mpi-measurement")
            clean_source(processes, manifest, plan.get("source_revision"))
            mpi_exit = measured.returncode
        finally:
            report = collect_wave(run_root, plan, mpi_exit)
        require(report["status"] == "COMPLETE", "Collection failed")
        result["status"], code = "COMPLETE", 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        result["error"] = str(error)
        if isinstance(error, StopRequested) or any(r["timed_out"] for r in processes.records):
            result["status"] = "UNKNOWN"
    finally:
        result.update(ended=runner.utc_timestamp(), exit_code=code, commands=processes.records)
        save(launch / "status.json", result)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect-runtime", "master", "observe", "worker"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--wave-sha256")
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args(argv)
    try:
        require(sys.platform.startswith("linux") and os.environ.get("PBS_ENVIRONMENT") == "PBS_BATCH"
                and re.fullmatch(r"[0-9]+\.[A-Za-z0-9_.-]+", os.environ.get("PBS_JOBID", "")),
                "An approved Sirius PBS batch allocation is required")
        plan = checked_json({"path": str(args.plan), "sha256": args.plan_sha256})
        if args.action == "collect-runtime":
            require(args.output_directory is not None and args.output_directory.is_absolute()
                    and (ROOT / "runs").resolve() in args.output_directory.resolve().parents,
                    "Runtime collection needs a fresh output directory under runs/")
            collect_runtime(plan, args.output_directory)
            return 0
        validate_plan(plan, os.environ["PBS_JOBID"])
        run_root = ROOT / "runs" / plan["run_id"]
        if args.action == "master":
            return run_master(run_root, plan)
        bind_worker(plan)
        if args.action == "observe":
            _, _, _, runtime = inputs(plan)
            diagnostic = (run_root / "launch" / str(plan["wave"]) / "observations" /
                          (os.environ["OMPI_COMM_WORLD_RANK"] + "-preflight-runtime-mismatch.json"))
            record = observe(plan, runtime, diagnostic_path=diagnostic)
            save(run_root / "launch" / str(plan["wave"]) / "observations" / (str(record["rank"]) + ".json"), record)
            return 0
        require(args.wave_sha256 is not None, "Worker requires the master wave hash")
        return run_worker(run_root, plan, args.wave_sha256)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print("Sirius production stopped: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    sys.exit(main())

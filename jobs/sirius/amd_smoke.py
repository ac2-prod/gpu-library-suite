#!/usr/bin/env python3
"""Allocated AMD smoke of immutable prebuilt artifacts; no submission.

The source which produced the binaries is verified before importing its runner.
The focused paths run numerical checks without loader tracing. The legacy
six-family diagnostic path retains its explicit LD_DEBUG checks. Neither path
provides performance evidence.
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
HELPERS = ("amd_smoke.py", "amd_smoke_source.py", "amd_smoke_evidence.py",
           "amd_smoke_campaign.py")
ENVIRONMENT_KEYS = (
    "PATH", "LD_LIBRARY_PATH", "ROCM_PATH", "HIP_PATH", "AOCL_ROOT", "LOADEDMODULES",
    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL", "OMP_DEFAULT_DEVICE", "OMP_TARGET_OFFLOAD",
    "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_THREAD_LIMIT", "OMP_PROC_BIND",
    "OMP_PLACES", "AOCLSPARSE_NUM_THREADS", "AOCL_ENABLE_INSTRUCTIONS",
    "BLIS_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "ROCBLAS_LAYER", "ROCBLAS_TENSILE_LIBPATH", "ROCBLAS_USE_HIPBLASLT",
    "HIPBLASLT_TENSILE_LIBPATH", "ROCFFT_RTC_CACHE_PATH", "XDG_CACHE_HOME", "TMPDIR",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class StopRequested(RuntimeError):
    """An external deadline/cancellation is not evidence of child completion."""


def request_stop(signum, frame):
    raise StopRequested("Received signal " + str(signum))


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    # Same canonical UTF-8 representation as the pinned strict_json.dump_bytes;
    # available before importing any unverified source.
    def normalize(item):
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            require(math.isfinite(item), "Nonfinite JSON value")
            return 0.0 if item == 0.0 else item
        if isinstance(item, (list, tuple)):
            return [normalize(part) for part in item]
        if isinstance(item, dict):
            require(all(isinstance(key, str) for key in item), "JSON object keys must be strings")
            return {key: normalize(part) for key, part in item.items()}
        raise ValueError("Unsupported JSON value")
    content = (json.dumps(normalize(value), sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    with Path(path).open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(path)


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Invalid JSON constant: " + value)
    def finite(value):
        number = float(value)
        require(math.isfinite(number), "Nonfinite JSON number")
        return number
    return json.loads(Path(path).read_text(encoding="utf-8", errors="strict"),
                      object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite)


def real_path(value, directory=True):
    path = Path(value)
    require(path.is_absolute() and path.resolve(strict=True) == path,
            "Canonical existing absolute path required")
    require(not path.is_symlink() and (path.is_dir() if directory else path.is_file()),
            "Expected a real directory/regular file")
    return path


def load_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def focused_selection(packet):
    require(not ("library" in packet and "libraries" in packet), "Ambiguous library selection")
    selection = packet.get("libraries", packet.get("library"))
    if selection is None:
        return None
    libraries = [selection] if isinstance(selection, str) else selection
    require(isinstance(libraries, list) and all(isinstance(name, str) for name in libraries)
            and libraries and len(libraries) == len(set(libraries))
            and set(libraries).issubset({"rocblas", "rocsparse", "rocsolver", "rocrand", "rocthrust"}),
            "Unsupported focused smoke libraries")
    return libraries


def validate_packet(path, expected_hash, environment):
    """No process or output creation occurs before all local approval gates."""
    require(re.fullmatch(r"[0-9a-f]{64}", expected_hash or ""), "Invalid packet hash")
    path = real_path(path, directory=False)
    require(sha(path) == expected_hash, "Packet hash differs from approval")
    packet = read_json(path)
    library = focused_selection(packet)
    require(type(packet.get("schema_version")) is int and packet["schema_version"] == 1
            and packet.get("purpose") == ("numerical-smoke" if library else "diagnostic-smoke"),
            "Smoke purpose differs from selected scope")
    if library:
        require(packet.get("example_count") == 3 * len(library) and packet.get("diagnostic_rows") == 0
                and packet.get("final_rows") == 96 * len(library), "Focused smoke coverage differs")
    require(environment.get("GPU_SUITE_SIRIUS_AMD_SMOKE_APPROVED") == "yes",
            "Explicit AMD smoke approval is required")
    require(re.fullmatch(r"[0-9]+\.[A-Za-z0-9_.-]+", environment.get("PBS_JOBID", ""))
            and environment.get("PBS_ENVIRONMENT") == "PBS_BATCH", "PBS batch allocation required")
    require(sys.platform.startswith("linux"), "Production smoke requires the allocated Linux host")
    source = real_path(packet["source_root"])
    root = real_path(packet["run_root"])
    require(root == source / "runs" / packet["run_id"] and path.parent == root,
            "Packet must be in its unique SOURCE_ROOT/runs/RUN_ID")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}", packet["run_id"])
            and ".." not in packet["run_id"], "Invalid smoke run ID")
    frozen = real_path(packet["snapshot_root"])
    require(frozen == root / "artifact-source", "Complete frozen artifact source must be in this run")
    build = real_path(packet["build_run_root"])
    require(build.parent == source / "runs" and build != root, "Invalid retained build run")
    require(type(packet["allocated_cpus"]) is int and packet["allocated_cpus"] >= 2
            and type(packet["allocated_gpus"]) is int and packet["allocated_gpus"] == 1,
            "Smoke needs two CPUs and one allocated APU")
    require(type(packet["timeout_seconds"]) is int and 30 <= packet["timeout_seconds"] <= 1800
            and type(packet["process_timeout_seconds"]) is int
            and 1 <= packet["process_timeout_seconds"] <= packet["timeout_seconds"], "Invalid timeout")
    require(hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) >= 2,
            "Cannot verify at least two allocated affinity CPUs")
    if environment.get("OMP_THREAD_LIMIT"):
        require(environment["OMP_THREAD_LIMIT"].isdigit()
                and int(environment["OMP_THREAD_LIMIT"]) >= 2, "OMP_THREAD_LIMIT prevents requested threads")
    for name in ("LD_PRELOAD", "LD_AUDIT", "LD_DEBUG", "LD_DEBUG_OUTPUT"):
        require(not environment.get(name), "Unsupported inherited loader override: " + name)
    require(set(packet["helper_sha256"]) == set(HELPERS), "Incomplete harness hash inventory")
    for name, digest in packet["helper_sha256"].items():
        require(re.fullmatch(r"[0-9a-f]{64}", digest or "") and sha(real_path(root / name, False)) == digest,
                "Harness hash mismatch: " + name)
    require(Path(__file__).resolve() == root / "amd_smoke.py", "Run the hash-pinned packet controller")
    require(not (root / "smoke").exists(), "An existing attempt cannot be retried or overwritten")
    return packet


def load_pinned_runner(snapshot):
    """Never mix currently edited Python with the binary's verified snapshot."""
    tools = Path(snapshot) / "tools"
    for name in sys.modules:
        require(not (name == "gpu_suite" or name.startswith("gpu_suite.")
                     or name == "rocfft_smoke_evidence"),
                "Suite helper was imported before frozen-source verification")
    sys.path.insert(0, str(tools))
    sys.path.insert(1, str(Path(snapshot) / "jobs/sirius"))
    from gpu_suite import runner
    check_module_origins(snapshot)
    return runner


def check_module_origins(snapshot):
    root = (Path(snapshot) / "tools/gpu_suite").resolve()
    for name, module in list(sys.modules.items()):
        if name == "gpu_suite" or name.startswith("gpu_suite."):
            location = Path(module.__file__).resolve()
            require(root in location.parents, "Mixed runner source: " + name)
        elif name == "rocfft_smoke_evidence":
            require(Path(module.__file__).resolve() == Path(snapshot) / "jobs/sirius/rocfft_smoke_evidence.py",
                    "Mixed frozen evidence helper")


class Processes:
    """Single-process-at-a-time, own-child timeout, exclusive immutable logs."""

    def __init__(self, root, timeout, process_timeout, popen=subprocess.Popen):
        self.root = Path(root)
        self.root.mkdir()
        self.deadline = time.monotonic() + timeout
        self.process_timeout = process_timeout
        self.popen = popen
        self.records = []
        self.stopped = False

    @staticmethod
    def stop_child(child):
        if type(getattr(child, "returncode", None)) is int:
            return child.returncode
        if getattr(child, "_smoke_reaped", False) is True:
            return child._smoke_exit
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
        try:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                code = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                code = child.wait()
            child._smoke_exit, child._smoke_reaped = code, True
            return code
        finally:
            # Pending signals may be delivered here, after the child is reaped.
            # A repeated cleanup never signals a recycled PID.
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    def run(self, argv, environment, label):
        require(not self.stopped, "No process after failure/timeout")
        remaining = self.deadline - time.monotonic()
        require(remaining > 0, "Overall smoke deadline exhausted")
        argv = [str(part) for part in argv]
        index = len(self.records) + 1
        prefix = self.root / ("%03d-" % index + label)
        stdout, stderr = Path(str(prefix) + ".stdout"), Path(str(prefix) + ".stderr")
        record = {"argv": argv, "started": utc(), "pid": None, "exit": None,
                  "timed_out": False, "stdout": str(stdout), "stderr": str(stderr)}
        write_json(str(prefix) + "-attempt.json", record)
        child = None
        try:
            with stdout.open("xb") as out, stderr.open("xb") as err:
                # Record ownership before an outer deadline can interrupt spawn.
                previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
                try:
                    child = self.popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                       env=environment, start_new_session=True,
                                       preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask))
                    record["pid"] = child.pid
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                try:
                    record["exit"] = child.wait(timeout=min(self.process_timeout, remaining))
                except (subprocess.TimeoutExpired, StopRequested) as error:
                    record["timed_out"] = True
                    record["stop_reason"] = str(error)
                    # Only our newly created child process group is signalled.
                    record["exit"] = self.stop_child(child)
        except StopRequested as error:
            record.update(timed_out=True, stop_reason=str(error))
            if child is not None:
                record["exit"] = self.stop_child(child)
        except (OSError, subprocess.SubprocessError) as error:
            record.update(error=str(error), launch_failed=child is None)
        finally:
            record["ended"] = utc()
            for key, file in (("stdout", stdout), ("stderr", stderr)):
                if file.exists():
                    record[key + "_sha256"] = sha(file)
                    record[key + "_bytes"] = file.stat().st_size
            self.records.append(record)
            write_json(str(prefix) + "-result.json", record)
        self.stopped = record["timed_out"] or record["exit"] != 0
        require(not self.stopped, "Child failed/unknown; no automatic retry: " + label)
        require(stdout.stat().st_size <= 64 * 1024 * 1024 and stderr.stat().st_size <= 64 * 1024 * 1024,
                "Diagnostic log exceeds read limit")
        return SimpleNamespace(returncode=record["exit"], pid=record["pid"],
                               stdout=stdout.read_text(encoding="utf-8", errors="strict"),
                               stderr=stderr.read_text(encoding="utf-8", errors="strict"), record=record)


def link_dependencies(entry, inventory):
    require(entry["artifact_id"] in inventory, "Missing verified artifact link evidence")
    return inventory[entry["artifact_id"]]


def preflight_link_evidence(proof, evidence, artifact_count=36):
    """Check every artifact's evidence before even the first ldd child."""
    entries = proof["entries"]
    inventory = proof["link_evidence"]
    ids = {entry["artifact_id"] for entry in entries}
    require(len(entries) == len(ids) == artifact_count and set(inventory) == ids,
            "Incomplete or duplicate artifact link evidence inventory")
    return {entry["artifact_id"]: evidence.validate_link_evidence(
        entry, proof["metadata_by_hash"][entry["build_metadata_sha256"]],
        link_dependencies(entry, inventory)) for entry in entries}


class TracedRunner:
    """Narrow subprocess.run adapter only for the approved runner schedule."""

    def __init__(self, processes, evidence, metadata, schedule, threads, link_evidence, expected=None, *, trace=True):
        self.processes, self.evidence, self.metadata = processes, evidence, metadata
        self.schedule = list(schedule)
        self.link_evidence = link_evidence
        self.threads, self.expected = threads, expected
        self.trace = trace
        self.observed = {}
        self.calls = 0

    def run(self, argv, *, text, capture_output, check, env):
        require(text is True and capture_output is True and check is False,
                "Unexpected runner subprocess contract")
        require(self.calls < len(self.schedule), "Unexpected extra artifact invocation")
        item = self.schedule[self.calls]
        require(list(argv) == item["argv"], "Command differs from approved schedule")
        self.calls += 1
        entry = item["entry"]
        require(sha(entry["executable_path"]) == entry["binary_sha256"], "Binary changed before launch")
        if not self.trace:
            # The normal runner validates exit status and numerical records.
            # Preserve stderr verbatim; no loader-hook interpretation is needed.
            return self.processes.run(argv, env, "benchmark-" + entry["target_name"])
        links = link_dependencies(entry, self.link_evidence)
        environment = dict(env, LD_DEBUG="libs")
        key = entry["artifact_id"] + ":" + str(self.threads) + ":" + item["scope"]
        result = self.processes.run(argv, environment, "benchmark-" + entry["target_name"])
        trace = self.evidence.validate_loaded_trace(
            entry, self.metadata[entry["build_metadata_sha256"]], result.stderr,
            expected_program=entry["executable_path"], expected_pid=result.pid,
            link_dependencies=links)
        objects = trace["initialized_objects"]
        if self.expected is not None:
            require(key in self.expected and self.expected[key] == objects,
                    "Runtime library initialization identity differs from preflight")
        require(key not in self.observed, "Duplicate diagnostic invocation")
        self.observed[key] = objects
        write_json(Path(result.record["stderr"] + ".evidence.json"), trace)
        result.stderr = trace["process"]["application_stderr"]
        return result


def smoke_environment(base, root, threads):
    env = dict(base)
    env.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE", OMP_TARGET_OFFLOAD="MANDATORY",
               OMP_DEFAULT_DEVICE="0", AOCLSPARSE_NUM_THREADS=str(threads),
               BLIS_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(root / "tmp"),
               XDG_CACHE_HOME=str(root / "cache"),
               ROCFFT_RTC_CACHE_PATH=str(root / "cache/rocfft.db"))
    # A stale parent campaign must not determine child provenance.
    for key in list(env):
        if key.startswith("GPU_SUITE_"):
            env.pop(key)
    return env


def campaign_context(config, run_id, job_id, environment):
    from gpu_suite.ordering import implementation_order
    return dict(config=config, cpu_threads=config["cpu_threads"], device=0,
                hostname=socket.gethostname(), node_index=0, run_id=run_id, wave=0,
                scheduler="PBS", scheduler_job_id=job_id, system_label="Sirius",
                implementation_order=implementation_order(0, 0, "amd"),
                aocl_sparse_environment={key: environment.get(key) for key in
                    ("AOCLSPARSE_NUM_THREADS", "OMP_NUM_THREADS", "OMP_DYNAMIC", "AOCL_ENABLE_INSTRUCTIONS")})


def execute_campaign(runner, campaigns, evidence, processes, proof, manifest, config, root,
                     run_id, job_id, environment, runtime, expected, *, library=None):
    from gpu_suite.results_io import load_raw_results
    root.mkdir()
    config_hash = write_json(root / "config.json", config)
    runtime = dict(runtime, run_id=run_id, config_sha256=config_hash,
                   environment={key: environment.get(key) for key in ENVIRONMENT_KEYS},
                   requested_cpu_threads=config["cpu_threads"], observed_cpu_worker_count=None)
    runtime_hash = write_json(root / "runtime-environment.json", runtime)
    context = campaign_context(config, run_id, job_id, environment)
    schedule = runner.build_schedule(config, manifest, context)
    invocations = 12 * len(campaigns.selected_libraries(library)) if library else 36
    require(len(schedule) == invocations and all(item["prerequisite_failure"] is None for item in schedule),
            "Incomplete benchmark schedule; no fallback")
    write_json(root / "schedule.json", schedule)
    adapter = TracedRunner(processes, evidence, proof["metadata_by_hash"], schedule,
                           config["cpu_threads"], proof["link_evidence"], expected, trace=library is None)
    raw = root / ("raw." + config["output_format"])
    previous_env, previous_subprocess = dict(os.environ), runner.subprocess
    try:
        os.environ.clear()
        os.environ.update(environment)
        runner.subprocess = adapter
        code = runner.execute_schedule(schedule, context, raw, config_hash, runtime_hash,
                                       proof["metadata_by_hash"], None,
                                       proof["artifact_source"]["source_snapshot_sha256"])
    finally:
        runner.subprocess = previous_subprocess
        os.environ.clear()
        os.environ.update(previous_env)
    require(code == 0 and adapter.calls == invocations, "Campaign did not complete all successful invocations")
    report = campaigns.validate_smoke_rows(load_raw_results(raw), config, manifest,
        context=context, config_hash=config_hash, runtime_hash=runtime_hash,
        source_hash=proof["artifact_source"]["source_snapshot_sha256"], git_diff_hash=None, library=library)
    write_json(root / "validation.json", report)
    if library is None:
        write_json(root / "initialized-objects.json", adapter.observed)
    return report, adapter.observed


def execute(packet, root, processes):
    packet_root = Path(packet["run_root"])
    source = load_helper("smoke_source", packet_root / "amd_smoke_source.py")
    proof_args = dict(expected_source_sha256=packet["artifact_source_sha256"],
        expected_summary_sha256=packet["build_summary_sha256"],
        expected_evidence_sha256=packet["build_evidence_sha256"],
        expected_build_run_id=Path(packet["build_run_root"]).name,
        expected_recorded_source_root=packet["source_root"],
        harness_path=packet_root / "amd_smoke.py", expected_harness_sha256=packet["helper_sha256"]["amd_smoke.py"])
    full_proof = source.verify_prebuilt_source(packet["snapshot_root"], packet["build_run_root"], **proof_args)
    write_json(root / "prebuilt-proof.json", full_proof)
    library = focused_selection(packet)
    proof = dict(full_proof)
    if library:
        proof["entries"] = [entry for entry in proof["entries"] if entry["library"] in library]
        proof["link_evidence"] = {entry["artifact_id"]: full_proof["link_evidence"][entry["artifact_id"]]
                                  for entry in proof["entries"]}
    runner = load_pinned_runner(packet["snapshot_root"])
    evidence = load_helper("smoke_evidence", packet_root / "amd_smoke_evidence.py")
    campaigns = load_helper("smoke_campaign", packet_root / "amd_smoke_campaign.py")
    from gpu_suite.manifest import merge_entries
    manifest = {"manifest_schema_version": 1, "build_metadata_sha256s": sorted(proof["metadata_by_hash"]),
                "entries": merge_entries([proof["entries"]])}
    write_json(root / "executables.json", manifest)
    runner.validate_manifest_artifacts(manifest, proof["metadata_by_hash"], "smoke", True)
    link_proof = preflight_link_evidence(proof, evidence, 6 * len(library) if library else 36)
    write_json(root / "link-evidence.json", link_proof)
    for name in ("tmp", "cache"):
        (root / name).mkdir()
    environment = smoke_environment(os.environ, root, 1)
    definitions = campaigns.campaign_configs(
        Path(packet["snapshot_root"]) / "jobs/sirius/amd-smoke.json",
        Path(packet["snapshot_root"]) / "jobs/sirius/amd-threaded-smoke.json", library=library)
    for name, config in definitions.items():
        context = campaign_context(config, packet["run_id"] + "-" + name, os.environ["PBS_JOBID"],
                                   smoke_environment(os.environ, root, config["cpu_threads"]))
        schedule = runner.build_schedule(config, manifest, context)
        require(len(schedule) == (12 * len(library) if library else 36) and all(item["prerequisite_failure"] is None for item in schedule),
                "Incomplete preflight schedule; no artifact will run")
    check_module_origins(packet["snapshot_root"])
    ldd = shutil.which("ldd", path=environment["PATH"])
    require(ldd is not None, "ldd is unavailable")
    runtime = {"kind": "numerical-smoke-not-performance" if library else "diagnostic-smoke-not-performance",
               "loader_trace": None if library else "LD_DEBUG=libs",
               "loaded_libraries_verified": False,
               "artifact_source_sha256": packet["artifact_source_sha256"],
               "build_summary_sha256": packet["build_summary_sha256"],
               "harness_sha256": packet["helper_sha256"], "ldd": {"path": ldd, "sha256": sha(ldd)},
               "python": {"path": str(Path(sys.executable).resolve()), "sha256": sha(sys.executable),
                          "version": sys.version},
               "kernel": list(os.uname()), "job_id": os.environ["PBS_JOBID"],
               "allocated_cpus": packet["allocated_cpus"], "allocated_gpus": packet["allocated_gpus"],
               "resolution": {}, "cpu_worker_count_verified": False}
    for entry in proof["entries"]:
        result = processes.run([ldd, entry["executable_path"]], environment, "ldd-" + entry["target_name"])
        runtime["resolution"][entry["artifact_id"]] = evidence.validate_artifact_resolution(
            entry, proof["metadata_by_hash"][entry["build_metadata_sha256"]], result.stdout,
            link_dependencies=link_dependencies(entry, proof["link_evidence"]))
    write_json(root / "resolution.json", runtime["resolution"])
    examples = []
    for entry in proof["entries"]:
        if entry["executable_role"] != "example":
            continue
        require(sha(entry["executable_path"]) == entry["binary_sha256"], "Example binary changed")
        result = processes.run([entry["executable_path"]], environment if library else dict(environment, LD_DEBUG="libs"),
                               "example-" + entry["target_name"])
        if library is None:
            trace = evidence.validate_loaded_trace(entry, proof["metadata_by_hash"][entry["build_metadata_sha256"]],
                result.stderr, expected_program=entry["executable_path"], expected_pid=result.pid,
                link_dependencies=link_dependencies(entry, proof["link_evidence"]))
            write_json(result.record["stderr"] + ".evidence.json", trace)
        examples.append({"artifact_id": entry["artifact_id"], "exit": result.returncode,
                         "stdout": result.record["stdout"], "stderr": result.record["stderr"]})
    require(len(examples) == (3 * len(library) if library else 18), "Incomplete teaching example coverage")
    write_json(root / "examples.json", examples)
    expected = {}
    diagnostic = {}
    # Both scopes and both requested thread settings are observed before fixing
    # the final runtime identity. Diagnostic rows never enter the final 288.
    for name in (() if library else ("serial-jsonl", "threaded-jsonl")):
        config = definitions[name]
        env = smoke_environment(os.environ, root, config["cpu_threads"])
        report, observed = execute_campaign(runner, campaigns, evidence, processes, proof, manifest,
            config, root / ("diagnostic-" + name), packet["run_id"] + "-diag-" + name,
            os.environ["PBS_JOBID"], env, runtime, None)
        require(not (set(expected) & set(observed)), "Duplicate preflight identity")
        expected.update(observed)
        diagnostic[name] = report
    if library is None:
        require(len(expected) == 72, "Incomplete runtime preflight")
        write_json(root / "diagnostic-validation.json", diagnostic)
        runtime["expected_initialized_objects"] = expected
    reports = {}
    for name, config in definitions.items():
        reports[name], _ = execute_campaign(runner, campaigns, evidence, processes, proof, manifest,
            config, root / name, packet["run_id"] + "-" + name, os.environ["PBS_JOBID"],
            smoke_environment(os.environ, root, config["cpu_threads"]), runtime, expected, library=library)
    final = campaigns.validate_campaign_set(reports, library=library)
    check_module_origins(packet["snapshot_root"])
    require(source.verify_prebuilt_source(packet["snapshot_root"], packet["build_run_root"], **proof_args) == full_proof,
            "Prebuilt evidence changed during smoke")
    for name, digest in packet["helper_sha256"].items():
        require(sha(packet_root / name) == digest, "Harness changed during smoke")
    final.update(examples=len(examples), diagnostic_rows=0 if library else 144,
                 numerical_only=bool(library), performance_measurement=False,
                 loaded_libraries_verified=False, diagnostic_timings_only=True,
                 cpu_worker_count_verified=False, artifact_source_sha256=packet["artifact_source_sha256"])
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--expected-packet-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        packet = validate_packet(args.packet, args.expected_packet_sha256, os.environ)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("AMD smoke stopped before execution: " + str(error), file=sys.stderr)
        return 2
    root = Path(packet["run_root"]) / "smoke"
    root.mkdir(mode=0o700)
    processes = Processes(root / "commands", packet["timeout_seconds"], packet["process_timeout_seconds"])
    result = {"status": "FAILED", "run_id": packet["run_id"], "job_id": os.environ["PBS_JOBID"],
              "started": utc(), "ssh_calls": 0, "scheduler_calls": 0, "builds": 0}
    code = 1
    handlers = {number: signal.signal(number, request_stop) for number in (signal.SIGTERM, signal.SIGINT)}
    try:
        result["validation"] = execute(packet, root, processes)
        result["status"] = "SMOKE_COMPLETE"
        code = 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        result["error"] = str(error)
        if isinstance(error, StopRequested) or any(record["timed_out"] for record in processes.records):
            result["status"] = "UNKNOWN"
    finally:
        result.update(ended=utc(), commands=processes.records)
        write_json(root / "summary.json", result)
        for number, handler in handlers.items():
            signal.signal(number, handler)
    print("GPU_SUITE_SIRIUS_AMD_SMOKE_END " + packet["run_id"] + " " + str(code))
    return code


if __name__ == "__main__":
    sys.exit(main())

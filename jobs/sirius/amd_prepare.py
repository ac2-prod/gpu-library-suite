#!/usr/bin/env python3
"""Same-allocation preparation after the 18-benchmark clean build; never submit."""
import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import amd_build
import amd_production as production


def identity(path):
    item = amd_build.file_identity(path)
    return {key: item[key] for key in ("path", "sha256")}


def full_host(nodefile, hostname, cpus):
    lines = nodefile.read_text(encoding="utf-8").splitlines()
    hosts = {line.split(".")[0].lower() for line in lines}
    production.require(len(lines) == 4 and all(production.HOST.fullmatch(s) for s in lines)
                       and hosts == {hostname.split(".")[0].lower()} and len(cpus) >= 96,
                       "Expected select=4/place=exclhost: four slots on this host and at least 96 allocated CPUs")
    return dict(hostname=hostname, nodefile=identity(nodefile), slots=4, inherited_cpus=sorted(cpus))


def preflight(job):
    """No compiler, GPU, MPI, scheduler or network call here."""
    env = job.env
    production.require(sys.platform == "linux" and env.get("PBS_ENVIRONMENT") == "PBS_BATCH",
                       "Preparation requires a Linux PBS batch allocation")
    nodefile = Path(env.get("PBS_NODEFILE", ""))
    production.require(nodefile.is_absolute() and nodefile.is_file(), "Missing PBS nodefile")
    record = full_host(nodefile, socket.gethostname(), os.sched_getaffinity(0))
    for name in ("mpirun", "ldd"):
        path = shutil.which(name, path=env.get("PATH"))
        production.require(path is not None, "Preparation tool unavailable: " + name)
        record[name] = identity(path)
    production.require("openmpi/5.0.10/rocm7.2.0" in env.get("LOADEDMODULES", "").split(":"),
                       "The approved pilot MPI module must be loaded before building")
    record["rocm_version_file"] = identity(job.args.rocm_version_file)
    record["scheduler_job_id"] = env["PBS_JOBID"]
    # Validate thread/loader controls before spending the build budget.
    production.worker_environment(env, job.root)
    amd_build.write_json(job.root / "preparation-inputs.json", record)
    return record


def cpu_list(text):
    result = set()
    for field in text.strip().split(","):
        production.require(re.fullmatch(r"[0-9]+(?:-[0-9]+)?", field), "Invalid kernel CPU list")
        limits = [int(v) for v in field.split("-")]
        first, last = limits[0], limits[-1]
        production.require(first <= last < 1048576, "Invalid kernel CPU range")
        result.update(range(first, last + 1))
    return result


def numa_binding(observation, allowed, sysfs=Path("/sys")):
    production.require(observation.get("hip_device") == 0
                       and type(observation.get("visible_hip_devices")) is int
                       and observation["visible_hip_devices"] > 0
                       and re.fullmatch(r"[0-9a-f]{32}", observation.get("gpu_uuid", ""))
                       and observation["gpu_uuid"] != "0" * 32, "Missing selected APU identity")
    match = re.fullmatch(r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])",
                        observation.get("pci_bus_id", ""))
    production.require(match is not None, "Invalid PCI bus identity")
    pci = "{:04x}:{:02x}:{:02x}.{:x}".format(*(int(s, 16) for s in match.groups()))
    numa_path = sysfs / "bus/pci/devices" / pci / "numa_node"
    numa = int(numa_path.read_text().strip())
    production.require(numa >= 0, "Kernel does not identify the APU NUMA node")
    cpulist = sysfs / "devices/system/node" / ("node" + str(numa)) / "cpulist"
    cpus = cpu_list(cpulist.read_text())
    production.require(len(cpus) == 24 and cpus <= set(allowed),
                       "The selected APU must have exactly 24 scheduler-allowed NUMA CPUs")
    return dict(pci_bus_id=pci, numa_node=numa, affinity_cpus=sorted(cpus),
                topology_files=[identity(numa_path), identity(cpulist)])


def check_worker(request, output, run=subprocess.run, sysfs=Path("/sys")):
    """One rank: identify device 0, bind its NUMA CPUs, then prove interop."""
    production.require(os.environ.get("PBS_JOBID") == request["scheduler_job_id"]
                       and os.environ.get("OMPI_COMM_WORLD_SIZE") == "1"
                       and os.environ.get("OMPI_COMM_WORLD_RANK") == "0"
                       and os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK") == "0", "Unexpected MPI allocation/rank")
    allocation = full_host(production.checked_file(request["nodefile"]), socket.gethostname(), os.sched_getaffinity(0))
    output.mkdir()
    record = dict(status="FAILED", allocation=allocation, scheduler_job_id=os.environ["PBS_JOBID"],
                  visibility={k: os.environ.get(k) for k in production.VISIBILITY}, commands=[])
    try:
        executable = production.checked_file(request["probe"])
        observations = []
        for mode in ("identity", "verify"):
            argv = [str(executable), mode]
            # The MPI parent has the common job deadline; this child is bounded
            # too. No benchmark, fallback device, sweep or retry is performed.
            with (output / (mode + ".stdout")).open("xb") as out, (output / (mode + ".stderr")).open("xb") as err:
                item = dict(argv=argv, started=amd_build.utc(), exit_status=None)
                record["commands"].append(item)
                try:
                    result = run(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=30, check=False)
                    item["exit_status"] = result.returncode
                finally:
                    item["ended"] = amd_build.utc()
            production.require(result.returncode == 0, "APU helper failed: " + mode)
            observation = amd_build.load_json(output / (mode + ".stdout"))
            observations.append(observation)
            if mode == "identity":
                binding = numa_binding(observation, allocation["inherited_cpus"], sysfs)
                os.sched_setaffinity(0, binding["affinity_cpus"])
                production.require(sorted(os.sched_getaffinity(0)) == binding["affinity_cpus"], "CPU binding did not take effect")
                record["binding"] = binding
        production.require(all(observations[0][key] == observations[1][key]
                               for key in ("hip_device", "pci_bus_id", "gpu_uuid", "visible_hip_devices"))
                           and observations[1].get("hip_openmp_same_device") is True,
                           "HIP/OpenMP identity changed or correspondence was not established")
        record.update(status="COMPLETE", observations=observations,
                      gpu_mask_modified=False, allocated_vnode=None,
                      limitation="Full-host NUMA/APU proof only; PBS vnode index is not inferred")
    finally:
        production.save(output / "placement.json", record)
    return record


class BuildProcesses:
    """Reuse the build's single deadline, subprocess boundary and command log."""
    def __init__(self, job):
        self.job, self.records = job, []

    def run(self, argv, env, label):
        try:
            stdout = self.job.run(label, argv, 30, env=env)
            return subprocess.CompletedProcess(argv, 0, stdout, "")
        finally:
            self.records.append(self.job.commands[-1])


def freeze_inputs(job, root):
    entries, metadata, links = [], [], {}
    for profile, result in job.profiles.items():
        metadata.append(identity(result["files"]["build-metadata.json"]["path"]))
        for target, artifact in result["artifacts"].items():
            entry = artifact["manifest_entry"]
            entries.append(entry)
            program = Path(entry["executable_path"])
            directory = program.parent / "CMakeFiles" / (target + ".dir")
            if profile == "hip":
                command = job.commands[result["target_builds"][target]["command_index"] - 1]
                links[entry["artifact_id"]] = dict(mode="native-hip-link-command",
                    link_command=identity(directory / "link.txt"), build_stdout=identity(command["stdout"]),
                    build_directory=str(program.parent))
            else:
                links[entry["artifact_id"]] = str(directory / "link.d")
    production.require(len(entries) == 18 and all(r["status"] == "BUILT" for r in job.slots), "18 successful benchmarks required")
    entries = production.runner.merge_entries([entries])
    production.save(root / "executables-manifest.json", dict(manifest_schema_version=1, entries=entries))
    config = production.load_config(job.args.source_root / "jobs/sirius/amd-benchmark.json")
    for family, settings in config["benchmarks"].items():
        cpu = next(e for e in entries if e["library"] == family and e["implementation"] == "cpu")
        for series in settings["series"]:
            if series["implementation"] == "cpu":
                production.require(series["cpu_backend"] in cpu["supported_cpu_backends"],
                                   "Production CPU backend was not built: " + family)
    production.save(root / "effective-config.json", config)
    production.save(root / "link-evidence.json", links)
    request = dict(sirius_runtime_request_schema_version=1,
                   manifest=identity(root / "executables-manifest.json"), config=identity(root / "effective-config.json"),
                   build_metadata=metadata, link_evidence=identity(root / "link-evidence.json"))
    request.update({k: job.preparation_inputs[k] for k in ("mpirun", "ldd", "rocm_version_file")})
    production.artifact_inputs(request)
    production.save(root / "runtime-request.json", request)
    return request


def prepare(job):
    root = job.root / "preparation"
    root.mkdir()
    summary = dict(status="FAILED", placement="NOT_RUN", runtime="NOT_RUN", benchmark_executions=0)
    try:
        request = freeze_inputs(job, root)
        metadata = job.profiles["openmp"]["build_metadata"]
        source = job.args.source_root / "jobs/sirius/apu_probe.cpp"
        executable = root / "apu_probe"
        flags = shlex.split(metadata["openmp"]["compile_flags"])
        linker = shlex.split(metadata["openmp"]["link_flags"])
        argv = [metadata["cxx"]["compiler_path"], "-std=c++17", "-D__HIP_PLATFORM_AMD__=1",
                "-isystem", metadata["rocm"]["hip_include"], *flags, str(source),
                *linker, metadata["dependencies"]["hip"]["path"], "-o", str(executable)]
        summary["placement"] = "RUNNING"
        job.run("placement-compile", argv, 120)
        production.save(root / "probe-build.json", dict(source=identity(source), binary=identity(executable), argv=argv))
        placement_request = dict(probe=identity(executable), scheduler_job_id=job.env["PBS_JOBID"],
                                 nodefile=job.preparation_inputs["nodefile"])
        production.save(root / "placement-request.json", placement_request)
        environment = production.worker_environment(job.env, job.root)
        job.run("placement-mpi", [request["mpirun"]["path"], "--hostfile", placement_request["nodefile"]["path"],
                "-np", "1", "-npernode", "1", "--bind-to", "none", "--display-map", sys.executable, "-B",
                Path(__file__).resolve(), "--worker-request", root / "placement-request.json",
                "--output", root / "placement"], 90, env=environment)
        placement = production.load(root / "placement/placement.json")
        production.require(placement["status"] == "COMPLETE", "Missing successful CPU/APU proof")
        summary["placement"] = "COMPLETE"
        summary["runtime"] = "RUNNING"
        production.collect_runtime(request, root / "runtime", processes=BuildProcesses(job), environment=job.env)
        summary.update(runtime="COMPLETE", status="PREPARED",
                       runtime_input=identity(root / "runtime/runtime-environment.json"))
        # Source changes after compile cannot silently become measurement inputs.
        after = amd_build.source_evidence(job)
        production.require(after["source_snapshot_sha256"] == job.args.expected_source_sha256,
                           "Source changed during preparation")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        summary.update(status="FAILED", error=str(error))
        for field in ("placement", "runtime"):
            if summary[field] == "RUNNING":
                summary[field] = "FAILED"
        raise
    finally:
        production.save(root / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        check_worker(production.load(args.worker_request), args.output)
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""Prepare actual six-host placement, then run one approved AMD wave in PBS.

Reuses the retained APU helper and 18 binaries. No build, submission, retry or
runtime recollection; the packet supplies the frozen runtime and source IDs.
"""
import argparse
import os
from pathlib import Path
import signal
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import amd_prepare as preparation
import amd_production as production


def assemble_placement(request, directory):
    rows = []
    for rank in range(production.NODES):
        path = directory / str(rank) / "placement.json"
        proof = production.load(path)
        rows.append(dict(rank=rank, local_rank=proof["local_rank"],
            hostname=proof["allocation"]["hostname"], allocated_vnode=None,
            gpu_uuid=proof["observations"][0]["gpu_uuid"],
            affinity_cpus=proof["binding"]["affinity_cpus"], visibility=proof["visibility"],
            proof=preparation.identity(path)))
    document = dict(status="site-verified", method="full-host-device0-numa",
                    scheduler_job_id=request["scheduler_job_id"], evidence=request["nodefile"],
                    rank_bindings=rows)
    production.validate_placement(document, request["scheduler_job_id"])
    production.require(len({r["gpu_uuid"] for r in rows}) == production.NODES,
                       "Duplicate physical APU identities")
    return document


def run(request, output):
    production.require(sys.platform == "linux" and os.environ.get("PBS_ENVIRONMENT") == "PBS_BATCH"
                       and os.environ.get("PBS_JOBID"), "An approved PBS batch allocation is required")
    production.require(request["wave"] == 0 and request.get("source_revision") is not None,
                       "This preparation is limited to a sealed wave 0")
    production.require(output.is_absolute() and (production.ROOT / "runs").resolve() in output.resolve().parents,
                       "Preparation output must be a fresh directory under runs/")
    run_root = production.ROOT / "runs" / request["run_id"]
    production.runner.validate_execution_context(request["run_id"], "Sirius", "master", 0, 0, 0)
    production.require(not run_root.exists() and run_root not in output.parents,
                       "Campaign output already exists or overlaps preparation")
    _, manifest, _, runtime = production.inputs(request)
    mpi = production.checked_file(request["mpirun"])
    production.require(os.access(mpi, os.X_OK), "MPI launcher is not executable")
    probe = production.checked_file(request["apu_probe"])
    production.require(os.access(probe, os.X_OK), "Retained APU helper is not executable")
    output.mkdir()
    for name in ("tmp", "cache", "placement"):
        (output / name).mkdir()
    processes = production.Processes(output / "commands", 180, 180)
    summary = dict(status="FAILED", started=production.runner.utc_timestamp(),
                   scheduler_job_id=os.environ["PBS_JOBID"], run_id=request["run_id"], builds=0)
    try:
        production.clean_source(processes, manifest, request["source_revision"])
        environment = production.worker_environment(os.environ, output)
        production.check_runtime_environment(environment, runtime["sirius_execution_checks"]["environment"],
                                             output / "runtime-mismatch.json")
        nodefile = Path(os.environ.get("PBS_NODEFILE", ""))
        production.require(nodefile.is_absolute() and nodefile.is_file(), "Missing PBS nodefile")
        frozen_nodefile = output / "pbs-nodefile.txt"
        with frozen_nodefile.open("xb") as stream:
            stream.write(nodefile.read_bytes())
        allocation = preparation.identity(frozen_nodefile)
        production.allocation_hosts(allocation)
        placement_request = dict(world_size=production.NODES, scheduler_job_id=os.environ["PBS_JOBID"],
                                 nodefile=allocation, probe=request["apu_probe"])
        preparation_path = output / "placement-request.json"
        production.save(preparation_path, placement_request)
        argv = [request["mpirun"]["path"], "--display-map", "--hostfile", str(frozen_nodefile),
                "-np", "6", "-npernode", "1", "--bind-to", "none",
                "-x", "PATH", "-x", "LD_LIBRARY_PATH", sys.executable, "-B",
                str(Path(preparation.__file__).resolve()), "--worker-request", str(preparation_path),
                "--output", str(output / "placement")]
        processes.run(argv, environment, "mpi-placement")
        placement = assemble_placement(placement_request, output / "placement")
        production.save(output / "placement.json", placement)
        plan = dict(request, placement=preparation.identity(output / "placement.json"),
                    allocation_nodefile=allocation)
        production.save(output / "launch-plan.json", plan)
        code = production.run_master(run_root, plan)
        summary["status"] = production.load(run_root / "launch/0/status.json")["status"]
        return code
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        summary["error"] = str(error)
        if isinstance(error, production.StopRequested) or any(r["timed_out"] for r in processes.records):
            summary["status"] = "UNKNOWN"
        print(str(error), file=sys.stderr)
        return 1
    finally:
        summary.update(ended=production.runner.utc_timestamp(), preparation_commands=processes.records)
        production.save(output / "status.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = production.checked_json({"path": str(args.request), "sha256": args.request_sha256})
    return run(request, args.output)


if __name__ == "__main__":
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, production.request_stop)
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print("Sirius wave stopped: " + str(error), file=sys.stderr)
        sys.exit(1)

"""Minimal preparation fixtures: no actual MPI, device, sysfs or subprocess."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "jobs/sirius"))
import amd_prepare as prepare
import amd_production as production
import amd_wave as wave


class SiriusPreparationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="amd-prepare-"))
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(subprocess, "Popen", side_effect=AssertionError("No real process")).start()
        self.nodefile = self.root / "nodefile"
        self.nodefile.write_text("node1\n" * 4)
        self.probe = self.root / "inert-probe"
        self.probe.write_text("synthetic, never executed")
        self.request = dict(scheduler_job_id="42.fixture", nodefile=prepare.identity(self.nodefile),
                            probe=prepare.identity(self.probe))
        self.observation = dict(hip_device=0, visible_hip_devices=4, pci_bus_id="0000:41:00.0",
                                gpu_uuid="1" * 32, hip_openmp_same_device=False)
        self.sysfs = self.root / "sys"
        for path, text in (("bus/pci/devices/0000:41:00.0/numa_node", "2\n"),
                           ("devices/system/node/node2/cpulist", "48-71\n")):
            target = self.sysfs / path
            target.parent.mkdir(parents=True)
            target.write_text(text)
        self.affinity = set(range(96))
        self.environment = dict(PBS_JOBID="42.fixture", OMPI_COMM_WORLD_SIZE="1",
            OMPI_COMM_WORLD_RANK="0", OMPI_COMM_WORLD_LOCAL_RANK="0", OMP_DEFAULT_DEVICE="0",
            ROCR_VISIBLE_DEVICES="0,1,2,3")
        mock.patch.dict(os.environ, self.environment, clear=True).start()
        mock.patch.object(prepare.socket, "gethostname", return_value="node1").start()
        mock.patch.object(prepare.os, "sched_getaffinity", create=True,
                          side_effect=lambda _: self.affinity.copy()).start()
        mock.patch.object(prepare.os, "sched_setaffinity", create=True,
                          side_effect=lambda _, cpus: setattr(self, "affinity", set(cpus))).start()
        self.calls = []

    def fake_probe(self, argv, **kwargs):
        self.calls.append(argv)
        observation = dict(self.observation)
        if argv[-1] == "verify":
            self.assertEqual(self.affinity, set(range(48, 72)))
            observation["hip_openmp_same_device"] = True
        kwargs["stdout"].write(json.dumps(observation).encode())
        return subprocess.CompletedProcess(argv, 0)

    def test_mapping_binds_exact_numa_cpus_and_preserves_gpu_masks(self):
        result = prepare.check_worker(self.request, self.root / "placement", self.fake_probe, self.sysfs)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["binding"]["affinity_cpus"], list(range(48, 72)))
        self.assertEqual(result["visibility"]["ROCR_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertIsNone(result["allocated_vnode"])
        self.assertEqual([a[-1] for a in self.calls], ["identity", "verify"])
        self.assertFalse(result["gpu_mask_modified"])

    def test_partial_allocation_missing_topology_or_changed_device_is_not_success(self):
        with self.assertRaisesRegex(ValueError, "four slots"):
            prepare.full_host(self.nodefile, "node1", set(range(24)))
        with self.assertRaisesRegex(ValueError, "scheduler-allowed"):
            prepare.numa_binding(self.observation, set(range(24)), self.sysfs)
        zero = dict(self.observation, gpu_uuid="0" * 32)
        with self.assertRaisesRegex(ValueError, "identity"):
            prepare.numa_binding(zero, set(range(96)), self.sysfs)
        def mismatch(argv, **kwargs):
            if argv[-1] == "verify":
                self.observation["gpu_uuid"] = "2" * 32
            return self.fake_probe(argv, **kwargs)
        output = self.root / "failed-placement"
        with self.assertRaisesRegex(ValueError, "identity changed"):
            prepare.check_worker(self.request, output, mismatch, self.sysfs)
        self.assertEqual(production.load(output / "placement.json")["status"], "FAILED")
        self.assertEqual(len(self.calls), 2)

    def test_six_full_hosts_freeze_actual_apu_proofs_without_changing_masks(self):
        self.nodefile.write_text("".join(("node" + str(i) + "\n") * 4 for i in range(6)))
        request = dict(self.request, world_size=6, nodefile=prepare.identity(self.nodefile))
        output = self.root / "six-ranks"
        output.mkdir()
        os.environ["OMPI_COMM_WORLD_SIZE"] = "6"
        for rank in range(6):
            self.affinity = set(range(96))
            self.observation["gpu_uuid"] = format(rank + 1, "032x")
            os.environ["OMPI_COMM_WORLD_RANK"] = str(rank)
            with mock.patch.object(prepare.socket, "gethostname", return_value="node" + str(rank)):
                prepare.check_worker(request, output / str(rank), self.fake_probe, self.sysfs)
        result = wave.assemble_placement(request, output)
        self.assertEqual(len(result["rank_bindings"]), 6)
        self.assertTrue(all(r["allocated_vnode"] is None and r["affinity_cpus"] == list(range(48, 72))
                            and r["visibility"]["ROCR_VISIBLE_DEVICES"] == "0,1,2,3"
                            for r in result["rank_bindings"]))
        self.assertEqual(len(self.calls), 12)
        result["rank_bindings"][0]["gpu_uuid"] = "f" * 32
        with self.assertRaisesRegex(ValueError, "physical APU"):
            production.validate_placement(result, "42.fixture")
        self.nodefile.write_text("node0\n" * 5 + "node1\n" * 3 +
                                 "".join(("node" + str(i) + "\n") * 4 for i in range(2, 6)))
        with self.assertRaisesRegex(ValueError, "four slots"):
            prepare.full_host(self.nodefile, "node0", set(range(96)), 6)

    def test_wave_preparation_stops_before_mpi_for_unsealed_source_or_failed_placement(self):
        self.probe.chmod(0o700)
        self.nodefile.write_text("".join(("node" + str(i) + "\n") * 4 for i in range(6)))
        request = dict(wave=0, run_id="fixture-wave", source_revision={"launcher_commit": None},
                       apu_probe=prepare.identity(self.probe), mpirun=prepare.identity(self.probe))
        calls = []
        runtime = {"sirius_execution_checks": {"environment": production.environment_controls(
            production.worker_environment(os.environ, self.root))}}
        processes = SimpleNamespace(records=[], run=lambda argv, env, label: calls.append((argv, label)))
        (self.root / "runs").mkdir()
        with mock.patch.object(production, "ROOT", self.root), \
             mock.patch.object(wave.sys, "platform", "linux"), \
             mock.patch.dict(os.environ, PBS_ENVIRONMENT="PBS_BATCH", PBS_NODEFILE=str(self.nodefile)), \
             mock.patch.object(production, "inputs", return_value=({}, {}, {}, runtime)), \
             mock.patch.object(production, "Processes", return_value=processes), \
             mock.patch.object(production, "clean_source", side_effect=ValueError("Unsealed source")) as clean, \
             mock.patch.object(production, "run_master") as master, \
             mock.patch.object(wave, "assemble_placement", side_effect=ValueError("Missing rank proof")) as assemble:
            self.assertEqual(wave.run(request, self.root / "runs/unsealed"), 1)
            self.assertFalse(calls)
            assemble.assert_not_called()
            master.assert_not_called()
            clean.side_effect = None
            self.assertEqual(wave.run(request, self.root / "runs/missing-rank"), 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], "mpi-placement")
            argv = calls[0][0]
            self.assertEqual(argv[argv.index("-np") + 1], "6")
            self.assertEqual(argv[argv.index("-npernode") + 1], "1")
            self.assertEqual(argv[argv.index("--bind-to") + 1], "none")
            python_index = argv.index(sys.executable)
            self.assertEqual([argv[i + 1] for i, value in enumerate(argv) if value == "-x"],
                             ["PATH", "LD_LIBRARY_PATH"])
            self.assertEqual(argv[python_index - 4:python_index],
                             ["-x", "PATH", "-x", "LD_LIBRARY_PATH"])
            master.assert_not_called()
            assemble.side_effect = None
            assemble.return_value = {"status": "fixture placement"}
            def measure(root, plan):
                self.assertEqual(production.checked_json(plan["placement"]), assemble.return_value)
                self.assertEqual(production.checked_file(plan["allocation_nodefile"]).read_bytes(),
                                 self.nodefile.read_bytes())
                target = root / "launch/0"
                target.mkdir(parents=True)
                production.save(target / "status.json", {"status": "COMPLETE"})
                return 0
            master.side_effect = measure
            self.assertEqual(wave.run(request, self.root / "runs/ready"), 0)
            master.assert_called_once()
            self.assertEqual(len(calls), 2)

    def test_same_job_handoff_shares_budget_and_stops_on_runtime_failure(self):
        metadata = dict(cxx=dict(compiler_path="/fixture/clang++"),
                        openmp=dict(compile_flags="-O3 -fopenmp --offload-arch=fixture", link_flags="-fopenmp"),
                        rocm=dict(hip_include="/fixture/include"), dependencies=dict(hip=dict(path="/fixture/libhip.so")))
        for failure in (False, True):
            with self.subTest(failure=failure):
                root = self.root / ("failed" if failure else "passed")
                root.mkdir()
                job = SimpleNamespace(root=root, env=dict(os.environ), commands=[],
                    args=SimpleNamespace(source_root=ROOT, expected_source_sha256="a" * 64),
                    profiles={"openmp": {"build_metadata": metadata}}, preparation_inputs={"nodefile": self.request["nodefile"]})
                stages = []
                def run(stage, argv, timeout=30, env=None):
                    stages.append(stage)
                    job.commands.append(dict(stage=stage, argv=[str(a) for a in argv]))
                    if stage == "placement-compile":
                        Path(argv[-1]).write_text("inert check executable")
                    if stage == "placement-mpi":
                        self.assertEqual([str(a) for a in argv[3:9]], ["-np", "1", "-npernode", "1", "--bind-to", "none"])
                        out = root / "preparation/placement"
                        out.mkdir()
                        production.save(out / "placement.json", {"status": "COMPLETE"})
                    return "fixture only\n"
                job.run = run
                def collect(request, output, *, processes, environment):
                    self.assertIs(processes.job, job)
                    self.assertIs(environment, job.env)
                    processes.run(["/fixture/ldd", "/fixture/benchmark"], environment, "fixture-resolution")
                    if failure:
                        raise ValueError("Missing library evidence")
                    output.mkdir()
                    production.save(output / "runtime-environment.json", {"fixture": True})
                with mock.patch.object(prepare, "freeze_inputs", return_value={"mpirun": {"path": "/fixture/mpirun"}}), \
                     mock.patch.object(production, "collect_runtime", side_effect=collect), \
                     mock.patch.object(prepare.amd_build, "source_evidence", return_value={"source_snapshot_sha256": "a" * 64}):
                    if failure:
                        with self.assertRaisesRegex(ValueError, "Missing library"):
                            prepare.prepare(job)
                    else:
                        self.assertEqual(prepare.prepare(job)["status"], "PREPARED")
                result = production.load(root / "preparation/summary.json")
                self.assertEqual(result["placement"], "COMPLETE")
                self.assertEqual(result["runtime"], "FAILED" if failure else "COMPLETE")
                self.assertEqual(stages, ["placement-compile", "placement-mpi", "fixture-resolution"])
                self.assertEqual(result["benchmark_executions"], 0)


if __name__ == "__main__":
    unittest.main()

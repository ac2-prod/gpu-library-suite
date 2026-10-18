"""Six-worker orchestration fixtures: no MPI, GPU, SSH or scheduler process."""

import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "jobs/sirius"))

import amd_production as production
from gpu_suite import runner
from gpu_suite.config import load_config
from gpu_suite.hashing import sha256_file
from gpu_suite.manifest import artifact_id
from gpu_suite.strict_json import dump_bytes, load
from test_sirius_amd_smoke_campaign import fixture


class SiriusProductionTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix="sirius-production-fixture-"))
        # Test runners own disposable local fixture trees; retain them for review.
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(subprocess, "Popen", side_effect=AssertionError("No real process in these tests")).start()
        self.config = load_config(ROOT / "jobs/sirius/amd-benchmark.json")
        self.prototypes, self.manifest, _ = fixture(self.config, "production")
        for entry in self.manifest["entries"]:
            entry["git_dirty"] = False
            entry["artifact_id"] = artifact_id(entry)
        self.metadata = {"2" * 64: {}}
        evidence = self.work / "binding-evidence.txt"
        evidence.write_text("fixture only; no hardware observation\n")
        mpi = self.work / "inert-mpirun"
        mpi.write_text("not executable code\n")
        mpi.chmod(0o700)
        nodefile = self.work / "pbs-nodefile.txt"
        nodefile.write_text("".join("node" + str(i) + "\n" for i in range(6)))
        self.bindings = [dict(rank=i, local_rank=0, hostname="node" + str(i),
            allocated_vnode="node" + str(i) + "[2]", gpu_uuid="fixture-gpu-" + str(i),
            affinity_cpus=list(range(48, 72)), visibility={key: (
                "0" if key in {"ROCR_VISIBLE_DEVICES", "OMP_DEFAULT_DEVICE"} else None)
                for key in production.VISIBILITY}) for i in range(6)]
        self.placement = dict(status="site-verified", method="inert fixture allocation",
            scheduler_job_id="42.fixture", evidence=self.identity(evidence), rank_bindings=self.bindings)
        placement_path = self.work / "placement.json"
        production.save(placement_path, self.placement)
        environment = production.worker_environment({}, self.work)
        self.runtime = dict(runtime_environment_schema_version=1, rocm_version="fixture",
            executables_manifest_sha256="1" * 64, build_metadata=[],
            sirius_execution_checks=dict(environment={k: environment.get(k) for k in production.ENVIRONMENT},
                                         files=[self.identity(evidence)]), resolved_libraries={})
        self.plan = dict(sirius_launch_schema_version=1, run_id="fixture-production", wave=0,
            config={"path": str(self.work / "config.json"), "sha256": production.digest(self.config)},
            manifest={"path": str(self.work / "manifest.json"), "sha256": production.digest(self.manifest)},
            runtime={"path": str(self.work / "runtime.json"), "sha256": production.digest(self.runtime)},
            build_metadata=[], mpirun=self.identity(mpi), mpi_options=["--display-map"],
            allocation_nodefile=self.identity(nodefile), placement=self.identity(placement_path),
            launcher_sha256=sha256_file(production.__file__), controller_timeout_seconds=5100,
            invocation_timeout_seconds=600)
        for field, document in (("config", self.config), ("manifest", self.manifest), ("runtime", self.runtime)):
            production.save(Path(self.plan[field]["path"]), document)
        self.input_patch = mock.patch.object(production, "inputs", return_value=(
            self.config, self.manifest, self.metadata, self.runtime))
        self.mock_inputs = self.input_patch.start()
        mock.patch.object(runner, "_validate_amd_dependencies").start()
        real_hash = runner.sha256_file
        mock.patch.object(runner, "sha256_file", side_effect=lambda p:
                          "1" * 64 if str(p).startswith("/fixture/") else real_hash(p)).start()
        self.run_root = self.work / self.plan["run_id"]
        self.process_count = 0
        self.fail_at = None
        self.fake_mpi_actions = []
        self.row_cache = {}

    @staticmethod
    def identity(path):
        return {"path": str(path), "sha256": sha256_file(path)}

    def observations(self):
        return [dict(rank=b["rank"], size=6, local_rank=0, hostname=b["hostname"],
            scheduler_job_id="42.fixture", affinity_cpus=b["affinity_cpus"], visibility=b["visibility"],
            runtime_environment_sha256=self.plan["runtime"]["sha256"]) for b in self.bindings]

    def rows_for(self, rank):
        cache_key = (self.plan["wave"], rank)
        if cache_key in self.row_cache:
            return self.row_cache[cache_key]
        context = production.context_for(self.config, self.plan, self.bindings[rank], self.runtime)
        def key(row):
            return (row["benchmark"], row["implementation"], row["scope"],
                    production.digest(runner.core_raw_parameters(row["benchmark"], row["parameters"])), row["trial"])
        prototypes = {key(r): r for r in self.prototypes}
        rows = []
        for item in production.schedule_for(self.config, self.manifest, context):
            for trial in range(5):
                parameters = runner.core_raw_parameters(item["benchmark"],
                    runner.expected_raw_parameters(item["benchmark"], item["parameters"], item["series"]["implementation"]))
                row = copy.deepcopy(prototypes[(item["benchmark"], item["series"]["implementation"],
                                                item["scope"], production.digest(parameters), trial)])
                row.update(run_id=self.plan["run_id"], wave=self.plan["wave"], node_index=rank,
                    hostname=context["hostname"], block_id="{0}|{1}|{2}".format(
                        self.plan["run_id"], self.plan["wave"], context["hostname"]),
                    scheduler_job_id="42.fixture", implementation_order=list(context["implementation_order"]),
                    runtime_environment_sha256=self.plan["runtime"]["sha256"], git_dirty=False,
                    source_snapshot_sha256=None, git_diff_sha256=None)
                rows.append(row)
        self.row_cache[cache_key] = rows
        return rows

    def fake_processes(self, root, timeout, process_timeout):
        test = self
        root.mkdir()
        class Fake:
            def __init__(self):
                self.records = []

            def run(self, argv, environment, label):
                if label.startswith("mpi-"):
                    action = "observe" if label == "mpi-preflight" else "worker"
                    test.fake_mpi_actions.append(action)
                    test.assertIn("-np", argv)
                    test.assertEqual(argv[argv.index("-np") + 1], "6")
                    test.assertEqual(argv[argv.index("-npernode") + 1], "1")
                    test.assertEqual(argv[argv.index("--bind-to") + 1], "none")
                    test.assertEqual(argv[argv.index("--hostfile") + 1], test.plan["allocation_nodefile"]["path"])
                    for rank in range(6):
                        if action == "observe":
                            production.save(test.run_root / "launch" / str(test.plan["wave"]) /
                                            "observations" / (str(rank) + ".json"), test.observations()[rank])
                        else:
                            wave_path = test.run_root / "waves" / str(test.plan["wave"]) / "wave-metadata.json"
                            with mock.patch.object(production, "observe", return_value=test.observations()[rank]):
                                code = production.run_worker(test.run_root, test.plan, sha256_file(wave_path))
                            test.assertEqual(code, 0)
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                test.process_count += 1
                index = len(self.records)
                rank = int(root.parent.name[4:])
                rows = test.rows_for(rank)[index * 5:(index + 1) * 5]
                stdout, stderr = root / (str(index) + ".stdout"), root / (str(index) + ".stderr")
                stdout.write_bytes(b"".join(dump_bytes(row) for row in rows))
                stderr.write_text("fixture stderr is kept separate\n")
                failed = test.fail_at == index
                self.records.append(dict(argv=argv, started="fixture-start", ended="fixture-end",
                    exit=None if failed else 0, timed_out=failed, stdout=str(stdout), stderr=str(stderr),
                    stdout_sha256=sha256_file(stdout), stderr_sha256=sha256_file(stderr)))
                if failed:
                    raise ValueError("fixture timeout; no retry")
                return SimpleNamespace(returncode=0, stdout=stdout.read_text(), stderr="")
        return Fake()

    def master(self):
        with mock.patch.dict(os.environ, PBS_JOBID="42.fixture"), \
                mock.patch.object(production, "Processes", side_effect=self.fake_processes), \
                mock.patch.object(production, "clean_source"):
            return production.run_master(self.run_root, self.plan)

    def test_six_workers_two_waves_have_exclusive_outputs_and_6480_valid_rows(self):
        self.assertEqual(self.master(), 0)
        first = (self.run_root / "run-metadata.json").read_bytes()
        self.plan["wave"] = 1
        self.assertEqual(self.master(), 0)
        self.assertEqual((self.run_root / "run-metadata.json").read_bytes(), first)
        counts = [load(self.run_root / "waves" / str(wave) / "artifact-manifest.json") for wave in (0, 1)]
        self.assertEqual([r["observed_rows"] for r in counts], [3240, 3240])
        self.assertEqual(self.process_count, 1296)
        self.assertEqual(self.fake_mpi_actions, ["observe", "worker", "observe", "worker"])
        self.assertEqual(len(list(self.run_root.glob("waves/*/nodes/*/raw.jsonl"))), 12)
        with self.assertRaises(FileExistsError):
            self.master()

    def test_unresolved_duplicate_or_incomplete_placement_never_starts_mpi(self):
        self.plan["placement"] = None
        with self.assertRaisesRegex(ValueError, "unresolved"):
            self.master()
        for mutate in (
            lambda p: p["rank_bindings"].pop(),
            lambda p: p["rank_bindings"][1].update(hostname="node0.example"),
            lambda p: p["rank_bindings"][0].update(affinity_cpus=[0]),
            lambda p: p.update(scheduler_job_id="99.fixture"),
        ):
            changed = copy.deepcopy(self.placement)
            mutate(changed)
            with self.assertRaises(ValueError):
                production.validate_placement(changed, "42.fixture")
        self.assertFalse(self.run_root.exists())
        self.assertEqual(self.fake_mpi_actions, [])

    def test_observation_checks_binding_and_keeps_job_facts_outside_runtime_identity(self):
        env = dict(PBS_JOBID="42.fixture", OMPI_COMM_WORLD_RANK="0", OMPI_COMM_WORLD_SIZE="6",
                   OMPI_COMM_WORLD_LOCAL_RANK="0", ROCR_VISIBLE_DEVICES="0")
        with mock.patch.object(production.socket, "gethostname", return_value="node0"), \
                mock.patch.object(production.os, "sched_getaffinity", return_value=set(range(48, 72)), create=True):
            self.assertEqual(production.observe(self.plan, self.runtime, env), self.observations()[0])
            with self.assertRaisesRegex(ValueError, "Runtime environment"):
                production.observe(self.plan, self.runtime, dict(env, HSA_XNACK="1"))
            with self.assertRaisesRegex(ValueError, "visibility"):
                production.observe(self.plan, self.runtime, dict(env, ROCR_VISIBLE_DEVICES="1"))

    def test_one_rank_per_allocated_host_and_cpu_subset(self):
        production.validate_plan(self.plan, "42.fixture")
        for options in (["-npernode", "2"], ["--map-by", "slot"], ["--bind-to", "core"],
                        ["--host", "other"], ["--oversubscribe"]):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "policy is fixed"):
                production.validate_plan(dict(self.plan, mpi_options=options), "42.fixture")
        nodefile = self.work / "other-nodefile.txt"
        nodefile.write_text("node0\nnode0.example\nnode1\nnode2\nnode3\nnode4\n")
        with self.assertRaisesRegex(ValueError, "six physical hosts"):
            production.validate_plan(dict(self.plan, allocation_nodefile=self.identity(nodefile)), "42.fixture")
        nodefile.write_text("".join("node" + str(i) + "\n" for i in range(1, 7)))
        with self.assertRaisesRegex(ValueError, "hosts differ"):
            production.validate_plan(dict(self.plan, allocation_nodefile=self.identity(nodefile)), "42.fixture")
        env = dict(PBS_JOBID="42.fixture", OMPI_COMM_WORLD_RANK="0", OMPI_COMM_WORLD_SIZE="6",
                   OMPI_COMM_WORLD_LOCAL_RANK="0", ROCR_VISIBLE_DEVICES="0")
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(production.socket, "gethostname", return_value="node0"), \
                mock.patch.object(production.os, "sched_getaffinity", create=True) as get_affinity, \
                mock.patch.object(production.os, "sched_setaffinity", create=True) as set_affinity:
            get_affinity.side_effect = [set(range(96)), set(range(48, 72))]
            production.bind_worker(self.plan)
            set_affinity.assert_called_once_with(0, set(range(48, 72)))
            set_affinity.reset_mock()
            get_affinity.side_effect = [set(range(24))]
            with self.assertRaisesRegex(ValueError, "exceed scheduler"):
                production.bind_worker(self.plan)
            set_affinity.assert_not_called()
            os.environ["ROCR_VISIBLE_DEVICES"] = "1"
            with self.assertRaisesRegex(ValueError, "do not replace"):
                production.bind_worker(self.plan)
            self.assertEqual(os.environ["ROCR_VISIBLE_DEVICES"], "1")
            set_affinity.assert_not_called()

    def test_runtime_collection_keeps_software_identity_and_separate_job_evidence(self):
        version = self.work / "rocm-version"
        version.write_text("7.2.0\n")
        library = self.work / "fixture.so"
        library.write_bytes(b"\x7fELFfixture identity only")
        compiler = self.work / "compiler"
        compiler.write_text("fixture compiler")
        links = self.work / "links.json"
        production.save(links, {e["artifact_id"]: "fixture" for e in self.manifest["entries"]})
        metadata = {e["build_metadata_sha256"]: {"cxx": {"compiler_path": str(compiler)}}
                    for e in self.manifest["entries"]}
        request = dict(self.plan, sirius_runtime_request_schema_version=1,
                       ldd=self.plan["mpirun"], rocm_version_file=self.identity(version),
                       link_evidence=self.identity(links))
        commands = []
        def run(argv, environment, label):
            self.assertEqual(argv[0], request["ldd"]["path"])
            self.assertTrue(label.startswith("ldd-"))
            self.assertNotIn("LD_DEBUG", environment)
            commands.append(argv)
            return SimpleNamespace(returncode=0, stdout="fixture ldd output", stderr="")
        processes = SimpleNamespace(records=[], run=run)
        def resolution(entry, metadata, text, link_dependencies):
            self.assertEqual(text, "fixture ldd output")
            return dict(resolved_objects={str(library): sha256_file(library)},
                        normalized_lines=["fixture.so => " + str(library)])
        env = dict(PATH="/bin:/usr/bin:/bin", LD_LIBRARY_PATH="/lib:/usr/lib:/lib", PBS_JOBID="42.fixture")
        collector = dict(git_commit="1" * 40, git_status=" M jobs/sirius/amd_smoke_evidence.py\n",
                         git_diff_sha256="2" * 64, files=[])
        with mock.patch.object(production, "artifact_inputs", return_value=(self.config, self.manifest, metadata)), \
                mock.patch.object(production, "preflight_link_evidence"), \
                mock.patch.object(production, "collector_source", return_value=collector), \
                mock.patch.object(production, "Processes", return_value=processes), \
                mock.patch.object(production.amd_smoke_evidence, "validate_artifact_resolution", side_effect=resolution), \
                mock.patch.dict(os.environ, env, clear=True):
            first = production.collect_runtime(request, self.work / "runtime-wave0")
            self.assertEqual(len(commands), 18)
            self.assertEqual(first["sirius_execution_checks"]["environment"]["PATH"], "/bin:/usr/bin")
            self.assertEqual(first["rocm_version"], "7.2.0")
            self.assertEqual(len(first["resolved_libraries"]), 18)
            self.assertFalse(first["loaded_libraries_verified"])
            self.assertIsNone(first["observed_cpu_worker_count"])
            self.assertIn(str(compiler), [f["path"] for f in first["sirius_execution_checks"]["files"]])
            os.environ.update(PBS_JOBID="43.fixture", ROCR_VISIBLE_DEVICES="1", TMPDIR="/different/job/cache")
            with mock.patch.object(production.socket, "gethostname", return_value="different-host"):
                second = production.collect_runtime(request, self.work / "runtime-wave1",
                                                    processes=processes, environment=dict(os.environ))
            self.assertEqual(first, second)
            evidence = load(self.work / "runtime-wave1/runtime-environment-evidence.json")
            self.assertEqual((evidence["hostname"], evidence["scheduler_job_id"]), ("different-host", "43.fixture"))
            self.assertEqual(evidence["collector_source"], collector)
            self.assertNotIn("collector_source", second)
            os.environ["LD_LIBRARY_PATH"] = "/different/library:/lib:/usr/lib"
            third = production.collect_runtime(request, self.work / "runtime-changed")
            self.assertNotEqual(production.digest(first), production.digest(third))
            with self.assertRaises(FileExistsError):
                production.collect_runtime(request, self.work / "runtime-wave0")
            count = len(commands)
            with mock.patch.object(production.amd_smoke_evidence, "validate_artifact_resolution",
                                   side_effect=ValueError("unresolved ldd dependency")):
                with self.assertRaisesRegex(ValueError, "unresolved ldd"):
                    production.collect_runtime(request, self.work / "runtime-failed")
            self.assertEqual(len(commands), count + 1)
            self.assertFalse((self.work / "runtime-failed/runtime-environment.json").exists())
            self.assertEqual(load(self.work / "runtime-failed/runtime-environment-evidence.json")["status"], "FAILED")
            with mock.patch.object(production, "collector_source", side_effect=[collector, dict(collector, files=["changed"])]):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    production.collect_runtime(request, self.work / "runtime-source-changed")
            self.assertFalse((self.work / "runtime-source-changed/runtime-environment.json").exists())
            version.write_text("changed\n")
            count = len(commands)
            with self.assertRaisesRegex(ValueError, "Missing or changed"):
                production.collect_runtime(request, self.work / "runtime-wrong-version")
            self.assertEqual(len(commands), count)

    def test_collection_records_tool_edits_but_measurement_still_requires_clean_source(self):
        def run(argv, environment, label):
            values = {"collector-head": "1" * 40 + "\n", "collector-status": " M jobs/sirius/amd_smoke_evidence.py\n",
                      "collector-diff": "fixture patch\n", "git-head": "1" * 40 + "\n",
                      "git-status": " M jobs/sirius/amd_smoke_evidence.py\n"}
            return SimpleNamespace(stdout=values[label])
        processes = SimpleNamespace(run=run)
        result = production.collector_source(processes)
        self.assertEqual(result["git_commit"], "1" * 40)
        self.assertTrue(result["git_status"].startswith(" M "))
        self.assertEqual(len(result["git_diff_sha256"]), 64)
        self.assertIn(str(ROOT / "jobs/sirius/amd_production.py"), [f["path"] for f in result["files"]])
        with self.assertRaisesRegex(ValueError, "must be clean"):
            production.clean_source(processes, self.manifest)

    def test_clean_tool_revision_reuses_unchanged_artifacts_without_relabelling(self):
        base = "1" * 40
        for entry in self.manifest["entries"]:
            entry["git_commit"] = base
            entry["artifact_id"] = artifact_id(entry)
        Path(self.plan["manifest"]["path"]).write_bytes(dump_bytes(self.manifest))
        self.plan["manifest"]["sha256"] = production.digest(self.manifest)
        self.runtime["executables_manifest_sha256"] = self.plan["manifest"]["sha256"]
        Path(self.plan["runtime"]["path"]).write_bytes(dump_bytes(self.runtime))
        self.plan["runtime"]["sha256"] = production.digest(self.runtime)
        revision = dict(artifact_commit=base, launcher_commit="a" * 40,
                        changed_paths=["jobs/sirius/amd_production.py"])
        outputs = {"git-head": "a" * 40 + "\n", "git-status": "",
                   "git-tool-revision": "jobs/sirius/amd_production.py\n"}
        processes = SimpleNamespace(run=lambda argv, env, label: SimpleNamespace(stdout=outputs[label]))
        production.clean_source(processes, self.manifest, revision)
        self.assertEqual({e["git_commit"] for e in self.manifest["entries"]}, {base})
        for changed in (dict(revision, launcher_commit=None),
                        dict(revision, artifact_commit="b" * 40),
                        dict(revision, changed_paths=["amd/c-cpp/rocfft/benchmarks/rocfft_hip.cpp"]),
                        dict(revision, changed_paths=[])):
            with self.subTest(revision=changed), self.assertRaises(ValueError):
                production.clean_source(processes, self.manifest, changed)
        outputs["git-status"] = " M jobs/sirius/amd_production.py\n"
        with self.assertRaisesRegex(ValueError, "must be clean"):
            production.clean_source(processes, self.manifest, revision)
        outputs["git-status"] = ""
        self.run_root.mkdir()
        self.plan["source_revision"] = revision
        production.prepare_wave(self.run_root, self.plan, self.bindings, self.observations())
        saved = load(self.run_root / "run-metadata.json")
        self.assertEqual(saved["git_commit"], base)
        self.assertEqual(saved["launcher"]["source_revision"], revision)
        changed = dict(self.plan, wave=1, source_revision=dict(revision, launcher_commit="c" * 40))
        with self.assertRaisesRegex(ValueError, "provenance"):
            production.prepare_wave(self.run_root, changed, self.bindings, self.observations())

    def test_input_inventory_hashes_and_clean_artifact_gate(self):
        self.input_patch.stop()
        compiler = self.work / "inert-compiler"
        compiler.write_text("fixture compiler identity only\n")
        library = self.work / "inert-library.so"
        library.write_text("fixture library identity only\n")
        manifest = copy.deepcopy(self.manifest)
        metadata = {}
        names = set().union(*(runner._amd_dependency_names(e) for e in manifest["entries"]))
        for entry in manifest["entries"]:
            profile = entry["build_profile"]
            if profile not in metadata:
                doc = {k: entry[k] for k in ("git_commit", "git_dirty", "git_metadata_available", "build_type", "build_profile")}
                doc[entry["compiler_language"]] = {k: entry[k] for k in ("compiler", "compiler_version", "global_configure_flags")}
                doc[entry["compiler_language"]]["compiler_path"] = str(compiler)
                doc["dependencies"] = {name: self.identity(library) for name in names}
                path = self.work / (profile + "-metadata.json")
                production.save(path, doc)
                metadata[profile] = self.identity(path)
            binary = self.work / entry["target_name"]
            binary.write_text("fixture binary identity only\n")
            entry.update(executable_path=str(binary), binary_sha256=sha256_file(binary),
                         build_metadata_sha256=metadata[profile]["sha256"])
            entry["artifact_id"] = artifact_id(entry)
        manifest["entries"] = runner.merge_entries([manifest["entries"]])
        path = self.work / "real-input-manifest.json"
        production.save(path, manifest)
        plan = dict(self.plan, manifest=self.identity(path), build_metadata=list(metadata.values()))
        runtime = copy.deepcopy(self.runtime)
        runtime.update(executables_manifest_sha256=plan["manifest"]["sha256"],
                       build_metadata=list(metadata.values()),
                       resolved_libraries={e["artifact_id"]: [self.identity(library)] for e in manifest["entries"]})
        runtime["sirius_execution_checks"]["files"] = [self.identity(compiler), plan["mpirun"], self.identity(Path(sys.executable))]
        runtime_path = self.work / "real-runtime.json"
        production.save(runtime_path, runtime)
        plan["runtime"] = self.identity(runtime_path)
        self.assertEqual(len(production.inputs(plan)[1]["entries"]), 18)
        # Dependency APIs remain mocked; no vendor toolchain is being certified.
        changed = copy.deepcopy(manifest)
        changed["entries"][0]["git_dirty"] = True
        with mock.patch.object(runner, "validate_build_metadata"), self.assertRaisesRegex(ValueError, "clean build"):
            runner.validate_manifest_artifacts(changed, runner.load_build_metadata(
                [Path(i["path"]) for i in metadata.values()]), "production", False)
        Path(manifest["entries"][0]["executable_path"]).write_text("changed fixture bytes\n")
        with self.assertRaisesRegex(ValueError, "binary hash"):
            production.inputs(plan)

    def test_changed_campaign_identity_is_rejected_before_second_measurement(self):
        self.run_root.mkdir()
        self.plan["wave"] = 0
        production.prepare_wave(self.run_root, self.plan, self.bindings, self.observations())
        self.plan["wave"] = 1
        original = self.plan["runtime"]["sha256"]
        for field in ("config", "manifest", "runtime"):
            changed = copy.deepcopy(self.plan)
            changed[field]["sha256"] = "f" * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "provenance"):
                production.prepare_wave(self.run_root, changed, self.bindings,
                    [dict(o, runtime_environment_sha256=changed["runtime"]["sha256"]) for o in self.observations()])
        self.assertEqual(self.plan["runtime"]["sha256"], original)
        self.assertFalse((self.run_root / "waves/1").exists())

    def test_timeout_preserves_partial_output_and_missing_nodes_are_not_success(self):
        self.run_root.mkdir()
        wave = production.prepare_wave(self.run_root, self.plan, self.bindings, self.observations())
        self.fail_at = 1
        with mock.patch.object(production, "Processes", side_effect=self.fake_processes), \
                mock.patch.object(production, "observe", return_value=self.observations()[0]):
            self.assertEqual(production.run_worker(self.run_root, self.plan, production.digest(wave)), 1)
        node = self.run_root / "waves/0/nodes/node0"
        self.assertEqual(load(node / "node-status.json")["status"], "UNKNOWN")
        self.assertEqual(self.process_count, 2)
        self.assertEqual(len((node / "raw.jsonl").read_text().splitlines()), 5)
        result = production.collect_wave(self.run_root, self.plan, None)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(len(result["errors"]), 7)
        self.assertTrue(result["files"])

    def test_collector_rejects_tampered_node_evidence(self):
        self.assertEqual(self.master(), 0)
        # A second collector output is never allowed to overwrite the first.
        with self.assertRaises(FileExistsError):
            production.collect_wave(self.run_root, self.plan, 0)
        rows = self.rows_for(0)
        for mutate in (
            lambda r: r.pop(),
            lambda r: r[0].update(runtime_environment_sha256="f" * 64),
            lambda r: next(x for x in r if x["implementation"] == "hip").update(gpu_uuid="wrong-device"),
        ):
            changed = copy.deepcopy(rows)
            mutate(changed)
            with self.assertRaises(ValueError):
                production.validate_rows(changed, self.plan, self.bindings[0], self.config, self.manifest, self.runtime)
        # Missing raw UUIDs remain explicitly unverified, as in saved AMD rows.
        report = production.validate_rows(rows, self.plan, self.bindings[0], self.config, self.manifest, self.runtime)
        self.assertFalse(report["gpu_uuid_observed_in_all_gpu_rows"])


if __name__ == "__main__":
    unittest.main()

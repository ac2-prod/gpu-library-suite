"""Local fake process checks; no SSH, scheduler, GPU, or real artifact launch."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("tested_amd_smoke", ROOT / "jobs/sirius/amd_smoke.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class AmdSmokeControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def packet(self):
        source = self.root / "source"
        run = source / "runs/smoke-test"
        run.mkdir(parents=True)
        frozen = run / "artifact-source"
        frozen.mkdir()
        build = source / "runs/old-build"
        build.mkdir()
        hashes = {}
        for name in smoke.HELPERS:
            file = run / name
            file.write_text("# pinned fake helper\n")
            hashes[name] = smoke.sha(file)
        document = dict(schema_version=1, purpose="diagnostic-smoke", run_id=run.name,
            source_root=str(source), run_root=str(run), snapshot_root=str(frozen),
            build_run_root=str(build), allocated_cpus=24, allocated_gpus=1,
            timeout_seconds=120, process_timeout_seconds=30, helper_sha256=hashes)
        path = run / "packet.json"
        smoke.write_json(path, document)
        env = {"GPU_SUITE_SIRIUS_AMD_SMOKE_APPROVED": "yes", "PBS_JOBID": "42.pbs",
               "PBS_ENVIRONMENT": "PBS_BATCH"}
        return path, document, env

    def validate(self, path, env):
        with mock.patch.object(smoke.sys, "platform", "linux"), \
             mock.patch.object(smoke, "__file__", str(path.parent / "amd_smoke.py")), \
             mock.patch.object(smoke.os, "sched_getaffinity", return_value={0, 1}, create=True):
            return smoke.validate_packet(path, smoke.sha(path), env)

    def test_packet_accepts_explicit_allocation_without_launch(self):
        path, packet, env = self.packet()
        with mock.patch.object(smoke.subprocess, "Popen") as launch:
            self.assertEqual(self.validate(path, env), packet)
            launch.assert_not_called()
        self.assertFalse((path.parent / "smoke").exists())

    def test_blas_packet_requires_explicit_numerical_scope(self):
        path, packet, env = self.packet()
        packet.update(library="rocblas", purpose="numerical-smoke", example_count=3,
                      diagnostic_rows=0, final_rows=96)
        path.write_text(json.dumps(packet))
        with mock.patch.object(smoke.subprocess, "Popen") as launch:
            self.assertEqual(self.validate(path, env), packet)
            for field, value in (("library", "rocfft"), ("purpose", "diagnostic-smoke"),
                                 ("example_count", 18), ("diagnostic_rows", 144), ("final_rows", 288)):
                path.write_text(json.dumps(dict(packet, **{field: value})))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    self.validate(path, env)
            launch.assert_not_called()

    def test_approval_batch_and_affinity_gates(self):
        path, _, env = self.packet()
        for key, value in (("GPU_SUITE_SIRIUS_AMD_SMOKE_APPROVED", "no"),
                           ("PBS_JOBID", ""), ("PBS_ENVIRONMENT", "PBS_INTERACTIVE"),
                           ("OMP_THREAD_LIMIT", "1")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(path, dict(env, **{key: value}))

    def test_inherited_loader_injection_rejected(self):
        path, _, env = self.packet()
        for key in ("LD_PRELOAD", "LD_AUDIT", "LD_DEBUG", "LD_DEBUG_OUTPUT"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(path, dict(env, **{key: "inherited"}))

    def test_bad_packet_hash_never_launches(self):
        path, _, env = self.packet()
        with mock.patch.object(smoke.subprocess, "Popen") as launch, self.assertRaises(ValueError):
            smoke.validate_packet(path, "0" * 64, env)
        launch.assert_not_called()

    def test_changed_helper_and_existing_attempt_rejected(self):
        path, _, env = self.packet()
        (path.parent / "smoke").mkdir()
        with self.assertRaises(ValueError):
            self.validate(path, env)
        (path.parent / "amd_smoke_source.py").write_text("changed")
        with self.assertRaises(ValueError):
            self.validate(path, env)

    def test_duplicate_json_rejected(self):
        path = self.root / "duplicate.json"
        path.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaises(ValueError):
            smoke.read_json(path)

    def test_exclusive_json_preserves_existing_bytes(self):
        path = self.root / "record.json"
        smoke.write_json(path, {"old": 1})
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            smoke.write_json(path, {"new": 1})
        self.assertEqual(before, path.read_bytes())

    def processes(self, exit_code=0, timeout=False):
        child = mock.Mock(pid=43210)
        child.wait.side_effect = [subprocess.TimeoutExpired(["fake"], 1), -15] if timeout else None
        child.wait.return_value = exit_code
        def fake_popen(argv, **kwargs):
            kwargs["stdout"].write(b"numeric-output\n")
            kwargs["stderr"].write(b"private trace\n")
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
            self.assertTrue(kwargs["start_new_session"])
            return child
        launch = mock.Mock(side_effect=fake_popen)
        return smoke.Processes(self.root / "commands", 30, 10, popen=launch), launch, child

    def test_process_streams_hashes_exit_and_pid_recorded(self):
        processes, launch, _ = self.processes()
        result = processes.run(["/fake/artifact"], {"X": "1"}, "example")
        self.assertEqual(result.stdout, "numeric-output\n")
        self.assertEqual(result.stderr, "private trace\n")
        self.assertEqual(result.pid, 43210)
        record = processes.records[0]
        self.assertEqual(record["exit"], 0)
        self.assertEqual(record["stdout_sha256"], hashlib.sha256(b"numeric-output\n").hexdigest())
        self.assertNotEqual(record["stdout_sha256"], record["stderr_sha256"])
        launch.assert_called_once()

    def test_timeout_signals_only_owned_child_and_never_retries(self):
        processes, launch, _ = self.processes(timeout=True)
        with mock.patch.object(smoke.os, "killpg") as kill:
            with self.assertRaises(ValueError):
                processes.run(["/fake/artifact"], {}, "timeout")
            kill.assert_called_once_with(43210, smoke.signal.SIGTERM)
        self.assertTrue(processes.records[0]["timed_out"])
        with self.assertRaises(ValueError):
            processes.run(["/fake/artifact"], {}, "retry")
        launch.assert_called_once()

    def test_nonzero_exit_stops_future_launches(self):
        processes, launch, _ = self.processes(exit_code=7)
        with self.assertRaises(ValueError):
            processes.run(["/fake/artifact"], {}, "failure")
        with self.assertRaises(ValueError):
            processes.run(["/fake/artifact"], {}, "retry")
        self.assertEqual(processes.records[0]["exit"], 7)
        launch.assert_called_once()

    def test_launch_error_is_not_success_or_retry(self):
        for index, error in enumerate((OSError("fake launch failure"),
                                       subprocess.SubprocessError("fake preexec failure"))):
            with self.subTest(error=type(error).__name__):
                launch = mock.Mock(side_effect=error)
                processes = smoke.Processes(self.root / ("commands-" + str(index)), 30, 10, popen=launch)
                with self.assertRaises(ValueError):
                    processes.run(["/fake/artifact"], {}, "failure")
                self.assertIsNone(processes.records[0]["exit"])
                self.assertIsNone(processes.records[0]["pid"])
                self.assertEqual(str(error), processes.records[0]["error"])
                self.assertTrue(processes.records[0]["launch_failed"])
                self.assertTrue(processes.stopped)
                with self.assertRaises(ValueError):
                    processes.run(["/fake/artifact"], {}, "retry")
                launch.assert_called_once()

    def test_smoke_environment_separates_requests_and_stale_provenance(self):
        original = {"PATH": "/fake", "GPU_SUITE_RUNTIME_ENVIRONMENT_SHA256": "old"}
        actual = smoke.smoke_environment(original, self.root, 2)
        self.assertEqual(actual["OMP_NUM_THREADS"], "2")
        self.assertEqual(actual["AOCLSPARSE_NUM_THREADS"], "2")
        self.assertEqual(actual["OMP_TARGET_OFFLOAD"], "MANDATORY")
        self.assertNotIn("GPU_SUITE_RUNTIME_ENVIRONMENT_SHA256", actual)
        self.assertNotIn("LD_DEBUG", actual)
        self.assertEqual(original["GPU_SUITE_RUNTIME_ENVIRONMENT_SHA256"], "old")

    def adapter(self, expected=None):
        binary = self.root / "fake-binary"
        binary.write_bytes(b"fake non-executable bytes")
        entry = {"artifact_id": "artifact", "executable_path": str(binary), "binary_sha256": smoke.sha(binary),
                 "target_name": "fft_cpu_bench", "build_metadata_sha256": "metadata"}
        item = {"argv": [str(binary), "--scope", "compute"], "entry": entry, "scope": "compute"}
        stderr = self.root / "trace.stderr"
        stderr.write_text("trace")
        processes = mock.Mock()
        processes.run.return_value = SimpleNamespace(stdout="row", stderr="trace", returncode=0,
            pid=43210, record={"stderr": str(stderr)})
        evidence = mock.Mock()
        objects = [{"path": "/fake/library.so", "sha256": "1" * 64}]
        evidence.validate_loaded_trace.return_value = {"initialized_objects": objects,
            "process": {"application_stderr": "application message"}}
        links = {"artifact": {"mode": "native-hip-link-command", "fixture": "verified source proof"}}
        adapter = smoke.TracedRunner(processes, evidence, {"metadata": {}}, [item], 2, links, expected)
        return adapter, processes, item, objects

    def test_trace_adapter_preserves_stdout_and_records_actual_pid(self):
        adapter, processes, item, objects = self.adapter()
        result = adapter.run(item["argv"], text=True, capture_output=True, check=False, env={})
        self.assertEqual(result.stdout, "row")
        self.assertEqual(result.stderr, "application message")
        self.assertEqual(adapter.observed, {"artifact:2:compute": objects})
        self.assertEqual(processes.run.call_args.args[1]["LD_DEBUG"], "libs")
        self.assertEqual(adapter.evidence.validate_loaded_trace.call_args.kwargs["expected_pid"], 43210)
        self.assertEqual(adapter.evidence.validate_loaded_trace.call_args.kwargs["link_dependencies"],
                         adapter.link_evidence["artifact"])

    def test_missing_source_link_descriptor_prevents_artifact_launch(self):
        adapter, processes, item, _ = self.adapter()
        adapter.link_evidence = {}
        with self.assertRaisesRegex(ValueError, "Missing verified artifact link evidence"):
            adapter.run(item["argv"], text=True, capture_output=True, check=False, env={})
        processes.run.assert_not_called()

    def link_proof(self):
        entries = [{"artifact_id": "artifact-" + str(index), "build_metadata_sha256": "metadata"}
                   for index in range(36)]
        return {"entries": entries, "metadata_by_hash": {"metadata": {"verified": True}},
                "link_evidence": {entry["artifact_id"]: "verified-depfile-" + entry["artifact_id"]
                                  for entry in entries}}

    def test_all_36_link_evidence_preflight_without_process(self):
        proof, evidence = self.link_proof(), mock.Mock()
        evidence.validate_link_evidence.side_effect = lambda entry, metadata, descriptor: descriptor
        with mock.patch.object(smoke.subprocess, "Popen") as launch:
            result = smoke.preflight_link_evidence(proof, evidence)
        self.assertEqual(result, proof["link_evidence"])
        self.assertEqual(evidence.validate_link_evidence.call_count, 36)
        launch.assert_not_called()

    def test_all_36_real_evidence_parsers_allow_historical_objects_but_reject_missing_libraries(self):
        # Exercise the production parser at the controller boundary, not a mock
        # that automatically accepts each descriptor. Only file bytes are fake.
        import test_sirius_amd_smoke_evidence as fixtures
        evidence = fixtures.evidence
        proof = {"entries": [], "metadata_by_hash": {}, "link_evidence": {}}
        objects = {}
        cpu = {"rocblas": ("generic_cblas", ["cpu-generic-cblas"]),
               "rocsolver": ("generic_lapacke", ["cpu-generic-lapacke"]),
               "rocsparse": ("aocl-sparse", ["cpu-aocl-sparse"]),
               "rocrand": ("std-random-serial", ["cpu-std-random-serial"]),
               "rocthrust": ("stl-serial", ["cpu-stl-serial"])}
        for profile in ("cpu", "hip", "openmp"):
            for family in sorted(evidence.FAMILIES):
                for role in ("example", "benchmark"):
                    key = "-".join((profile, family, role))
                    fixture = fixtures.SiriusAmdSmokeEvidenceTests()
                    fixture.base = self.root / key
                    fixture.base.mkdir()
                    fixture.program = fixture.file("program")
                    fixture.loader = fixture.file("loader.so")
                    fixture.depfile = fixture.base / "link.d"
                    if profile == "hip":
                        entry, metadata, descriptor, _ = fixture.native_hip_inputs(family, role)
                    else:
                        if profile == "cpu":
                            metadata = fixture.cpu_metadata()
                            if family == "rocfft":
                                threaded = role == "benchmark"
                                variant = "fftw3f-serial+threaded" if threaded else "fftw3f-serial"
                                backends = ["cpu-fftw-serial"] + (["cpu-fftw-threaded"] if threaded else [])
                            else:
                                variant, backends = cpu[family]
                        else:
                            metadata = fixture.gpu_metadata(profile)
                            variant, backends = profile + "-" + family, []
                        entry = fixture.entry(family, profile, role, variant, backends)
                        selected = evidence.select_artifact_dependencies(entry, metadata)
                        paths = [value["path"] for value in selected.values() if value["kind"] != "header"]
                        paths.append("CMakeFiles/program.dir/input.cpp.o")
                        if profile == "openmp":
                            missing = fixture.base / "tmp/program.openmp.image.wrapper-5a1183.o"
                            self.assertFalse(missing.exists())
                            objects[key] = str(missing)
                            paths.insert(0, str(missing))
                        descriptor = fixture.links(paths)
                    entry.update(artifact_id=key, build_metadata_sha256=key)
                    proof["entries"].append(entry)
                    proof["metadata_by_hash"][key] = metadata
                    proof["link_evidence"][key] = descriptor
        with mock.patch.object(smoke.subprocess, "Popen") as launch:
            actual = smoke.preflight_link_evidence(proof, evidence)
            self.assertEqual(set(actual), set(proof["link_evidence"]))
            for key, missing in objects.items():
                self.assertIn(missing, json.dumps(actual[key]))
            # The last artifact's missing persistent input still rejects the
            # whole preflight; passing earlier entries never launches a child.
            last = proof["entries"][-1]
            depfile = proof["link_evidence"][last["artifact_id"]]
            depfile.write_text(depfile.read_text().rstrip() + " /missing/required-library.so\n")
            with self.assertRaises((ValueError, OSError)):
                smoke.preflight_link_evidence(proof, evidence)
            launch.assert_not_called()

    def test_incomplete_duplicate_or_extra_link_inventory_rejected(self):
        for mutation in ("missing", "extra", "duplicate"):
            proof, evidence = self.link_proof(), mock.Mock()
            if mutation == "missing":
                proof["link_evidence"].pop("artifact-35")
            elif mutation == "extra":
                proof["link_evidence"]["unapproved"] = "another-file"
            else:
                proof["entries"][-1] = proof["entries"][0]
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "inventory"):
                smoke.preflight_link_evidence(proof, evidence)
            evidence.validate_link_evidence.assert_not_called()

    def test_last_artifact_evidence_failure_prevents_any_child(self):
        proof, evidence, processes = self.link_proof(), mock.Mock(), mock.Mock()
        evidence.validate_link_evidence.side_effect = [None] * 35 + [ValueError("missing HIP link command")]
        packet = {"run_root": str(self.root), "artifact_source_sha256": "a" * 64,
            "build_summary_sha256": "b" * 64, "build_evidence_sha256": {},
            "build_run_root": str(self.root / "old"), "source_root": str(self.root),
            "snapshot_root": str(self.root / "frozen"), "helper_sha256": {"amd_smoke.py": "c" * 64}}
        source = mock.Mock()
        source.verify_prebuilt_source.return_value = proof
        with mock.patch.object(smoke, "load_helper", side_effect=[source, evidence, mock.Mock()]), \
             mock.patch.object(smoke, "load_pinned_runner"), \
             mock.patch("gpu_suite.manifest.merge_entries", return_value=proof["entries"]), \
             self.assertRaisesRegex(ValueError, "missing HIP link command"):
            smoke.execute(packet, self.root, processes)
        self.assertEqual(evidence.validate_link_evidence.call_count, 36)
        processes.run.assert_not_called()

    def test_trace_identity_mismatch_rejected(self):
        adapter, _, item, _ = self.adapter({"artifact:2:compute": []})
        with self.assertRaisesRegex(ValueError, "identity differs"):
            adapter.run(item["argv"], text=True, capture_output=True, check=False, env={})

    def test_wrong_command_and_changed_binary_never_launched(self):
        adapter, processes, item, _ = self.adapter()
        with self.assertRaises(ValueError):
            adapter.run(["/different"], text=True, capture_output=True, check=False, env={})
        Path(item["argv"][0]).write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            adapter.run(item["argv"], text=True, capture_output=True, check=False, env={})
        processes.run.assert_not_called()

    def test_wrong_module_origin_rejected(self):
        modules = {"gpu_suite.runner": SimpleNamespace(__file__=str(self.root / "wrong/runner.py"))}
        with mock.patch.object(smoke.sys, "modules", modules), self.assertRaises(ValueError):
            smoke.check_module_origins(self.root / "frozen")

    def test_source_failure_prevents_runner_import_and_process(self):
        source = mock.Mock()
        source.verify_prebuilt_source.side_effect = ValueError("source mismatch")
        packet = {"run_root": str(self.root), "artifact_source_sha256": "a" * 64,
            "build_summary_sha256": "b" * 64, "build_evidence_sha256": {},
            "build_run_root": str(self.root / "old"), "source_root": str(self.root),
            "snapshot_root": str(self.root / "frozen"), "helper_sha256": {"amd_smoke.py": "c" * 64}}
        processes = mock.Mock()
        with mock.patch.object(smoke, "load_helper", return_value=source), \
             mock.patch.object(smoke, "load_pinned_runner") as load, self.assertRaises(ValueError):
            smoke.execute(packet, self.root, processes)
        load.assert_not_called()
        processes.run.assert_not_called()


class AmdSmokeTraceIntegrationTests(unittest.TestCase):
    """Use real trace validation and owned-process records; only spawn is fake."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="amd-smoke-trace-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def traced_child(self, *, exit_code=0, timeout=False, omit_provider=False):
        # All bytes and paths are inert local fixtures. The lifecycle mirrors
        # glibc: loader/libc initialization can lack matching fini messages.
        # An arbitrarily named extra object prevents a libc/loader whitelist.
        import test_sirius_amd_smoke_evidence as fixtures
        evidence = fixtures.evidence
        files = {}
        for name in ("program", "libblas.so", "libc.so", "loader.so", "provider-neutral.so"):
            path = self.root / name
            path.write_bytes(b"\x7fELFfixture:" + name.encode("ascii"))
            files[name] = path
        program, provider = files["program"], files["libblas.so"]
        entry = {"artifact_id": "cpu-blas", "library": "rocblas", "implementation": "cpu",
                 "build_profile": "cpu", "executable_role": "benchmark",
                 "backend_variant": "generic_cblas", "supported_cpu_backends": ["cpu-generic-cblas"],
                 "executable_path": str(program), "binary_sha256": smoke.sha(program),
                 "target_name": "blas_cpu_bench", "build_metadata_sha256": "metadata"}
        metadata = {"vendor": "amd", "build_profile": "cpu",
                    "dependencies": {"cblas": {"path": str(provider), "sha256": smoke.sha(provider)}},
                    "amd_build_inputs": {"GPU_SUITE_CPU_BLAS_BACKEND": "GENERIC_CBLAS",
                        "CBLAS_INTERFACE_LINK_LIBRARIES": str(provider) + ";Threads::Threads;dl;m"}}
        depfile = self.root / "link.d"
        depfile.write_text("program: " + str(provider) + "\n")
        pending = [files["loader.so"], files["libc.so"], files["provider-neutral.so"]]
        observed = pending + ([] if omit_provider else [provider])
        messages = ["find library=libblas.so [0]; searching",
                    " search cache=/etc/ld.so.cache", " trying file=" + str(provider)]
        messages += ["calling init: " + str(path) for path in observed]
        messages += ["initialize program: " + str(program), "transferring control: " + str(program),
                     "calling fini:  [0]"]
        if not omit_provider:
            messages.append("calling fini: " + str(provider) + " [0]")
        trace = "".join("  43210:\t" + message + "\n" for message in messages)
        trace += "application diagnostic\n"
        child = mock.Mock(pid=43210)
        child.wait.return_value = exit_code
        if timeout:
            child.wait.side_effect = [subprocess.TimeoutExpired([str(program)], 1), -15]

        def fake_popen(argv, **kwargs):
            self.assertEqual(argv, [str(program), "--scope", "compute"])
            self.assertEqual(kwargs["env"]["LD_DEBUG"], "libs")
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
            self.assertTrue(kwargs["start_new_session"])
            kwargs["stdout"].write(b"numerical-output\n")
            kwargs["stderr"].write(trace.encode("utf-8"))
            return child

        launch = mock.Mock(side_effect=fake_popen)
        processes = smoke.Processes(self.root / "commands", 30, 10, popen=launch)
        item = {"argv": [str(program), "--scope", "compute"], "entry": entry, "scope": "compute"}
        adapter = smoke.TracedRunner(processes, evidence, {"metadata": metadata}, [item], 1,
                                     {entry["artifact_id"]: depfile})
        return adapter, processes, item, launch, trace, pending

    def test_real_trace_parser_accepts_unfinalized_objects_after_owned_child_success(self):
        adapter, processes, item, launch, trace, pending = self.traced_child()
        result = adapter.run(item["argv"], text=True, capture_output=True, check=False, env={})
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "numerical-output\n")
        self.assertEqual(result.stderr, "application diagnostic")
        record = processes.records[0]
        self.assertEqual(record["exit"], 0)
        self.assertFalse(record["timed_out"])
        self.assertEqual(record["pid"], 43210)
        self.assertEqual(Path(record["stderr"]).read_text(), trace)
        proof = smoke.read_json(Path(record["stderr"] + ".evidence.json"))
        self.assertEqual(proof["process"]["initializations_without_observed_finalization"],
                         sorted(map(str, pending)))
        self.assertTrue(proof["process"]["main_finalization_observed"])
        self.assertEqual(proof["process"]["trace_sha256"], record["stderr_sha256"])
        self.assertEqual(adapter.observed["cpu-blas:1:compute"], proof["initialized_objects"])
        self.assertEqual(set(proof["dependencies"]), {"cblas"})
        self.assertEqual(len(proof["initialized_objects"]), 4)
        launch.assert_called_once()

    def test_real_trace_parser_not_reached_after_nonzero_child(self):
        adapter, processes, item, launch, _, _ = self.traced_child(exit_code=7)
        with mock.patch.object(adapter.evidence, "validate_loaded_trace",
                               wraps=adapter.evidence.validate_loaded_trace) as parse:
            with self.assertRaisesRegex(ValueError, "Child failed/unknown"):
                adapter.run(item["argv"], text=True, capture_output=True, check=False, env={})
            with self.assertRaises(ValueError):
                adapter.run(item["argv"], text=True, capture_output=True, check=False, env={})
            parse.assert_not_called()
        self.assertEqual(processes.records[0]["exit"], 7)
        self.assertTrue(processes.stopped)
        self.assertEqual(adapter.observed, {})
        self.assertFalse(list(processes.root.glob("*.evidence.json")))
        launch.assert_called_once()

    def test_real_trace_parser_not_reached_after_owned_child_timeout(self):
        adapter, processes, item, launch, _, _ = self.traced_child(timeout=True)
        with mock.patch.object(smoke.os, "killpg") as kill, \
             mock.patch.object(adapter.evidence, "validate_loaded_trace",
                               wraps=adapter.evidence.validate_loaded_trace) as parse:
            with self.assertRaisesRegex(ValueError, "Child failed/unknown"):
                adapter.run(item["argv"], text=True, capture_output=True, check=False, env={})
            with self.assertRaises(ValueError):
                adapter.run(item["argv"], text=True, capture_output=True, check=False, env={})
            parse.assert_not_called()
            kill.assert_called_once_with(43210, smoke.signal.SIGTERM)
        self.assertEqual(processes.records[0]["exit"], -15)
        self.assertTrue(processes.records[0]["timed_out"])
        self.assertTrue(processes.stopped)
        self.assertEqual(adapter.observed, {})
        self.assertFalse(list(processes.root.glob("*.evidence.json")))
        launch.assert_called_once()

    def test_real_trace_parser_rejects_missing_required_provider_after_success(self):
        adapter, processes, item, launch, _, _ = self.traced_child(omit_provider=True)
        with self.assertRaisesRegex(ValueError, "required ELF path/hash not observed"):
            adapter.run(item["argv"], text=True, capture_output=True, check=False, env={})
        self.assertEqual(processes.records[0]["exit"], 0)
        self.assertFalse(processes.records[0]["timed_out"])
        self.assertEqual(adapter.observed, {})
        self.assertFalse(list(processes.root.glob("*.evidence.json")))
        launch.assert_called_once()


# Integration tests retain the canonical runner and replace only process effects.
import copy
import io
import signal

sys.path.insert(0, str(ROOT / "tools"))
from gpu_suite import runner
from gpu_suite.manifest import artifact_id, merge_entries
from gpu_suite.results_io import ExclusiveRawWriter, load_raw_results
from gpu_suite.strict_json import dump_bytes, load
from test_sirius_amd_smoke_campaign import SOURCE_HASH, campaign, fixture

CONTROLLER_SPEC = importlib.util.spec_from_file_location("sirius_amd_smoke_controller", ROOT / "jobs/sirius/amd_smoke.py")
controller = importlib.util.module_from_spec(CONTROLLER_SPEC)
CONTROLLER_SPEC.loader.exec_module(controller)


def artifact_fixture(root, config, label):
    """Supply real file/hash metadata without compiling or executing anything."""
    rows, manifest, arguments = fixture(config, label)
    dependencies = {}
    for name in ("fftw3f", "fftw3f_omp", "fftw_openmp_runtime_0", "cblas", "lapacke",
                 "aocl_sparse", "aocl_sparse_header", "hip", "rocfft", "rocblas",
                 "rocsparse", "rocsolver", "rocrand", "rocthrust", "rocprim"):
        path = root / ("aoclsparse.h" if name == "aocl_sparse_header" else name + ".fixture")
        path.write_bytes(("inert dependency: " + name).encode())
        dependencies[name] = {"path": str(path), "sha256": controller.sha(path)}
    metadata_by_hash, profile_hashes = {}, {}
    exemplar = manifest["entries"][0]
    for profile in ("cpu", "hip", "openmp"):
        metadata = {key: exemplar[key] for key in
                    ("git_commit", "git_dirty", "git_metadata_available", "build_type")}
        metadata.update(vendor="amd", build_profile=profile, dependencies=dependencies,
                        fftw_threading={"requested": "OPENMP", "backend": "openmp",
                            "compile_options": [], "link_options": [],
                            "dependency_names": ["fftw3f", "fftw3f_omp", "fftw_openmp_runtime_0"]},
                        amd_build_inputs={"GPU_SUITE_CPU_SPARSE_BACKEND": "AOCLSPARSE",
                            "AOCLSparse_INTERFACE_INCLUDE_DIRECTORIES": str(root),
                            "AOCLSparse_INTERFACE_LINK_LIBRARIES": dependencies["aocl_sparse"]["path"]})
        for language in ("c", "cxx", "hip"):
            metadata[language] = {key: exemplar[key] for key in
                                  ("compiler", "compiler_version", "global_configure_flags")}
        path = root / (profile + "-metadata.json")
        path.write_bytes(dump_bytes(metadata))
        digest = controller.sha(path)
        metadata_by_hash[digest] = metadata
        profile_hashes[profile] = digest
    for entry in manifest["entries"]:
        path = root / entry["target_name"]
        path.write_bytes(("not an executable: " + entry["target_name"]).encode())
        entry.update(executable_path=str(path), binary_sha256=controller.sha(path),
                     build_metadata_sha256=profile_hashes[entry["build_profile"]])
        entry["artifact_id"] = artifact_id(entry)
    manifest["entries"] = merge_entries([manifest["entries"]])
    manifest["build_metadata_sha256s"] = sorted(metadata_by_hash)
    entries = {(entry["library"], entry["implementation"]): entry for entry in manifest["entries"]}
    for row in rows:
        entry = entries[row["benchmark"], row["implementation"]]
        row["binary_sha256"] = entry["binary_sha256"]
        row["parameters"]["runtime"]["build_metadata_sha256"] = entry["build_metadata_sha256"]
    proof = {"entries": manifest["entries"], "metadata_by_hash": metadata_by_hash,
             "artifact_source": {"source_snapshot_sha256": SOURCE_HASH},
             "link_evidence": {entry["artifact_id"]: "inert verified evidence descriptor"
                               for entry in manifest["entries"]}}
    return rows, manifest, arguments, proof


class FixtureProcesses:
    """Emit fixture records in the requested format, with captured launch hashes."""

    def __init__(self, root, config, rows, manifest, mutate=None, *, trace=True):
        self.root, self.config, self.rows = root, config, rows
        root.mkdir()
        self.entries = {(entry["library"], entry["implementation"]): entry for entry in manifest["entries"]}
        self.calls, self.mutate = [], mutate
        self.trace = trace

    def run(self, argv, environment, label):
        index = len(self.calls)
        rows = copy.deepcopy(self.rows[2 * index:2 * index + 2])
        if len(rows) != 2:
            raise AssertionError("Unexpected extra fixture process")
        entry = self.entries[rows[0]["benchmark"], rows[0]["implementation"]]
        if argv[0] != entry["executable_path"] or environment.get("LD_DEBUG") != ("libs" if self.trace else None):
            raise AssertionError("Unexpected invocation or missing child-only loader trace")
        for row in rows:
            row["config_sha256"] = environment["GPU_SUITE_CONFIG_SHA256"]
            row["runtime_environment_sha256"] = environment["GPU_SUITE_RUNTIME_ENVIRONMENT_SHA256"]
            row["source_snapshot_sha256"] = environment["GPU_SUITE_SOURCE_SNAPSHOT_SHA256"]
        if self.mutate:
            self.mutate(rows, index)
        stdout = self.root / (str(index) + "." + self.config["output_format"])
        stderr = self.root / (str(index) + ".stderr")
        with ExclusiveRawWriter(stdout, self.config["output_format"]) as writer:
            for row in rows:
                writer.write(row)
        stderr.write_text("inert loader diagnostic\n")
        self.calls.append({"argv": list(argv), "environment": dict(environment), "label": label})
        return SimpleNamespace(returncode=0, pid=1000 + index,
                               stdout=stdout.read_text(), stderr=stderr.read_text(),
                               record={"stdout": str(stdout), "stderr": str(stderr)})


class SiriusAmdSmokeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="amd-smoke-controller-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name).resolve()
        self.configs = campaign.campaign_configs(ROOT / "jobs/sirius/amd-smoke.json",
                                                 ROOT / "jobs/sirius/amd-threaded-smoke.json")
        for name in ("subprocess.run", "subprocess.Popen"):
            patcher = mock.patch(name, side_effect=AssertionError("No actual process is allowed"))
            patcher.start()
            self.addCleanup(patcher.stop)

    def data(self, label):
        artifacts = self.work / (label + "-artifacts")
        artifacts.mkdir()
        return artifact_fixture(artifacts, self.configs[label], label)

    def test_blas_campaigns_run_96_rows_without_loader_trace_or_diagnostic_runs(self):
        self.configs = campaign.campaign_configs(ROOT / "jobs/sirius/amd-smoke.json",
            ROOT / "jobs/sirius/amd-threaded-smoke.json", library="rocblas")
        reports = {}
        for label, config in self.configs.items():
            rows, manifest, arguments, proof = self.data(label)
            processes = FixtureProcesses(self.work / (label + "-processes"), config, rows, manifest, trace=False)
            evidence = mock.Mock()
            root = self.work / label
            with mock.patch.object(controller.socket, "gethostname", return_value="node0"):
                report, observed = controller.execute_campaign(runner, campaign, evidence, processes,
                    proof, manifest, config, root, arguments["context"]["run_id"], "42.fixture",
                    controller.smoke_environment({}, self.work, config["cpu_threads"]),
                    {"loaded_libraries_verified": False}, None, library="rocblas")
            self.assertEqual(len(processes.calls), 12)
            self.assertTrue(all("LD_DEBUG" not in call["environment"] for call in processes.calls))
            self.assertTrue(all("rocblas" in call["argv"][0] for call in processes.calls))
            self.assertEqual(evidence.mock_calls, [])
            self.assertEqual(observed, {})
            self.assertFalse((root / "initialized-objects.json").exists())
            self.assertFalse(load(root / "runtime-environment.json")["loaded_libraries_verified"])
            self.assertTrue(all(path.read_text() == "inert loader diagnostic\n"
                                for path in processes.root.glob("*.stderr")))
            reports[label] = report
        self.assertEqual(sum(report["row_count"] for report in reports.values()), 96)

    def test_blas_numerical_failure_stops_campaign(self):
        self.configs = campaign.campaign_configs(ROOT / "jobs/sirius/amd-smoke.json",
            ROOT / "jobs/sirius/amd-threaded-smoke.json", library="rocblas")
        config = self.configs["serial-jsonl"]
        rows, manifest, arguments, proof = self.data("serial-jsonl")
        def corrupt(rows, index):
            rows[0]["verification_metrics"]["max_abs_error"] = 1.0
            rows[0]["verification_status"] = "failure"
            rows[0].update(status="failure", exit_code=1, failure_origin="verification")
        processes = FixtureProcesses(self.work / "processes", config, rows, manifest, corrupt, trace=False)
        with mock.patch.object(controller.socket, "gethostname", return_value="node0"), self.assertRaises(ValueError):
            controller.execute_campaign(runner, campaign, mock.Mock(), processes, proof, manifest,
                config, self.work / "failed", arguments["context"]["run_id"], "42.fixture",
                controller.smoke_environment({}, self.work, 1), {}, None, library="rocblas")
        self.assertEqual(len(processes.calls), 1)
        self.assertFalse((self.work / "failed/validation.json").exists())

    def test_json_bytes_and_hash_match_canonical_unicode_and_negative_zero(self):
        document = {"日本語": {"é": -0.0, "零": [0.0, -0.0]}, "tuple": (True, None, 1.5)}
        path = self.work / "canonical.json"
        digest = controller.write_json(path, document)
        self.assertEqual(path.read_bytes(), dump_bytes(document))
        self.assertEqual(digest, hashlib.sha256(dump_bytes(document)).hexdigest())
        self.assertEqual(controller.read_json(path), load(path))
        self.assertNotIn(b"-0.0", path.read_bytes())
        with self.assertRaises(FileExistsError):
            controller.write_json(path, {"changed": True})
        self.assertEqual(path.read_bytes(), dump_bytes(document))

    def test_json_rejects_invalid_values_before_creating_output(self):
        for index, value in enumerate((float("nan"), float("inf"), -float("inf"), {1: "key"}, {"set": {1}})):
            with self.subTest(value=value):
                path = self.work / (str(index) + ".json")
                with self.assertRaises(ValueError):
                    controller.write_json(path, value)
                self.assertFalse(path.exists())
        path = self.work / "invalid-input.json"
        for content in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e9999}', b'"\xff"'):
            path.write_bytes(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                controller.read_json(path)

    def test_execute_campaign_four_formats_match_saved_hashes_and_288_rows(self):
        reports = {}
        original_environment, original_subprocess = dict(os.environ), runner.subprocess
        for label, config in self.configs.items():
            with self.subTest(campaign=label):
                rows, manifest, arguments, proof = self.data(label)
                # Validate actual artifact/dependency bytes; only child execution is fake.
                runner.validate_manifest_artifacts(manifest, proof["metadata_by_hash"], "smoke", True)
                manifest_path = self.work / (label + "-executables.json")
                controller.write_json(manifest_path, manifest)
                processes = FixtureProcesses(self.work / (label + "-processes"), config, rows, manifest)
                evidence = SimpleNamespace(validate_loaded_trace=mock.Mock(return_value={
                    "initialized_objects": ["inert loader object"], "process": {"application_stderr": ""}}))
                environment = controller.smoke_environment({"PATH": "/fixture", "PBS_JOBID": "42.fixture"},
                                                           self.work, config["cpu_threads"])
                schedule = runner.build_schedule(config, manifest, arguments["context"])
                expected = {item["entry"]["artifact_id"] + ":" + str(config["cpu_threads"]) + ":" + item["scope"]:
                            ["inert loader object"] for item in schedule}
                root = self.work / label
                with mock.patch.object(controller.socket, "gethostname", return_value="node0"):
                    report, observed = controller.execute_campaign(
                        runner, campaign, evidence, processes, proof, manifest, config, root,
                        arguments["context"]["run_id"], "42.fixture", environment,
                        {"comment": "日本語 runtime", "zero": -0.0}, expected)
                self.assertEqual(report["status"], "PASS")
                self.assertEqual(report["row_count"], 72)
                self.assertEqual(report["config_sha256"], controller.sha(root / "config.json"))
                self.assertEqual(report["runtime_environment_sha256"], controller.sha(root / "runtime-environment.json"))
                self.assertEqual(report["manifest_sha256"], controller.sha(manifest_path))
                self.assertEqual(load(root / "validation.json"), report)
                self.assertEqual(load(root / "initialized-objects.json"), observed)
                self.assertEqual(observed, expected)
                self.assertEqual(len(processes.calls), 36)
                self.assertEqual(evidence.validate_loaded_trace.call_count, 36)
                saved_rows = load_raw_results(root / ("raw." + config["output_format"]))
                self.assertEqual(len(saved_rows), 72)
                self.assertTrue(all(row["status"] == "success" and row["verification_status"] == "pass"
                                    for row in saved_rows))
                self.assertEqual({row["config_sha256"] for row in saved_rows}, {report["config_sha256"]})
                self.assertEqual({row["runtime_environment_sha256"] for row in saved_rows},
                                 {report["runtime_environment_sha256"]})
                self.assertEqual(os.environ, original_environment)
                self.assertIs(runner.subprocess, original_subprocess)
                reports[label] = report
        # Binary identities legitimately differ between these independent inert fixtures.
        self.assertEqual(sum(report["row_count"] for report in reports.values()), 288)

    def test_campaign_rejects_wrong_child_provenance_and_restores_environment(self):
        label = "serial-jsonl"
        config = self.configs[label]
        rows, manifest, arguments, proof = self.data(label)
        def corrupt(rows, index):
            if index == 0:
                rows[0]["config_sha256"] = "0" * 64
        processes = FixtureProcesses(self.work / "processes", config, rows, manifest, corrupt)
        evidence = SimpleNamespace(validate_loaded_trace=mock.Mock(return_value={
            "initialized_objects": [], "process": {"application_stderr": ""}}))
        previous_environment, previous_subprocess = dict(os.environ), runner.subprocess
        environment = controller.smoke_environment({"PATH": "/fixture"}, self.work, 1)
        with mock.patch.object(controller.socket, "gethostname", return_value="node0"), \
                mock.patch.object(sys, "stderr", new_callable=io.StringIO), \
                self.assertRaisesRegex(ValueError, "Campaign did not complete"):
            controller.execute_campaign(runner, campaign, evidence, processes, proof, manifest,
                config, self.work / "rejected", arguments["context"]["run_id"], "42.fixture",
                environment, {}, None)
        self.assertEqual(os.environ, previous_environment)
        self.assertIs(runner.subprocess, previous_subprocess)
        self.assertFalse((self.work / "rejected/validation.json").exists())
        self.assertTrue(any(row["status"] != "success" for row in load_raw_results(self.work / "rejected/raw.jsonl")))

    def test_preloaded_suite_or_rocfft_helper_is_rejected_before_import(self):
        original_path = list(sys.path)
        for name in ("gpu_suite", "gpu_suite.runner", "rocfft_smoke_evidence"):
            with self.subTest(module=name), mock.patch.object(controller.sys, "modules", {name: SimpleNamespace()}), \
                    self.assertRaisesRegex(ValueError, "before frozen-source verification"):
                controller.load_pinned_runner(self.work / "unverified-snapshot")
        self.assertEqual(sys.path, original_path)

    def test_loaded_module_origins_reject_mixed_current_source(self):
        frozen = self.work / "frozen"
        for name, path in (("gpu_suite.runner", ROOT / "tools/gpu_suite/runner.py"),
                           ("rocfft_smoke_evidence", ROOT / "jobs/sirius/rocfft_smoke_evidence.py")):
            with self.subTest(module=name), \
                    mock.patch.object(controller.sys, "modules", {name: SimpleNamespace(__file__=str(path))}), \
                    self.assertRaisesRegex(ValueError, "Mixed"):
                controller.check_module_origins(frozen)

    def setup_packet(self):
        packet = {"run_root": str(self.work / "packet"), "snapshot_root": str(self.work / "frozen"),
                  "build_run_root": str(self.work / "build-run"), "source_root": str(self.work / "source"),
                  "run_id": "fixture-smoke", "artifact_source_sha256": SOURCE_HASH,
                  "build_summary_sha256": "d" * 64, "build_evidence_sha256": "e" * 64,
                  "helper_sha256": {"amd_smoke.py": "f" * 64}}
        root = self.work / "execution"
        root.mkdir()
        return packet, root

    def test_source_proof_failure_stops_before_runner_import_or_process(self):
        packet, root = self.setup_packet()
        source = SimpleNamespace(verify_prebuilt_source=mock.Mock(side_effect=ValueError("Source proof failed")))
        processes = SimpleNamespace(run=mock.Mock(side_effect=AssertionError("No process before source proof")))
        with mock.patch.object(controller, "load_helper", return_value=source), \
                mock.patch.object(controller, "load_pinned_runner") as importer, \
                self.assertRaisesRegex(ValueError, "Source proof failed"):
            controller.execute(packet, root, processes)
        importer.assert_not_called()
        processes.run.assert_not_called()
        self.assertEqual(list(root.iterdir()), [])

    def test_invalid_pinned_config_stops_before_any_artifact_process(self):
        packet, root = self.setup_packet()
        _, _, _, proof = self.data("serial-jsonl")
        directory = Path(packet["snapshot_root"]) / "jobs/sirius"
        directory.mkdir(parents=True)
        invalid = copy.deepcopy(self.configs["serial-jsonl"])
        invalid["continue_on_failure"] = False
        (directory / "amd-smoke.json").write_bytes(dump_bytes(invalid))
        (directory / "amd-threaded-smoke.json").write_bytes(dump_bytes(self.configs["threaded-jsonl"]))
        source = SimpleNamespace(verify_prebuilt_source=mock.Mock(return_value=proof))
        evidence = SimpleNamespace()
        helpers = {"smoke_source": source, "smoke_campaign": campaign, "smoke_evidence": evidence}
        processes = SimpleNamespace(run=mock.Mock(side_effect=AssertionError("No process before config proof")))
        with mock.patch.object(controller, "load_helper", side_effect=lambda name, path: helpers[name]), \
                mock.patch.object(controller, "load_pinned_runner", return_value=runner), \
                mock.patch.object(controller, "preflight_link_evidence", return_value={}), \
                mock.patch.object(controller, "check_module_origins") as origins, \
                self.assertRaisesRegex(ValueError, "diagnostic continuation"):
            controller.execute(packet, root, processes)
        source.verify_prebuilt_source.assert_called_once()
        processes.run.assert_not_called()
        origins.assert_not_called()
        self.assertFalse((root / "resolution.json").exists())

    def test_stop_requested_terminates_only_own_child_and_forbids_retry(self):
        for interrupt_cleanup in (False, True):
            with self.subTest(interrupt_cleanup=interrupt_cleanup):
                child = mock.Mock(pid=43210)
                child.wait.side_effect = [controller.StopRequested("fixture signal"),
                                          subprocess.TimeoutExpired(["inert"], 5), -signal.SIGKILL]
                popen = mock.Mock(return_value=child)
                directory = self.work / ("commands-" + str(interrupt_cleanup))
                processes = controller.Processes(directory, 30, 10, popen=popen)
                saved_mask, restored = {signal.SIGUSR1}, []
                def fake_mask(action, values):
                    if action == signal.SIG_SETMASK:
                        restored.append(values)
                        if interrupt_cleanup and len(restored) == 2:
                            raise controller.StopRequested("pending cleanup signal")
                    return saved_mask
                with mock.patch.object(controller.os, "killpg") as killpg, \
                        mock.patch.object(controller.signal, "pthread_sigmask", side_effect=fake_mask) as mask:
                    with self.assertRaisesRegex(ValueError, "no automatic retry"):
                        processes.run(["/inert/program"], {"PATH": "/fixture"}, "fixture")
                    self.assertEqual(killpg.call_args_list,
                                     [mock.call(child.pid, signal.SIGTERM), mock.call(child.pid, signal.SIGKILL)])
                    with self.assertRaisesRegex(ValueError, "No process after failure"):
                        processes.run(["/inert/program"], {}, "retry")
                    self.assertEqual(mask.call_args_list,
                                     [mock.call(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT}),
                                      mock.call(signal.SIG_SETMASK, saved_mask)] * 2)
                    self.assertIs(child._smoke_reaped, True)
                    self.assertEqual(processes.stop_child(child), -signal.SIGKILL)
                    self.assertEqual(processes.stop_child(mock.Mock(returncode=0)), 0)
                    self.assertEqual(killpg.call_count, 2)
                    self.assertEqual(child.wait.call_count, 3)
                    self.assertEqual(mask.call_count, 4)
                    restore_child_mask = popen.call_args.kwargs["preexec_fn"]
                    self.assertTrue(callable(restore_child_mask))
                    restore_child_mask()
                    mask.assert_called_with(signal.SIG_SETMASK, saved_mask)
                popen.assert_called_once()
                self.assertTrue(popen.call_args.kwargs["start_new_session"])
                self.assertEqual(popen.call_args.kwargs["stdin"], subprocess.DEVNULL)
                self.assertTrue(processes.stopped)
                self.assertEqual(len(processes.records), 1)
                record = processes.records[0]
                self.assertTrue(record["timed_out"])
                self.assertEqual(record["stop_reason"], "pending cleanup signal" if interrupt_cleanup else "fixture signal")
                self.assertEqual(record["exit"], -signal.SIGKILL)
                self.assertEqual(load(directory / "001-fixture-result.json"), record)
                self.assertEqual(record["stdout_sha256"], controller.sha(record["stdout"]))
                self.assertEqual(record["stderr_sha256"], controller.sha(record["stderr"]))

    def test_stop_at_spawn_unmask_records_owned_child_and_main_reports_unknown(self):
        child = mock.Mock(pid=43211)
        child.wait.return_value = -signal.SIGTERM
        popen = mock.Mock(return_value=child)
        processes = controller.Processes(self.work / "spawn-commands", 30, 10, popen=popen)
        with mock.patch.object(controller.signal, "pthread_sigmask",
                               side_effect=[set(), controller.StopRequested("pending signal"), set(), set()]), \
                mock.patch.object(controller.os, "killpg") as killpg, \
                self.assertRaisesRegex(ValueError, "no automatic retry"):
            processes.run(["/inert/program"], {}, "fixture")
        killpg.assert_called_once_with(child.pid, signal.SIGTERM)
        self.assertEqual(processes.records[0]["pid"], child.pid)
        self.assertTrue(processes.records[0]["timed_out"])
        self.assertTrue(processes.stopped)
        packet_root = self.work / "main-packet"
        packet_root.mkdir()
        packet = {"run_root": str(packet_root), "run_id": "fixture-main",
                  "timeout_seconds": 30, "process_timeout_seconds": 10}
        old_handlers = {signal.SIGTERM: object(), signal.SIGINT: object()}
        with mock.patch.object(controller, "validate_packet", return_value=packet), \
                mock.patch.object(controller, "execute", side_effect=controller.StopRequested("main signal")), \
                mock.patch.object(controller.signal, "signal", side_effect=lambda number, handler: old_handlers[number]) as handlers, \
                mock.patch.dict(os.environ, {"PBS_JOBID": "42.fixture"}), \
                mock.patch.object(sys, "stdout", new_callable=io.StringIO):
            self.assertEqual(controller.main(["--packet", str(packet_root / "packet.json"),
                                              "--expected-packet-sha256", "1" * 64]), 1)
        summary = load(packet_root / "smoke/summary.json")
        self.assertEqual(summary["status"], "UNKNOWN")
        self.assertEqual(summary["commands"], [])
        self.assertEqual(handlers.call_args_list,
                         [mock.call(signal.SIGTERM, controller.request_stop),
                          mock.call(signal.SIGINT, controller.request_stop),
                          mock.call(signal.SIGTERM, old_handlers[signal.SIGTERM]),
                          mock.call(signal.SIGINT, old_handlers[signal.SIGINT])])


if __name__ == "__main__":
    unittest.main()

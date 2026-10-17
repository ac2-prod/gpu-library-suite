"""Synthetic retained-build/source proofs; no artifact, agent or process runs."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gpu_suite.hashing import source_snapshot_sha256
from gpu_suite.manifest import artifact_id

SPEC = importlib.util.spec_from_file_location("sirius_smoke_source", ROOT / "jobs/sirius/amd_smoke_source.py")
proof = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proof)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


class RetainedBuild:
    """Small independent fake files in the existing build evidence format."""

    def __init__(self, root):
        self.source = root / "old-source"
        self.snapshot = root / "saved-source"
        self.run_id = "sirius-amd-build-fixture"
        self.run = self.source / "runs" / self.run_id
        self.run.mkdir(parents=True)
        self.harness = root / "new-harness.py"
        save(self.harness, b"# never executed new harness\n")
        paths = []
        for name in ("amd_build.py", "amd_build_job.sh", "rocfft_compute_modules.sh"):
            relative = "jobs/sirius/" + name
            save(self.snapshot / relative, ("# synthetic " + name + "\n").encode())
            save(self.run / name, (self.snapshot / relative).read_bytes())
            paths.append(relative)
        save(self.snapshot / "benchmark.c", b"/* immutable source fixture */\n")
        paths.append("benchmark.c")
        self.paths = sorted(paths)
        self.source_hash = source_snapshot_sha256(self.snapshot, self.paths)
        self.head = "a" * 40
        self.evidence = {"git_head": self.head, "git_status": "## fixture\n M benchmark.c\n",
                         "paths": self.paths, "source_snapshot_sha256": self.source_hash,
                         "exclusions": ["runs/", ".fuse_hidden* in any path component"]}
        for name in proof.SOURCE_FILES:
            save(self.run / name, self.evidence)
        save(self.run / "source-preflight.json", dict(self.evidence, expected_source_snapshot_sha256=self.source_hash))
        save(self.run / "job.sh", b"# synthetic job, never run\n")
        helpers = {name: sha(self.run / name) for name in (
            "amd_build.py", "amd_build_job.sh", "rocfft_compute_modules.sh", "job.sh")}
        self.packet = {"run_id": self.run_id, "remote_source_root": str(self.source),
                       "remote_run_root": str(self.run), "source_snapshot_sha256": self.source_hash,
                       "requested_artifact_slots": 36, "libraries": list(proof.FAMILIES),
                       "profiles": list(proof.PROFILES), "helper_sha256": helpers}
        save(self.run / "manifest.json", self.packet)
        save(self.run / "prepared-files-sha256.json", dict(helpers, **{"manifest.json": sha(self.run / "manifest.json")}))
        self.summary = {"run_id": self.run_id, "status": "BUILT", "counts": {"BUILT": 36},
                        "job_id": "123.pbs", "artifact_execution_count": 0, "gpu_query_count": 0,
                        "scheduler_call_count": 0, "ssh_call_count": 0, "slots": [], "commands": [], "profiles": {}}
        stems = {"rocblas": "blas", "rocfft": "fft", "rocsparse": "sparse",
                 "rocsolver": "solver", "rocrand": "rand", "rocthrust": "reduce"}
        for profile in ("cpu", "hip", "openmp"):
            base = self.run / "amd-build" / ("build-" + profile)
            compiler = {"compiler": "fixture", "compiler_path": "/fixture/compiler",
                        "compiler_version": "1", "global_configure_flags": "-O3"}
            metadata = {"vendor": "amd", "build_profile": profile, "build_type": "Release",
                        "git_metadata_available": True, "git_commit": self.head, "git_dirty": True,
                        "dependencies": {}, "c": dict(compiler), "cxx": dict(compiler), "hip": dict(compiler)}
            save(base / "build-metadata.json", metadata)
            metadata_hash = sha(base / "build-metadata.json")
            descriptors, entries, artifacts, targets = [], [], {}, {}
            for family, stem in stems.items():
                for role in ("example", "benchmark"):
                    name = "openmp_" + family if profile == "openmp" else stem + ("_cpu" if profile == "cpu" else "_gpu")
                    if role == "benchmark":
                        name += "_bench"
                    binary = base / "amd/c-cpp" / family / name
                    save(binary, ("nonexecutable bytes " + profile + name).encode())
                    language = "hip" if profile == "hip" else "cxx" if profile == "openmp" or family in ("rocrand", "rocthrust") else "c"
                    descriptor = {"library": family, "implementation": profile, "executable_role": role,
                                  "target_name": name, "build_profile": profile, "backend_variant": "fixture",
                                  "executable_path": str(binary), "compiler_language": language, "cpu_backends": []}
                    descriptors.append(descriptor)
                    entry = {key: value for key, value in descriptor.items() if key != "cpu_backends"}
                    entry.update(compiler, supported_cpu_backends=[], binary_sha256=sha(binary),
                                 build_metadata_sha256=metadata_hash, build_type="Release",
                                 git_metadata_available=True, git_commit=self.head, git_dirty=True)
                    entry["artifact_id"] = artifact_id(entry)
                    entries.append(entry)
                    artifacts[name] = {"identity": self.identity(binary), "manifest_entry": entry}
                    link_dir = binary.parent / "CMakeFiles" / (name + ".dir")
                    output = b"fixture stdout\n"
                    if profile == "hip":
                        line = "/fixture/compiler --hip-link CMakeFiles/" + name + ".dir/source.o -o " + name
                        save(link_dir / "link.txt", (line + "\n").encode())
                        output = ("cd " + str(binary.parent) + " && /fixture/cmake -E cmake_link_script "
                                  "CMakeFiles/" + name + ".dir/link.txt --verbose=1\n" + line + "\n"
                                  "[100%] Built target " + name + "\n").encode()
                    else:
                        save(link_dir / "link.d", (str(binary) + ": /fixture/lib.so\n").encode())
                    command = self.command("target-" + profile + "-" + name,
                                           ["/fixture/cmake-launcher", "--build", str(base), "--parallel", "1", "--target", name],
                                           output)
                    self.summary["commands"].append(command)
                    index = len(self.summary["commands"])
                    targets[name] = {"command_index": index, "compile_status": "SUCCEEDED"}
                    self.summary["slots"].append({"library": family, "profile": profile, "role": role,
                                                  "target": name, "status": "BUILT", "compile_status": "SUCCEEDED",
                                                  "failure_stage": None, "command_index": index})
            descriptor = {"build_metadata_path": str(base / "build-metadata.json"), "entries": descriptors}
            for name in ("partial-manifest-input.json", "built-manifest-input.json"):
                save(base / name, descriptor)
            save(base / "partial-manifest.json", {"manifest_schema_version": 1, "build_metadata_sha256": metadata_hash,
                                                  "entries": entries})
            save(base / "CMakeCache.txt", b"CMAKE_COMMAND:INTERNAL=/fixture/cmake\n")
            save(base / "compile_commands.json", [])
            files = {name: self.identity(base / name) for name in proof.PROFILE_FILES}
            self.summary["profiles"][profile] = {
                "files": files, "configured_file_identities": {name: files[name] for name in
                                                               ("build-metadata.json", "partial-manifest-input.json")},
                "build_metadata": metadata, "dependencies": {}, "artifacts": artifacts,
                "target_builds": targets, "configured_targets": sorted(targets)}
        save(self.run / "amd-build/summary.json", self.summary)
        marker = "GPU_SUITE_SIRIUS_AMD_BUILD_END " + self.run_id + " 0"
        save(self.run / "job.stdout", ("fixture output\n" + marker + "\n").encode())
        save(self.run / "job.stderr", b"")
        save(self.run / "amd-build-job-exit.txt", b"0\n")
        terminal = self.command("pbs-terminal", ["/opt/pbs/bin/qstat", "-xf", "123.pbs"],
                                b"Job Id: 123.pbs\n    job_state = F\n    Exit_status = 0\n    run_count = 1\n    Rerunable = False\n")
        self.dispatch = {"status": "BUILD_COMPLETE", "build_status": "BUILT", "pbs_job_state": "F",
                         "pbs_exit_status": 0, "qsub_exit": 0, "job_submissions": 1,
                         "artifact_counts": {"BUILT": 36}, "requested_artifact_slots": 36,
                         "run_id": self.run_id, "job_id": "123.pbs", "purpose": "build-only",
                         "run_root": str(self.run), "source_root": str(self.source),
                         "proof_sha256": {}, "completion_marker": marker, "terminal_observation": terminal}
        argv = ["/opt/pbs/bin/qsub", str(self.run / "job.sh")]
        save(self.run / "submitted-job.json", {"argv": argv, "job_id": "123.pbs", "run_id": self.run_id, "qsub_exit": 0})
        save(self.run / "submission-exit.json", self.command("qsub", argv, b"123.pbs\n"))
        self.refresh_approval()

    @staticmethod
    def identity(path):
        return {"path": str(path), "resolved_path": str(path), "sha256": sha(path)}

    def command(self, name, argv, stdout=b"fixture stdout\n"):
        out, err = self.run / "logs" / (name + ".stdout"), self.run / "logs" / (name + ".stderr")
        save(out, stdout)
        save(err, b"")
        return {"argv": argv, "stdout": str(out), "stderr": str(err), "stdout_sha256": sha(out),
                "stderr_sha256": sha(err), "exit_status": 0, "exit": 0,
                "process_started": True, "timed_out": False}

    def refresh_approval(self):
        save(self.run / "amd-build/summary.json", self.summary)
        self.dispatch["proof_sha256"] = {name: sha(self.run / name) for name in (
            "amd-build-job-exit.txt", "amd-build/summary.json", "job.stdout", "job.stderr")}
        save(self.run / "dispatch-result.json", self.dispatch)
        self.arguments = dict(expected_source_sha256=self.source_hash,
                              expected_summary_sha256=sha(self.run / "amd-build/summary.json"),
                              expected_evidence_sha256={name: sha(self.run / name) for name in proof.EVIDENCE_FILES},
                              expected_build_run_id=self.run_id, expected_recorded_source_root=str(self.source),
                              harness_path=self.harness, expected_harness_sha256=sha(self.harness))

    def verify(self, **overrides):
        return proof.verify_prebuilt_source(self.snapshot, self.run, **dict(self.arguments, **overrides))


class AmdSmokeSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="amd-smoke-source-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.fixture = RetainedBuild(self.root)

    def test_complete_retained_build_preserves_files_and_separates_harness(self):
        before = {str(p): (p.stat().st_mtime_ns, p.stat().st_mode, sha(p))
                  for p in self.fixture.run.rglob("*") if p.is_file()}
        with mock.patch.object(subprocess, "run", side_effect=AssertionError("No subprocess allowed")), \
                mock.patch.object(subprocess, "Popen", side_effect=AssertionError("No subprocess allowed")):
            result = self.fixture.verify()
        after = {str(p): (p.stat().st_mtime_ns, p.stat().st_mode, sha(p))
                 for p in self.fixture.run.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(result["status"], "VERIFIED")
        self.assertFalse(result["path_mapping_used"])
        self.assertEqual(len(result["entries"]), 36)
        self.assertEqual(result["artifact_source"]["source_snapshot_sha256"], self.fixture.source_hash)
        self.assertNotEqual(result["harness"]["sha256"], self.fixture.source_hash)
        self.assertFalse(result["runtime_dependencies_verified"])
        self.assertFalse(result["loaded_libraries_verified"])
        self.assertEqual(set(result["link_evidence"]), {entry["artifact_id"] for entry in result["entries"]})
        for entry in result["entries"]:
            link = result["link_evidence"][entry["artifact_id"]]
            if entry["implementation"] == "hip":
                self.assertEqual(link["mode"], "native-hip-link-command")
                self.assertEqual(link["build_directory"], str(Path(entry["executable_path"]).parent))
                self.assertEqual(link["link_command"]["sha256"], sha(Path(link["link_command"]["path"])))
                self.assertFalse(Path(link["link_command"]["path"]).with_name("link.d").exists())
            else:
                self.assertEqual(Path(link).name, "link.d")
                self.assertIn(str(Path(link).relative_to(self.fixture.run)), result["evidence_sha256"])
        self.assertEqual(proof.snapshot_identity(self.fixture.snapshot, self.fixture.paths)["sha256"],
                         source_snapshot_sha256(self.fixture.snapshot, self.fixture.paths))

    def test_saved_source_rejects_extra_missing_bytes_mode_and_symlink(self):
        source = self.fixture.snapshot / "benchmark.c"
        original, mode = source.read_bytes(), source.stat().st_mode
        source.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "Saved source bytes"):
            self.fixture.verify()
        source.write_bytes(original)
        source.chmod(mode | 0o100)
        with self.assertRaisesRegex(ValueError, "executable bits"):
            self.fixture.verify()
        source.chmod(mode)
        extra = self.fixture.snapshot / "extra.c"
        extra.write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "extra or missing"):
            self.fixture.verify()
        # Rename our temporary fixture instead of changing any retained evidence.
        extra.rename(self.root / "extra-saved")
        source.rename(self.root / "missing-source")
        with self.assertRaisesRegex(ValueError, "extra or missing"):
            self.fixture.verify()
        (self.root / "missing-source").rename(source)
        original_method = Path.is_symlink
        with mock.patch.object(Path, "is_symlink", lambda p: p == source or original_method(p)):
            with self.assertRaisesRegex(ValueError, "Symbolic link"):
                self.fixture.verify()

    def test_source_evidence_must_match_every_profile_and_full_snapshot(self):
        path = self.fixture.run / "amd-build/source-before-hip.json"
        changed = dict(self.fixture.evidence, git_head="b" * 40)
        save(path, changed)
        self.fixture.refresh_approval()
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.fixture.verify()
        save(path, self.fixture.evidence)
        self.fixture.refresh_approval()
        with self.assertRaisesRegex(ValueError, "Build source differs"):
            self.fixture.verify(expected_source_sha256="b" * 64)
        for content in (b'{"paths":[],"paths":[]}', b'{"value":NaN}', b'{"value":1e999}',
                        b'{"value":"\xff"}', b'[]'):
            with self.subTest(json=content):
                save(path, content)
                self.fixture.refresh_approval()
                with self.assertRaises(ValueError):
                    self.fixture.verify()

    def test_pinned_evidence_rejects_changed_summary_logs_and_binary(self):
        paths = [self.fixture.run / "amd-build/summary.json",
                 Path(self.fixture.summary["commands"][0]["stdout"]),
                 Path(next(iter(self.fixture.summary["profiles"]["cpu"]["artifacts"].values()))["identity"]["path"])]
        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"tamper")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    self.fixture.verify()
                path.write_bytes(original)

    def test_approval_inventory_and_production_mapping_are_not_bypassable(self):
        approved = dict(self.fixture.arguments["expected_evidence_sha256"])
        approved.pop("source-preflight.json")
        with self.assertRaisesRegex(ValueError, "complete and exact"):
            self.fixture.verify(expected_evidence_sha256=approved)
        mapped = self.root / "mounted" / self.fixture.run_id
        shutil.copytree(self.fixture.run, mapped)
        with self.assertRaisesRegex(ValueError, "Production build root"):
            proof.verify_prebuilt_source(self.fixture.snapshot, mapped, **self.fixture.arguments)
        result = proof.verify_prebuilt_source(self.fixture.snapshot, mapped,
                                              **dict(self.fixture.arguments, _allow_path_mapping=True))
        self.assertEqual(result["status"], "VERIFIED")
        self.assertNotEqual(result["build_run_root"], result["recorded_build_run_root"])
        self.assertTrue(result["path_mapping_used"])
        self.assertTrue(all((item["link_command"]["path"] if isinstance(item, dict) else item).startswith(str(self.fixture.run))
                            for item in result["link_evidence"].values()))

    def test_cpu_and_openmp_require_existing_nonempty_link_depfiles(self):
        for profile, target in (("cpu", "blas_cpu"), ("openmp", "openmp_rocblas")):
            path = self.fixture.run / ("amd-build/build-" + profile) / "amd/c-cpp/rocblas/CMakeFiles" / (target + ".dir/link.d")
            original = path.read_bytes()
            path.rename(path.with_name("saved-link.d"))
            with self.subTest(profile=profile, missing=True), self.assertRaises(OSError):
                self.fixture.verify()
            path.with_name("saved-link.d").rename(path)
            path.write_bytes(b"\n")
            with self.subTest(profile=profile, empty=True), self.assertRaisesRegex(ValueError, "Empty linker"):
                self.fixture.verify()
            path.write_bytes(original)

    def test_hip_link_command_requires_exact_successful_target_binding(self):
        path = self.fixture.run / "amd-build/build-hip/amd/c-cpp/rocblas/CMakeFiles/blas_gpu.dir/link.txt"
        original = path.read_bytes()
        cases = (
            (original.replace(b"/fixture/compiler", b"/different/compiler"), "compiler or output"),
            (original.replace(b"-o blas_gpu", b"-o another_target"), "output differs"),
            (original.replace(b" --hip-link", b" --hip-link -O3"), "absent or ambiguous"),
            (original + original, "exactly one command"),
        )
        for content, message in cases:
            with self.subTest(content=content):
                path.write_bytes(content)
                with self.assertRaisesRegex(ValueError, message):
                    self.fixture.verify()
        path.write_bytes(original)
        cache = self.fixture.run / "amd-build/build-hip/CMakeCache.txt"
        saved_cache = cache.read_bytes()
        for content, message in ((b"", "Missing or ambiguous configured"),
                                 (saved_cache + saved_cache, "Missing or ambiguous configured"),
                                 (b"CMAKE_COMMAND:INTERNAL=/different/cmake\n", "directory or target")):
            with self.subTest(cache=content):
                cache.write_bytes(content)
                self.fixture.summary["profiles"]["hip"]["files"]["CMakeCache.txt"] = self.fixture.identity(cache)
                self.fixture.refresh_approval()
                with self.assertRaisesRegex(ValueError, message):
                    self.fixture.verify()
        cache.write_bytes(saved_cache)
        self.fixture.summary["profiles"]["hip"]["files"]["CMakeCache.txt"] = self.fixture.identity(cache)
        self.fixture.refresh_approval()
        path.rename(path.with_name("saved-link.txt"))
        with self.assertRaises(OSError):
            self.fixture.verify()

    def test_hip_link_stdout_is_pinned_and_cannot_borrow_another_target_log(self):
        index = self.fixture.summary["profiles"]["hip"]["target_builds"]["blas_gpu"]["command_index"] - 1
        command = self.fixture.summary["commands"][index]
        path = Path(command["stdout"])
        original = path.read_bytes()
        path.write_bytes(original + b"changed\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.fixture.verify()
        for content, message in (
            (original.replace(b"/rocblas &&", b"/rocfft &&"), "directory or target"),
            (original.replace(b"Built target blas_gpu", b"Built target fft_gpu"), "subsequent successful"),
            (original + original, "absent or ambiguous"),
        ):
            with self.subTest(content=content):
                path.write_bytes(content)
                command["stdout_sha256"] = sha(path)
                self.fixture.refresh_approval()
                with self.assertRaisesRegex(ValueError, message):
                    self.fixture.verify()
        other = self.fixture.summary["commands"][index + 1]
        command["stdout"], command["stdout_sha256"] = other["stdout"], other["stdout_sha256"]
        self.fixture.refresh_approval()
        with self.assertRaisesRegex(ValueError, "absent or ambiguous"):
            self.fixture.verify()

    def test_dispatch_rejects_unknown_failed_duplicate_or_wrong_completion(self):
        stdout = self.fixture.run / "job.stdout"
        original = stdout.read_bytes()
        for content in (b"missing\n", original + original, original.replace(b"fixture 0", b"different 0"),
                        original.replace(b"fixture 0", b"fixture 1")):
            with self.subTest(content=content):
                stdout.write_bytes(content)
                self.fixture.refresh_approval()
                with self.assertRaisesRegex(ValueError, "completion marker"):
                    self.fixture.verify()
        stdout.write_bytes(original)
        self.fixture.dispatch["status"] = "UNKNOWN"
        self.fixture.refresh_approval()
        with self.assertRaisesRegex(ValueError, "completion proof"):
            self.fixture.verify()

    def test_pbs_terminal_proof_requires_finished_single_nonrerunnable_job(self):
        terminal = self.fixture.dispatch["terminal_observation"]
        path = Path(terminal["stdout"])
        original = path.read_bytes()
        for source, replacement in ((b"job_state = F", b"job_state = R"),
                                    (b"Exit_status = 0", b"Exit_status = 1"),
                                    (b"run_count = 1", b"run_count = 2"),
                                    (b"Rerunable = False", b"Rerunable = True"),
                                    (b"123.pbs", b"456.pbs")):
            with self.subTest(replacement=replacement):
                path.write_bytes(original.replace(source, replacement))
                terminal["stdout_sha256"] = sha(path)
                self.fixture.refresh_approval()
                with self.assertRaisesRegex(ValueError, "PBS terminal output"):
                    self.fixture.verify()

    def test_all_36_slots_and_zero_execution_are_mandatory(self):
        summary = self.fixture.summary
        summary["slots"][-1] = dict(summary["slots"][0])
        self.fixture.refresh_approval()
        with self.assertRaisesRegex(ValueError, "artifact slot"):
            self.fixture.verify()
        summary["artifact_execution_count"] = 1
        self.fixture.refresh_approval()
        with self.assertRaisesRegex(ValueError, "execution counters"):
            self.fixture.verify()

    def test_recorded_path_escape_and_noncanonical_alias_are_rejected(self):
        command = self.fixture.summary["commands"][0]
        original = command["stdout"]
        for value in (str(self.root / "outside"), str(self.fixture.run) + "/../outside",
                      str(self.fixture.run) + "//logs/target.stdout"):
            with self.subTest(path=value):
                command["stdout"] = value
                self.fixture.refresh_approval()
                with self.assertRaises(ValueError):
                    self.fixture.verify()
        command["stdout"] = original

    def test_manifest_descriptor_and_metadata_have_independent_bindings(self):
        for label, name in (("compiler", "partial-manifest.json"), ("descriptor", "built-manifest-input.json"),
                            ("metadata", "build-metadata.json"), ("duplicate", "partial-manifest.json")):
            with self.subTest(binding=label):
                fixture = RetainedBuild(self.root / label)
                profile = fixture.summary["profiles"]["cpu"]
                path = fixture.run / "amd-build/build-cpu" / name
                document = json.loads(path.read_text())
                if label == "compiler":
                    document["entries"][0]["compiler_version"] = "wrong"
                elif label == "descriptor":
                    document["entries"][0]["backend_variant"] = "different"
                elif label == "duplicate":
                    document["entries"][-1] = dict(document["entries"][0])
                else:
                    document["git_commit"] = "b" * 40
                save(path, document)
                profile["files"][name] = fixture.identity(path)
                fixture.refresh_approval()
                with self.assertRaises(ValueError):
                    fixture.verify()

    def test_source_hash_cannot_be_relabelled_as_new_harness_identity(self):
        self.fixture.harness.write_bytes(b"new launcher revision\n")
        with self.assertRaisesRegex(ValueError, "harness hash"):
            self.fixture.verify()
        new_hash = sha(self.fixture.harness)
        result = self.fixture.verify(expected_harness_sha256=new_hash)
        self.assertEqual(result["artifact_source"]["source_snapshot_sha256"], self.fixture.source_hash)
        self.assertEqual(result["harness"]["sha256"], new_hash)
        with self.assertRaisesRegex(ValueError, "Build source differs"):
            self.fixture.verify(expected_source_sha256=new_hash, expected_harness_sha256=new_hash)

    def test_changes_during_multifile_inspection_are_rejected(self):
        original_read = proof._read
        path = self.fixture.run / "amd-build/summary.json"
        reads = 0
        def changing_read(target):
            nonlocal reads
            result = original_read(target)
            if target == path:
                reads += 1
                if reads == 2:
                    return result[0] + b"\n", result[1]
            return result
        with mock.patch.object(proof, "_read", side_effect=changing_read):
            with self.assertRaisesRegex(ValueError, "hash mismatch|changed during"):
                self.fixture.verify()


if __name__ == "__main__":
    unittest.main()

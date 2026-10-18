"""Host-only regressions for the allocated rocFFT helper; no GPU/site commands."""

import importlib.util
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "rocfft_compute_smoke", ROOT / "jobs/sirius/rocfft_compute_smoke.py")
compute = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compute)
sys.path.insert(0, str(ROOT / "tools"))
from gpu_suite.hashing import source_snapshot_sha256


class RocfftComputeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rocfft-compute-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.source = self.base / "source"
        self.source.mkdir()
        (self.source / "runs").mkdir()

    def executable(self, name):
        path = self.base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake executable; never launched\n")
        path.chmod(0o700)
        return str(path)

    def test_tool_lookup_preserves_cpp_driver_invocation(self):
        invocation = self.executable("bin/clang++")
        job = object.__new__(compute.Job)
        job.env = {"PATH": str(self.base / "bin")}
        # Model a driver symlink without creating any symlink locally.
        with mock.patch.object(compute.shutil, "which", return_value=invocation), \
                mock.patch.object(Path, "resolve", return_value=self.base / "clang-22"):
            self.assertEqual(job.executable("clang++"), invocation)

    def test_cpu_compilers_prefer_module_gnu_drivers_to_system_aliases(self):
        job = object.__new__(compute.Job)
        job.env = {"PATH": str(self.base / "module/bin") + ":" + str(self.base / "system/bin")}
        for language, module_name, system_name in (("c", "gcc", "cc"), ("cxx", "g++", "c++")):
            with self.subTest(language=language):
                selected = self.executable("module/bin/" + module_name)
                self.executable("system/bin/" + system_name)
                job.run = mock.Mock(return_value=(0, "GNU module compiler"))
                self.assertEqual(compute.cpu_compiler(job, language), (selected, "GNU module compiler"))
                self.assertEqual(job.run.call_args.args[1], [selected, "--version"])
                job.run.assert_called_once()

    def test_cpu_compiler_default_fallback_is_checked_and_returned(self):
        selected = self.executable("bin/cc")
        job = object.__new__(compute.Job)
        job.env = {"PATH": str(self.base / "bin")}
        job.run = mock.Mock(return_value=(0, "fallback compiler version"))
        self.assertEqual(compute.cpu_compiler(job, "c"), (selected, "fallback compiler version"))
        job.run.assert_called_once_with("cpu-c-version", [selected, "--version"],
                                        timeout=20, required=False)

    def test_explicit_cpu_compilers_override_path_and_preserve_cpp_driver(self):
        for language, variable, name in (("c", "CPU_CC", "gcc"), ("cxx", "CPU_CXX", "g++")):
            with self.subTest(language=language):
                invocation = self.executable("selected/bin/" + name)
                job = mock.Mock()
                job.env = {variable: invocation, "PATH": "/unselected/bin"}
                job.run.return_value = (0, "explicit compiler version")
                with mock.patch.object(Path, "resolve", side_effect=AssertionError("must preserve driver")):
                    self.assertEqual(compute.cpu_compiler(job, language),
                                     (invocation, "explicit compiler version"))
                job.run.assert_called_once_with(compute.CPU_COMPILERS[language][2],
                                                [invocation, "--version"], timeout=20, required=False)
                job.tool.assert_not_called()
                job.executable.assert_not_called()

    def test_invalid_explicit_cpu_compilers_stop_without_fallback(self):
        non_executable = Path(self.executable("not-executable"))
        non_executable.chmod(0o600)
        invalid = ("", "gcc", "gcc -O2", str(self.base / "missing"),
                   str(self.base), str(non_executable))
        for language, variable in (("c", "CPU_CC"), ("cxx", "CPU_CXX")):
            for invocation in invalid:
                with self.subTest(language=language, invocation=invocation):
                    job = mock.Mock()
                    job.env = {variable: invocation}
                    with self.assertRaisesRegex(ValueError, "absolute executable file"):
                        compute.cpu_compiler(job, language)
                    job.run.assert_not_called()
                    job.tool.assert_not_called()

    def test_explicit_cpu_compiler_version_failure_stops_without_fallback(self):
        for language, variable in (("c", "CPU_CC"), ("cxx", "CPU_CXX")):
            with self.subTest(language=language):
                invocation = self.executable(variable)
                job = mock.Mock()
                job.env = {variable: invocation}
                job.run.return_value = (127, "")
                with self.assertRaisesRegex(ValueError, "--version failed; no fallback"):
                    compute.cpu_compiler(job, language)
                job.run.assert_called_once()
                job.tool.assert_not_called()

    def test_amd_compiler_versions_use_logical_driver_not_realpath(self):
        invocation = self.executable("bin/amdclang++")
        job = mock.Mock()
        job.executable.side_effect = lambda name: invocation if name == "amdclang++" else None
        job.run.return_value = (0, "AMD clang version test")
        with mock.patch.object(Path, "resolve", return_value=self.base / "amdllvm"):
            self.assertEqual(compute.amd_compiler(job, self.base), invocation)
        job.run.assert_called_once_with("amd-cxx-version", [invocation, "--version"],
                                        timeout=20, required=False)

    def test_rocm_discovery_uses_realpath_only_for_prefix_evidence(self):
        invocation = self.executable("front/amdclang++")
        rocm = self.base / "rocm"
        for header in ("include/hip/hip_runtime_api.h", "include/rocfft/rocfft.h"):
            path = rocm / header
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic header location\n")
        original_resolve = Path.resolve

        def resolved(path, *args, **kwargs):
            if str(path) == invocation:
                return rocm / "lib/llvm/bin/amdllvm"
            return original_resolve(path, *args, **kwargs)

        job = mock.Mock()
        job.env = {}
        job.executable.side_effect = lambda name: invocation if name == "amdclang++" else None
        job.run.return_value = (0, "AMD clang version test")
        with mock.patch.object(Path, "resolve", resolved):
            self.assertEqual(compute.find_rocm(job), rocm)
        self.assertEqual(job.run.call_args.args[1][0], invocation)

    def test_tool_provenance_does_not_replace_invocation(self):
        invocation = self.executable("bin/clang++")
        resolved = self.base / "clang-22"
        with mock.patch.object(Path, "resolve", return_value=resolved):
            evidence = compute.tool_identity(invocation)
        self.assertEqual(evidence["invocation"], invocation)
        self.assertEqual(evidence["resolved_path"], str(resolved))
        self.assertEqual(len(evidence["sha256"]), 64)

    def test_fftw_root_hints_are_forwarded_without_inventing_a_prefix(self):
        self.assertEqual(compute.fftw_cmake_flags({}), [])
        environment = {name: str(self.base / name) for name in compute.FFTW_HINT_VARIABLES}
        self.assertEqual(compute.fftw_cmake_flags(environment),
                         ["-D" + name + "=" + environment[name]
                          for name in compute.FFTW_HINT_VARIABLES[:4]])

    def test_fftw_threading_selection_is_explicit_and_invalid_values_fail(self):
        for backend in ("AUTO", "PTHREAD", "OPENMP", "COMBINED", "NONE"):
            self.assertEqual(compute.fftw_cmake_flags({"FFTW3f_THREADING": backend}),
                             ["-DFFTW3f_THREADING=" + backend])
        for value in ("", "openmp", "OPENMP;PTHREAD"):
            with self.assertRaisesRegex(ValueError, "Invalid FFTW3f_THREADING"):
                compute.fftw_cmake_flags({"FFTW3f_THREADING": value})

    def test_cpu_threads_are_bounded_by_allocation_and_process_affinity(self):
        with mock.patch.object(compute.os, "sched_getaffinity", return_value={4, 5}, create=True):
            self.assertEqual(compute.cpu_thread_allocation(2, 24, {}), {
                "requested_threads": 2, "effective_threads": 2, "allocated_cpus": 24,
                "affinity_cpu_count": 2, "confirmed_thread_limit": 2, "openmp_thread_limit": None})
            for requested, allocated in ((3, 24), (2, 1)):
                with self.assertRaisesRegex(ValueError, "exceeds verified allocation or affinity"):
                    compute.cpu_thread_allocation(requested, allocated)
        for requested, allocated in ((0, 24), (-1, 24), (True, 24), (2, 0), (2, "24")):
            with self.assertRaises(ValueError):
                compute.cpu_thread_allocation(requested, allocated)

    def test_failed_or_empty_affinity_is_not_treated_as_unlimited(self):
        with mock.patch.object(compute.os, "sched_getaffinity", side_effect=OSError("denied"), create=True):
            with self.assertRaisesRegex(ValueError, "Cannot confirm process CPU affinity"):
                compute.cpu_thread_allocation(2, 24)
        with mock.patch.object(compute.os, "sched_getaffinity", return_value=set(), create=True):
            with self.assertRaisesRegex(ValueError, "affinity is empty"):
                compute.cpu_thread_allocation(2, 24)

    def test_openmp_thread_limit_cannot_silently_reduce_requested_parallelism(self):
        with mock.patch.object(compute.os, "sched_getaffinity", return_value={0, 1}, create=True):
            for value in ("0", "", "not-a-count", "1"):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "OMP_THREAD_LIMIT"):
                    compute.cpu_thread_allocation(2, 24, {"OMP_THREAD_LIMIT": value})
            self.assertEqual(compute.cpu_thread_allocation(2, 24, {"OMP_THREAD_LIMIT": "2"})["openmp_thread_limit"], 2)
    def test_threaded_smoke_uses_separate_primary_backend_and_effective_thread_count(self):
        serial_before = (ROOT / "jobs/sirius/rocfft-smoke.json").read_bytes()
        for count in (1, 2, 4):
            config = compute.smoke_config(ROOT, count)
            benchmark = config["benchmarks"]["rocfft"]
            cpu = [series for series in benchmark["series"] if series["implementation"] == "cpu"]
            expected = "cpu-fftw-serial" if count == 1 else "cpu-fftw-threaded"
            self.assertEqual(len(cpu), 1)
            self.assertEqual(cpu[0]["cpu_backend"], expected)
            self.assertEqual(cpu[0]["series_role"], "primary")
            self.assertEqual(cpu[0]["cpu_backend_role"], "production")
            self.assertEqual(cpu[0]["cpu_threads_effective"], count)
            self.assertEqual(config["cpu_threads"], count)
            self.assertEqual(benchmark["default_speedup_cpu_backend"], expected)
            self.assertEqual(set(benchmark["cases"][0]["scopes"]), {"compute", "end-to-end"})
            canonical = json.loads((ROOT / "jobs/sirius" / (
                "rocfft-smoke.json" if count == 1 else "rocfft-threaded-smoke.json")).read_text())
            self.assertEqual(benchmark["cases"], canonical["benchmarks"]["rocfft"]["cases"])
            self.assertEqual(benchmark["verification"], canonical["benchmarks"]["rocfft"]["verification"])
            self.assertFalse(config["continue_on_failure"])
            self.assertEqual(len(benchmark["cases"]), 2)
        self.assertEqual((ROOT / "jobs/sirius/rocfft-smoke.json").read_bytes(), serial_before)

    def test_serial_and_threaded_campaigns_are_explicitly_separate(self):
        self.assertEqual(compute.campaign_threads(1), [(None, 1)])
        self.assertEqual(compute.campaign_threads(2), [(None, 2)])
        self.assertEqual(compute.campaign_threads(2, True), [("serial", 1), ("threaded", 2)])
        with self.assertRaisesRegex(ValueError, "requires --cpu-threads"):
            compute.campaign_threads(1, True)

    def test_missing_or_failed_profile_stops_before_any_benchmark(self):
        complete = {profile: self.base / profile for profile in ("cpu", "hip", "openmp")}
        compute.require_all_profiles(complete, [])
        for omitted in complete:
            with self.subTest(omitted=omitted), self.assertRaisesRegex(ValueError, "All CPU/HIP/OpenMP"):
                compute.require_all_profiles({key: path for key, path in complete.items() if key != omitted}, [])
        with self.assertRaisesRegex(ValueError, "All CPU/HIP/OpenMP"):
            compute.require_all_profiles(complete, [{"stage": "example-openmp"}])

    def verified_rows(self, config, run_id="fixture", config_hash="a" * 64,
                      runtime_hash="b" * 64, source_hash="c" * 64):
        from test_rocfft_tools import campaign
        _, manifest, _, _, rows = campaign()
        cpu = config["benchmarks"]["rocfft"]["series"][0]
        verification = config["benchmarks"]["rocfft"]["verification"]
        for row in rows:
            row.update(run_id=run_id, block_id=run_id + "|0|node0", system_label="Sirius",
                       config_sha256=config_hash, runtime_environment_sha256=runtime_hash,
                       source_snapshot_sha256=source_hash, scheduler="PBS", scheduler_job_id="42.fixture",
                       verification_metrics={"dc_relative_error": 0.0, "non_dc_max_abs_error": 0.0},
                       verification_primary_metric="non_dc_max_abs_error",
                       verification_thresholds={
                           "dc_relative_error": {"method": "relative-upper-bound", "upper_bound": verification["rel_tolerance"]},
                           "non_dc_max_abs_error": {"method": "absolute-upper-bound", "upper_bound": verification["abs_tolerance"]}})
            if row["implementation"] == "cpu":
                row.update(cpu_backend=cpu["cpu_backend"], library_name=cpu["cpu_backend"],
                           cpu_threads_requested=config["cpu_threads"], cpu_threads_effective=config["cpu_threads"],
                           cpu_parallelism=cpu["cpu_parallelism"])
        return manifest, rows

    def test_strict_smoke_verifies_all_24_rows_in_serial_and_threaded_formats(self):
        from gpu_suite.results_io import ExclusiveRawWriter, load_raw_results
        for count in (1, 2):
            config = compute.smoke_config(ROOT, count)
            manifest, rows = self.verified_rows(config)
            for fmt in ("jsonl", "csv"):
                with self.subTest(count=count, fmt=fmt):
                    path = self.base / (str(count) + "." + fmt)
                    with ExclusiveRawWriter(path, fmt) as writer:
                        for row in rows:
                            writer.write(row)
                    report = compute.validate_smoke_rows(load_raw_results(path), config, manifest,
                                                        "a" * 64, "b" * 64, "fixture", "c" * 64,
                                                        "PBS", "42.fixture")
                    self.assertEqual(report["status_counts"], {"success": 24})
                    self.assertEqual(report["validation_status"], "pass")

    def test_strict_smoke_rejects_missing_duplicate_and_unconfigured_rows(self):
        config = compute.smoke_config(ROOT, 2)
        manifest, rows = self.verified_rows(config)
        for mutation in ("missing", "duplicate", "extra-block", "extra-invocation"):
            changed = copy.deepcopy(rows)
            if mutation == "missing":
                changed.pop()
            elif mutation == "duplicate":
                changed.append(copy.deepcopy(changed[0]))
            elif mutation == "extra-block":
                extra = copy.deepcopy(rows)
                for row in extra:
                    row.update(hostname="node1", node_index=6, block_id="fixture|0|node1")
                changed.extend(extra)
            else:
                extra = copy.deepcopy(changed[0])
                extra["parameters"]["nfft"] = 128
                extra["problem_size"] = 128
                changed.append(extra)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compute.validate_smoke_rows(changed, config, manifest, "a" * 64, "b" * 64, "fixture", "c" * 64,
                                           "PBS", "42.fixture")

    def test_strict_smoke_rejects_skipped_verification_and_false_numerical_pass(self):
        config = compute.smoke_config(ROOT, 2)
        manifest, rows = self.verified_rows(config)
        for mutation in ("verification-skipped", "metric-outside-tolerance", "negative-metric",
                         "changed-tolerance", "missing-metric", "missing-primary"):
            changed = copy.deepcopy(rows)
            row = changed[0]
            if mutation == "verification-skipped":
                row["verification_status"] = "skipped"
            elif mutation == "metric-outside-tolerance":
                row["verification_metrics"]["dc_relative_error"] = 1.0
            elif mutation == "negative-metric":
                row["verification_metrics"]["dc_relative_error"] = -1.0
            elif mutation == "changed-tolerance":
                row["verification_thresholds"]["dc_relative_error"]["upper_bound"] = 1.0
            elif mutation == "missing-primary":
                row["verification_primary_metric"] = None
            else:
                row["verification_metrics"].pop("dc_relative_error")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compute.validate_smoke_rows(changed, config, manifest, "a" * 64, "b" * 64, "fixture", "c" * 64,
                                           "PBS", "42.fixture")

    def test_strict_smoke_rejects_uniform_but_wrong_campaign_provenance(self):
        config = compute.smoke_config(ROOT, 2)
        manifest, rows = self.verified_rows(config)
        for field in ("run_id", "runtime_environment_sha256", "source_snapshot_sha256", "cpu_threads_requested", "scheduler_job_id"):
            changed = copy.deepcopy(rows)
            for row in changed:
                if field == "cpu_threads_requested":
                    if row["implementation"] == "cpu":
                        row[field] = 7
                elif field == "run_id":
                    row.update(run_id="other", block_id="other|0|node0")
                elif field == "scheduler_job_id":
                    row[field] = "99.other"
                else:
                    row[field] = "d" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                compute.validate_smoke_rows(changed, config, manifest, "a" * 64, "b" * 64, "fixture", "c" * 64,
                                           "PBS", "42.fixture")

    def fake_campaign_job(self, include_serial=True):
        from gpu_suite.results_io import ExclusiveRawWriter
        from gpu_suite.strict_json import load
        root = self.base / "compute"
        root.mkdir()
        args = SimpleNamespace(source_root=ROOT, cpu_threads=2 if include_serial else 1,
                               allocated_cpus=24, include_serial_campaign=include_serial,
                               run_id="fixture", scheduler="PBS", job_id="42.fixture")
        job = object.__new__(compute.Job)
        job.args, job.root, job.failures = args, root, []
        job.env = {"OMP_NUM_THREADS": str(args.cpu_threads), "OMP_DYNAMIC": "FALSE"}
        job.commands = []
        manifest, _ = self.verified_rows(compute.smoke_config(ROOT, args.cpu_threads))
        manifest_path = root / "executables.json"
        compute.write_json(manifest_path, manifest)

        def run(stage, argv, **kwargs):
            argv = list(map(str, argv))
            job.commands.append({"stage": stage, "argv": argv, "environment": job.env.copy()})
            if stage.startswith("benchmark-"):
                option = lambda name: argv[argv.index(name) + 1]
                config_path = Path(option("--config"))
                config = load(config_path)
                _, rows = self.verified_rows(config, option("--run-id"),
                    hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    option("--runtime-environment-sha256"), option("--source-snapshot-sha256"))
                with ExclusiveRawWriter(Path(option("--output")), config["output_format"]) as writer:
                    for row in rows:
                        writer.write(row)
            return 0, "fixture; no process was launched"

        job.run = mock.Mock(side_effect=run)
        builds = {profile: self.base / ("build-" + profile) for profile in ("cpu", "hip", "openmp")}
        runtime = {"run_id": "fixture", "environment": job.env.copy(), "build_metadata": {}}
        return job, builds, runtime, manifest_path

    def test_campaign_execution_uses_distinct_configs_runtime_hashes_and_thread_environments(self):
        job, builds, runtime, manifest = self.fake_campaign_job()
        before = copy.deepcopy(runtime)
        with mock.patch.object(compute.subprocess, "Popen") as launch, \
                mock.patch.object(compute.os, "sched_getaffinity", return_value=set(range(24)), create=True):
            compute.execute_campaigns(job, builds, runtime, "c" * 64, manifest, "/fixture/python")
            launch.assert_not_called()
        self.assertEqual(job.failures, [])
        self.assertEqual(runtime, before)
        commands = [item for item in job.commands if item["stage"].startswith("benchmark-")]
        self.assertEqual(len(commands), 4)
        identities, hashes, config_hashes = set(), set(), set()
        for command in commands:
            argv = command["argv"]
            option = lambda name: argv[argv.index(name) + 1]
            run_id = option("--run-id")
            label, fmt = run_id.removeprefix("fixture-").split("-")
            root = job.root / label
            config_path = root / ("config-" + fmt + ".json")
            evidence_path = root / ("runtime-environment-" + fmt + ".json")
            evidence = json.loads(evidence_path.read_text())
            self.assertEqual(command["environment"]["OMP_NUM_THREADS"], "1" if label == "serial" else "2")
            self.assertEqual(command["environment"]["OMP_DYNAMIC"], "FALSE")
            self.assertEqual(evidence["environment"], command["environment"])
            self.assertEqual(evidence["run_id"], run_id)
            self.assertEqual(evidence["config_sha256"], hashlib.sha256(config_path.read_bytes()).hexdigest())
            self.assertEqual(option("--runtime-environment-sha256"), hashlib.sha256(evidence_path.read_bytes()).hexdigest())
            self.assertEqual(evidence["cpu_thread_allocation"]["effective_threads"], 1 if label == "serial" else 2)
            self.assertEqual(json.loads((root / ("strict-validation-" + fmt + ".json")).read_text())["record_count"], 24)
            identities.add(run_id)
            hashes.add(option("--runtime-environment-sha256"))
            config_hashes.add(evidence["config_sha256"])
        self.assertEqual((len(identities), len(hashes), len(config_hashes)), (4, 4, 4))
        self.assertEqual(job.env["OMP_NUM_THREADS"], "2")

    def test_default_serial_campaign_retains_existing_output_paths(self):
        job, builds, runtime, manifest = self.fake_campaign_job(include_serial=False)
        with mock.patch.object(compute.os, "sched_getaffinity", return_value=set(range(24)), create=True):
            compute.execute_campaigns(job, builds, runtime, "c" * 64, manifest, "/fixture/python")
        self.assertEqual(job.failures, [])
        for fmt in ("jsonl", "csv"):
            self.assertTrue((job.root / ("config-" + fmt + ".json")).is_file())
            self.assertTrue((job.root / ("raw." + fmt)).is_file())
        self.assertFalse((job.root / "serial").exists())

    def test_campaign_failure_stops_remaining_campaigns_and_restores_environment(self):
        job, builds, runtime, manifest = self.fake_campaign_job()
        def failure(stage, argv, **kwargs):
            if stage.startswith("benchmark-"):
                job.fail(stage, 7)
                return 7, "failed fixture"
            return 1, "raw file absent"
        job.run.side_effect = failure
        with mock.patch.object(compute.os, "sched_getaffinity", return_value=set(range(24)), create=True):
            compute.execute_campaigns(job, builds, runtime, "c" * 64, manifest, "/fixture/python")
        stages = [call.args[0] for call in job.run.call_args_list]
        self.assertEqual(stages, ["benchmark-serial-jsonl", "validator-serial-jsonl"])
        self.assertTrue(job.failures)
        self.assertFalse((job.root / "threaded").exists())
        self.assertFalse((job.root / "serial/config-csv.json").exists())
        self.assertEqual(job.env["OMP_NUM_THREADS"], "2")

    def test_threaded_smoke_requires_built_manifest_capability_and_backend_provenance(self):
        entry = {"target_name": "fft_cpu_bench", "library": "rocfft", "implementation": "cpu",
                 "supported_cpu_backends": ["cpu-fftw-serial", "cpu-fftw-threaded"]}
        manifest = {"entries": [entry]}
        for backend in ("pthread", "openmp", "combined"):
            names = ["fftw3f"] + {"pthread": ["fftw3f_threads"],
                      "openmp": ["fftw3f_omp", "fftw_openmp_runtime_0"], "combined": []}[backend]
            metadata = {"dependencies": {name: {} for name in names}, "fftw_threading": {
                "requested": "AUTO", "backend": backend, "compile_options": [],
                "link_options": [], "dependency_names": names}}
            compute.require_threaded_capability(manifest, metadata, 2)
            metadata["dependencies"]["fftw3f_threads" if backend != "pthread" else "fftw3f_omp"] = {}
            with self.assertRaisesRegex(ValueError, "dependency identities differ"):
                compute.require_threaded_capability(manifest, metadata, 2)
        for bad in ({"entries": []}, {"entries": [dict(entry, supported_cpu_backends=["cpu-fftw-serial"])]},
                    {"entries": [entry, entry]}):
            with self.assertRaisesRegex(ValueError, "manifest-verified.*no serial fallback"):
                compute.require_threaded_capability(bad, {"fftw_threading": {"backend": "pthread"}}, 2)
        for metadata in ({}, {"fftw_threading": {"backend": "none"}}):
            with self.assertRaisesRegex(ValueError, "threading provenance"):
                compute.require_threaded_capability(manifest, metadata, 2)
        compute.require_threaded_capability({"entries": []}, {}, 1)

    def test_thread_environment_and_allocation_are_recorded_without_process_launch(self):
        root = self.source / "runs/threaded"
        root.mkdir()
        args = SimpleNamespace(source_root=self.source, cpu_threads=2, allocated_cpus=24,
                               allocated_gpus=1, scheduler="PBS", job_id="fake.pbs", run_id="threaded")
        with mock.patch.object(compute.os, "sched_getaffinity", return_value={1, 2, 3, 4}, create=True), \
                mock.patch.object(compute.subprocess, "Popen") as launch:
            job = compute.Job(args, root)
            launch.assert_not_called()
        self.assertEqual(job.env["OMP_NUM_THREADS"], "2")
        self.assertEqual(job.env["OMP_DYNAMIC"], "FALSE")
        self.assertEqual(job.env["OMP_TARGET_OFFLOAD"], "MANDATORY")
        evidence = json.loads((root / "allocation-start.json").read_text())
        self.assertEqual(evidence["cpu_thread_allocation"], job.cpu_allocation)
        self.assertEqual(job.cpu_allocation["effective_threads"], 2)

    def test_direct_runs_child_is_accepted_and_no_files_are_created(self):
        output = self.source / "runs/new-run"
        self.assertEqual(compute.validate_run_location(self.source, output, "new-run"),
                         (self.source, output))
        self.assertFalse(output.exists())

    def test_outside_source_nested_and_ancestor_outputs_are_rejected(self):
        for output in (self.base / "new-run", self.source, self.base,
                       self.source / "build/new-run", self.source / "runs/deeper/new-run"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                compute.validate_run_location(self.source, output)

    def test_run_id_mismatch_and_noncanonical_path_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "match the run ID"):
            compute.validate_run_location(self.source, self.source / "runs/one", "two")
        with self.assertRaisesRegex(ValueError, "canonical"):
            compute.validate_run_location(self.source, self.source / "runs/nested/../one")

    def test_symlink_run_parent_is_rejected_without_creating_symlinks(self):
        original = Path.is_symlink

        def is_symlink(path):
            return path == self.source / "runs" or original(path)

        with mock.patch.object(Path, "is_symlink", is_symlink), \
                self.assertRaisesRegex(ValueError, "symlinks"):
            compute.validate_run_location(self.source, self.source / "runs/one")

    def test_run_logs_and_fuse_tombstones_do_not_enter_source_hash(self):
        (self.source / "code.cpp").write_text("int source;\n")
        paths = ["code.cpp", "runs/old/log", "runs/new/log", ".fuse_hidden01",
                 "cmake/.fuse_hidden02", ".fuse_hidden03/child"]
        selected = compute.snapshot_paths(paths)
        self.assertEqual(selected, ["code.cpp"])
        before = source_snapshot_sha256(self.source, selected)
        # Excluded paths need not even exist; no bytes are read from them.
        after = source_snapshot_sha256(self.source, compute.snapshot_paths(paths + ["runs/later/log"]))
        self.assertEqual(before, after)
        (self.source / "code.cpp").write_text("int changed_source;\n")
        self.assertNotEqual(before, source_snapshot_sha256(self.source, selected))


if __name__ == "__main__":
    unittest.main()

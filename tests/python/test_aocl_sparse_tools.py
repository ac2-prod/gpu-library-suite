"""AOCL Sparse contracts using inert files and mocked subprocesses only."""

import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from gpu_suite.config import load_config, validate_config
from gpu_suite.hashing import deterministic_json_sha256, sha256_file
from gpu_suite.manifest import artifact_id, merge_entries
from gpu_suite.plot_display import BACKEND_LABELS
from gpu_suite.results_io import load_raw_results
from gpu_suite.runner import (
    AOCL_SPARSE_ENVIRONMENT_KEYS, _expected_verification, _validate_amd_dependencies,
    aocl_sparse_environment_overrides, build_schedule, core_raw_parameters,
    dry_run_document, execute_schedule, synthetic_result, validate_manifest_artifacts,
    validate_subprocess_record,
)
from gpu_suite.schema import validate_raw_result
from gpu_suite.strict_json import dumps
from gpu_suite.validation import validate_campaign, validate_rocfft_provenance
from test_rocfft_tools import ZERO, amd_entry

ROOT = Path(__file__).resolve().parents[2]


def sparse_campaign():
    config = load_config(ROOT / "configs/amd/smoke.json")
    config["benchmarks"] = {"rocsparse": config["benchmarks"]["rocsparse"]}
    definition = config["benchmarks"]["rocsparse"]
    definition["default_speedup_cpu_backend"] = "cpu-aocl-sparse"
    cpu = definition["series"][0]
    cpu.update(cpu_backend="cpu-aocl-sparse", cpu_threads_effective=None)
    entries = []
    for impl in ("cpu", "hip", "openmp"):
        entry = amd_entry(impl)
        target = {"cpu": "sparse_cpu_bench", "hip": "sparse_gpu_bench",
                  "openmp": "openmp_rocsparse_bench"}[impl]
        entry.update(library="rocsparse", target_name=target, executable_path="/fixture/" + target,
                     backend_variant="aocl-sparse" if impl == "cpu" else impl + "-rocsparse",
                     supported_cpu_backends=["cpu-aocl-sparse"] if impl == "cpu" else [])
        entry["artifact_id"] = artifact_id(entry)
        entries.append(entry)
    manifest = {"manifest_schema_version": 1, "entries": merge_entries([entries])}
    context = dict(config=config, cpu_threads=1, device=0, hostname="node0", node_index=0,
                   run_id="run-1", wave=0, scheduler=None, scheduler_job_id=None,
                   system_label="local", implementation_order=("cpu", "hip", "openmp"))
    metadata = {"2" * 64: {language: {name: entries[0][name] for name in
                 ("compiler", "compiler_version", "global_configure_flags")}
                 for language in ("c", "hip", "cxx")}}
    schedule = build_schedule(validate_config(config), manifest, context)
    records = []
    for item in schedule:
        for trial in range(item["scope_settings"]["trials"]):
            row = synthetic_result(item, context, ZERO, ZERO, metadata, trial, False,
                                   "prerequisite", "fixture", None, ZERO, None)
            thresholds, metrics, primary = _expected_verification(item, context)
            row.update(attempted=True, failure_origin=None, elapsed_total_sec=0.01,
                       elapsed_sec=0.01 / row["repeat"], status="success", exit_code=0,
                       verification_status="pass", verification_thresholds=thresholds,
                       verification_metrics={key: 0.0 for key in metrics},
                       verification_primary_metric=primary, message="",
                       measurement_start_timestamp="2026-07-14T00:00:00.000Z",
                       measurement_end_timestamp="2026-07-14T00:00:00.010Z",
                       library_version="fixture-version")
            if row["implementation"] == "cpu":
                row["parameters"]["runtime"]["aocl_sparse_environment"] = dict(
                    aocl_sparse_environment_overrides(item, context),
                    AOCL_ENABLE_INSTRUCTIONS=None)
            else:
                row["gpu_name"] = "fixture GPU, never executed"
                row["parameters"]["runtime"].update(hip_runtime_version=70200000,
                                                       hip_driver_version=70200000)
            records.append(validate_raw_result(row))
    return config, manifest, context, schedule, records, metadata


class AoclSparseToolsTests(unittest.TestCase):
    def setUp(self):
        base = Path("/tmp/gpu-library-suite-local-build")
        base.mkdir(exist_ok=True)
        requested = Path(tempfile.gettempdir()).resolve()
        if requested == base.resolve() or base.resolve() in requested.parents:
            base = requested
        temporary = tempfile.TemporaryDirectory(prefix="aocl-sparse-tools-", dir=base)
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)

    def dependency_fixture(self):
        _, manifest, _, _, _, _ = sparse_campaign()
        entry = copy.deepcopy(next(row for row in manifest["entries"] if row["implementation"] == "cpu"))
        binary = self.work / entry["target_name"]
        binary.write_bytes(b"inert fixture binary; never executed")
        primary, extra = self.work / "libaoclsparse.so", self.work / "libgomp.so"
        header = self.work / "aoclsparse.h"
        primary.write_bytes(b"fixture sparse provider")
        extra.write_bytes(b"fixture threading runtime")
        header.write_bytes(b"fixture AOCL Sparse public header")
        metadata = {name: entry[name] for name in (
            "build_profile", "build_type", "git_commit", "git_dirty", "git_metadata_available")}
        metadata.update(vendor="amd", c={name: entry[name] for name in
                         ("compiler", "compiler_version", "global_configure_flags")},
                        dependencies={name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                                      for name, path in (("aocl_sparse", primary),
                                                         ("aocl_sparse_link_1", extra),
                                                         ("aocl_sparse_header", header))},
                        amd_build_inputs={"GPU_SUITE_CPU_SPARSE_BACKEND": "AOCLSPARSE",
                                          "AOCLSparse_INTERFACE_INCLUDE_DIRECTORIES": str(self.work),
                                          "AOCLSparse_INTERFACE_LINK_LIBRARIES": str(primary) + ";" + str(extra) + ";Threads::Threads;dl;m"})
        entry.update(executable_path=str(binary), binary_sha256=sha256_file(binary))
        entry["artifact_id"] = artifact_id(entry)
        return entry, metadata, primary, extra

    def test_explicit_backend_config_and_manifest_are_required(self):
        config, manifest, _, _, _, _ = sparse_campaign()
        validate_config(config)
        merge_entries([manifest["entries"]])
        self.assertEqual(BACKEND_LABELS["cpu-aocl-sparse"], "AOCL Sparse")
        for mutation in ("wrong-family", "reference-role", "invented-effective"):
            altered = copy.deepcopy(config)
            cpu = altered["benchmarks"]["rocsparse"]["series"][0]
            if mutation == "wrong-family":
                altered["benchmarks"]["cusparse"] = altered["benchmarks"].pop("rocsparse")
                altered.pop("vendor")
            elif mutation == "reference-role": cpu["cpu_backend_role"] = "reference"
            else: cpu["cpu_threads_effective"] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_config(altered)
        cpu = next(row for row in manifest["entries"] if row["implementation"] == "cpu")
        for supported in ([], ["cpu-onemkl"], ["cpu-aocl-sparse", "cpu-onemkl"]):
            altered = dict(cpu, supported_cpu_backends=supported)
            with self.subTest(supported=supported), self.assertRaises(ValueError):
                merge_entries([[altered]])

    def test_ordered_provider_inputs_and_explicit_selection_are_bound(self):
        entry, metadata, _, _ = self.dependency_fixture()
        _validate_amd_dependencies(entry, metadata)
        for mutation in ("missing", "wrong-backend", "wrong-order", "missing-extra", "extra-index", "unresolved", "missing-header", "wrong-header"):
            altered = copy.deepcopy(metadata)
            if mutation == "missing": altered["dependencies"].pop("aocl_sparse")
            elif mutation == "wrong-backend": altered["amd_build_inputs"]["GPU_SUITE_CPU_SPARSE_BACKEND"] = "ONEMKL"
            elif mutation == "wrong-order":
                altered["amd_build_inputs"]["AOCLSparse_INTERFACE_LINK_LIBRARIES"] = ";".join(
                    reversed(altered["amd_build_inputs"]["AOCLSparse_INTERFACE_LINK_LIBRARIES"].split(";")))
            elif mutation == "missing-extra": altered["dependencies"].pop("aocl_sparse_link_1")
            elif mutation == "extra-index": altered["dependencies"]["aocl_sparse_link_2"] = altered["dependencies"].pop("aocl_sparse_link_1")
            elif mutation == "missing-header": altered["dependencies"].pop("aocl_sparse_header")
            elif mutation == "wrong-header": altered["amd_build_inputs"]["AOCLSparse_INTERFACE_INCLUDE_DIRECTORIES"] = str(self.work / "wrong")
            else: altered["amd_build_inputs"]["AOCLSparse_INTERFACE_LINK_LIBRARIES"] += ";unresolved-provider"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                _validate_amd_dependencies(entry, altered)

    def test_dependency_bytes_and_binary_bytes_cannot_change(self):
        entry, metadata, primary, extra = self.dependency_fixture()
        manifest = {"entries": [entry]}
        metadata_by_hash = {entry["build_metadata_sha256"]: metadata}
        validate_manifest_artifacts(manifest, metadata_by_hash, "smoke", True)
        for path in (primary, extra, self.work / "aoclsparse.h", Path(entry["executable_path"])):
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                validate_manifest_artifacts(manifest, metadata_by_hash, "smoke", True)
            path.write_bytes(original)

    def test_explicit_thread_launch_overrides_are_aocl_only_and_visible(self):
        config, manifest, context, schedule, _, _ = sparse_campaign()
        context["cpu_threads"] = 3
        with mock.patch.dict(os.environ, {"AOCLSPARSE_NUM_THREADS": "99", "OMP_NUM_THREADS": "88",
                                        "OMP_DYNAMIC": "TRUE", "AOCL_ENABLE_INSTRUCTIONS": "AVX2"}):
            doc = dry_run_document(config, manifest, schedule, context, ZERO, ZERO)
            for command in doc["commands"]:
                if command["series"]["implementation"] == "cpu":
                    self.assertEqual(command["environment_overrides"], {
                        "AOCLSPARSE_NUM_THREADS": "3", "OMP_NUM_THREADS": "3", "OMP_DYNAMIC": "FALSE"})
                    self.assertEqual(command["inherited_aocl_enable_instructions"], "AVX2")
                else:
                    self.assertNotIn("environment_overrides", command)
            self.assertEqual(os.environ["AOCLSPARSE_NUM_THREADS"], "99")
        for backend in ("cpu-onemkl", "cpu-reference-csr"):
            item = copy.deepcopy(next(item for item in schedule if item["series"]["implementation"] == "cpu"))
            item["series"]["cpu_backend"] = backend
            self.assertEqual(aocl_sparse_environment_overrides(item, context), {})

    def test_runner_passes_requested_environment_without_mutating_parent(self):
        _, _, context, schedule, records, metadata = sparse_campaign()
        item = next(item for item in schedule if item["series"]["implementation"] == "cpu")
        rows = [copy.deepcopy(row) for row in records if row["implementation"] == "cpu" and row["scope"] == item["scope"]]
        for row in rows:
            row["parameters"]["runtime"]["aocl_sparse_environment"]["AOCL_ENABLE_INSTRUCTIONS"] = "AVX2"
        with mock.patch.dict(os.environ, {"AOCLSPARSE_NUM_THREADS": "99", "OMP_NUM_THREADS": "88",
                                        "OMP_DYNAMIC": "TRUE", "AOCL_ENABLE_INSTRUCTIONS": "AVX2"}), \
             mock.patch("gpu_suite.runner._validate_amd_dependencies"), \
             mock.patch("gpu_suite.runner.sha256_file", return_value="1" * 64), \
             mock.patch("gpu_suite.runner.subprocess.run", return_value=SimpleNamespace(
                 returncode=0, stderr="", stdout="\n".join(dumps(row) for row in rows) + "\n")) as launch:
            output = self.work / "raw.jsonl"
            self.assertEqual(execute_schedule([item], context, output, ZERO, ZERO, metadata, ZERO, None), 0)
            environment = launch.call_args.kwargs["env"]
            self.assertEqual({key: environment[key] for key in AOCL_SPARSE_ENVIRONMENT_KEYS}, {
                "AOCLSPARSE_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE",
                "AOCL_ENABLE_INSTRUCTIONS": "AVX2"})
            self.assertEqual(os.environ["AOCLSPARSE_NUM_THREADS"], "99")
            self.assertEqual(load_raw_results(output), rows)
        launch.assert_called_once()
        for name, stdout, returncode in (("empty", "", 1), ("malformed", "not JSON", 0)):
            with self.subTest(output=name), \
                 mock.patch.dict(os.environ, {"AOCLSPARSE_NUM_THREADS": "99", "OMP_NUM_THREADS": "88",
                                              "OMP_DYNAMIC": "TRUE", "AOCL_ENABLE_INSTRUCTIONS": "AVX2"}), \
                 mock.patch("gpu_suite.runner._validate_amd_dependencies"), \
                 mock.patch("gpu_suite.runner.sha256_file", return_value="1" * 64), \
                 mock.patch("gpu_suite.runner.subprocess.run", return_value=SimpleNamespace(
                     returncode=returncode, stderr="fixture failure", stdout=stdout)) as launch:
                output = self.work / (name + ".jsonl")
                self.assertEqual(execute_schedule([item], context, output, ZERO, ZERO,
                                                  metadata, ZERO, None), 1)
                failures = load_raw_results(output)
                self.assertTrue(any(row["attempted"] for row in failures))
                for row in failures:
                    self.assertEqual(row["parameters"]["runtime"]["aocl_sparse_environment"], {
                        "AOCLSPARSE_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                        "OMP_DYNAMIC": "FALSE", "AOCL_ENABLE_INSTRUCTIONS": "AVX2"})
                launch.assert_called_once()
        prelaunch = synthetic_result(item, context, ZERO, ZERO, metadata, 0, False,
                                     "prerequisite", "not launched", None, ZERO, None)
        self.assertNotIn("aocl_sparse_environment", prelaunch["parameters"]["runtime"])

    def test_startup_environment_version_and_worker_observation_are_required(self):
        _, _, context, schedule, records, _ = sparse_campaign()
        item = next(item for item in schedule if item["series"]["implementation"] == "cpu")
        row = next(copy.deepcopy(row) for row in records if row["implementation"] == "cpu" and row["scope"] == item["scope"])
        with mock.patch.dict(os.environ, {}, clear=True):
            validate_subprocess_record(row, item, context, ZERO, ZERO, ZERO, None)
            for mutation in ("missing-env", "wrong-threads", "dynamic", "isa", "version", "effective"):
                altered = copy.deepcopy(row)
                runtime = altered["parameters"]["runtime"]
                if mutation == "missing-env": runtime.pop("aocl_sparse_environment")
                elif mutation == "wrong-threads": runtime["aocl_sparse_environment"]["AOCLSPARSE_NUM_THREADS"] = "99"
                elif mutation == "dynamic": runtime["aocl_sparse_environment"]["OMP_DYNAMIC"] = "TRUE"
                elif mutation == "isa": runtime["aocl_sparse_environment"]["AOCL_ENABLE_INSTRUCTIONS"] = "AVX2"
                elif mutation == "version": altered["library_version"] = None
                else: altered["cpu_threads_effective"] = 1
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    validate_subprocess_record(altered, item, context, ZERO, ZERO, ZERO, None)

    def test_campaign_rejects_same_mathematics_with_mixed_provenance(self):
        config, manifest, _, _, records, _ = sparse_campaign()
        self.assertEqual(validate_campaign(records, config, ZERO, manifest)["validation_status"], "pass")
        first = next(row for row in records if row["implementation"] == "cpu")
        for mutation in ("runtime", "source", "config", "binary", "compiler", "flags", "build-metadata", "version", "isa"):
            other = copy.deepcopy(first)
            if mutation in {"runtime", "source", "config", "binary"}:
                key = {"runtime": "runtime_environment_sha256", "source": "source_snapshot_sha256",
                       "config": "config_sha256", "binary": "binary_sha256"}[mutation]
                other[key] = "9" * 64
            elif mutation == "compiler": other["compiler_version"] = "different"
            elif mutation == "flags": other["global_configure_flags"] = "different"
            elif mutation == "build-metadata": other["parameters"]["runtime"]["build_metadata_sha256"] = "9" * 64
            elif mutation == "version": other["library_version"] = "different"
            else: other["parameters"]["runtime"]["aocl_sparse_environment"]["AOCL_ENABLE_INSTRUCTIONS"] = "AVX2"
            self.assertEqual(core_raw_parameters("rocsparse", first["parameters"]),
                             core_raw_parameters("rocsparse", other["parameters"]))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_rocfft_provenance([first, other])
        failed = copy.deepcopy(first)
        failed.update(status="failure", failure_origin="benchmark", verification_status="skipped",
                      exit_code=1, message="fixture API failure")
        validate_raw_result(failed)
        validate_rocfft_provenance([first, failed])
        for baseline in (first, failed):
            for mutation in ("version", "isa", "thread-controls"):
                other = copy.deepcopy(failed)
                if mutation == "version": other["library_version"] = "different"
                elif mutation == "isa": other["parameters"]["runtime"]["aocl_sparse_environment"]["AOCL_ENABLE_INSTRUCTIONS"] = "AVX2"
                else: other["parameters"]["runtime"]["aocl_sparse_environment"]["AOCLSPARSE_NUM_THREADS"] = "99"
                validate_raw_result(other)
                with self.subTest(status=baseline["status"], failed_mutation=mutation), self.assertRaises(ValueError):
                    validate_rocfft_provenance([baseline, other])
        unknown = copy.deepcopy(failed)
        unknown["library_version"] = None
        unknown["parameters"]["runtime"].pop("aocl_sparse_environment")
        validate_raw_result(unknown)
        validate_rocfft_provenance([unknown, first, failed, unknown])
        # Even when an incomplete failure lacks runtime metadata, an actual API
        # version is still observed evidence, unlike a null placeholder.
        incomplete = copy.deepcopy(failed)
        incomplete["parameters"].pop("runtime")
        validate_raw_result(incomplete)
        validate_rocfft_provenance([incomplete, first])
        incomplete["library_version"] = "different"
        with self.assertRaises(ValueError):
            validate_rocfft_provenance([first, incomplete])
        changed = copy.deepcopy(manifest)
        next(entry for entry in changed["entries"] if entry["implementation"] == "cpu")["supported_cpu_backends"] = ["cpu-onemkl"]
        with self.assertRaises(ValueError):
            validate_campaign(records, config, ZERO, changed)

    def test_missing_or_wrong_backend_is_not_a_reference_fallback(self):
        config, manifest, context, _, _, _ = sparse_campaign()
        for mode in ("missing", "onemkl", "reference"):
            changed = copy.deepcopy(manifest)
            if mode == "missing":
                changed["entries"] = [entry for entry in changed["entries"] if entry["implementation"] != "cpu"]
            else:
                entry = next(entry for entry in changed["entries"] if entry["implementation"] == "cpu")
                entry.update(supported_cpu_backends=["cpu-onemkl" if mode == "onemkl" else "cpu-reference-csr"],
                             backend_variant="onemkl-sparse" if mode == "onemkl" else "reference-csr")
                entry["artifact_id"] = artifact_id(entry)
            with self.subTest(mode=mode), mock.patch("gpu_suite.runner.subprocess.run") as launch:
                schedule = build_schedule(config, changed, context)
                cpus = [item for item in schedule if item["series"]["implementation"] == "cpu"]
                self.assertTrue(cpus)
                self.assertTrue(all(item["prerequisite_failure"] for item in cpus))
                launch.assert_not_called()

    def test_nvidia_configs_and_backend_identity_remain_unchanged(self):
        for name in ("benchmark.json", "pilot.json"):
            config = load_config(ROOT / "configs" / name)
            self.assertEqual(config["benchmarks"]["cusparse"]["default_speedup_cpu_backend"], "cpu-onemkl")
        entry = amd_entry("cpu")
        entry.update(library="rocsparse", supported_cpu_backends=["cpu-onemkl"], backend_variant="onemkl-sparse")
        entry["artifact_id"] = artifact_id(entry)
        merge_entries([[entry]])
        with mock.patch("gpu_suite.runner.sha256_file", return_value=ZERO), \
             mock.patch("pathlib.Path.is_file", return_value=True):
            _validate_amd_dependencies(entry, {"vendor": "amd", "dependencies": {
                "mkl_sparse": {"path": "/fixture/mkl", "sha256": ZERO}}})

    def test_sirius_smoke_configs_select_provider_and_distinct_thread_campaigns(self):
        portable = load_config(ROOT / "configs/amd/smoke.json")
        documents = [load_config(ROOT / "jobs/sirius" / name)
                     for name in ("amd-smoke.json", "amd-threaded-smoke.json")]
        self.assertNotEqual(deterministic_json_sha256(documents[0]),
                            deterministic_json_sha256(documents[1]))
        expected = {"rocblas": "cpu-generic-cblas", "rocsolver": "cpu-generic-lapacke",
                    "rocsparse": "cpu-aocl-sparse"}
        for threads, config in enumerate(documents, 1):
            self.assertEqual(config["cpu_threads"], threads)
            for family, definition in config["benchmarks"].items():
                self.assertEqual(definition["cases"], portable["benchmarks"][family]["cases"])
                cpu = next(row for row in definition["series"] if row["implementation"] == "cpu")
                if family in expected:
                    self.assertEqual(cpu["cpu_backend"], expected[family])
                    self.assertIsNone(cpu["cpu_threads_effective"])
                elif family == "rocfft":
                    self.assertEqual(cpu["cpu_backend"], "cpu-fftw-serial" if threads == 1 else "cpu-fftw-threaded")
                    self.assertEqual(cpu["cpu_threads_effective"], threads)
                else:
                    self.assertEqual(cpu["cpu_parallelism"], "serial")
                    self.assertEqual(cpu["cpu_threads_effective"], 1)
                self.assertEqual(definition["default_speedup_cpu_backend"], cpu["cpu_backend"])
            subset = copy.deepcopy(config)
            subset["benchmarks"] = {"rocsparse": subset["benchmarks"]["rocsparse"]}
            _, manifest, context, _, _, _ = sparse_campaign()
            context.update(config=subset, cpu_threads=threads)
            self.assertTrue(all(item["prerequisite_failure"] is None
                                for item in build_schedule(subset, manifest, context)))

    def test_campaign_rejects_rocm_and_runtime_environment_changes_across_waves(self):
        config, manifest, _, _, records, _ = sparse_campaign()
        for mutation in ("rocm", "isa", "environment-hash"):
            wave = copy.deepcopy(records)
            for row in wave:
                row.update(wave=6, block_id="run-1|6|node0")
                if mutation == "rocm" and row["implementation"] != "cpu":
                    row["parameters"]["runtime"]["hip_runtime_version"] += 1
                if mutation == "isa" and row["implementation"] == "cpu":
                    row["parameters"]["runtime"]["aocl_sparse_environment"]["AOCL_ENABLE_INSTRUCTIONS"] = "AVX2"
                if mutation == "environment-hash": row["runtime_environment_sha256"] = "9" * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_campaign(records + wave, config, ZERO, manifest)


if __name__ == "__main__":
    unittest.main()

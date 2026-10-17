"""Six-family smoke coverage with synthetic rows only; no executable is run."""

import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gpu_suite.families import AMD_BENCHMARKS
from gpu_suite.manifest import artifact_id, merge_entries
from gpu_suite.results_io import ExclusiveRawWriter, load_raw_results
from gpu_suite.runner import (_expected_verification, aocl_sparse_environment_overrides,
                              build_schedule, expected_raw_parameters, synthetic_result)
from gpu_suite.strict_json import dump_bytes
from test_rocfft_tools import amd_entry

SPEC = importlib.util.spec_from_file_location("sirius_amd_smoke_campaign", ROOT / "jobs/sirius/amd_smoke_campaign.py")
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)

SOURCE_HASH = "c" * 64


def fixture(config, label):
    entries = []
    for library in config["benchmarks"]:
        for impl in ("cpu", "hip", "openmp"):
            entry = amd_entry(impl)
            target = library + "_" + impl + "_bench"
            entry.update(library=library, target_name=target, executable_path="/fixture/" + target)
            if impl == "cpu":
                entry["supported_cpu_backends"] = (["cpu-fftw-serial", "cpu-fftw-threaded"] if library == "rocfft"
                                                   else [campaign.CPU_BACKENDS[library]])
                entry["backend_variant"] = "aocl-sparse" if library == "rocsparse" else "fixture-" + library
            entry["artifact_id"] = artifact_id(entry)
            entries.append(entry)
    manifest = {"manifest_schema_version": 1, "entries": merge_entries([entries])}
    environment = {"AOCLSPARSE_NUM_THREADS": str(config["cpu_threads"]),
                   "OMP_NUM_THREADS": str(config["cpu_threads"]), "OMP_DYNAMIC": "FALSE",
                   "AOCL_ENABLE_INSTRUCTIONS": None}
    context = dict(config=config, cpu_threads=config["cpu_threads"], device=0,
                   hostname="node0", node_index=0, run_id="fixture-" + label, wave=0,
                   scheduler="PBS", scheduler_job_id="42.fixture", system_label="Sirius",
                   implementation_order=("cpu", "hip", "openmp"), aocl_sparse_environment=environment)
    config_hash = hashlib.sha256(dump_bytes(config)).hexdigest()
    runtime_hash = hashlib.sha256(label.encode()).hexdigest()
    metadata = {"2" * 64: {language: {name: entries[0][name] for name in
                 ("compiler", "compiler_version", "global_configure_flags")}
                 for language in ("c", "hip", "cxx")}}
    rows = []
    for item in build_schedule(config, manifest, context):
        if aocl_sparse_environment_overrides(item, context):
            item["aocl_sparse_environment"] = dict(environment)
        for trial in range(item["scope_settings"]["trials"]):
            row = synthetic_result(item, context, config_hash, runtime_hash, metadata, trial,
                                   False, "prerequisite", "fixture", None, None, SOURCE_HASH)
            runtime = row["parameters"]["runtime"]
            row["parameters"] = expected_raw_parameters(item["benchmark"], item["parameters"], item["series"]["implementation"])
            row["parameters"]["runtime"] = runtime
            thresholds, metric_names, primary = _expected_verification(item, context)
            metrics = dict.fromkeys(metric_names, 0.0)
            if item["benchmark"] == "rocrand":
                metrics.update(observed_min=0.01, observed_max=0.99,
                               sample_mean=thresholds["sample_mean"]["expected_mean"],
                               second_central_moment_about_half=thresholds["second_central_moment_about_half"]["expected_second_central_moment"])
            row.update(attempted=True, failure_origin=None, elapsed_total_sec=0.01,
                       elapsed_sec=0.01 / row["repeat"], status="success", exit_code=0,
                       verification_status="pass", verification_thresholds=thresholds,
                       verification_metrics=metrics, verification_primary_metric=primary, message="",
                       measurement_start_timestamp="2026-07-14T00:00:00.000Z",
                       measurement_end_timestamp="2026-07-14T00:00:00.010Z",
                       library_version="fixture-version")
            if item["benchmark"] == "rocsolver":
                row.update(getrf_info=0, getrs_info=0)
            if row["implementation"] != "cpu":
                row["gpu_name"] = "inert fixture GPU identity"
                runtime.update(hip_runtime_version=70200000, hip_driver_version=70200000)
            rows.append(row)
    return rows, manifest, dict(context=context, config_hash=config_hash,
                               runtime_hash=runtime_hash, source_hash=SOURCE_HASH, git_diff_hash=None)


class SiriusAmdSmokeCampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="amd-smoke-campaign-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name).resolve()
        self.serial, self.threaded = self.work / "serial.json", self.work / "threaded.json"
        self.serial.write_bytes((ROOT / "jobs/sirius/amd-smoke.json").read_bytes())
        self.threaded.write_bytes((ROOT / "jobs/sirius/amd-threaded-smoke.json").read_bytes())
        self.configs = campaign.campaign_configs(self.serial, self.threaded)
        for name in ("subprocess.run", "subprocess.Popen"):
            patcher = mock.patch(name, side_effect=AssertionError("pure validation must not launch a process"))
            patcher.start()
            self.addCleanup(patcher.stop)

    def data(self, label="serial-jsonl"):
        config = self.configs[label]
        rows, manifest, arguments = fixture(config, label)
        return config, rows, manifest, arguments

    def check(self, config, rows, manifest, arguments):
        return campaign.validate_smoke_rows(rows, config, manifest, **arguments)

    def test_four_campaigns_verify_288_rows_in_both_formats(self):
        reports = {}
        for label, config in self.configs.items():
            rows, manifest, arguments = fixture(config, label)
            path = self.work / (label + "." + config["output_format"])
            with ExclusiveRawWriter(path, config["output_format"]) as writer:
                for row in rows:
                    writer.write(row)
            report = self.check(config, load_raw_results(path), manifest, arguments)
            self.assertEqual((report["status"], report["row_count"], report["invocation_count"]), ("PASS", 72, 36))
            self.assertEqual(report["library_counts"], dict.fromkeys(AMD_BENCHMARKS, 12))
            reports[label] = report
        combined = campaign.validate_campaign_set(reports)
        self.assertEqual((combined["status"], combined["row_count"], combined["campaign_count"]), ("PASS", 288, 4))

    def test_blas_only_96_rows_and_unknown_effective_threads(self):
        configs = campaign.campaign_configs(self.serial, self.threaded, library="rocblas")
        reports = {}
        for label, config in configs.items():
            self.assertFalse(config["continue_on_failure"])
            self.assertEqual(set(config["benchmarks"]), {"rocblas"})
            definition = config["benchmarks"]["rocblas"]
            self.assertEqual(definition["cases"][1]["parameters"],
                             {"size": 31, "alpha": 1.25, "beta": 0.5})
            self.assertIsNone(definition["series"][0]["cpu_threads_effective"])
            rows, manifest, arguments = fixture(config, label)
            reports[label] = campaign.validate_smoke_rows(rows, config, manifest, **arguments, library="rocblas")
        self.assertEqual(campaign.validate_campaign_set(reports, library="rocblas")["row_count"], 96)
        with self.assertRaises(ValueError):
            campaign.validate_campaign_set(reports)

    def test_remaining_four_libraries_384_rows_with_provider_thread_policies(self):
        libraries = ["rocsparse", "rocsolver", "rocrand", "rocthrust"]
        configs = campaign.campaign_configs(self.serial, self.threaded, library=libraries)
        reports = {}
        for label, config in configs.items():
            rows, manifest, arguments = fixture(config, label)
            self.assertEqual({row["benchmark"] for row in rows}, set(libraries))
            self.assertEqual(len(rows), 96)
            for row in rows:
                if row["implementation"] == "cpu":
                    self.assertEqual(row["cpu_threads_effective"],
                                     1 if row["benchmark"] in {"rocrand", "rocthrust"} else None)
            reports[label] = campaign.validate_smoke_rows(rows, config, manifest, **arguments, library=libraries)
            with self.assertRaises(ValueError):
                campaign.validate_smoke_rows(rows[:-1], config, manifest, **arguments, library=libraries)
        self.assertEqual(campaign.validate_campaign_set(reports, library=libraries)["row_count"], 384)

    def test_blas_rejects_missing_failed_nonfinite_or_wrong_provenance_rows(self):
        config = campaign.campaign_configs(self.serial, self.threaded, library="rocblas")["serial-jsonl"]
        rows, manifest, arguments = fixture(config, "serial-jsonl")
        for mutation in ("missing", "duplicate", "failed", "skipped", "metric", "nonfinite", "source", "runtime", "threads"):
            changed = copy.deepcopy(rows)
            if mutation == "missing": changed.pop()
            elif mutation == "duplicate": changed[-1] = copy.deepcopy(changed[0])
            elif mutation == "failed": changed[0]["status"] = "failure"
            elif mutation == "skipped": changed[0]["verification_status"] = "skipped"
            elif mutation in {"metric", "nonfinite"}:
                changed[0]["verification_metrics"]["max_abs_error"] = 1.0 if mutation == "metric" else float("nan")
            elif mutation == "source": changed[0]["source_snapshot_sha256"] = "d" * 64
            elif mutation == "runtime": changed[0]["runtime_environment_sha256"] = "d" * 64
            else: next(row for row in changed if row["implementation"] == "cpu")["cpu_threads_effective"] = 2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                campaign.validate_smoke_rows(changed, config, manifest, **arguments, library="rocblas")

    def test_explicit_config_files_are_independent_and_not_modified(self):
        before = self.serial.read_bytes(), self.threaded.read_bytes()
        self.configs["serial-csv"]["benchmarks"]["rocfft"]["cases"][0]["parameters"]["nfft"] = 128
        self.assertEqual(self.configs["serial-jsonl"]["benchmarks"]["rocfft"]["cases"][0]["parameters"]["nfft"], 32)
        self.assertEqual((self.serial.read_bytes(), self.threaded.read_bytes()), before)
        with self.assertRaises(FileNotFoundError):
            campaign.load_smoke_config(self.work / "missing", campaign="serial", output_format="jsonl")

    def test_config_rejects_incomplete_matrix_or_cpu_policy(self):
        for mutation in ("missing-family", "disabled", "extra-case", "scope", "trial", "warmup", "repeat",
                         "reference", "threads", "effective", "rand-threaded", "continuation", "mode", "format"):
            config = copy.deepcopy(self.configs["serial-jsonl"])
            definition = config["benchmarks"]["rocsparse"]
            if mutation == "missing-family": del config["benchmarks"]["rocthrust"]
            elif mutation == "disabled": definition["enabled"] = False
            elif mutation == "extra-case": definition["cases"].append(copy.deepcopy(definition["cases"][0]))
            elif mutation == "scope": del definition["cases"][0]["scopes"]["compute"]
            elif mutation in {"trial", "warmup", "repeat"}:
                definition["cases"][0]["scopes"]["compute"]["trials" if mutation == "trial" else mutation] = 3
            elif mutation == "reference": definition["series"][0]["cpu_backend"] = "reference"
            elif mutation == "threads": config["cpu_threads"] = 3
            elif mutation == "effective": definition["series"][0]["cpu_threads_effective"] = 1
            elif mutation == "rand-threaded": config["benchmarks"]["rocrand"]["series"][0]["cpu_threads_effective"] = 2
            elif mutation == "continuation": config["continue_on_failure"] = False
            elif mutation == "mode": config["run_mode"] = "production"
            else: config["output_format"] = "xml"
            self.serial.write_bytes(dump_bytes(config))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                campaign.load_smoke_config(self.serial, campaign="serial", output_format="jsonl")

    def test_config_rejects_duplicate_keys_and_unknown_campaign(self):
        self.serial.write_text('{"cpu_threads":1,"cpu_threads":2}')
        with self.assertRaises(ValueError):
            campaign.load_smoke_config(self.serial, campaign="serial", output_format="jsonl")
        for name, fmt in (("extra", "jsonl"), ("serial", "xml")):
            with self.subTest(name=name, fmt=fmt), self.assertRaises(ValueError):
                campaign.load_smoke_config(self.threaded, campaign=name, output_format=fmt)

    def test_rows_reject_missing_duplicate_and_out_of_range_trials(self):
        config, rows, manifest, arguments = self.data()
        for mutation in ("missing", "extra", "replace-duplicate", "trial"):
            changed = copy.deepcopy(rows)
            if mutation == "missing": changed.pop()
            elif mutation == "extra": changed.append(copy.deepcopy(changed[0]))
            elif mutation == "replace-duplicate": changed[-1] = copy.deepcopy(changed[0])
            else: changed[0]["trial"] = 2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.check(config, changed, manifest, arguments)

    def test_rows_bind_library_case_backend_and_thread_request(self):
        config, rows, manifest, arguments = self.data()
        for mutation in ("library", "case", "backend", "requested", "effective", "scope", "config"):
            changed = copy.deepcopy(rows)
            row = next(row for row in changed if row["benchmark"] == "rocfft" and row["implementation"] == "cpu")
            if mutation == "library": row["benchmark"] = "rocblas"
            elif mutation == "case": row["parameters"]["nfft"] = row["problem_size"] = 128
            elif mutation == "backend": row["cpu_backend"] = "cpu-fftw-threaded"
            elif mutation == "requested": row["cpu_threads_requested"] = 3
            elif mutation == "effective": row["cpu_threads_effective"] = 3
            elif mutation == "scope": row["scope"] = "end-to-end" if row["scope"] == "compute" else "compute"
            else: row["config_sha256"] = "d" * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.check(config, changed, manifest, arguments)

    def test_rows_reject_uniform_but_wrong_identity(self):
        config, rows, manifest, arguments = self.data()
        for field in ("run_id", "runtime_environment_sha256", "source_snapshot_sha256", "git_diff_sha256",
                      "hostname", "scheduler_job_id", "binary_sha256", "compiler_version"):
            changed = copy.deepcopy(rows)
            for row in changed:
                row[field] = "d" * 64 if field.endswith("sha256") else "different"
                if field in {"run_id", "hostname"}:
                    row["block_id"] = row["run_id"] + "|0|" + row["hostname"]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(config, changed, manifest, arguments)

    def test_manifest_requires_all_supported_unambiguous_artifacts(self):
        config, rows, manifest, arguments = self.data()
        for mutation in ("missing", "duplicate", "wrong-backend", "changed-binary"):
            changed = copy.deepcopy(manifest)
            if mutation == "missing": changed["entries"].pop()
            elif mutation == "duplicate": changed["entries"].append(copy.deepcopy(changed["entries"][0]))
            else:
                entry = next(item for item in changed["entries"] if item["implementation"] == "cpu")
                if mutation == "wrong-backend": entry["supported_cpu_backends"] = []
                else: entry["binary_sha256"] = "d" * 64
                entry["artifact_id"] = artifact_id(entry)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.check(config, rows, changed, arguments)

    def test_numerical_values_are_checked_for_all_families(self):
        config, rows, manifest, arguments = self.data()
        for family in AMD_BENCHMARKS:
            prototype = next(row for row in rows if row["benchmark"] == family)
            for metric in prototype["verification_metrics"]:
                changed = copy.deepcopy(rows)
                row = next(row for row in changed if row["benchmark"] == family)
                row["verification_metrics"][metric] = 100.0
                with self.subTest(family=family, metric=metric), self.assertRaises(ValueError):
                    self.check(config, changed, manifest, arguments)

    def test_numerical_fields_cannot_fake_a_pass(self):
        config, rows, manifest, arguments = self.data()
        for mutation in ("null", "nan", "negative", "missing", "threshold", "verification-skipped"):
            changed = copy.deepcopy(rows)
            row = next(row for row in changed if row["benchmark"] == "rocfft")
            if mutation == "null": row["verification_metrics"]["dc_relative_error"] = None
            elif mutation == "nan": row["verification_metrics"]["dc_relative_error"] = float("nan")
            elif mutation == "negative": row["verification_metrics"]["dc_relative_error"] = -1.0
            elif mutation == "missing": del row["verification_metrics"]["dc_relative_error"]
            elif mutation == "threshold": row["verification_thresholds"]["dc_relative_error"]["upper_bound"] = 1.0
            else: row["verification_status"] = "skipped"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.check(config, changed, manifest, arguments)

    def test_solver_status_cannot_be_nonzero_or_unknown(self):
        config, rows, manifest, arguments = self.data()
        for field in ("getrf_info", "getrs_info"):
            for value in (None, 1, -1):
                changed = copy.deepcopy(rows)
                next(row for row in changed if row["benchmark"] == "rocsolver")[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.check(config, changed, manifest, arguments)

    def test_synthetic_failure_and_skip_never_complete_smoke(self):
        config, rows, manifest, arguments = self.data()
        for attempted in (True, False):
            changed = copy.deepcopy(rows)
            changed[0].update(status="failure" if attempted else "skipped", attempted=attempted,
                              failure_origin="parse" if attempted else "prerequisite", exit_code=1 if attempted else None,
                              verification_status="skipped", verification_metrics={}, verification_thresholds={},
                              verification_primary_metric=None, measurement_start_timestamp=None,
                              measurement_end_timestamp=None, elapsed_total_sec=None, elapsed_sec=None, message="fixture")
            with self.subTest(attempted=attempted), self.assertRaises(ValueError):
                self.check(config, changed, manifest, arguments)

    def test_aocl_environment_is_independent_of_rows_and_validator_host(self):
        config, rows, manifest, arguments = self.data("threaded-jsonl")
        with mock.patch.dict(os.environ, {"AOCL_ENABLE_INSTRUCTIONS": "unrelated-validator-host", "OMP_NUM_THREADS": "99"}):
            self.assertEqual(self.check(config, rows, manifest, arguments)["status"], "PASS")
        for mutation in ("row", "missing-context", "context-controls"):
            changed, kwargs = copy.deepcopy(rows), copy.deepcopy(arguments)
            if mutation == "row":
                row = next(row for row in changed if row["benchmark"] == "rocsparse" and row["implementation"] == "cpu")
                row["parameters"]["runtime"]["aocl_sparse_environment"]["AOCL_ENABLE_INSTRUCTIONS"] = "AVX2"
            elif mutation == "missing-context": del kwargs["context"]["aocl_sparse_environment"]
            else: kwargs["context"]["aocl_sparse_environment"]["OMP_NUM_THREADS"] = "99"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.check(config, changed, manifest, kwargs)

    def test_aggregate_rejects_missing_failed_or_mixed_reports(self):
        reports = {label: self.check(*self.data(label)) for label in self.configs}
        for mutation in ("missing", "failed", "rows", "duplicate-run", "source", "manifest", "label"):
            changed = copy.deepcopy(reports)
            report = changed["threaded-csv"]
            if mutation == "missing": del changed["serial-csv"]
            elif mutation == "failed": report["status"] = "FAIL"
            elif mutation == "rows": report["row_count"] = 71
            elif mutation == "duplicate-run":
                report["run_id"] = report["validation"]["run_id"] = changed["serial-jsonl"]["run_id"]
            elif mutation == "source": report["source_snapshot_sha256"] = "d" * 64
            elif mutation == "manifest": report["manifest_sha256"] = "d" * 64
            else: report["campaign"] = "serial-jsonl"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                campaign.validate_campaign_set(changed)


if __name__ == "__main__":
    unittest.main()

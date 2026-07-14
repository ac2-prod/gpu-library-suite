import copy
import tempfile
import unittest
from pathlib import Path

import validate_results

from gpu_suite.hashing import sha256_file
from gpu_suite.ordering import implementation_order
from gpu_suite.runner import build_schedule, synthetic_result
from gpu_suite.strict_json import dump_bytes, dumps, load
from gpu_suite.validation import CampaignValidationError, validate_campaign

from support import ROOT, ZERO_HASH, pilot_config


def prerequisite_campaign():
    config = pilot_config()
    context = {
        "config": config,
        "cpu_threads": 48,
        "device": 0,
        "hostname": "node0",
        "implementation_order": implementation_order(0, 0),
        "node_index": 0,
        "run_id": "run-1",
        "scheduler": None,
        "scheduler_job_id": None,
        "system_label": "local",
        "wave": 0,
    }
    manifest = {"entries": [], "manifest_schema_version": 1}
    records = []
    for item in build_schedule(config, manifest, context):
        for trial in range(item["scope_settings"]["trials"]):
            records.append(synthetic_result(
                item, context, ZERO_HASH, ZERO_HASH, {}, trial, False,
                "prerequisite", "not built", None, None, None,
            ))
    return config, manifest, records


class ValidationTests(unittest.TestCase):
    def test_recovered_verification_failure_is_campaign_valid(self):
        config = pilot_config()
        config["benchmarks"]["cufft"]["cases"][0]["scopes"]["compute"][
            "trials"
        ] = 3
        entry = {
            "artifact_id": "a" * 64,
            "target_name": "fft_cpu_bench",
            "build_profile": "cpu-cuda",
            "backend_variant": "fftw-threaded",
            "executable_path": "/not/executed/fft_cpu_bench",
            "library": "cufft",
            "implementation": "cpu",
            "executable_role": "benchmark",
            "build_type": "Release",
            "binary_sha256": "b" * 64,
            "build_metadata_sha256": "c" * 64,
            "compiler": "TestCompiler",
            "compiler_language": "c",
            "compiler_version": "1.0",
            "global_configure_flags": "-O3",
            "git_metadata_available": True,
            "git_commit": "abc",
            "git_dirty": False,
            "supported_cpu_backends": [
                "cpu-fftw-threaded", "cpu-fftw-serial"
            ],
        }
        manifest = {"entries": [entry], "manifest_schema_version": 1}
        metadata = {
            "c" * 64: {
                "c": {
                    "compiler": "TestCompiler",
                    "global_configure_flags": "-O3",
                    "compiler_version": "1.0",
                }
            }
        }
        context = {
            "config": config,
            "cpu_threads": 48,
            "device": 0,
            "hostname": "node0",
            "implementation_order": implementation_order(0, 0),
            "node_index": 0,
            "run_id": "run-1",
            "scheduler": None,
            "scheduler_job_id": None,
            "system_label": "local",
            "wave": 0,
        }
        records = []
        for item in build_schedule(config, manifest, context):
            selected = (
                item["benchmark"] == "cufft"
                and item["scope"] == "compute"
                and item["series"]["cpu_backend"] == "cpu-fftw-threaded"
            )
            for trial in range(item["scope_settings"]["trials"]):
                if selected:
                    origin = "verification" if trial == 0 else "benchmark"
                    record = synthetic_result(
                        item, context, ZERO_HASH, ZERO_HASH, metadata, trial,
                        True, origin, "temporary", 1, None, None,
                    )
                    record.update({
                        "measurement_start_timestamp":
                            "2026-07-14T00:00:00.000Z",
                        "measurement_end_timestamp":
                            "2026-07-14T00:00:01.000Z",
                        "elapsed_total_sec": 2.0,
                        "elapsed_sec": 1.0,
                    })
                    if trial == 0:
                        record.update({
                            "failure_origin": "verification",
                            "verification_status": "failure",
                            "status": "failure",
                            "message": "recoverable verification failure",
                        })
                    else:
                        record.update({
                            "failure_origin": None,
                            "verification_status": "pass",
                            "exit_code": 0,
                            "status": "success",
                            "message": "",
                        })
                else:
                    record = synthetic_result(
                        item, context, ZERO_HASH, ZERO_HASH, metadata, trial,
                        False, "prerequisite", "not built", None, None, None,
                    )
                records.append(record)
        report = validate_campaign(records, config, ZERO_HASH, manifest)
        self.assertEqual(report["validation_status"], "failure")
        self.assertEqual(report["status_counts"]["failure"], 1)
        self.assertEqual(report["status_counts"]["success"], 2)

    def test_complete_prerequisite_campaign_is_valid(self):
        config, manifest, records = prerequisite_campaign()
        report = validate_campaign(records, config, ZERO_HASH, manifest)
        self.assertEqual(report["validation_status"], "failure")
        self.assertEqual(report["block_count"], 1)
        self.assertEqual(report["record_count"], 38)
        self.assertEqual(report["status_counts"], {"skipped": 38})

    def test_validation_cli_reports_complete_failed_campaign_as_failure(self):
        _config, manifest, records = prerequisite_campaign()
        config_path = ROOT / "configs" / "pilot.json"
        config_sha256 = sha256_file(config_path)
        for record in records:
            record["config_sha256"] = config_sha256
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = directory / "manifest.json"
            raw_path = directory / "raw-results.jsonl"
            report_path = directory / "validation-report.json"
            manifest_path.write_bytes(dump_bytes(manifest))
            raw_path.write_text(
                "".join(dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            result = validate_results.main([
                "--config", str(config_path),
                "--manifest", str(manifest_path),
                "--report-output", str(report_path),
                str(raw_path),
            ])
            self.assertEqual(result, 1)
            self.assertEqual(load(report_path)["validation_status"], "failure")

    def test_duplicate_missing_and_mixed_provenance_are_rejected(self):
        config, manifest, records = prerequisite_campaign()
        with self.assertRaises(CampaignValidationError):
            validate_campaign(records + [copy.deepcopy(records[0])],
                              config, ZERO_HASH, manifest)
        with self.assertRaises(CampaignValidationError):
            validate_campaign(records[:-1], config, ZERO_HASH, manifest)
        mixed = copy.deepcopy(records)
        mixed[-1]["runtime_environment_sha256"] = "1" * 64
        with self.assertRaises(CampaignValidationError):
            validate_campaign(mixed, config, ZERO_HASH, manifest)

    def test_ordering_formula_mismatch_is_rejected(self):
        config, manifest, records = prerequisite_campaign()
        changed = copy.deepcopy(records)
        changed[0]["implementation_order"] = ["cuda", "cpu", "openacc"]
        with self.assertRaises(CampaignValidationError):
            validate_campaign(changed, config, ZERO_HASH, manifest)


if __name__ == "__main__":
    unittest.main()

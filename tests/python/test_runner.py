import contextlib
import copy
import io
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_suite
from gpu_suite.hashing import sha256_bytes, sha256_file
from gpu_suite.manifest import artifact_id, merge_entries
from gpu_suite.ordering import implementation_order
from gpu_suite.results_io import load_raw_results
from gpu_suite.runner import (
    RunnerError,
    build_schedule,
    execute_schedule,
    load_manifest,
    synthetic_result,
    validate_subprocess_record,
)
from gpu_suite.schema import validate_raw_result
from gpu_suite.strict_json import dump_bytes, dumps, load, loads
from gpu_suite.validation import validate_campaign

from support import ROOT, ZERO_HASH, pilot_config


def cpu_integration_fixture_config():
    config = pilot_config()
    parameters = {
        "cufft": {"batch": 8, "nfft": 256},
        "cublas": {"size": 128},
        "cusparse": {"size": 4096},
        "cusolver": {"nrhs": 2, "size": 64},
        "curand": {"size": 65536},
        "thrust": {"size": 65536},
    }
    for benchmark, overrides in parameters.items():
        case = copy.deepcopy(config["benchmarks"][benchmark]["cases"][0])
        case["parameters"].update(overrides)
        case["scopes"] = {
            "compute": {
                "repeat": 1 if benchmark == "cusolver" else 2,
                "trials": 1,
                "warmup": 1,
            },
            "end-to-end": {"repeat": 1, "trials": 1, "warmup": 1},
        }
        config["benchmarks"][benchmark]["cases"] = [case]
    return config


def execution_fixture(trials=3, continue_on_failure=True):
    config = pilot_config()
    config["continue_on_failure"] = continue_on_failure
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
        "supported_cpu_backends": ["cpu-fftw-threaded", "cpu-fftw-serial"],
    }
    metadata = {
        entry["build_metadata_sha256"]: {
            "build_profile": "cpu-cuda",
            "build_type": "Release",
            "git_metadata_available": True,
            "git_commit": "abc",
            "git_dirty": False,
            "c": {
                "compiler": "TestCompiler",
                "global_configure_flags": "-O3",
                "compiler_version": "1.0",
            },
        }
    }
    case = config["benchmarks"]["cufft"]["cases"][0]
    item = {
        "artifact_id": entry["artifact_id"],
        "argv": [entry["executable_path"], "--output", "-"],
        "benchmark": "cufft",
        "case_index": 0,
        "entry": entry,
        "parameters": dict(case["parameters"]),
        "prerequisite_failure": None,
        "scope": "compute",
        "scope_settings": {
            "warmup": 1,
            "repeat": 2,
            "trials": trials,
        },
        "series": {
            "implementation": "cpu",
            "cpu_backend": "cpu-fftw-threaded",
            "cpu_backend_role": "production",
            "cpu_parallelism": "threaded",
            "cpu_threads_effective": 48,
            "series_role": "primary",
        },
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
    return item, context, metadata


def successful_record(item, context, metadata, trial):
    record = synthetic_result(
        item, context, ZERO_HASH, ZERO_HASH, metadata, trial, True,
        "subprocess", "temporary", 1, None, None,
    )
    record.update({
        "record_timestamp": "2026-07-14T00:00:01.000Z",
        "measurement_start_timestamp": "2026-07-14T00:00:00.000Z",
        "measurement_end_timestamp": "2026-07-14T00:00:01.000Z",
        "attempted": True,
        "failure_origin": None,
        "elapsed_total_sec": 2.0,
        "elapsed_sec": 1.0,
        "cpu_threads_effective": 48,
        "cpu_parallelism": "threaded",
        "verification_metrics": {
            "dc_relative_error": 0.0,
            "non_dc_max_abs_error": 0.0,
        },
        "verification_thresholds": {
            "dc_relative_error": {
                "method": "relative-upper-bound",
                "upper_bound": 1e-5,
            },
            "non_dc_max_abs_error": {
                "method": "absolute-upper-bound",
                "upper_bound": 1e-4,
            },
        },
        "verification_primary_metric": "non_dc_max_abs_error",
        "verification_status": "pass",
        "exit_code": 0,
        "status": "success",
        "message": "",
    })
    return validate_raw_result(record)


def verification_failure_record(item, context, metadata, trial):
    record = successful_record(item, context, metadata, trial)
    record.update({
        "failure_origin": "verification",
        "verification_status": "failure",
        "exit_code": 1,
        "status": "failure",
        "message": "recoverable verification failure",
    })
    return validate_raw_result(record)


class RunnerTests(unittest.TestCase):
    def assert_subprocess_mismatch(self, mutate, field):
        item, context, metadata = execution_fixture()
        record = successful_record(item, context, metadata, 0)
        mutate(record)
        validate_raw_result(record)
        with self.assertRaisesRegex(RunnerError, field):
            validate_subprocess_record(
                record, item, context, ZERO_HASH, ZERO_HASH, None, None
            )

    def test_rejects_requested_thread_mismatch(self):
        self.assert_subprocess_mismatch(
            lambda record: record.__setitem__("cpu_threads_requested", 47),
            "cpu_threads_requested",
        )

    def test_rejects_effective_thread_mismatch(self):
        self.assert_subprocess_mismatch(
            lambda record: record.__setitem__("cpu_threads_effective", 47),
            "cpu_threads_effective",
        )

    def test_rejects_device_mismatch(self):
        self.assert_subprocess_mismatch(
            lambda record: record.__setitem__("device_id", 1), "device_id"
        )

    def test_rejects_precision_mismatch(self):
        self.assert_subprocess_mismatch(
            lambda record: record.__setitem__("precision", "fp64"), "precision"
        )

    def test_rejects_problem_size_mismatch(self):
        self.assert_subprocess_mismatch(
            lambda record: record.__setitem__("problem_size", 257),
            "problem_size",
        )

    def test_rejects_secondary_size_mismatch(self):
        self.assert_subprocess_mismatch(
            lambda record: record.__setitem__("secondary_size", 9),
            "secondary_size",
        )

    def test_rejects_unexpected_parameter(self):
        def mutate(record):
            record["parameters"]["unowned_parameter"] = 1

        self.assert_subprocess_mismatch(mutate, "parameters")

    def test_rejects_scheduler_mismatch(self):
        item, context, metadata = execution_fixture()
        context["scheduler"] = "NQSV"
        context["scheduler_job_id"] = "0:866211.nqsv"
        record = successful_record(item, context, metadata, 0)
        record["scheduler"] = "PBS"
        validate_raw_result(record)
        with self.assertRaisesRegex(RunnerError, "scheduler"):
            validate_subprocess_record(
                record, item, context, ZERO_HASH, ZERO_HASH, None, None
            )

    def test_synthetic_failure_preserves_raw_nqsv_scheduler_job_id(self):
        item, context, metadata = execution_fixture()
        context["scheduler"] = "NQSV"
        context["scheduler_job_id"] = "0:866211.nqsv"
        record = synthetic_result(
            item, context, ZERO_HASH, ZERO_HASH, metadata, 0, True,
            "subprocess", "fixture failure", 2, None, None,
        )
        self.assertEqual(record["scheduler"], "NQSV")
        self.assertEqual(record["scheduler_job_id"], "0:866211.nqsv")

    def test_rejects_source_hash_mismatch(self):
        self.assert_subprocess_mismatch(
            lambda record: record.__setitem__(
                "source_snapshot_sha256", "d" * 64
            ),
            "source_snapshot_sha256",
        )

    def test_rejects_verification_threshold_mismatch(self):
        def mutate(record):
            record["verification_thresholds"]["dc_relative_error"][
                "upper_bound"
            ] = 2e-5

        self.assert_subprocess_mismatch(mutate, "verification thresholds")

    def test_rejects_single_ulp_config_threshold_operand_mismatch(self):
        def mutate(record):
            threshold = record["verification_thresholds"][
                "dc_relative_error"
            ]
            threshold["upper_bound"] = math.nextafter(
                threshold["upper_bound"], math.inf
            )

        self.assert_subprocess_mismatch(
            mutate, "verification threshold operand mismatch"
        )

    def test_rejects_library_mismatch(self):
        self.assert_subprocess_mismatch(
            lambda record: record.__setitem__("library_name", "wrong-library"),
            "library_name",
        )

    def test_duplicate_benchmark_manifest_entry_is_defensively_rejected(self):
        _, context, _ = execution_fixture()
        entry = {
            "artifact_id": "a" * 64,
            "executable_role": "benchmark",
            "implementation": "cpu",
            "library": "cufft",
        }
        duplicate = dict(entry)
        duplicate["artifact_id"] = "b" * 64
        with self.assertRaisesRegex(
            RunnerError, "duplicate benchmark manifest entry: cufft/cpu"
        ):
            build_schedule(
                pilot_config(),
                {"entries": [entry, duplicate], "manifest_schema_version": 1},
                context,
            )

    def test_schedule_uses_config_owned_cpu_metadata_and_thresholds(self):
        item, context, _ = execution_fixture()
        schedule = build_schedule(
            pilot_config(),
            {"entries": [item["entry"]], "manifest_schema_version": 1},
            context,
        )
        threaded = next(
            candidate for candidate in schedule
            if candidate["benchmark"] == "cufft"
            and candidate["series"]["cpu_backend"] == "cpu-fftw-threaded"
            and candidate["scope"] == "compute"
        )
        command = threaded["argv"]

        def value(name):
            return command[command.index(name) + 1]

        self.assertEqual(value("--cpu-backend-role"), "production")
        self.assertEqual(value("--series-role"), "primary")
        self.assertEqual(value("--cpu-parallelism"), "threaded")
        self.assertEqual(value("--cpu-threads-effective"), "48")
        self.assertEqual(value("--abs-tolerance"), "0.0001")
        self.assertEqual(value("--rel-tolerance"), "1e-05")

    def test_verification_failure_can_be_followed_by_restored_trials(self):
        item, context, metadata = execution_fixture()
        emitted = [
            verification_failure_record(item, context, metadata, 0),
            successful_record(item, context, metadata, 1),
            successful_record(item, context, metadata, 2),
        ]
        completed = SimpleNamespace(
            returncode=1,
            stdout="".join(dumps(record) + "\n" for record in emitted),
            stderr="",
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "raw.jsonl"
            with mock.patch("gpu_suite.runner.subprocess.run", return_value=completed):
                status = execute_schedule(
                    [item], context, output, ZERO_HASH, ZERO_HASH,
                    metadata, None, None,
                )
            records = load_raw_results(output)
        self.assertEqual(status, 1)
        self.assertEqual(len(records), 3)
        self.assertEqual(
            [record["status"] for record in records],
            ["failure", "success", "success"],
        )
        self.assertTrue(all(record["attempted"] for record in records))

    def test_contamination_preserves_completed_and_synthesizes_remainder(self):
        item, context, metadata = execution_fixture()
        stdout = dumps(successful_record(item, context, metadata, 0)) + "\nprogress\n"
        completed = SimpleNamespace(returncode=0, stdout=stdout, stderr="diagnostic")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "raw.jsonl"
            with mock.patch("gpu_suite.runner.subprocess.run", return_value=completed):
                with contextlib.redirect_stderr(io.StringIO()):
                    status = execute_schedule(
                        [item], context, output, ZERO_HASH, ZERO_HASH,
                        metadata, None, None,
                    )
            records = load_raw_results(output)
        self.assertEqual(status, 1)
        self.assertEqual([record["trial"] for record in records], [0, 1, 2])
        self.assertEqual([record["status"] for record in records],
                         ["success", "failure", "skipped"])
        self.assertEqual(records[1]["failure_origin"], "output-validation")
        self.assertEqual(records[2]["failure_origin"], "prior-failure")

    def test_signal_creates_one_attempted_failure_and_later_skips(self):
        item, context, metadata = execution_fixture()
        completed = SimpleNamespace(returncode=-9, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "raw.jsonl"
            with mock.patch("gpu_suite.runner.subprocess.run", return_value=completed):
                status = execute_schedule(
                    [item], context, output, ZERO_HASH, ZERO_HASH,
                    metadata, None, None,
                )
            records = load_raw_results(output)
        self.assertEqual(status, 1)
        self.assertEqual(sum(record["attempted"] for record in records), 1)
        self.assertEqual(records[0]["failure_origin"], "subprocess")
        self.assertEqual(records[0]["exit_code"], 137)
        self.assertTrue(all(record["status"] == "skipped" for record in records[1:]))

    def test_success_exit_with_missing_stdout_is_output_failure(self):
        item, context, metadata = execution_fixture(trials=2)
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "raw.jsonl"
            with mock.patch("gpu_suite.runner.subprocess.run", return_value=completed):
                status = execute_schedule(
                    [item], context, output, ZERO_HASH, ZERO_HASH,
                    metadata, None, None,
                )
            records = load_raw_results(output)
        self.assertEqual(status, 1)
        self.assertEqual(records[0]["status"], "failure")
        self.assertEqual(records[0]["failure_origin"], "output-validation")
        self.assertEqual(records[1]["status"], "skipped")

    def test_stop_policy_marks_later_invocations_prior_failure(self):
        item, context, metadata = execution_fixture(trials=2, continue_on_failure=False)
        later = copy.deepcopy(item)
        later["scope"] = "end-to-end"
        later["scope_settings"] = {"warmup": 1, "repeat": 1, "trials": 2}
        completed = SimpleNamespace(returncode=1, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "raw.jsonl"
            with mock.patch("gpu_suite.runner.subprocess.run", return_value=completed) as launched:
                execute_schedule(
                    [item, later], context, output, ZERO_HASH, ZERO_HASH,
                    metadata, None, None,
                )
            records = load_raw_results(output)
        self.assertEqual(launched.call_count, 1)
        self.assertEqual([record["failure_origin"] for record in records[2:]],
                         ["prior-failure", "prior-failure"])
        self.assertTrue(all(not record["attempted"] for record in records[2:]))

    def test_dry_run_is_deterministic_and_side_effect_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            executable = directory / "fft_cpu_bench"
            executable.write_bytes(b"test executable\n")
            metadata_document = {
                "build_profile": "cpu-cuda",
                "build_type": "Release",
                "c": {
                    "compiler": "TestCompiler",
                    "global_configure_flags": "-O3",
                    "compiler_version": "1.0",
                },
                "git_metadata_available": True,
                "git_commit": "abc",
                "git_dirty": False,
            }
            metadata_path = directory / "build-metadata.json"
            metadata_path.write_bytes(dump_bytes(metadata_document))
            entry = {
                "artifact_id": "",
                "target_name": "fft_cpu_bench",
                "build_profile": "cpu-cuda",
                "backend_variant": "fftw-threaded",
                "executable_path": str(executable),
                "library": "cufft",
                "implementation": "cpu",
                "executable_role": "benchmark",
                "build_type": "Release",
                "binary_sha256": sha256_file(executable),
                "build_metadata_sha256": sha256_file(metadata_path),
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
            entry["artifact_id"] = artifact_id(entry)
            manifest_path = directory / "executables.json"
            manifest_path.write_bytes(dump_bytes({
                "entries": [entry], "manifest_schema_version": 1
            }))
            output = directory / "not-created" / "raw.jsonl"
            arguments = [
                "--config", str(ROOT / "configs" / "pilot.json"),
                "--manifest", str(manifest_path),
                "--build-metadata", str(metadata_path),
                "--output", str(output),
                "--run-id", "run-1",
                "--system-label", "local",
                "--wave", "0",
                "--node-index", "0",
                "--hostname", "node0",
                "--runtime-environment-sha256", ZERO_HASH,
                "--dry-run",
            ]

            def snapshot():
                return {
                    str(path.relative_to(directory)): path.read_bytes()
                    for path in directory.rglob("*") if path.is_file()
                }

            before = snapshot()
            outputs = []
            with mock.patch("run_suite.execute_schedule") as execute:
                for _ in range(2):
                    stream = io.StringIO()
                    with contextlib.redirect_stdout(stream):
                        self.assertEqual(run_suite.main(arguments), 0)
                    outputs.append(stream.getvalue())
            self.assertFalse(execute.called)
            self.assertEqual(before, snapshot())
            self.assertFalse(output.exists())
            self.assertEqual(outputs[0], outputs[1])
            document = loads(outputs[0])
            self.assertEqual(document["dry_run_schema_version"], 1)
            self.assertTrue(all(
                command["argv"] is None or
                command["argv"][command["argv"].index("--output") + 1] == "-"
                for command in document["commands"]
            ))

    @unittest.skipUnless(
        os.environ.get("GPU_SUITE_FFTW_FIXTURE_MANIFEST") and
        os.environ.get("GPU_SUITE_FFTW_FIXTURE_METADATA") and
        os.environ.get("GPU_SUITE_CPU_PROVIDER_FIXTURE_MANIFEST") and
        os.environ.get("GPU_SUITE_CPU_PROVIDER_FIXTURE_METADATA"),
        "synthetic suite manifest is not set",
    )
    def test_real_fixture_runs_through_single_writer_and_validates(self):
        fftw_manifest_path = Path(
            os.environ["GPU_SUITE_FFTW_FIXTURE_MANIFEST"]
        )
        fftw_metadata_path = Path(
            os.environ["GPU_SUITE_FFTW_FIXTURE_METADATA"]
        )
        cpu_manifest_path = Path(
            os.environ["GPU_SUITE_CPU_PROVIDER_FIXTURE_MANIFEST"]
        )
        cpu_metadata_path = Path(
            os.environ["GPU_SUITE_CPU_PROVIDER_FIXTURE_METADATA"]
        )
        fftw_manifest = load(fftw_manifest_path)
        cpu_manifest = load(cpu_manifest_path)
        fftw_libraries = {"cufft", "curand", "thrust"}
        cpu_provider_libraries = {"cublas", "cusparse", "cusolver"}
        entries = merge_entries([
            [
                entry for entry in fftw_manifest["entries"]
                if entry["library"] in fftw_libraries
            ],
            [
                entry for entry in cpu_manifest["entries"]
                if entry["library"] in cpu_provider_libraries
            ],
        ])
        self.assertEqual(
            {entry["library"] for entry in entries},
            fftw_libraries | cpu_provider_libraries,
        )
        config = cpu_integration_fixture_config()
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "fixture-config.json"
            config_path.write_bytes(dump_bytes(config))
            config_sha256 = sha256_file(config_path)
            manifest_path = Path(temporary) / "executables.json"
            manifest_path.write_bytes(dump_bytes({
                "build_metadata_sha256s": sorted({
                    fftw_manifest["build_metadata_sha256"],
                    cpu_manifest["build_metadata_sha256"],
                }),
                "entries": entries,
                "manifest_schema_version": 1,
            }))
            output = Path(temporary) / "raw-results.jsonl"
            arguments = [
                "--config", str(config_path),
                "--manifest", str(manifest_path),
                "--build-metadata", str(fftw_metadata_path),
                "--build-metadata", str(cpu_metadata_path),
                "--output", str(output),
                "--run-id", "fixture-run",
                "--system-label", "local-fixture",
                "--wave", "0",
                "--node-index", "0",
                "--runtime-environment-sha256", ZERO_HASH,
                "--source-snapshot-sha256", ZERO_HASH,
            ]
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(run_suite.main(arguments), 1)
            records = load_raw_results(output)
            manifest, _ = load_manifest(manifest_path)
        self.assertEqual(len(records), 36)
        self.assertEqual(len({
            (record["benchmark"], record["implementation"],
             record["cpu_backend"], record["scope"], record["trial"])
            for record in records
        }), 36)
        self.assertTrue(any(record["status"] == "success" for record in records))
        self.assertTrue(any(record["status"] == "skipped" for record in records))
        attempted = {
            (
                record["benchmark"], record["implementation"],
                record["cpu_backend"], record["scope"],
            )
            for record in records if record["attempted"]
        }
        self.assertEqual(attempted, {
            ("cufft", "cpu", "cpu-fftw-threaded", "compute"),
            ("cufft", "cpu", "cpu-fftw-threaded", "end-to-end"),
            ("cublas", "cpu", "cpu-onemkl", "compute"),
            ("cublas", "cpu", "cpu-onemkl", "end-to-end"),
            ("cusparse", "cpu", "cpu-onemkl", "compute"),
            ("cusparse", "cpu", "cpu-onemkl", "end-to-end"),
            ("cusolver", "cpu", "cpu-onemkl", "compute"),
            ("cusolver", "cpu", "cpu-onemkl", "end-to-end"),
            ("curand", "cpu", "cpu-std-random-serial", "compute"),
            ("curand", "cpu", "cpu-std-random-serial", "end-to-end"),
            ("thrust", "cpu", "cpu-stl-serial", "compute"),
            ("thrust", "cpu", "cpu-stl-serial", "end-to-end"),
        })
        self.assertTrue(all(
            record["status"] == "success"
            for record in records if record["attempted"]
        ))
        report = validate_campaign(
            records, config, config_sha256, manifest
        )
        self.assertEqual(report["validation_status"], "failure")


if __name__ == "__main__":
    unittest.main()

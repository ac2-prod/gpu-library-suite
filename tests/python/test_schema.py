import copy
import unittest

from gpu_suite.schema import SchemaError, validate_raw_result, validate_trial_indices


ZERO_HASH = "0" * 64


def success_record(trial=0):
    return {
        "result_schema_version": 1,
        "run_id": "run-1",
        "record_timestamp": "2026-07-14T00:00:00.011Z",
        "measurement_start_timestamp": "2026-07-14T00:00:00.000Z",
        "measurement_end_timestamp": "2026-07-14T00:00:00.010Z",
        "system_label": "local",
        "wave": 0,
        "node_index": 0,
        "hostname": "node0",
        "block_id": "run-1|0|node0",
        "scheduler": None,
        "scheduler_job_id": None,
        "implementation_order": ["cpu", "cuda", "openacc"],
        "benchmark": "cufft",
        "implementation": "cpu",
        "scope": "compute",
        "problem_size": 256,
        "secondary_size": 8,
        "parameters": {"batch": 8, "nfft": 256, "transform": "c2c-forward"},
        "precision": "fp32",
        "cpu_backend": "cpu-fftw-threaded",
        "cpu_backend_role": "production",
        "series_role": "primary",
        "cpu_threads_requested": 48,
        "cpu_threads_effective": 48,
        "cpu_parallelism": "threaded",
        "warmup": 1,
        "repeat": 2,
        "trial": trial,
        "attempted": True,
        "failure_origin": None,
        "elapsed_total_sec": 0.01,
        "elapsed_sec": 0.005,
        "clock_id": "CLOCK_MONOTONIC",
        "clock_resolution_sec": 1e-9,
        "verification_metrics": {},
        "verification_thresholds": {},
        "verification_primary_metric": None,
        "verification_status": "pass",
        "getrf_info": None,
        "getrs_info": None,
        "device_id": None,
        "gpu_name": None,
        "gpu_uuid": None,
        "cuda_driver_version": None,
        "compiler": "AppleClang",
        "compiler_version": "21",
        "global_configure_flags": "-O3",
        "library_name": "fftw3f_threads",
        "library_version": None,
        "cuda_runtime_version": None,
        "git_metadata_available": True,
        "git_commit": "abc",
        "git_dirty": True,
        "git_diff_sha256": ZERO_HASH,
        "source_snapshot_sha256": None,
        "config_sha256": ZERO_HASH,
        "runtime_environment_sha256": ZERO_HASH,
        "binary_sha256": ZERO_HASH,
        "exit_code": 0,
        "status": "success",
        "message": "",
    }


class SchemaTests(unittest.TestCase):
    def test_success_and_failure_and_skipped(self):
        record = success_record()
        self.assertEqual(validate_raw_result(record)["status"], "success")

        failure = copy.deepcopy(record)
        failure["failure_origin"] = "verification"
        failure["verification_status"] = "failure"
        failure["status"] = "failure"
        failure["message"] = "verification failed"
        self.assertEqual(validate_raw_result(failure)["status"], "failure")

        skipped = copy.deepcopy(record)
        skipped.update(
            {
                "attempted": False,
                "failure_origin": "prior-failure",
                "measurement_start_timestamp": None,
                "measurement_end_timestamp": None,
                "elapsed_total_sec": None,
                "elapsed_sec": None,
                "verification_status": "skipped",
                "exit_code": None,
                "status": "skipped",
                "message": "not started",
            }
        )
        self.assertEqual(validate_raw_result(skipped)["status"], "skipped")

    def test_invalid_attempt_and_unknown_field(self):
        record = success_record()
        record["attempted"] = False
        with self.assertRaises(SchemaError):
            validate_raw_result(record)
        record = success_record()
        record["unexpected"] = 1
        with self.assertRaises(SchemaError):
            validate_raw_result(record)

    def test_trial_indices(self):
        records = [success_record(0), success_record(1)]
        validate_trial_indices(records, 2)
        with self.assertRaises(SchemaError):
            validate_trial_indices([records[0], records[0]], 2)
        with self.assertRaises(SchemaError):
            validate_trial_indices([success_record(2)], 2)

    def test_unavailable_git_metadata_is_not_clean(self):
        record = success_record()
        record["git_metadata_available"] = False
        record["git_commit"] = None
        record["git_dirty"] = None
        record["git_diff_sha256"] = None
        self.assertIsNone(validate_raw_result(record)["git_dirty"])

        record["git_dirty"] = False
        with self.assertRaises(SchemaError):
            validate_raw_result(record)


if __name__ == "__main__":
    unittest.main()

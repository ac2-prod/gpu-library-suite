import subprocess
import sys
import unittest

from gpu_suite.scheduler import (
    SchedulerIdentityError,
    parse_pbs_job_token,
    scheduler_job_path_token,
    validate_scheduler_identity,
)

from support import ROOT


RAW_NQSV_JOB_ID = "0:866211.nqsv"


class SchedulerIdentityTests(unittest.TestCase):
    def test_nqsv_raw_identity_and_filesystem_token_are_distinct(self):
        self.assertEqual(
            validate_scheduler_identity("NQSV", RAW_NQSV_JOB_ID),
            ("NQSV", RAW_NQSV_JOB_ID),
        )
        self.assertEqual(parse_pbs_job_token(RAW_NQSV_JOB_ID), "866211")
        self.assertEqual(parse_pbs_job_token("866211.nqsv"), "866211")
        self.assertEqual(
            scheduler_job_path_token("NQSV", RAW_NQSV_JOB_ID), "866211"
        )

    def test_local_null_identity_is_valid(self):
        self.assertEqual(validate_scheduler_identity(None, None), (None, None))
        self.assertIsNone(scheduler_job_path_token(None, None))
        for scheduler, job_id in (("NQSV", None), (None, RAW_NQSV_JOB_ID)):
            with self.subTest(scheduler=scheduler, job_id=job_id):
                with self.assertRaises(SchedulerIdentityError):
                    validate_scheduler_identity(scheduler, job_id)

    def test_invalid_path_and_control_values_are_rejected(self):
        for raw_job_id in (
            "", "unexpected/value", "unexpected\\value",
            "866211.nqsv\n", "866211.nqsv\x1f", "866211.nqsv\x7f",
        ):
            with self.subTest(raw_job_id=repr(raw_job_id)):
                with self.assertRaises(SchedulerIdentityError):
                    parse_pbs_job_token(raw_job_id)
                with self.assertRaises(SchedulerIdentityError):
                    validate_scheduler_identity("NQSV", raw_job_id)

    def test_tracked_cli_has_the_four_pbs_acceptance_cases(self):
        script = ROOT / "jobs" / "pegasus" / "scheduler_job_id.py"
        cases = (
            ("0:866071.nqsv", 0, "866071"),
            ("866071.nqsv", 0, "866071"),
            ("", 2, ""),
            ("unexpected/value", 2, ""),
        )
        for raw_job_id, returncode, token in cases:
            with self.subTest(raw_job_id=raw_job_id):
                completed = subprocess.run(
                    [
                        sys.executable, str(script), "token",
                        "--scheduler", "NQSV", "--job-id", raw_job_id,
                    ],
                    text=True, capture_output=True, check=False,
                )
                self.assertEqual(completed.returncode, returncode)
                self.assertEqual(completed.stdout.strip(), token)


if __name__ == "__main__":
    unittest.main()

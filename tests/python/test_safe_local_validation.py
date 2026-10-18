"""Focused wrapper tests; never recursively build or start real SSH."""

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_profiled_unittests as profiles
import run_safe_local_validation as safe


class SafeLocalValidationTests(unittest.TestCase):
    def temporary(self):
        temporary = tempfile.TemporaryDirectory(prefix="safe-entry-test-")
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def test_policy_selection_excludes_exact_four_ctest_cases(self):
        inventory = {"tests": [{"name": name} for name in sorted(safe.CTEST_ALL)]}
        selected, records = safe.ctest_selection(inventory, {"cxx": {"compiler": "Clang"}}, True)
        self.assertEqual(set(safe.CTEST_ALL) - set(selected), set(safe.CTEST_EXCLUSIONS))
        self.assertEqual({row["status"] for row in records}, {"POLICY_EXCLUDED"})

    def test_ctest_unknown_duplicate_and_removed_ids_fail_closed(self):
        names = sorted(safe.CTEST_ALL)
        for altered in (names + ["new-unreviewed"], names + [names[0]], names[1:]):
            with self.subTest(altered=altered[-1]):
                with self.assertRaises(safe.ValidationError):
                    safe.ctest_selection({"tests": [{"name": name} for name in altered]},
                                         {"cxx": {"compiler": "Clang"}}, True)

    def test_missing_shellcheck_is_dependency_not_pass(self):
        names = safe.CTEST_ALL - safe.CTEST_SHELLCHECK
        selected, rows = safe.ctest_selection({"tests": [{"name": name} for name in names]},
                                               {"cxx": {"compiler": "Clang"}}, False)
        missing = [row for row in rows if row["status"] == "NOT_RUN_DEPENDENCY"]
        self.assertEqual({row["id"].removeprefix("ctest:") for row in missing}, safe.CTEST_SHELLCHECK)
        self.assertFalse(set(selected) & safe.CTEST_SHELLCHECK)

    def test_python_new_removed_or_renamed_case_requires_audit(self):
        root = self.temporary()
        path = root / "test_fixture.py"
        source = "class Case:\n    def test_a(self): pass\n"
        path.write_text(source)
        digest = hashlib.sha256(b"test_fixture.Case.test_a").hexdigest()
        with mock.patch.object(profiles, "AUDITED_MODULE_IDS", {"test_fixture": digest}), \
                mock.patch.object(profiles, "POLICY_EXCLUSIONS", {}):
            self.assertEqual(profiles.audit_inventory(root), {"test_fixture.Case.test_a"})
            for altered in (source + "    def test_new(self): pass\n", source.replace("test_a", "test_b"), "class Case: pass\n"):
                path.write_text(altered)
                with self.assertRaisesRegex(ValueError, "inventory"):
                    profiles.audit_inventory(root)

    def test_profile_rules_preserve_existing_cpu_selection(self):
        class Case:
            def __init__(self, name):
                self.name = name
            def id(self):
                return self.name
        self.assertTrue(profiles.requires_cpu_fixture(Case("test_cufft_fixture_runtime.Case.test_a")))
        self.assertTrue(profiles.requires_cpu_fixture(Case(next(iter(profiles.CPU_FIXTURE_TEST_IDS)))))
        self.assertFalse(profiles.requires_cpu_fixture(Case("test_sirius_remote.Case.test_a")))
        self.assertEqual(len(profiles.POLICY_EXCLUSIONS), 2)

    def test_missing_unittest_result_is_failure(self):
        result = profiles.RecordedResult(unittest.runner._WritelnDecorator(io.StringIO()), True, 0)
        rows = profiles.complete_records(["Case.test_missing"], result)
        self.assertEqual(rows[0]["status"], "FAIL")
        self.assertIn("no recorded", rows[0]["reason"])

    def test_unittest_class_skip_is_recorded_for_all_selected_cases(self):
        class Skipped(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                raise unittest.SkipTest("fixture absent")
            def test_one(self):
                self.fail("must not run")
            def test_two(self):
                self.fail("must not run")
        cases = list(profiles.iter_test_cases(unittest.defaultTestLoader.loadTestsFromTestCase(Skipped)))
        ids = [case.id() for case in cases]
        result = unittest.TextTestRunner(stream=io.StringIO(), resultclass=profiles.RecordedResult).run(unittest.TestSuite(cases))
        rows = profiles.complete_records(ids, result)
        for test_id in ids:
            self.assertEqual(next(row for row in rows if row["id"] == test_id)["status"], "SKIP")

    def test_failed_subtest_does_not_become_pass(self):
        class Subtests(unittest.TestCase):
            def test_one(self):
                with self.subTest(value=1):
                    self.assertEqual(1, 2)
        cases = list(profiles.iter_test_cases(unittest.defaultTestLoader.loadTestsFromTestCase(Subtests)))
        ids = [case.id() for case in cases]
        result = unittest.TextTestRunner(stream=io.StringIO(), resultclass=profiles.RecordedResult).run(unittest.TestSuite(cases))
        self.assertEqual(profiles.complete_records(ids, result)[0]["status"], "FAIL")

    def test_junit_outcomes_and_missing_evidence(self):
        path = self.temporary() / "ctest.xml"
        path.write_text('<testsuite><testcase name="a" status="run"/><testcase name="b"><failure/></testcase><testcase name="c"><skipped message="fixture"/></testcase></testsuite>')
        rows = safe.junit_results(path, ["a", "b", "c", "missing"])
        self.assertEqual({row["id"]: row["status"] for row in rows},
                         {"ctest:a": "PASS", "ctest:b": "FAIL", "ctest:c": "SKIP", "ctest:missing": "FAIL"})

    def test_junit_duplicate_or_unselected_case_fails(self):
        path = self.temporary() / "ctest.xml"
        for content in ('<testsuite><testcase name="a"/><testcase name="a"/></testsuite>',
                        '<testsuite><testcase name="unknown"/></testsuite>'):
            path.write_text(content)
            with self.assertRaises(safe.ValidationError):
                safe.junit_results(path, ["a"])

    def test_junit_without_explicit_completion_status_is_failure(self):
        path = self.temporary() / "ctest.xml"
        path.write_text('<testsuite><testcase name="a"/></testsuite>')
        self.assertEqual(safe.junit_results(path, ["a"])[0]["status"], "FAIL")

    def test_python_report_rejects_duplicates_extra_and_missing_evidence(self):
        test_id = "test_cufft_fixture_runtime.Case.test_a"
        valid = {"inventory": [test_id], "selected": [test_id], "profile": "cpu-fixture",
                 "results": [{"id": test_id, "status": "PASS", "reason": "completed"}]}
        self.assertEqual(len(safe.validated_python_report(valid, {test_id})), 1)
        for change in ("duplicate", "unknown", "missing", "unselected", "excluded"):
            document = json.loads(json.dumps(valid))
            if change == "duplicate":
                document["results"].append(dict(document["results"][0]))
            elif change == "unknown":
                document["results"].append({"id": "unknown", "status": "PASS", "reason": "wrong"})
            elif change == "missing":
                document["results"] = []
            elif change == "unselected":
                document["selected"] = []
            else:
                document["results"][0]["status"] = "POLICY_EXCLUDED"
            with self.subTest(change=change):
                with self.assertRaises(safe.ValidationError):
                    safe.validated_python_report(document, {test_id})

    def test_finish_accounts_for_unexecuted_planned_case(self):
        root = self.temporary()
        source = root / "source"
        source.mkdir()
        run = object.__new__(safe.ValidationRun)
        run.source, run.work, run.source_files = source, root, {}
        run.results = []
        run.document = {"planned_inventory": ["ctest:missing"]}
        with mock.patch("builtins.print"):
            self.assertEqual(run.finish(), 1)
        self.assertEqual(run.results[0]["id"], "ctest:missing")
        self.assertEqual(run.results[0]["status"], "FAIL")

    def test_probe_checks_parse_without_running_probe(self):
        run = object.__new__(safe.ValidationRun)
        run.runtime_source = Path("/fixture/source")
        run.results = []
        with mock.patch.object(safe.shutil, "which", side_effect=["/bin/bash", None]), \
                mock.patch.object(run, "command", return_value=True) as command:
            run.check_probe_syntax()
        command.assert_called_once_with("probe-bash-syntax", ["/bin/bash", "-n", Path("/fixture/source/jobs/sirius/probe.sh")])
        self.assertEqual(run.results[0]["status"], "NOT_RUN_DEPENDENCY")
        self.assertEqual(run.results[0]["id"], "command:probe-shellcheck")

    def test_path_outside_local_base_is_rejected(self):
        with self.assertRaises(safe.ValidationError):
            safe.validated_path(Path("/outside-validation-root"))

    def test_symlink_path_is_rejected_without_creating_links(self):
        with mock.patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(safe.ValidationError, "symbolic link"):
                safe.validated_path(safe.LOCAL_BASE / "mock-link")

    def test_existing_overlap_and_remote_filesystem_are_rejected(self):
        base = self.temporary()
        source = base / "source"
        source.mkdir()
        (source / "CMakeLists.txt").write_text("project(fixture)\n")
        existing = base / "existing"
        existing.mkdir()
        with mock.patch.object(safe, "LOCAL_BASE", base):
            for output in (source / "work", source, existing):
                with self.assertRaises(safe.ValidationError):
                    safe.validate_paths(source, output)
            with mock.patch.object(safe, "local_filesystem", side_effect=safe.ValidationError("remote filesystem")):
                with self.assertRaisesRegex(safe.ValidationError, "remote"):
                    safe.validate_paths(source, base / "new")

    def test_environment_routes_caches_and_preserves_home(self):
        root = self.temporary()
        with mock.patch.dict(os.environ, {"HOME": "/user-home", "GPU_SUITE_STALE": "x", "GIT_DIR": "bad"}):
            env = safe.make_environment(root / "source", root)
        self.assertEqual(env["HOME"], "/user-home")
        self.assertNotIn("GPU_SUITE_STALE", env)
        self.assertNotIn("GIT_DIR", env)
        for name in ("TMPDIR", "TMP", "TEMP", "PYTHONPYCACHEPREFIX", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
            self.assertIn(root, Path(env[name]).parents)

    def test_report_rejects_duplicate_keys_and_nonstandard_numbers(self):
        path = self.temporary() / "bad.json"
        for content in ('{"x": 1, "x": 2}', '{"x": NaN}'):
            path.write_text(content)
            with self.assertRaises(safe.ValidationError):
                safe.strict_json(path)

    def test_json_reports_are_never_overwritten(self):
        path = self.temporary() / "report.json"
        safe.write_json(path, {"first": True})
        with self.assertRaises(FileExistsError):
            safe.write_json(path, {"first": False})
        self.assertEqual(json.loads(path.read_text()), {"first": True})

    def test_command_failure_records_argv_exit_and_separate_logs(self):
        root = self.temporary()
        (root / "logs").mkdir()
        run = object.__new__(safe.ValidationRun)
        run.work, run.environment, run.results, run.commands = root, {}, [], []
        def fake_process(argv, **kwargs):
            kwargs["stdout"].write(b"standard-out")
            kwargs["stderr"].write(b"standard-error")
            return type("Completed", (), {"returncode": 7})()
        with mock.patch.object(safe.subprocess, "run", side_effect=fake_process) as call:
            self.assertFalse(run.command("fake", ["fake-tool", "argument"]))
        self.assertEqual(call.call_count, 1)
        self.assertEqual(run.commands[0]["exit_code"], 7)
        self.assertEqual(run.results[0]["status"], "FAIL")
        self.assertEqual((root / "logs/fake.stdout.log").read_bytes(), b"standard-out")
        self.assertEqual((root / "logs/fake.stderr.log").read_bytes(), b"standard-error")

    def test_input_snapshot_mutation_fails_completion(self):
        root = self.temporary()
        source = root / "source"
        source.mkdir()
        path = source / "source.txt"
        path.write_text("before")
        run = object.__new__(safe.ValidationRun)
        run.source, run.work, run.source_files = source, root, safe.snapshot_files(source)
        run.results, run.document = [], {}
        path.write_text("after")
        with mock.patch("builtins.print"):
            self.assertEqual(run.finish(), 1)
        self.assertEqual(run.results[0]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()

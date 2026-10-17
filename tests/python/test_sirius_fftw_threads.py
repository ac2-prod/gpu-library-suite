"""Read-only threading inventory; all command results and providers are fake."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("fftw_thread_inventory", ROOT / "jobs/sirius/fftw_thread_inventory.py")
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class SiriusFftwThreadsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="fftw-threads-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.commands = []

    def file(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    @staticmethod
    def symbol_table(names, undefined=(), hidden=()):
        rows = []
        for index, name in enumerate(names):
            rows.append(" %d: 00000000 10 FUNC GLOBAL %s %s %s@@FFTW_3" % (
                index, "HIDDEN" if name in hidden else "DEFAULT",
                "UND" if name in undefined else "12", name))
        return "\n".join(rows)

    def runner(self, argv):
        self.commands.append(argv)
        if "--dyn-syms" in argv:
            content = self.symbol_table(inventory.APIS)
        elif "--dynamic" in argv:
            content = " 0x00000001 (NEEDED) Shared library: [libgomp.so.1]\n"
        else:
            raise AssertionError("unapproved command: " + repr(argv))
        return dict(argv=argv, status="COMPLETE", exit=0, stdout=content, stderr="")

    def collect(self, runner=None, locate=None):
        return inventory.Inventory(self.root, runner=runner or self.runner,
            locate=locate or (lambda name: "/fake/bin/" + name)).collect()

    def test_three_variants_hashes_declared_version_and_no_mutation(self):
        files = [self.file("include/fftw3.h", b"X(init_threads) X(plan_with_nthreads) X(cleanup_threads)"),
                 self.file("lib/pkgconfig/fftw3f.pc", b"Version: 3.3.10\nLibs: -lfftw3f\n"),
                 self.file("lib/libfftw3f.so.3.6.10", b"\x7fELFbase"),
                 self.file("lib64/libfftw3f_threads.so", b"\x7fELFthreads"),
                 self.file("lib_LP64/libfftw3f_omp.so", b"\x7fELFomp")]
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_mode) for p in files}
        result = self.collect()
        self.assertEqual(result["status"], "COMPLETE_STATIC_INVENTORY")
        self.assertEqual({x["variant"] for x in result["libraries"]}, {"base", "pthread", "openmp"})
        self.assertEqual(result["pkg_config"][0]["version_text"], "3.3.10")
        for item in result["libraries"]:
            self.assertEqual(item["sha256"], hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest())
            self.assertTrue(item["inspection"]["all_thread_api_exports"])
            self.assertEqual(item["inspection"]["needed"], ["libgomp.so.1"])
        self.assertTrue(all(result["headers"][0]["thread_api_tokens"].values()))
        self.assertEqual(len(self.commands), 6)
        self.assertTrue(all(command[0] == "/fake/bin/readelf" for command in self.commands))
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_mode) for p in files})

    def test_undefined_hidden_and_partial_symbols_do_not_prove_threading(self):
        table = self.symbol_table(inventory.APIS,
            undefined=("fftwf_init_threads",), hidden=("fftwf_cleanup_threads",))
        parsed = inventory.readelf_symbols(table)
        self.assertTrue(parsed["fftwf_init_threads"]["undefined_reference"])
        self.assertFalse(parsed["fftwf_init_threads"]["defined_export"])
        self.assertFalse(parsed["fftwf_cleanup_threads"]["defined_export"])
        self.assertTrue(parsed["fftwf_plan_with_nthreads"]["defined_export"])
        self.assertFalse(inventory.readelf_symbols("")["fftwf_execute"]["defined_export"])

    def test_filtered_command_evidence_retains_target_lines_and_output_digest(self):
        output = self.symbol_table(("fftwf_init_threads", "fftwf_unrelated_codelet"))
        original = dict(status="COMPLETE", exit=0, stdout=output, stderr="", argv=["/fake/readelf"])
        record = inventory.compact_command(original)
        self.assertIn("fftwf_init_threads", record["stdout"])
        self.assertNotIn("fftwf_unrelated_codelet", record["stdout"])
        self.assertEqual(record["stdout_sha256"], hashlib.sha256(output.encode()).hexdigest())
        self.assertEqual(original["stdout"], output)

    def test_nm_fallback_reports_needed_unknown_and_undefined_weak_symbols(self):
        self.file("lib/libfftw3f.so", b"\x7fELFfake")
        def runner(argv):
            self.commands.append(argv)
            return dict(argv=argv, status="COMPLETE", exit=0, stderr="",
                stdout="fftwf_init_threads U\nfftwf_plan_with_nthreads T 01 20\nfftwf_cleanup_threads w\n")
        result = self.collect(runner=runner, locate=lambda name: "/fake/nm" if name == "nm" else None)
        self.assertEqual(result["status"], "INCOMPLETE")
        evidence = result["libraries"][0]["inspection"]
        self.assertEqual(evidence["source"], "nm")
        self.assertEqual(evidence["needed_status"], "UNKNOWN")
        self.assertFalse(evidence["all_thread_api_exports"])
        self.assertTrue(evidence["symbols"]["fftwf_cleanup_threads"]["undefined_reference"])
        self.assertEqual(self.commands[0][1:3], ["-D", "--format=posix"])

    def test_unavailable_tools_and_non_elf_are_unknown_without_execution(self):
        self.file("lib/libfftw3f.so", b"\x7fELFfake")
        self.file("lib/libfftw3f_threads.a", b"!<arch>fake")
        result = self.collect(locate=lambda name: None)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(self.commands, [])
        self.assertTrue(all(x["inspection"]["status"] == "UNKNOWN" for x in result["libraries"]))

    def test_failed_or_truncated_queries_never_count_as_symbol_evidence(self):
        self.file("lib/libfftw3f.so", b"\x7fELFfake")
        def runner(argv):
            return dict(argv=argv, status="INCOMPLETE", exit=-9, reason="output_limit",
                        stdout=self.symbol_table(inventory.APIS), stderr="truncated")
        result = self.collect(runner=runner)
        evidence = result["libraries"][0]["inspection"]
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(evidence["status"], "UNKNOWN")
        self.assertNotIn("all_thread_api_exports", evidence)

    def test_bounded_directories_do_not_walk_nested_or_unrelated_files(self):
        self.file("lib/nested/libfftw3f.so", b"\x7fELFnested")
        self.file("unrelated/libfftw3f.so", b"\x7fELFunrelated")
        self.file("lib/libfftw3.so", b"\x7fELFdouble")
        self.file("lib/secret.txt", b"must not appear")
        result = self.collect()
        self.assertEqual(result["libraries"], [])
        self.assertEqual(self.commands, [])
        self.assertNotIn("must not appear", json.dumps(result))

    def test_oversized_library_text_and_directory_limits_are_incomplete(self):
        self.file("lib/libfftw3f.so", b"\x7fELFoversized")
        self.file("include/fftw3.h", b"oversized header")
        with mock.patch.object(inventory, "MAX_LIBRARY_BYTES", 4), mock.patch.object(inventory, "MAX_TEXT_BYTES", 4):
            result = self.collect()
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["libraries"][0]["status"], "UNKNOWN")
        self.assertEqual(result["headers"][0]["status"], "UNKNOWN")
        self.assertEqual(self.commands, [])
        with mock.patch.object(inventory, "MAX_DIRECTORY_ENTRIES", 0):
            result = self.collect()
        self.assertTrue(any(x["reason"] == "directory_entry_limit" for x in result["issues"]))

    def test_root_and_outside_real_paths_are_rejected_without_symlink_creation(self):
        with self.assertRaises(ValueError):
            inventory.Inventory("relative")
        with self.assertRaises(ValueError):
            inventory.Inventory(str(self.root / ".."))
        target = self.file("lib/libfftw3f.so", b"\x7fELFfake")
        instance = inventory.Inventory(self.root, runner=self.runner, locate=lambda name: "/fake/" + name)
        with mock.patch.object(Path, "resolve", return_value=Path("/outside-provider/libfftw3f.so")):
            record = instance.library(target, "base")
        self.assertEqual(record["status"], "UNKNOWN")
        self.assertEqual(self.commands, [])

    def test_collection_time_budget_stops_before_another_utility_launch(self):
        instance = inventory.Inventory(self.root, runner=self.runner, locate=lambda name: "/fake/" + name)
        instance.deadline = 0
        record = instance.query(["/fake/readelf", "--dyn-syms", "/fake/lib"])
        self.assertEqual(record["status"], "INCOMPLETE")
        self.assertEqual(record["reason"], "collection_time_limit")
        self.assertEqual(self.commands, [])

    def test_cli_json_only_and_nonzero_for_incomplete_or_invalid_root(self):
        self.file("lib/libfftw3f.so", b"\x7fELFfake")
        with mock.patch.object(inventory.shutil, "which", return_value=None):
            # Bind explicit fake locate: defaults were captured at definition time.
            with mock.patch.object(inventory, "Inventory", return_value=inventory.Inventory(self.root, locate=lambda name: None)):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    status = inventory.main(["--aocl-root", str(self.root)])
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "INCOMPLETE")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = inventory.main(["--aocl-root", "relative"])
        self.assertEqual(status, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "STOPPED")

    def fake_child(self, chunks):
        child = mock.Mock()
        child.stdout = mock.Mock()
        child.stderr = mock.Mock()
        child.wait.return_value = 0
        selector = mock.MagicMock()
        selector.__enter__.return_value = selector
        selector.get_map.side_effect = [True] * (len(chunks) + 1) + [False]
        selector.select.return_value = [(SimpleNamespace(fd=20, data="stdout", fileobj=child.stdout), 1)]
        return child, selector

    def test_command_output_limit_kills_only_its_child_and_marks_incomplete(self):
        child, selector = self.fake_child([b"abcdefgh"])
        with mock.patch.object(inventory.subprocess, "Popen", return_value=child) as launch, \
                mock.patch.object(inventory.selectors, "DefaultSelector", return_value=selector), \
                mock.patch.object(inventory.os, "read", return_value=b"abcdefgh"), \
                mock.patch.object(inventory, "MAX_COMMAND_BYTES", 4):
            result = inventory.bounded_command(["/fake/readelf", "--dyn-syms", "/fake/lib"])
        self.assertEqual(result["reason"], "output_limit")
        self.assertEqual(result["stdout"], "abcd")
        self.assertEqual(result["status"], "INCOMPLETE")
        child.kill.assert_called_once_with()
        self.assertFalse(launch.call_args.kwargs["shell"])
        self.assertEqual(launch.call_args.kwargs["stdin"], inventory.subprocess.DEVNULL)

    def test_command_timeout_and_launch_failure_remain_unknown(self):
        child, selector = self.fake_child([])
        with mock.patch.object(inventory.subprocess, "Popen", return_value=child), \
                mock.patch.object(inventory.selectors, "DefaultSelector", return_value=selector), \
                mock.patch.object(inventory.time, "monotonic", side_effect=[0, 11, 11]):
            result = inventory.bounded_command(["/fake/readelf", "--dynamic", "/fake/lib"])
        self.assertEqual(result["reason"], "timeout")
        child.kill.assert_called_once_with()
        with mock.patch.object(inventory.subprocess, "Popen", side_effect=OSError("fake launch failure")):
            result = inventory.bounded_command(["/fake/nm", "-D", "/fake/lib"])
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertIsNone(result["exit"])


if __name__ == "__main__":
    unittest.main()

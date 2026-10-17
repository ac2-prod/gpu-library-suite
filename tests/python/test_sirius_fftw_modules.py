"""Static module inventory fixtures; no real modules, providers or network."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "fftw_module_inventory", ROOT / "jobs/sirius/fftw_module_inventory.py")
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class SiriusFftwModuleInventoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="fftw-static-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, path, text):
        target = self.root / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text if isinstance(text, bytes) else text.encode())
        return target

    def run_inventory(self, **kwargs):
        return inventory.run_inventory(system_root=self.root, env=kwargs.pop("env", {}), **kwargs)

    def records(self, records, kind):
        return [record for record in records if record["kind"] == kind]

    def test_tcl_configuration_chain_and_declared_provider(self):
        self.write("/etc/profile.d/modules.sh", "export MODULESHOME=/provider/modules\n")
        self.write("/provider/modules/init/.modulespath", "/observed/modules # site tree\n")
        self.write("/observed/modules/aocl/5.1.0/gcc14.2.1", """#%Module
set root /provider/aocl
prereq gcc/14.2.1
setenv AOCL_ROOT $root
prepend-path CPATH $root/include
prepend-path LD_LIBRARY_PATH $root/lib64
""")
        self.write("/provider/aocl/include/fftw3.h", "fixture header")
        self.write("/provider/aocl/lib64/libfftw3f.so", "fixture library")
        self.write("/provider/aocl/lib64/libfftw3f_threads.so.3", "fixture runtime")
        records = self.run_inventory()
        self.assertEqual(self.records(records, "modulefile")[0]["module"], inventory.MODULES[0])
        requirements = self.records(records, "compiler_requirement")
        self.assertEqual(requirements[0]["declaration"], "prereq gcc/14.2.1")
        self.assertEqual(requirements[0]["line"], 3)
        files = {row["path"]: row for row in self.records(records, "candidate_file")}
        self.assertEqual(files["/provider/aocl/include/fftw3.h"]["role"], "header")
        self.assertEqual(files["/provider/aocl/lib64/libfftw3f.so"]["role"], "link_library")
        self.assertEqual(files["/provider/aocl/lib64/libfftw3f_threads.so.3"]["role"], "versioned_runtime")
        self.assertFalse(records[-1]["effective_environment_verified"])

    def test_lua_literal_join_and_dependency_are_not_evaluated(self):
        self.write("/modules/aocl/5.1.0/aocc5.0.0.lua", """local root = "/provider with spaces/aocl"
depends_on("aocc/5.0.0")
conflict("gcc")
prepend_path("CPATH", pathJoin(root, "fftw/include"))
prepend_path("LD_LIBRARY_PATH", pathJoin(root, "fftw/lib"))
""")
        self.write("/provider with spaces/aocl/fftw/include/fftw3.h", "header")
        self.write("/provider with spaces/aocl/fftw/lib/libfftw3f_threads.a", "archive")
        records = self.run_inventory(module_roots=("/modules",))
        files = [row["path"] for row in self.records(records, "candidate_file")]
        self.assertIn("/provider with spaces/aocl/fftw/include/fftw3.h", files)
        self.assertIn("/provider with spaces/aocl/fftw/lib/libfftw3f_threads.a", files)
        self.assertEqual(len(self.records(records, "compiler_requirement")), 2)
        self.assertEqual([row["relation"] for row in self.records(records, "compiler_requirement")],
                         ["depends_on", "conflict"])

    def test_observed_lmod_home_sidecar_and_explicit_config(self):
        self.write("/module-system/lmod/init/.modulespath", "/modules/lua\n")
        self.write("/site/modules.conf", "module use /modules/site\n")
        records = self.run_inventory(env={"LMOD_CMD": "/module-system/lmod/libexec/lmod"},
                                     configs=("/site/modules.conf",))
        self.assertEqual({row["path"] for row in self.records(records, "module_root")},
                         {"/modules/lua", "/modules/site"})

    def test_module_command_and_modulepath_may_both_be_missing(self):
        (self.root / "etc/profile.d").mkdir(parents=True)
        records = self.run_inventory()
        self.assertTrue(self.records(records, "missing_file"))
        self.assertFalse(self.records(records, "modulefile"))
        self.assertFalse(records[-1]["compiler_link_verified"])
        self.assertTrue(records[-1]["unresolved"])
        self.assertEqual(records[-1]["missing_modules"], sorted(inventory.MODULES))
        self.assertEqual(records[-1]["found_modules"], [])
        self.assertEqual(records[-1]["requested_modules"], list(inventory.MODULES))

    def test_dynamic_code_and_load_declarations_never_run(self):
        marker = self.root / "must-not-be-created"
        self.write("/modules/aocl/5.1.0/gcc14.2.1", "set root [exec touch " + str(marker) + "]\n"
                   "module load gcc/14.2.1\nexec touch " + str(marker) + "\n"
                   "setenv FFTW_ROOT $root\n")
        records = self.run_inventory(module_roots=("/modules",))
        self.assertFalse(marker.exists())
        self.assertTrue(self.records(records, "unresolved"))
        self.assertEqual(self.records(records, "compiler_requirement")[0]["status"], "declared_only_not_loaded")
        self.assertFalse(self.records(records, "candidate_file"))

    def test_absolute_paths_are_mapped_inside_fixture(self):
        self.write("/modules/aocl/5.1.0/gcc14.2.1", "set root /usr\nsetenv FFTW_ROOT $root\n")
        self.write("/usr/include/fftw3.h", "fixture-only")
        records = self.run_inventory(module_roots=("/modules",))
        self.assertEqual([row["path"] for row in self.records(records, "candidate_file")], ["/usr/include/fftw3.h"])
        bad = self.run_inventory(module_roots=("/../outside",))
        self.assertTrue(any(row.get("reason") == "nonabsolute_or_parent_path" for row in bad))

    def test_conditional_and_reassigned_variables_remain_candidates(self):
        self.write("/modules/aocl/5.1.0/gcc14.2.1", "if {dynamic} {\nset root /first\n}\n"
                   "set root /second\nsetenv FFTW_ROOT $root\n")
        records = self.run_inventory(module_roots=("/modules",))
        self.assertTrue(self.records(records, "modulefile")[0]["conditional"])
        self.assertTrue(any(row.get("reason") == "variable_reassigned" for row in records))
        self.assertFalse(records[-1]["effective_environment_verified"])

    def test_unknown_modulepath_tail_keeps_literal_candidate(self):
        self.write("/etc/profile.d/site-modules.sh", "MODULEPATH=/known/modules:$UNKNOWN # not executed\n")
        records = self.run_inventory()
        self.assertEqual([row["path"] for row in self.records(records, "module_root")], ["/known/modules"])
        self.assertTrue(any(row.get("reason") == "dynamic_path_expression" for row in records))

    def test_self_reference_and_config_declarations_are_resolved_once(self):
        self.write("/site/config", 'MODULEPATH="$MODULEPATH:/new"\nmodule use --append /append\n')
        self.write("/site/config.lua", 'local base = "/lua"\nprepend_path("MODULEPATH", pathJoin(base, "modules")) -- note\n')
        records = self.run_inventory(env={"MODULEPATH": "/old"}, configs=("/site/config", "/site/config.lua"))
        self.assertEqual([row["path"] for row in self.records(records, "module_root")],
                         ["/old", "/new", "/append", "/lua/modules"])

    def test_unsupported_tcl_braces_and_lua_concatenation_are_unresolved(self):
        self.write("/modules/aocl/5.1.0/gcc14.2.1", "set root /provider\nprepend-path CPATH {$root/include}\n")
        self.write("/modules/aocl/5.1.0/aocc5.0.0.lua", 'local root = "/provider"\nprepend_path("CPATH", root .. "/include")\n')
        records = self.run_inventory(module_roots=("/modules",))
        unresolved = [row for row in self.records(records, "unresolved")
                      if row.get("reason") == "dynamic_path_expression"]
        self.assertEqual(len(unresolved), 2)

    def test_file_and_total_byte_limits_are_reported(self):
        self.write("/site/config", "x" * 33)
        with mock.patch.object(inventory, "MAX_FILE_BYTES", 32):
            records = self.run_inventory(configs=("/site/config",))
        self.assertTrue(any(row.get("reason") == "bytes" for row in self.records(records, "limit")))
        with mock.patch.object(inventory, "MAX_TOTAL_BYTES", 32):
            records = self.run_inventory(configs=("/site/config",))
        self.assertTrue(self.records(records, "limit"))

    def test_config_depth_and_file_limits_are_reported(self):
        for number in range(7):
            self.write("/site/c" + str(number), "source /site/c" + str(number + 1) + "\n")
        records = self.run_inventory(configs=("/site/c0",))
        self.assertTrue(any(row.get("reason") == "config_depth" for row in records))
        with mock.patch.object(inventory, "MAX_FILES", 1):
            records = self.run_inventory(configs=("/site/c0",))
        self.assertEqual(records[-1]["files_read"], 1)
        self.assertTrue(any(row.get("reason") == "files" for row in records))

    def test_root_and_shallow_directory_limits_are_reported(self):
        for name in ("a", "b", "c"):
            self.write("/etc/profile.d/" + name + "-modules.sh", "module use /" + name + "\n")
        with mock.patch.object(inventory, "MAX_ENTRIES", 2):
            records = self.run_inventory()
        self.assertTrue(any(row.get("reason") == "directory_entries" for row in records))
        with mock.patch.object(inventory, "MAX_ROOTS", 1):
            records = self.run_inventory(module_roots=("/one", "/two"))
        self.assertEqual(len(self.records(records, "module_root")), 1)
        self.assertTrue(any(row.get("reason") == "module_roots" for row in records))

    def test_config_queue_and_candidate_directory_limits_are_reported(self):
        self.write("/site/config", "source /site/one\nsource /site/two\n")
        with mock.patch.object(inventory, "CONFIGS", ()), mock.patch.object(inventory, "MAX_CONFIGS", 2):
            records = self.run_inventory(configs=("/site/config",))
        self.assertTrue(any(row.get("reason") == "config_candidates" for row in records))
        self.write("/modules/aocl/5.1.0/gcc14.2.1", "setenv FFTW_ROOT /provider\n")
        with mock.patch.object(inventory, "MAX_CANDIDATE_DIRS", 1):
            records = self.run_inventory(module_roots=("/modules",))
        self.assertTrue(any(row.get("reason") == "candidate_directories" for row in records))

    def test_invalid_utf8_is_unresolved_not_silently_parsed(self):
        self.write("/site/config", b"MODULEPATH=/bad\xff\n")
        records = self.run_inventory(configs=("/site/config",))
        self.assertTrue(any(row.get("error") == "UnicodeDecodeError" for row in records))
        self.assertFalse(self.records(records, "module_root"))

    def test_inventory_preserves_file_contents_and_mtimes(self):
        paths = [self.write("/modules/aocl/5.1.0/gcc14.2.1", "setenv FFTW_ROOT /provider\n"),
                 self.write("/provider/include/fftw3.h", "header"),
                 self.write("/provider/lib/libfftw3f_omp.so.3", "runtime")]
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
        records = self.run_inventory(module_roots=("/modules",))
        self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths})
        self.assertEqual(self.records(records, "candidate_file")[-1]["role"], "versioned_runtime")

    def test_cli_gate_arguments_and_json_output(self):
        for environment, arguments in (({}, []), ({"GPU_SUITE_FFTW_PROBE_APPROVED": "yes"}, ["--module-root", "relative"]),
                                       ({"GPU_SUITE_FFTW_PROBE_APPROVED": "yes"}, ["--system-root", str(self.root)])):
            with mock.patch.dict(os.environ, environment, clear=True), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    inventory.main(arguments)
            self.assertEqual(error.exception.code, 2)
        output = io.StringIO()
        with mock.patch.dict(os.environ, {"GPU_SUITE_FFTW_PROBE_APPROVED": "yes"}, clear=True), \
                mock.patch.object(inventory, "run_inventory", return_value=[{"kind": "fixture"}]) as run, \
                redirect_stdout(output):
            self.assertEqual(inventory.main(["--config", "/observed/config"]), 0)
        self.assertEqual(json.loads(output.getvalue()), {"kind": "fixture"})
        run.assert_called_once_with(module_roots=[], configs=["/observed/config"])


if __name__ == "__main__":
    unittest.main()

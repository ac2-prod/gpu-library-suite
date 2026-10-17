"""Read-only FFTW inventory with fake commands; no connection or real providers."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "jobs/sirius/fftw_probe.sh"


class SiriusFftwProbeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="fftw-inventory-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "commands.log"
        self.prefix = self.root / "provider with spaces"
        self.environment = {
            "PATH": str(self.bin), "HOME": str(self.root),
            "QUERY_LOG": str(self.log), "FIXTURE_PREFIX": str(self.prefix),
            "FFTW_ROOT": str(self.prefix), "GPU_SUITE_FFTW_PROBE_APPROVED": "yes",
        }

    def fake(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/bash\nprintf '%s' " + name + " >> \"$QUERY_LOG\"\n"
                        "printf ' <%s>' \"$@\" >> \"$QUERY_LOG\"\n"
                        "printf '\\n' >> \"$QUERY_LOG\"\n" + body + "\n")
        path.chmod(0o700)

    def identities(self):
        for name, output in (("id", "fixture-user"), ("hostname", "fixture-host"),
                             ("date", "2000-01-01T00:00:00Z")):
            self.fake(name, "printf '%s\\n' '" + output + "'")
        self.fake("python3", "printf '%s\\n' '{\"kind\":\"fixture_module_inventory\",\"status\":\"static_only\"}'")

    def invoke(self, *args):
        return subprocess.run(["/bin/bash", str(SCRIPT)] + list(args),
                              env=self.environment, capture_output=True, text=True,
                              check=False, timeout=10)

    def test_unapproved_or_relative_prefix_stops_before_any_query(self):
        self.identities()
        self.environment.pop("GPU_SUITE_FFTW_PROBE_APPROVED")
        result = self.invoke()
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.log.exists())
        self.environment["GPU_SUITE_FFTW_PROBE_APPROVED"] = "yes"
        result = self.invoke("relative/prefix")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.log.exists())

    def test_fake_inventory_preserves_prefix_and_uses_only_read_queries(self):
        self.identities()
        self.fake("module", "case \"$*\" in\n"
                  " '-t list') printf '%s\\n' 'fixture/compiler' ;;\n"
                  " '-t avail') printf '%s\\n' 'fixture/fftw-single' 'aocl/5.1.0/gcc14.2.1' 'unrelated/module' ;;\n"
                  " 'show aocl/5.1.0/gcc14.2.1'|'show aocl/5.1.0/aocc5.0.0') printf '%s\\n' 'fixture AOCL prerequisite and prefix' >&2 ;;\n"
                  " *) exit 91 ;;\nesac")
        self.fake("pkg-config", "case \"$1\" in\n"
                  " --variable=prefix) printf '%s\\n' \"$FIXTURE_PREFIX\" ;;\n"
                  " --modversion) printf '%s\\n' 'fixture-version' ;;\n"
                  " --cflags|--libs|--variable=includedir|--variable=libdir) printf '%s\\n' 'fixture-flags' ;;\n"
                  " *) exit 92 ;;\nesac")
        self.fake("ldconfig", "[[ \"$*\" == '-p' ]] || exit 93\n"
                  "printf '%s\\n' 'libfftw3f.so.3 => /fixture/libfftw3f.so.3' 'unrelated.so'")
        files = [self.prefix / "include/fftw3.h", self.prefix / "lib64/libfftw3f.so.3",
                 self.prefix / "lib64/libfftw3f_threads.a"]
        for path in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not a real library; must never be executed")
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}
        result = self.invoke(str(self.prefix))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FFTW_READ_ONLY_INVENTORY_END", result.stdout)
        self.assertIn("fixture_module_inventory", result.stdout)
        self.assertIn("MODULE_INSPECTION=static", result.stdout)
        self.assertIn("libfftw3f_threads.a", result.stdout)
        self.assertIn("CANDIDATE_FILE=", result.stdout)
        self.assertNotIn("unrelated/module", result.stdout)
        self.assertNotIn("unrelated.so", result.stdout)
        commands = self.log.read_text().splitlines()
        self.assertEqual({line.split()[0] for line in commands},
                         {"id", "hostname", "date", "python3", "pkg-config", "ldconfig"})
        self.assertEqual([line for line in commands if line.startswith("module")],
                         [])
        self.assertEqual([line for line in commands if line.startswith("python3")],
                         ["python3 <-I> <-B> <" + str(SCRIPT.with_name("fftw_module_inventory.py")) + ">"])
        self.assertEqual([line for line in commands if line.startswith("ldconfig")],
                         ["ldconfig <-p>"])
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files})

    def test_missing_tools_and_package_are_reported_without_fallback(self):
        self.identities()
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        for marker in ("MODULE_INSPECTION=static", "PKG_CONFIG_UNAVAILABLE", "LDCONFIG_UNAVAILABLE"):
            self.assertIn(marker, result.stdout)
        self.fake("pkg-config", "printf '%s\\n' 'fftw3f package unavailable' >&2\nexit 1")
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("QUERY_EXIT=1"), 6)
        self.assertIn("fftw3f package unavailable", result.stderr)
        self.assertIn("no compile/link validation", result.stdout)

    def test_bash_syntax(self):
        result = subprocess.run(["/bin/bash", "-n", str(SCRIPT)], capture_output=True,
                                text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_failed_read_queries_preserve_diagnostics(self):
        self.identities()
        self.fake("python3", "printf '%s\\n' 'module inventory denied' >&2\nexit 7")
        self.fake("ldconfig", "printf '%s\\n' 'cache read denied' >&2\nexit 8")
        result = self.invoke()
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("QUERY_EXIT=7", result.stdout)
        self.assertIn("QUERY_EXIT=8", result.stdout)
        self.assertIn("module inventory denied", result.stderr)
        self.assertIn("cache read denied", result.stderr)

    def test_missing_python_stops_before_queries(self):
        for name in ("id", "hostname", "date"):
            self.fake(name, "exit 99")
        result = self.invoke()
        self.assertEqual(result.returncode, 2)
        self.assertIn("Python 3.9+", result.stderr)
        self.assertFalse(self.log.exists())

    def test_missing_companion_stops_before_queries(self):
        self.identities()
        isolated = self.root / "fftw_probe.sh"
        isolated.write_bytes(SCRIPT.read_bytes())
        result = subprocess.run(["/bin/bash", str(isolated)], env=self.environment,
                                capture_output=True, text=True, check=False, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("companion", result.stderr)
        self.assertFalse(self.log.exists())

    @unittest.skipUnless(shutil.which("shellcheck"), "ShellCheck dependency unavailable")
    def test_shellcheck(self):
        result = subprocess.run([shutil.which("shellcheck"), str(SCRIPT)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

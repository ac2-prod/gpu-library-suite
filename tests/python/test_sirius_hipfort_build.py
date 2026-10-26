"""Offline dependency fixtures only; no real compiler, installer, SSH or GPU."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "jobs/sirius"))
import hipfort_build as dependency


class HipfortBuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="hipfort-fixture-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archives = self.root / "archives"
        self.archives.mkdir()
        self.lock = dict(schema_version=1, rocm_version="10.1.0", sources={})
        for name in ("hipfort", "rocm-cmake"):
            path = self.archives / (name + ".tar.gz")
            relative = "CMakeLists.txt" if name == "hipfort" else "share/rocmcmakebuildtools/cmake/ROCmCMakeBuildToolsConfig.cmake"
            with tarfile.open(path, "w:gz") as archive:
                item = tarfile.TarInfo(name + "-" + "a" * 40 + "/" + relative)
                raw = b"fixture only\n"
                item.size = len(raw)
                archive.addfile(item, io.BytesIO(raw))
            self.lock["sources"][name] = dict(archive=path.name, commit="a" * 40,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        lock = self.root / "lock.json"
        lock.write_text(json.dumps(self.lock))
        rocm = self.root / "rocm"
        (rocm / ".info").mkdir(parents=True)
        (rocm / ".info/version").write_text("10.1.0\n")
        (rocm / "bin").mkdir()
        for name in ("cmake", "amdclang", "amdflang", "gcc"):
            p = rocm / "bin" / name
            p.write_text("never executed")
            p.chmod(0o700)
        self.args = SimpleNamespace(archives=self.archives, output=self.root / "output", rocm_root=rocm,
                                    cmake=rocm / "bin/cmake", lock=lock, timeout_seconds=900)
        self.env = dict(GPU_SUITE_SIRIUS_AMD_BUILD_APPROVED="yes", PBS_JOBID="42.fixture", PBS_ENVIRONMENT="PBS_BATCH",
                        CPU_CC=str(rocm / "bin/gcc"))
        self.calls = []
        patch = mock.patch.object(subprocess, "Popen", side_effect=AssertionError("Real process prohibited"))
        patch.start()
        self.addCleanup(patch.stop)

    def fake_run(self, argv, **kwargs):
        self.calls.append(argv)
        kwargs["stdout"].write(b"fake stdout\n")
        kwargs["stderr"].write(b"fake stderr\n")
        if "--install" in argv:
            paths = ["build/CMakeCache.txt", "build/install_manifest.txt",
                     "install/lib/fortran/amdflang/libhipfort-amdgcn.a"] + [
                         "install/include/fortran/amdflang/hipfort/amdgcn/" + name + ".mod" for name in dependency.MODULES]
            for relative in paths:
                p = self.args.output / relative
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("fixture " + relative)
        return subprocess.CompletedProcess(argv, 0)

    def test_build_pins_offline_inputs_flags_and_output_identities(self):
        record = dependency.build(self.args, environment=self.env, run=self.fake_run)
        self.assertEqual(record["status"], "BUILT")
        self.assertEqual(len(self.calls), 7)
        flags = self.calls[4]
        gcc_flag = "--gcc-toolchain=" + str(self.args.rocm_root)
        for arg in ("-DCMAKE_REQUIRE_FIND_PACKAGE_ROCmCMakeBuildTools=ON", "-DBUILD_TESTING=OFF",
                    "-DHIPFORT_EXTENDED_TESTS=OFF", "-DHIPFORT_BUILD_NVPTX=OFF",
                    "-DCMAKE_C_FLAGS=" + gcc_flag,
                    "-DCMAKE_Fortran_FLAGS=" + gcc_flag + " -fno-fast-math -ffp-contract=off"):
            self.assertIn(arg, flags)
        self.assertEqual(self.calls[3], [self.env["CPU_CC"], "--version"])
        self.assertEqual(record["tools"]["gcc"]["path"], self.env["CPU_CC"])
        self.assertIn("-DROCmCMakeBuildTools_DIR=" + str(self.args.output / "rocm-cmake/share/rocmcmakebuildtools/cmake"), flags)
        self.assertEqual(len(record["outputs"]), 9)
        self.assertTrue(all(len(p["sha256"]) == 64 for p in record["outputs"]))
        self.assertEqual(record["compatibility"], "UNVERIFIED_UNTIL_SUITE_BUILD_AND_SMOKE")
        self.assertEqual(Path(record["commands"][4]["stdout"]).read_text(), "fake stdout\n")
        self.assertEqual(Path(record["commands"][4]["stderr"]).read_text(), "fake stderr\n")
        with self.assertRaises(FileExistsError):
            dependency.build(self.args, environment=self.env, run=self.fake_run)
        self.assertEqual(len(self.calls), 7)

    def test_missing_approval_or_changed_archive_stops_before_commands(self):
        with self.assertRaisesRegex(ValueError, "approval"):
            dependency.build(self.args, environment={}, run=self.fake_run)
        self.assertFalse(self.args.output.exists())
        for compiler in ("", "relative/gcc", str(self.root / "missing/bin/gcc")):
            with self.subTest(compiler=compiler), self.assertRaisesRegex(ValueError, "Loaded GCC"):
                dependency.build(self.args, environment=dict(self.env, CPU_CC=compiler), run=self.fake_run)
            self.assertFalse(self.args.output.exists())
        self.assertEqual(self.calls, [])
        (self.archives / "hipfort.tar.gz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            dependency.build(self.args, environment=self.env, run=self.fake_run)
        self.assertEqual(self.calls, [])
        self.assertEqual(json.loads((self.args.output / "summary.json").read_text())["status"], "FAILED")

    def test_failed_command_preserves_evidence_without_retry(self):
        def failed(argv, **kwargs):
            self.calls.append(argv)
            return subprocess.CompletedProcess(argv, 9)
        with self.assertRaisesRegex(ValueError, "command failed"):
            dependency.build(self.args, environment=self.env, run=failed)
        record = json.loads((self.args.output / "summary.json").read_text())
        self.assertEqual(record["status"], "FAILED")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(record["commands"][0]["exit_code"], 9)


if __name__ == "__main__":
    unittest.main()

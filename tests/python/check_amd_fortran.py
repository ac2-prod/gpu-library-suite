#!/usr/bin/env python3
"""Focused built-CPU check: no GPU, scheduler, remote access or new packages."""
import argparse
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gpu_suite.config import load_config
from gpu_suite.runner import load_manifest, load_build_metadata, validate_manifest_artifacts, _amd_dependency_names
from gpu_suite.results_io import parse_stdout_prefix
from gpu_suite.schema import validate_raw_result


class AmdFortranTests(unittest.TestCase):
    def test_config_and_manifest_identity(self):
        config = load_config(ROOT / "configs/amd/fortran/smoke.json")
        self.assertEqual(config["source_language"], "fortran")
        self.assertEqual(config["vendor"], "amd")
        manifest, _ = load_manifest(BUILD / "partial-manifest.json")
        metadata = load_build_metadata([BUILD / "build-metadata.json"])
        validate_manifest_artifacts(manifest, metadata, "smoke", True)
        # The producer must preserve language identity for CPU and HIP entries.
        self.assertTrue(manifest["entries"])
        for entry in manifest["entries"]:
            self.assertEqual(entry["compiler_language"], "fortran")
        # GPU binaries require the hipFORT identity in addition to roc/HIP libs.
        self.assertEqual(_amd_dependency_names({
            "library": "rocfft", "implementation": "hip", "compiler_language": "fortran"}),
            {"hip", "rocfft", "hipfort"})

    def test_intrinsic_trials_scopes_and_formats(self):
        for library, stem in (("rocrand", "rand"), ("rocthrust", "reduce")):
            for scope in ("compute", "end-to-end"):
                for fmt in ("jsonl", "csv"):
                    with self.subTest(library=library, scope=scope, format=fmt):
                        command = self.command(library, stem, scope, fmt)
                        run = subprocess.run(command, text=True, capture_output=True, check=False)
                        self.assertEqual(run.returncode, 0, run.stderr)
                        rows, error = parse_stdout_prefix(run.stdout, fmt)
                        self.assertIsNone(error)
                        self.assertEqual(len(rows), 2)
                        for row in rows:
                            validate_raw_result(row)
                            self.assertEqual(row["benchmark"], library)
                            self.assertEqual(row["parameters"]["source_language"], "fortran")
                            self.assertEqual(row["verification_status"], "pass")
                            self.assertEqual(row["cpu_threads_effective"], 1)
                        self.assertEqual(rows[0]["verification_metrics"], rows[1]["verification_metrics"])

    def test_reject_wrong_cpu_backend(self):
        command = self.command("rocrand", "rand", "compute", "jsonl")
        command += ["--cpu-backend", "cpu-std-random-serial"]
        run = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("not compiled in", run.stderr)

    @staticmethod
    def command(library, stem, scope, fmt):
        executable = BUILD / "amd/fortran" / library / (stem + "_cpu_bench")
        return [str(executable), "--size", "65536", "--scope", scope,
                "--warmup", "1", "--repeat", "2", "--trials", "2",
                "--verify", "true", "--output", "-", "--format", fmt]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, type=Path)
    args = parser.parse_args()
    BUILD = args.build.resolve()
    unittest.main(argv=[sys.argv[0]], verbosity=2)

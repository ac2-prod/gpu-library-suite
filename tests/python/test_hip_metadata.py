"""Exercise HIP metadata with fake APIs and the real common result/JSON lifecycle.

The executable uses host compilers only. Success records are synthetic metadata
fixtures, never evidence of GPU execution or numerical verification.
"""
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from gpu_suite.results_io import load_raw_results
from gpu_suite.schema import validate_raw_result

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("cmake") and shutil.which("cc") and shutil.which("c++"),
                     "CMake and host C/C++ compilers are required")
class HipMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory(prefix="hip-metadata-")
        cls.addClassCleanup(temporary.cleanup)
        cls.directory = Path(temporary.name)
        cls.environment = {key: value for key, value in os.environ.items()
                           if not key.startswith("GPU_SUITE_")}
        cls.build = cls.directory / "build"
        cls.executable = cls.directory / "hip-metadata"
        cls.checked(["cmake", "-S", ROOT, "-B", cls.build,
                     "-DGPU_SUITE_BUILD_CPU=OFF", "-DGPU_SUITE_BUILD_CUDA=OFF",
                     "-DGPU_SUITE_BUILD_OPENACC=OFF", "-DGPU_SUITE_BUILD_HIP=OFF",
                     "-DGPU_SUITE_BUILD_OPENMP=OFF", "-DGPU_SUITE_BUILD_EXAMPLES=OFF",
                     "-DGPU_SUITE_BUILD_BENCHMARKS=OFF", "-DGPU_SUITE_BUILD_TESTS=OFF"])
        cls.checked(["cmake", "--build", cls.build, "--target", "gpu_suite_common",
                     "--parallel", "2"])
        cls.checked(["c++", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic",
                     "-I", ROOT / "tests/fixtures/rocm/include",
                     "-I", ROOT / "common/c-cpp/include",
                     ROOT / "tests/cpp/test_hip_metadata.cpp",
                     cls.build / "common/c-cpp/libgpu_suite_common.a",
                     "-o", cls.executable])

    @classmethod
    def checked(cls, argv):
        result = subprocess.run([str(arg) for arg in argv], env=cls.environment,
                                capture_output=True, text=True, check=False)
        if result.returncode:
            raise AssertionError("command failed: {}\n{}{}".format(
                argv, result.stdout, result.stderr))
        return result

    def invoke(self, scenario="success", output_format="jsonl", implementation="hip"):
        return subprocess.run([str(self.executable), scenario, output_format, implementation],
                              env=self.environment, capture_output=True, text=True, check=False)

    def check_success(self, output_format):
        for implementation in ("hip", "openmp"):
            with self.subTest(implementation=implementation):
                result = self.invoke(output_format=output_format, implementation=implementation)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, "")
                path = self.directory / (implementation + "." + output_format)
                path.write_text(result.stdout)
                rows = load_raw_results(path)
                self.assertEqual(len(rows), 1)
                row = validate_raw_result(rows[0])
                self.assertEqual(row["implementation"], implementation)
                self.assertEqual(row["status"], "success")
                self.assertEqual(row["verification_status"], "skipped")
                self.assertEqual(row["library_version"], "fixture-rocfft-1.0")
                self.assertEqual(row["gpu_name"], "Fixture GPU")
                self.assertIsNone(row["gpu_uuid"])
                self.assertIsNone(row["cuda_driver_version"])
                self.assertIsNone(row["cuda_runtime_version"])
                runtime = row["parameters"]["runtime"]
                self.assertEqual(set(runtime), {"build_metadata_sha256",
                    "hip_runtime_version", "hip_driver_version"})
                self.assertRegex(runtime["build_metadata_sha256"], "^[0-9a-f]{64}$")
                self.assertEqual(runtime["hip_runtime_version"], 70200000)
                self.assertEqual(runtime["hip_driver_version"], 70100000)
                if output_format == "jsonl":
                    # Check emitted bytes, not only the parser's surviving value.
                    self.assertEqual(result.stdout.count('"runtime":'), 1)
                    self.assertEqual(result.stdout.count('"build_metadata_sha256":'), 1)
                else:
                    self.assertEqual(result.stdout.count("result_schema_version"), 1)
                    self.assertEqual(len(list(csv.DictReader(io.StringIO(result.stdout)))), 1)

    def test_real_result_lifecycle_augments_runtime_jsonl(self):
        self.check_success("jsonl")

    def test_real_result_lifecycle_augments_runtime_csv(self):
        self.check_success("csv")

    def test_each_runtime_api_failure_names_operation_and_status(self):
        for operation in ("hipGetDeviceProperties", "hipRuntimeGetVersion",
                          "hipDriverGetVersion", "rocfft_get_version_string"):
            with self.subTest(operation=operation):
                result = self.invoke(operation)
                self.assertEqual(result.returncode, 1)
                self.assertIn(operation + " failed (status=1", result.stderr)
                self.assertEqual(set(json.loads(result.stdout)["runtime"]),
                                 {"build_metadata_sha256"})

    def test_repeated_registration_rejects_without_overwriting_metadata(self):
        result = self.invoke("duplicate")
        self.assertEqual(result.returncode, 1)
        self.assertIn("runtime versions are already registered", result.stderr)
        self.assertEqual(json.loads(result.stdout)["runtime"]["hip_runtime_version"], 70200000)

    def test_missing_or_invalid_runtime_rejects_before_api_calls(self):
        for scenario in ("missing-runtime", "nonobject-runtime", "missing-build"):
            with self.subTest(scenario=scenario):
                result = self.invoke(scenario)
                self.assertEqual(result.returncode, 1)
                self.assertIn("runtime build metadata is missing or invalid", result.stderr)

    def test_node_local_gpu_mismatch_rejects_without_overwriting_metadata(self):
        result = self.invoke("device-mismatch")
        self.assertEqual(result.returncode, 1)
        self.assertIn("differs from node-local GPU metadata", result.stderr)
        self.assertEqual(set(json.loads(result.stdout)["runtime"]), {"build_metadata_sha256"})

    def test_borrowed_object_lookup_and_duplicate_key_ownership(self):
        result = self.invoke("getter")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()

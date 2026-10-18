"""Provider/runtime provenance checks using inert local files only."""

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gpu_suite.fftw_metadata import validate_fftw_threading
from gpu_suite.hashing import deterministic_json_sha256, sha256_file
from gpu_suite.runner import _validate_amd_dependencies, core_raw_parameters
from gpu_suite.validation import validate_campaign

SPEC = importlib.util.spec_from_file_location(
    "threading_smoke_evidence", ROOT / "jobs/sirius/rocfft_smoke_evidence.py")
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


class FftwMetadataTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="fftw-provenance-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.entry = {"library": "rocfft", "implementation": "cpu",
                      "supported_cpu_backends": ["cpu-fftw-serial", "cpu-fftw-threaded"]}

    def metadata(self, backend):
        names = ["fftw3f"]
        if backend == "pthread":
            names.append("fftw3f_threads")
        elif backend == "openmp":
            names += ["fftw3f_omp", "fftw_openmp_runtime_0"]
        dependencies = {}
        for name in names:
            path = self.root / (name + ".so")
            path.write_bytes(("inert fixture " + name).encode())
            dependencies[name] = {"path": str(path), "sha256": sha256_file(path)}
        return {"vendor": "amd", "build_profile": "cpu", "dependencies": dependencies,
                "fftw_threading": {"requested": "AUTO", "backend": backend,
                                   "compile_options": ["-fopenmp"] if backend == "openmp" else [],
                                   "link_options": ["-fopenmp"] if backend == "openmp" else [],
                                   "dependency_names": names}}

    def test_all_three_provider_identities_are_accepted_without_loading(self):
        for backend in ("pthread", "openmp", "combined"):
            with self.subTest(backend=backend):
                metadata = self.metadata(backend)
                _validate_amd_dependencies(self.entry, metadata)
                evidence.check_configured_dependencies(metadata, {
                    item["path"]: item["sha256"] for item in metadata["dependencies"].values()})

    def test_legacy_pthread_and_serial_metadata_remain_readable(self):
        metadata = self.metadata("pthread")
        del metadata["fftw_threading"]
        _validate_amd_dependencies(self.entry, metadata)
        metadata["dependencies"].pop("fftw3f_threads")
        with self.assertRaisesRegex(ValueError, "legacy threaded"):
            _validate_amd_dependencies(self.entry, metadata)
        self.entry["supported_cpu_backends"] = ["cpu-fftw-serial"]
        _validate_amd_dependencies(self.entry, metadata)

    def test_mislabeled_or_mixed_providers_are_rejected(self):
        for backend in ("pthread", "openmp", "combined"):
            for wrong in ("pthread", "openmp", "combined", "none"):
                if wrong == backend:
                    continue
                metadata = self.metadata(backend)
                metadata["fftw_threading"]["backend"] = wrong
                with self.subTest(backend=backend, wrong=wrong), self.assertRaises(ValueError):
                    _validate_amd_dependencies(self.entry, metadata)

    def test_explicit_request_mismatch_and_unknown_modes_are_rejected(self):
        for request in ("OPENMP", "NONE", "invented"):
            metadata = self.metadata("pthread")
            metadata["fftw_threading"]["requested"] = request
            with self.subTest(request=request), self.assertRaises(ValueError):
                validate_fftw_threading(metadata)

    def test_incomplete_duplicate_and_malformed_metadata_are_rejected(self):
        for mutation in ("missing", "extra", "duplicate", "missing_dependency", "options"):
            metadata = self.metadata("pthread")
            thread = metadata["fftw_threading"]
            if mutation == "missing":
                del thread["backend"]
            elif mutation == "extra":
                thread["guessed_version"] = "3.6.10"
            elif mutation == "duplicate":
                thread["dependency_names"].append("fftw3f")
            elif mutation == "missing_dependency":
                thread["dependency_names"].remove("fftw3f_threads")
            else:
                thread["link_options"] = "-pthread"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_fftw_threading(metadata)

    def test_thread_runtime_content_drift_is_rejected_before_launch(self):
        metadata = self.metadata("openmp")
        runtime = Path(metadata["dependencies"]["fftw_openmp_runtime_0"]["path"])
        runtime.write_bytes(b"different runtime")
        with self.assertRaisesRegex(ValueError, "dependency changed"):
            _validate_amd_dependencies(self.entry, metadata)

    def test_concrete_pthread_runtime_is_bound_for_separate_and_combined(self):
        for backend in ("pthread", "combined"):
            metadata = self.metadata(backend)
            runtime = self.root / "pthread-runtime.a"
            runtime.write_bytes(b"inert static thread runtime")
            metadata["dependencies"]["fftw_pthread_runtime_0"] = {
                "path": str(runtime), "sha256": sha256_file(runtime)}
            metadata["fftw_threading"]["dependency_names"].append("fftw_pthread_runtime_0")
            _validate_amd_dependencies(self.entry, metadata)
            runtime.write_bytes(b"runtime changed")
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "dependency changed"):
                _validate_amd_dependencies(self.entry, metadata)

    def test_runtime_resolving_to_another_path_is_rejected(self):
        metadata = self.metadata("openmp")
        resolved = {item["path"]: item["sha256"] for item in metadata["dependencies"].values()}
        item = metadata["dependencies"]["fftw_openmp_runtime_0"]
        del resolved[item["path"]]
        other = self.root / "other-runtime.so"
        other.write_bytes(Path(item["path"]).read_bytes())
        resolved[str(other)] = sha256_file(other)
        with self.assertRaisesRegex(ValueError, "loaded dependency differs"):
            evidence.check_configured_dependencies(metadata, resolved)

    def test_openmp_needs_runtime_evidence_and_valid_runtime_names(self):
        metadata = self.metadata("openmp")
        metadata["dependencies"].pop("fftw_openmp_runtime_0")
        thread = metadata["fftw_threading"]
        thread["dependency_names"].remove("fftw_openmp_runtime_0")
        thread["compile_options"] = []
        thread["link_options"] = []
        with self.assertRaisesRegex(ValueError, "runtime link evidence"):
            validate_fftw_threading(metadata)
        metadata = self.metadata("openmp")
        metadata["dependencies"]["fftw_openmp_runtime_3"] = metadata["dependencies"].pop("fftw_openmp_runtime_0")
        with self.assertRaisesRegex(ValueError, "runtime dependency names"):
            validate_fftw_threading(metadata)

    def test_provider_and_link_flags_change_campaign_identity_not_math(self):
        from test_rocfft_tools import campaign, ZERO
        config, manifest, _, _, records = campaign()
        original = self.metadata("pthread")
        digest = deterministic_json_sha256(original)
        for entry in manifest["entries"]:
            entry["build_metadata_sha256"] = digest
        for row in records:
            row["parameters"]["runtime"]["build_metadata_sha256"] = digest
        self.assertEqual(validate_campaign(records, config, ZERO, manifest)["validation_status"], "pass")
        for mutation in ("backend", "flags"):
            metadata = self.metadata("combined") if mutation == "backend" else copy.deepcopy(original)
            if mutation == "flags":
                metadata["fftw_threading"]["link_options"] = ["-pthread"]
            changed = copy.deepcopy(records)
            changed[0]["parameters"]["runtime"]["build_metadata_sha256"] = deterministic_json_sha256(metadata)
            self.assertEqual(core_raw_parameters("rocfft", changed[0]["parameters"]),
                             core_raw_parameters("rocfft", records[0]["parameters"]))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_campaign(changed, config, ZERO, manifest)


if __name__ == "__main__":
    unittest.main()

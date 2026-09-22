import copy
import unittest
from pathlib import Path

from gpu_suite.manifest import (
    ManifestError,
    artifact_id,
    merge_entries,
    validate_build_metadata,
)
from gpu_suite.strict_json import dump_bytes, load


ROOT = Path(__file__).resolve().parents[2]


def manifest_entry(target="fft_cpu", role="example", path="/build/fft_cpu"):
    entry = {
        "artifact_id": "",
        "target_name": target,
        "build_profile": "cpu-cuda",
        "backend_variant": "fftw-threaded",
        "executable_path": path,
        "library": "cufft",
        "implementation": "cpu",
        "executable_role": role,
        "build_type": "Release",
        "binary_sha256": "1" * 64,
        "build_metadata_sha256": "2" * 64,
        "compiler": "AppleClang",
        "compiler_language": "c",
        "compiler_version": "21",
        "global_configure_flags": "-O3",
        "git_metadata_available": True,
        "git_commit": "abc",
        "git_dirty": True,
        "supported_cpu_backends": ["cpu-fftw-threaded", "cpu-fftw-serial"],
    }
    entry["artifact_id"] = artifact_id(entry)
    return entry


class ManifestTests(unittest.TestCase):
    def test_artifact_identity_is_deterministic(self):
        first = manifest_entry()
        second = dict(reversed(list(first.items())))
        self.assertEqual(artifact_id(first), artifact_id(second))

    def test_merge_rejects_semantic_and_build_duplicates(self):
        entry = manifest_entry()
        with self.assertRaises(ManifestError):
            merge_entries([[entry], [copy.deepcopy(entry)]])

        other = manifest_entry()
        other["library"] = "cublas"
        other["executable_role"] = "benchmark"
        other["artifact_id"] = artifact_id(other)
        with self.assertRaises(ManifestError):
            merge_entries([[entry, other]])

    def test_conflicting_and_duplicate_paths_are_rejected(self):
        first = manifest_entry()
        second = manifest_entry(
            target="fft_cpu_bench", role="benchmark", path=first["executable_path"]
        )
        second["binary_sha256"] = "3" * 64
        second["artifact_id"] = artifact_id(second)
        with self.assertRaises(ManifestError):
            merge_entries([[first, second]])

        second["binary_sha256"] = first["binary_sha256"]
        second["artifact_id"] = artifact_id(second)
        with self.assertRaises(ManifestError):
            merge_entries([[first, second]])

    def test_build_profile_and_filename_conflicts_are_rejected(self):
        entry = manifest_entry()
        entry["build_profile"] = "openacc"
        entry["artifact_id"] = artifact_id(entry)
        with self.assertRaises(ManifestError):
            merge_entries([[entry]])
        entry = manifest_entry(path="/build/not-the-target")
        with self.assertRaises(ManifestError):
            merge_entries([[entry]])

    def test_build_metadata_comparison(self):
        entry = manifest_entry()
        metadata = {
            "git_metadata_available": entry["git_metadata_available"],
            "git_commit": entry["git_commit"],
            "git_dirty": entry["git_dirty"],
            "build_type": entry["build_type"],
            "build_profile": entry["build_profile"],
            "c": {
                "compiler": entry["compiler"],
                "global_configure_flags": "-O3",
                "compiler_version": entry["compiler_version"],
            },
        }
        validate_build_metadata(entry, metadata)
        metadata["c"]["compiler"] = "wrong"
        with self.assertRaises(ManifestError):
            validate_build_metadata(entry, metadata)

    def test_unavailable_git_metadata_is_explicit(self):
        entry = manifest_entry()
        entry["git_metadata_available"] = False
        entry["git_commit"] = None
        entry["git_dirty"] = None
        self.assertEqual(entry, merge_entries([[entry]])[0])
        entry["git_dirty"] = False
        with self.assertRaises(ManifestError):
            merge_entries([[entry]])

    def test_canonical_example_manifest_is_valid_and_deterministic(self):
        path = ROOT / "configs" / "executables.json.example"
        document = load(path)
        entries = merge_entries([document["entries"]])
        self.assertEqual(len(entries), 1)
        self.assertEqual(path.read_bytes(), dump_bytes(document))


if __name__ == "__main__":
    unittest.main()

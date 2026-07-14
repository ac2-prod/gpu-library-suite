import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hash_source_snapshot import git_source_paths
from gpu_suite.hashing import source_snapshot_sha256


class SourceSnapshotHashTests(unittest.TestCase):
    def test_order_is_canonical_and_content_changes_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "a.txt").write_bytes(b"alpha\n")
            (root / "nested" / "b.txt").write_bytes(b"beta\n")
            first = source_snapshot_sha256(
                root, ["nested/b.txt", "a.txt"]
            )
            self.assertEqual(
                first,
                source_snapshot_sha256(root, ["a.txt", "nested/b.txt"]),
            )
            (root / "nested" / "b.txt").write_bytes(b"changed\n")
            self.assertNotEqual(
                first,
                source_snapshot_sha256(root, ["a.txt", "nested/b.txt"]),
            )

    @unittest.skipUnless(os.name == "posix", "POSIX executable mode test")
    def test_executable_bit_is_part_of_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "script.sh"
            path.write_bytes(b"#!/bin/sh\n")
            path.chmod(0o644)
            regular = source_snapshot_sha256(root, ["script.sh"])
            path.chmod(0o755)
            executable = source_snapshot_sha256(root, ["script.sh"])
            self.assertNotEqual(regular, executable)

    def test_duplicate_traversal_and_non_file_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "file").write_bytes(b"data")
            (root / "directory").mkdir()
            with self.assertRaises(ValueError):
                source_snapshot_sha256(root, ["file", "file"])
            with self.assertRaises(ValueError):
                source_snapshot_sha256(root, ["../file"])
            with self.assertRaises(ValueError):
                source_snapshot_sha256(root, ["directory"])

    def test_empty_snapshot_has_stable_versioned_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            value = source_snapshot_sha256(Path(temporary), [])
            self.assertEqual(len(value), 64)
            self.assertEqual(value, source_snapshot_sha256(temporary, []))

    def test_git_inventory_unions_tracked_and_nonignored_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tracked").write_bytes(b"tracked")
            (root / "untracked").write_bytes(b"untracked")
            completed = mock.Mock(
                returncode=0,
                stdout=b"tracked\x00deleted\x00untracked\x00",
                stderr=b"",
            )
            with mock.patch("hash_source_snapshot.subprocess.run",
                            return_value=completed) as run:
                self.assertEqual(
                    git_source_paths(root), ["tracked", "untracked"]
                )
            arguments = run.call_args.args[0]
            self.assertEqual(arguments[:3], ["git", "-C", str(root)])
            self.assertIn("--cached", arguments)
            self.assertIn("--others", arguments)
            self.assertIn("--exclude-standard", arguments)


if __name__ == "__main__":
    unittest.main()

import sys
from pathlib import Path

from gpu_suite.hashing import sha256_file
from gpu_suite.manifest import artifact_id, merge_entries
from gpu_suite.strict_json import load


def main():
    if len(sys.argv) < 3:
        print("usage: check_partial_manifest.py MANIFEST TARGET...", file=sys.stderr)
        return 2
    manifest = load(Path(sys.argv[1]))
    if not isinstance(manifest, dict) or manifest.get("manifest_schema_version") != 1:
        raise ValueError("invalid partial manifest document")
    entries = merge_entries([manifest.get("entries", [])])
    actual_targets = {entry["target_name"] for entry in entries}
    expected_targets = set(sys.argv[2:])
    if actual_targets != expected_targets:
        raise ValueError(
            "target mismatch: actual={0} expected={1}".format(
                sorted(actual_targets), sorted(expected_targets)
            )
        )
    for entry in entries:
        if entry["artifact_id"] != artifact_id(entry):
            raise ValueError("artifact identity mismatch")
        if entry["binary_sha256"] != sha256_file(entry["executable_path"]):
            raise ValueError("binary hash mismatch")
        if entry["build_profile"] != "cpu-cuda":
            raise ValueError("unexpected build profile")
    return 0


if __name__ == "__main__":
    sys.exit(main())

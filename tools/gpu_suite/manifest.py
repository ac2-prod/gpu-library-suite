"""Deterministic executable-manifest identity and merge validation."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .hashing import deterministic_json_sha256


class ManifestError(ValueError):
    pass


REQUIRED_FIELDS = {
    "artifact_id",
    "target_name",
    "build_profile",
    "backend_variant",
    "executable_path",
    "library",
    "implementation",
    "executable_role",
    "build_type",
    "binary_sha256",
    "build_metadata_sha256",
    "compiler",
    "compiler_language",
    "compiler_version",
    "global_configure_flags",
    "git_metadata_available",
    "git_commit",
    "git_dirty",
    "supported_cpu_backends",
}
IDENTITY_FIELDS = (
    "library",
    "implementation",
    "executable_role",
    "target_name",
    "build_profile",
    "backend_variant",
    "binary_sha256",
)


def artifact_id(entry: Mapping[str, Any]) -> str:
    identity = {name: entry[name] for name in IDENTITY_FIELDS}
    return deterministic_json_sha256(identity, trailing_newline=False)


def validate_entry(entry: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise ManifestError("manifest entry must be an object")
    missing = REQUIRED_FIELDS.difference(entry.keys())
    if missing:
        raise ManifestError("missing manifest fields: {0}".format(sorted(missing)))
    if entry["build_profile"] not in {"cpu-cuda", "openacc"}:
        raise ManifestError("invalid build_profile")
    if entry["implementation"] not in {"cpu", "cuda", "openacc"}:
        raise ManifestError("invalid implementation")
    expected_profile = "openacc" if entry["implementation"] == "openacc" else "cpu-cuda"
    if entry["build_profile"] != expected_profile:
        raise ManifestError("implementation and build_profile conflict")
    if entry["executable_role"] not in {"example", "benchmark"}:
        raise ManifestError("invalid executable_role")
    if entry["compiler_language"] not in {"c", "cxx", "cuda"}:
        raise ManifestError("invalid compiler_language")
    if Path(entry["executable_path"]).name != entry["target_name"]:
        raise ManifestError("target_name and executable filename differ")
    if entry["artifact_id"] != artifact_id(entry):
        raise ManifestError("artifact_id does not match entry identity")
    for name in ("binary_sha256", "build_metadata_sha256", "artifact_id"):
        value = entry[name]
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ManifestError("invalid {0}".format(name))
    available = entry["git_metadata_available"]
    if not isinstance(available, bool):
        raise ManifestError("git_metadata_available must be boolean")
    if available:
        if not isinstance(entry["git_commit"], str) or entry["git_commit"] == "":
            raise ManifestError("available Git metadata requires a commit")
        if not isinstance(entry["git_dirty"], bool):
            raise ManifestError("available Git metadata requires dirty state")
    elif entry["git_commit"] is not None or entry["git_dirty"] is not None:
        raise ManifestError("unavailable Git metadata requires null fields")
    supported = entry["supported_cpu_backends"]
    if (
        not isinstance(supported, list)
        or any(not isinstance(value, str) or value == "" for value in supported)
        or len(supported) != len(set(supported))
    ):
        raise ManifestError("invalid supported_cpu_backends")
    if entry["implementation"] != "cpu" and supported:
        raise ManifestError("GPU artifacts cannot advertise CPU backends")
    if not isinstance(entry["global_configure_flags"], str):
        raise ManifestError("global_configure_flags must be a string")
    for name in REQUIRED_FIELDS.difference({
        "git_metadata_available", "git_commit", "git_dirty",
        "global_configure_flags", "supported_cpu_backends",
    }):
        if not isinstance(entry[name], str) or entry[name] == "":
            raise ManifestError("{0} must be a nonempty string".format(name))
    return dict(entry)


def merge_entries(partials: Iterable[Iterable[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    """Merge partial manifests while rejecting every ambiguous identity."""

    merged = []  # type: List[Dict[str, Any]]
    semantic = set()
    build = set()
    artifact_ids = set()
    paths = {}  # type: Dict[str, Tuple[str, str]]
    for partial in partials:
        for raw_entry in partial:
            entry = validate_entry(raw_entry)
            semantic_key = (
                entry["library"],
                entry["implementation"],
                entry["executable_role"],
            )
            build_key = (
                entry["target_name"],
                entry["build_profile"],
                entry["backend_variant"],
            )
            if semantic_key in semantic:
                raise ManifestError("duplicate semantic manifest key")
            if build_key in build:
                raise ManifestError("duplicate build manifest key")
            if entry["artifact_id"] in artifact_ids:
                raise ManifestError("duplicate artifact_id")
            path_identity = (entry["binary_sha256"], entry["build_metadata_sha256"])
            old_path_identity = paths.get(entry["executable_path"])
            if old_path_identity is not None:
                if old_path_identity != path_identity:
                    raise ManifestError("conflicting executable path")
                raise ManifestError("duplicate executable path")
            semantic.add(semantic_key)
            build.add(build_key)
            artifact_ids.add(entry["artifact_id"])
            paths[entry["executable_path"]] = path_identity
            merged.append(entry)
    merged.sort(
        key=lambda item: (
            item["library"],
            item["implementation"],
            item["executable_role"],
            item["target_name"],
        )
    )
    return merged


def validate_build_metadata(entry: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    """Confirm the runner-selected artifact matches generated build metadata."""

    comparisons = {
        "git_commit": "git_commit",
        "git_dirty": "git_dirty",
        "git_metadata_available": "git_metadata_available",
        "build_type": "build_type",
        "build_profile": "build_profile",
    }
    for entry_name, metadata_name in comparisons.items():
        if entry.get(entry_name) != metadata.get(metadata_name):
            raise ManifestError(
                "build metadata mismatch for {0}".format(entry_name)
            )
    language = entry["compiler_language"]
    compiler = metadata.get(language)
    if not isinstance(compiler, Mapping):
        raise ManifestError("build metadata lacks compiler language " + language)
    if entry["compiler"] != compiler.get("compiler"):
        raise ManifestError("build metadata mismatch for compiler")
    if entry["compiler_version"] != compiler.get("compiler_version"):
        raise ManifestError("build metadata mismatch for compiler_version")
    if entry["global_configure_flags"] != compiler.get(
        "global_configure_flags"
    ):
        raise ManifestError("build metadata mismatch for global_configure_flags")

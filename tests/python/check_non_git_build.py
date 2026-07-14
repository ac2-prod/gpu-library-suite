#!/usr/bin/env python3
"""Configure, build, and run core CTest checks from a source copy without .git."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List


class NonGitBuildError(RuntimeError):
    pass


def run(arguments: List[str], cwd: Path) -> None:
    completed = subprocess.run(
        arguments, cwd=str(cwd), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    if completed.returncode != 0:
        raise NonGitBuildError(
            "command failed: {0}\n{1}".format(" ".join(arguments),
                                               completed.stdout)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--fixture-root", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        arguments.fixture_root.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="non-git-",
                                     dir=str(arguments.fixture_root)))
        source_copy = work / "source"
        build = work / "build"
        shutil.copytree(
            arguments.source, source_copy,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )
        if (source_copy / ".git").exists():
            raise NonGitBuildError("source copy unexpectedly contains .git")
        run([
            arguments.cmake, "-S", str(source_copy), "-B", str(build),
            "-DCMAKE_BUILD_TYPE=Release", "-DGPU_SUITE_BUILD_CPU=ON",
            "-DGPU_SUITE_BUILD_CUDA=OFF", "-DGPU_SUITE_BUILD_OPENACC=OFF",
            "-DGPU_SUITE_BUILD_TESTS=ON",
        ], work)
        metadata_path = build / "build-metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("git_metadata_available") is not False:
            raise NonGitBuildError("non-Git metadata availability is not false")
        if metadata.get("git_commit") is not None or metadata.get("git_dirty") is not None:
            raise NonGitBuildError("unknown Git fields must be null")
        run([arguments.cmake, "--build", str(build), "--parallel", "2"], work)
        run([
            arguments.cmake, "--build", str(build), "--target",
            "gpu_suite_partial_manifest", "--parallel", "2",
        ], work)
        manifest = json.loads(
            (build / "partial-manifest.json").read_text(encoding="utf-8")
        )
        if not manifest.get("entries"):
            raise NonGitBuildError("non-Git manifest has no entries")
        for entry in manifest["entries"]:
            if entry.get("git_metadata_available") is not False:
                raise NonGitBuildError("manifest Git availability is not false")
            if entry.get("git_commit") is not None or entry.get("git_dirty") is not None:
                raise NonGitBuildError("manifest unknown Git fields must be null")
        ctest = str(Path(arguments.cmake).with_name("ctest"))
        run([
            ctest, "--test-dir", str(build), "--output-on-failure", "-E",
            "^non_git_source_build$",
        ], work)
    except (OSError, ValueError, NonGitBuildError) as error:
        print("non-Git build regression failed: {0}".format(error),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

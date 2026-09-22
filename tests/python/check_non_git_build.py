#!/usr/bin/env python3
"""Configure, build, and run core CTest checks from a source copy without .git."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Set


class NonGitBuildError(RuntimeError):
    pass


def run(
    arguments: List[str],
    cwd: Path,
    environment: Optional[Mapping[str, str]] = None,
) -> None:
    completed = subprocess.run(
        arguments, cwd=str(cwd), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
        env=None if environment is None else dict(environment),
    )
    if completed.returncode != 0:
        raise NonGitBuildError(
            "command failed: {0}\n{1}".format(" ".join(arguments),
                                               completed.stdout)
        )


def isolated_git_environment(source_copy: Path) -> Dict[str, str]:
    environment = dict(os.environ)
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR"):
        environment.pop(name, None)
    environment["GIT_CEILING_DIRECTORIES"] = str(source_copy.resolve().parent)
    return environment


def assert_not_git_worktree(
    source_copy: Path,
    environment: Mapping[str, str],
) -> None:
    completed = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=str(source_copy),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=dict(environment),
    )
    if completed.returncode == 0:
        raise NonGitBuildError(
            "source copy is unexpectedly inside a Git worktree: {0}".format(
                completed.stdout.strip()
            )
        )


def source_copy_ignore(source_root: Path) -> Callable[[str, List[str]], Set[str]]:
    resolved_root = source_root.resolve()

    def ignore(directory: str, names: List[str]) -> Set[str]:
        current = Path(directory).resolve()
        relative = current.relative_to(resolved_root)
        ignored = {
            name for name in names
            if name == ".git" or name == "__pycache__" or name.endswith(".pyc")
        }
        ignored.update(name for name in names if name == "test-fixtures")
        if not relative.parts:
            ignored.update(
                name for name in names
                if name in {"build", "manual-validation", "results"}
            )
        if relative.parts == ("jobs", "pegasus") and "generated" in names:
            ignored.add("generated")
        for name in names:
            child = current / name
            if (
                name in {"CMakeFiles", "_deps"}
                or (child / "CMakeCache.txt").is_file()
                or (child / "CMakeFiles").is_dir()
            ):
                ignored.add(name)
        return ignored

    return ignore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--source", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        with tempfile.TemporaryDirectory(
            prefix="gpu-library-suite-non-git-"
        ) as temporary_directory:
            work = Path(temporary_directory)
            source_copy = work / "source"
            build = work / "build"
            shutil.copytree(
                arguments.source, source_copy,
                ignore=source_copy_ignore(arguments.source),
            )
            if (source_copy / ".git").exists():
                raise NonGitBuildError("source copy unexpectedly contains .git")
            environment = isolated_git_environment(source_copy)
            assert_not_git_worktree(source_copy, environment)
            run([
                arguments.cmake, "-S", str(source_copy), "-B", str(build),
                "-DCMAKE_BUILD_TYPE=Release", "-DGPU_SUITE_BUILD_CPU=ON",
                "-DGPU_SUITE_BUILD_CUDA=OFF",
                "-DGPU_SUITE_BUILD_OPENACC=OFF",
                "-DGPU_SUITE_BUILD_TESTS=ON",
            ], work, environment)
            metadata_path = build / "build-metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("git_metadata_available") is not False:
                raise NonGitBuildError(
                    "non-Git metadata availability is not false"
                )
            if (
                metadata.get("git_commit") is not None
                or metadata.get("git_dirty") is not None
            ):
                raise NonGitBuildError("unknown Git fields must be null")
            run([
                arguments.cmake, "--build", str(build), "--parallel", "2",
            ], work, environment)
            run([
                arguments.cmake, "--build", str(build), "--target",
                "gpu_suite_partial_manifest", "--parallel", "2",
            ], work, environment)
            manifest = json.loads(
                (build / "partial-manifest.json").read_text(encoding="utf-8")
            )
            if not manifest.get("entries"):
                raise NonGitBuildError("non-Git manifest has no entries")
            for entry in manifest["entries"]:
                if entry.get("git_metadata_available") is not False:
                    raise NonGitBuildError(
                        "manifest Git availability is not false"
                    )
                if (
                    entry.get("git_commit") is not None
                    or entry.get("git_dirty") is not None
                ):
                    raise NonGitBuildError(
                        "manifest unknown Git fields must be null"
                    )
            ctest = str(Path(arguments.cmake).with_name("ctest"))
            run([
                ctest, "--test-dir", str(build), "--output-on-failure", "-E",
                "^non_git_source_build$",
            ], work, environment)
    except (OSError, ValueError, NonGitBuildError) as error:
        print("non-Git build regression failed: {0}".format(error),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Hash all tracked and untracked, non-ignored repository source files."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from gpu_suite.hashing import source_snapshot_sha256


def git_source_paths(repository: Path) -> List[str]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError("git ls-files failed: " + diagnostic)
    fields = completed.stdout.split(b"\x00")
    if fields and fields[-1] == b"":
        fields.pop()
    paths = []  # type: List[str]
    for field in fields:
        try:
            relative = field.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError("source path is not valid UTF-8") from error
        path = repository / relative
        if path.exists() or path.is_symlink():
            paths.append(relative)
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    arguments = parser.parse_args(argv)
    try:
        repository = arguments.repository.resolve()
        paths = git_source_paths(repository)
        print(source_snapshot_sha256(repository, paths))
    except (OSError, ValueError) as error:
        print("source snapshot hashing failed: {0}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

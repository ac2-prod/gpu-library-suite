#!/usr/bin/env python3
"""Validate a scheduler identity and print its filesystem-safe token."""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from gpu_suite.scheduler import scheduler_job_path_token  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    token = subparsers.add_parser("token")
    token.add_argument("--scheduler", required=True)
    token.add_argument("--job-id", required=True)
    arguments = parser.parse_args(argv)
    try:
        value = scheduler_job_path_token(
            arguments.scheduler, arguments.job_id
        )
    except ValueError as error:
        print(
            "scheduler job ID validation failed: {0}".format(error),
            file=sys.stderr,
        )
        return 2
    if value is None:
        print("scheduler job ID validation failed: token is unavailable",
              file=sys.stderr)
        return 2
    print(value)
    return 0


if __name__ == "__main__":
    sys.exit(main())

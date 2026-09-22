#!/usr/bin/env python3
"""Validate a generated CTest inventory and selected test environments."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


class InventoryError(RuntimeError):
    pass


def property_value(test, name):
    for entry in test.get("properties", []):
        if entry.get("name") == name:
            return entry.get("value")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctest", required=True)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--require-test", action="append", default=[])
    parser.add_argument("--forbid-test", action="append", default=[])
    parser.add_argument("--environment-test")
    parser.add_argument(
        "--forbid-environment-prefix", action="append", default=[]
    )
    parser.add_argument("--require-label-prefix")
    arguments = parser.parse_args()

    completed = subprocess.run(
        [
            arguments.ctest,
            "--test-dir", str(arguments.build),
            "--show-only=json-v1",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise InventoryError(
            "CTest inventory command failed:\n{0}".format(completed.stdout)
        )
    document = json.loads(completed.stdout)
    tests = document.get("tests")
    if not isinstance(tests, list):
        raise InventoryError("CTest inventory has no tests array")
    by_name = {}
    for test in tests:
        name = test.get("name")
        if not isinstance(name, str) or not name:
            raise InventoryError("CTest inventory has an invalid test name")
        if name in by_name:
            raise InventoryError("duplicate CTest name: {0}".format(name))
        by_name[name] = test

    missing = sorted(set(arguments.require_test) - set(by_name))
    forbidden = sorted(set(arguments.forbid_test) & set(by_name))
    if missing:
        raise InventoryError(
            "required tests are absent: {0}".format(", ".join(missing))
        )
    if forbidden:
        raise InventoryError(
            "profile-forbidden tests are registered: {0}".format(
                ", ".join(forbidden)
            )
        )

    if arguments.require_label_prefix:
        unlabeled = []
        for name, test in by_name.items():
            labels = property_value(test, "LABELS")
            if not isinstance(labels, list):
                unlabeled.append(name)
                continue
            if not any(
                isinstance(label, str)
                and label.startswith(arguments.require_label_prefix)
                for label in labels
            ):
                unlabeled.append(name)
        if unlabeled:
            raise InventoryError(
                "tests lack a profile label: {0}".format(
                    ", ".join(sorted(unlabeled))
                )
            )

    if arguments.environment_test:
        test = by_name.get(arguments.environment_test)
        if test is None:
            raise InventoryError(
                "environment test is absent: {0}".format(
                    arguments.environment_test
                )
            )
        environment = property_value(test, "ENVIRONMENT")
        if not isinstance(environment, list):
            raise InventoryError(
                "test has no environment list: {0}".format(
                    arguments.environment_test
                )
            )
        contaminated = sorted(
            value for value in environment
            if isinstance(value, str)
            and any(
                value.startswith(prefix)
                for prefix in arguments.forbid_environment_prefix
            )
        )
        if contaminated:
            raise InventoryError(
                "profile-forbidden environment entries: {0}".format(
                    ", ".join(contaminated)
                )
            )

    print("CTest inventory validation: {0} tests passed".format(len(tests)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (InventoryError, OSError, ValueError) as error:
        print("CTest inventory validation failed: {0}".format(error),
              file=sys.stderr)
        sys.exit(1)

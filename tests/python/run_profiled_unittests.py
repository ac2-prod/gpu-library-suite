#!/usr/bin/env python3
"""Run the profile-independent or CPU-fixture Python unittest subset."""

import argparse
import sys
import unittest
from pathlib import Path


CPU_FIXTURE_MODULE_PREFIXES = (
    "test_cufft_fixture_runtime.",
    "test_overflow_runtime.",
    "test_phase3_fixture_runtime.",
)
CPU_FIXTURE_TEST_IDS = frozenset({
    "test_runner.RunnerTests."
    "test_real_fixture_runs_through_single_writer_and_validates",
})


def iter_test_cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_test_cases(item)
        else:
            yield item


def requires_cpu_fixture(test_case):
    test_id = test_case.id()
    return (
        test_id in CPU_FIXTURE_TEST_IDS
        or test_id.startswith(CPU_FIXTURE_MODULE_PREFIXES)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-directory", required=True, type=Path)
    parser.add_argument(
        "--profile", required=True, choices=("independent", "cpu-fixture")
    )
    arguments = parser.parse_args()

    start_directory = arguments.start_directory.resolve()
    sys.path.insert(0, str(start_directory))
    loader = unittest.TestLoader()
    discovered = loader.discover(str(start_directory), pattern="test_*.py")
    if loader.errors:
        for error in loader.errors:
            print(error, file=sys.stderr)
        return 1

    all_cases = list(iter_test_cases(discovered))
    cpu_cases = [case for case in all_cases if requires_cpu_fixture(case)]
    if arguments.profile == "cpu-fixture":
        selected = cpu_cases
    else:
        selected = [case for case in all_cases if not requires_cpu_fixture(case)]
    if not selected:
        print(
            "profile {0} selected no Python tests".format(arguments.profile),
            file=sys.stderr,
        )
        return 1

    print(
        "Python unittest profile {0}: {1} selected, {2} CPU-fixture, "
        "{3} total".format(
            arguments.profile, len(selected), len(cpu_cases), len(all_cases)
        )
    )
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite(selected)
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())

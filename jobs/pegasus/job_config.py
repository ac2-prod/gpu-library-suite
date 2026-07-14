#!/usr/bin/env python3
"""Strict Pegasus configuration loading shared by render and build tools."""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Set


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from gpu_suite.strict_json import load  # noqa: E402


class PegasusConfigError(ValueError):
    pass


CPU_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OMP_DYNAMIC",
    "MKL_NUM_THREADS",
    "MKL_DYNAMIC",
    "MKL_THREADING_LAYER",
    "OPENBLAS_NUM_THREADS",
)
TOP_LEVEL_KEYS = {
    "pegasus_config_schema_version",
    "account",
    "queue",
    "nodes",
    "walltime",
    "mpi_version",
    "cpu_threads",
    "cpu_cuda_build_modules",
    "openacc_build_modules",
    "benchmark_runtime_modules",
    "module_purge",
    "cmake_generator",
    "build_type",
    "build_parallelism",
    "cuda_architectures",
    "cuda_toolkit_root",
    "cuda_toolkit_version",
    "nvhpc_cuda_home",
    "nvhpc_gpu_target",
    "shared_result_root",
    "local_scratch_root",
    "pbs_stdout",
    "pbs_stderr",
    "openmpi_options",
    "environment_overrides",
    "cpu_runtime_environment",
}
SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+@/-]*$")
SAFE_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
WALLTIME_RE = re.compile(r"^\d{2,3}:\d{2}:\d{2}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PegasusConfigError(message)


def _exact_keys(value: Mapping[str, Any], expected: Set[str], name: str) -> None:
    missing = expected.difference(value.keys())
    extra = set(value.keys()).difference(expected)
    _require(not missing, "{0} missing keys: {1}".format(name, sorted(missing)))
    _require(not extra, "{0} unknown keys: {1}".format(name, sorted(extra)))


def _configured_string(value: Any, name: str) -> str:
    _require(isinstance(value, str) and value != "", name + " must be set")
    upper = value.upper()
    _require(
        "REPLACE" not in upper and "<" not in value and ">" not in value,
        name + " still contains a placeholder",
    )
    _require("\n" not in value and "\r" not in value and "\x00" not in value,
             name + " contains an unsafe character")
    return value


def _absolute_path(value: Any, name: str) -> str:
    configured = _configured_string(value, name)
    _require(Path(configured).is_absolute(), name + " must be an absolute path")
    return configured


def _module_list(value: Any, name: str) -> None:
    _require(isinstance(value, list) and value, name + " must be a nonempty array")
    _require(len(value) == len(set(value)), name + " contains duplicates")
    for module in value:
        _require(
            isinstance(module, str) and SAFE_TOKEN_RE.fullmatch(module) is not None,
            name + " contains an unsafe module name",
        )


def validate_pegasus_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    _require(isinstance(config, Mapping), "Pegasus configuration must be an object")
    _exact_keys(config, TOP_LEVEL_KEYS, "Pegasus configuration")
    _require(config["pegasus_config_schema_version"] == 1,
             "unsupported Pegasus configuration schema")
    for name in ("account", "queue", "mpi_version"):
        value = _configured_string(config[name], name)
        _require(SAFE_TOKEN_RE.fullmatch(value) is not None, "unsafe " + name)
    nodes = config["nodes"]
    _require(isinstance(nodes, int) and not isinstance(nodes, bool) and 1 <= nodes <= 150,
             "nodes must be from 1 through 150")
    threads = config["cpu_threads"]
    _require(isinstance(threads, int) and not isinstance(threads, bool)
             and 1 <= threads <= 48, "cpu_threads must be from 1 through 48")
    walltime = _configured_string(config["walltime"], "walltime")
    _require(WALLTIME_RE.fullmatch(walltime) is not None, "invalid walltime")
    hours, minutes, seconds = (int(part) for part in walltime.split(":"))
    _require(hours <= 24 and minutes < 60 and seconds < 60, "invalid walltime")
    for name in (
        "cpu_cuda_build_modules", "openacc_build_modules",
        "benchmark_runtime_modules",
    ):
        _module_list(config[name], name)
    _require(isinstance(config["module_purge"], bool), "module_purge must be boolean")
    _configured_string(config["cmake_generator"], "cmake_generator")
    _require(config["build_type"] == "Release", "Pegasus build_type must be Release")
    parallelism = config["build_parallelism"]
    _require(isinstance(parallelism, int) and not isinstance(parallelism, bool)
             and 1 <= parallelism <= 48, "invalid build_parallelism")
    for name in (
        "cuda_architectures", "cuda_toolkit_version", "nvhpc_gpu_target",
    ):
        _configured_string(config[name], name)
    toolkit_root = _absolute_path(config["cuda_toolkit_root"], "cuda_toolkit_root")
    nvhpc_home = _absolute_path(config["nvhpc_cuda_home"], "nvhpc_cuda_home")
    _require(toolkit_root == nvhpc_home,
             "cuda_toolkit_root and nvhpc_cuda_home must select one Toolkit")
    for name in (
        "shared_result_root", "local_scratch_root", "pbs_stdout", "pbs_stderr",
    ):
        _absolute_path(config[name], name)
    _require(
        not any(character.isspace() for character in config["pbs_stdout"])
        and not any(character.isspace() for character in config["pbs_stderr"]),
        "PBS output paths cannot contain whitespace",
    )
    options = config["openmpi_options"]
    _require(isinstance(options, list), "openmpi_options must be an array")
    for option in options:
        _require(isinstance(option, str) and option != "" and "\n" not in option,
                 "invalid OpenMPI option")
    overrides = config["environment_overrides"]
    _require(isinstance(overrides, Mapping), "environment_overrides must be an object")
    for name, value in overrides.items():
        _require(SAFE_ENVIRONMENT_NAME_RE.fullmatch(name) is not None,
                 "invalid environment override name")
        _require(name not in CPU_ENVIRONMENT_KEYS,
                 "CPU runtime variables belong in cpu_runtime_environment")
        _configured_string(value, "environment override " + name)
    cpu_environment = config["cpu_runtime_environment"]
    _require(isinstance(cpu_environment, Mapping),
             "cpu_runtime_environment must be an object")
    _exact_keys(cpu_environment, set(CPU_ENVIRONMENT_KEYS),
                "cpu_runtime_environment")
    for name in CPU_ENVIRONMENT_KEYS:
        _configured_string(cpu_environment[name], name)
    _require(cpu_environment["OMP_DYNAMIC"] == "FALSE", "OMP_DYNAMIC must be FALSE")
    _require(cpu_environment["MKL_DYNAMIC"] == "FALSE", "MKL_DYNAMIC must be FALSE")
    try:
        omp_threads = int(cpu_environment["OMP_NUM_THREADS"])
    except ValueError as error:
        raise PegasusConfigError("OMP_NUM_THREADS must be an integer") from error
    _require(omp_threads == threads, "OMP_NUM_THREADS must equal cpu_threads")
    for name in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        try:
            count = int(cpu_environment[name])
        except ValueError as error:
            raise PegasusConfigError(name + " must be an integer") from error
        _require(1 <= count <= 48, "invalid " + name)
    return dict(config)


def load_pegasus_config(path: Path) -> Dict[str, Any]:
    return validate_pegasus_config(load(path))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    modules = subparsers.add_parser("modules")
    modules.add_argument(
        "--profile",
        required=True,
        choices=("cpu-cuda", "openacc", "benchmark-runtime"),
    )
    get_value = subparsers.add_parser("get")
    get_value.add_argument("name", choices=tuple(sorted(TOP_LEVEL_KEYS)))
    arguments = parser.parse_args(argv)
    try:
        config = load_pegasus_config(arguments.config)
        if arguments.command == "modules":
            key = {
                "cpu-cuda": "cpu_cuda_build_modules",
                "openacc": "openacc_build_modules",
                "benchmark-runtime": "benchmark_runtime_modules",
            }[arguments.profile]
            for module in config[key]:
                print(module)
        else:
            value = config[arguments.name]
            if isinstance(value, bool):
                print("true" if value else "false")
            elif isinstance(value, (str, int)):
                print(value)
            else:
                raise PegasusConfigError("get supports only scalar values")
    except (OSError, ValueError) as error:
        print("Pegasus configuration failed: {0}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Collect a stable, deterministic runtime software environment document."""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from gpu_suite.hashing import sha256_file  # noqa: E402
from gpu_suite.manifest import validate_build_metadata  # noqa: E402
from gpu_suite.runner import load_manifest  # noqa: E402
from gpu_suite.strict_json import dump_bytes, load  # noqa: E402
from job_config import CPU_ENVIRONMENT_KEYS  # noqa: E402


class RuntimeEnvironmentError(ValueError):
    pass


CommandExecutor = Callable[[Sequence[str]], Tuple[int, str]]
HEX_ADDRESS_RE = re.compile(r"\s+\(0x[0-9A-Fa-f]+\)\s*$")


def _default_executor(arguments: Sequence[str]) -> Tuple[int, str]:
    completed = subprocess.run(
        list(arguments), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    return completed.returncode, completed.stdout


def _normalized_lines(value: str) -> List[str]:
    return [line.rstrip() for line in value.replace("\r\n", "\n").splitlines()]


def command_record(
    arguments: Sequence[str], executor: CommandExecutor = _default_executor,
    which: Callable[[str], Optional[str]] = shutil.which,
) -> Dict[str, Any]:
    executable = which(arguments[0])
    if executable is None:
        return {
            "arguments": list(arguments),
            "executable": None,
            "output": [],
            "return_code": None,
            "status": "unavailable",
        }
    return_code, output = executor([executable] + list(arguments[1:]))
    return {
        "arguments": list(arguments),
        "executable": executable,
        "output": _normalized_lines(output),
        "return_code": return_code,
        "status": "success" if return_code == 0 else "failure",
    }


def normalize_ldd_output(lines: Sequence[str]) -> List[str]:
    """Remove ASLR load addresses while retaining every dependency mapping."""

    return [HEX_ADDRESS_RE.sub("", line).rstrip() for line in lines]


def resolved_library_paths(lines: Sequence[str]) -> List[str]:
    paths = set()
    for line in lines:
        text = HEX_ADDRESS_RE.sub("", line).strip()
        if "=>" in text:
            candidate = text.split("=>", 1)[1].strip()
        else:
            candidate = text.split(" ", 1)[0]
        if candidate.startswith("/"):
            paths.add(candidate)
    return sorted(paths)


def _library_probe(
    package_names: Sequence[str], executor: CommandExecutor,
    which: Callable[[str], Optional[str]],
) -> Dict[str, Any]:
    probes = [
        command_record(["pkg-config", "--modversion", name], executor, which)
        for name in package_names
    ]
    successful = [probe for probe in probes if probe["status"] == "success"]
    return {
        "probes": probes,
        "status": "success" if successful else "unavailable",
        "version": successful[0]["output"][0]
        if successful and successful[0]["output"] else None,
    }


def collect_runtime_environment(
    manifest_path: Path, build_metadata_paths: Sequence[Path], module_list_path: Path,
    configured_cuda_root: str, configured_cuda_version: str,
    nvhpc_cuda_home: str, environment: Mapping[str, str] = os.environ,
    executor: CommandExecutor = _default_executor,
    which: Callable[[str], Optional[str]] = shutil.which,
    require_ldd: bool = False, require_gpu_tools: bool = False,
) -> Dict[str, Any]:
    manifest, manifest_sha256 = load_manifest(manifest_path)
    cpu_environment = {}  # type: Dict[str, str]
    for name in CPU_ENVIRONMENT_KEYS:
        value = environment.get(name)
        if value is None or value == "":
            raise RuntimeEnvironmentError("missing CPU runtime variable " + name)
        cpu_environment[name] = value
    selected_home = environment.get("NVHPC_CUDA_HOME", nvhpc_cuda_home)
    if selected_home != nvhpc_cuda_home or selected_home != configured_cuda_root:
        raise RuntimeEnvironmentError("runtime CUDA Toolkit selection is inconsistent")

    build_metadata = []
    metadata_by_hash = {}  # type: Dict[str, Dict[str, Any]]
    for path in build_metadata_paths:
        document = load(path)
        if not isinstance(document, dict):
            raise RuntimeEnvironmentError("build metadata must be an object")
        digest = sha256_file(path)
        metadata_by_hash[digest] = document
        build_metadata.append({
            "metadata": document,
            "sha256": digest,
        })
    build_metadata.sort(key=lambda item: item["sha256"])

    for entry in manifest["entries"]:
        metadata = metadata_by_hash.get(entry["build_metadata_sha256"])
        if metadata is None:
            raise RuntimeEnvironmentError(
                "manifest entry has no supplied build metadata"
            )
        validate_build_metadata(entry, metadata)
        if entry["implementation"] in {"cuda", "openacc"}:
            cuda = metadata.get("cuda")
            if not isinstance(cuda, dict):
                raise RuntimeEnvironmentError("GPU build metadata lacks CUDA data")
            metadata_root = cuda.get("toolkit_root")
            metadata_version = cuda.get("toolkit_version")
            if (
                not isinstance(metadata_root, str)
                or Path(metadata_root).resolve() != Path(configured_cuda_root).resolve()
                or metadata_version != configured_cuda_version
            ):
                raise RuntimeEnvironmentError(
                    "build and runtime CUDA Toolkit selections differ"
                )
        if entry["implementation"] == "openacc":
            nvhpc = metadata.get("nvhpc")
            if (
                not isinstance(nvhpc, dict)
                or Path(str(nvhpc.get("cuda_home", ""))).resolve()
                != Path(selected_home).resolve()
            ):
                raise RuntimeEnvironmentError(
                    "OpenACC build metadata and NVHPC CUDA selection differ"
                )

    binaries = []
    all_resolved_paths = set()
    for entry in manifest["entries"]:
        path = Path(entry["executable_path"])
        if not path.is_file() or sha256_file(path) != entry["binary_sha256"]:
            raise RuntimeEnvironmentError(
                "manifest binary is absent or changed: {0}".format(path)
            )
        ldd = command_record(["ldd", str(path)], executor, which)
        ldd["output"] = normalize_ldd_output(ldd["output"])
        resolved = resolved_library_paths(ldd["output"])
        all_resolved_paths.update(resolved)
        if require_ldd and ldd["status"] != "success":
            raise RuntimeEnvironmentError("ldd failed for " + str(path))
        binaries.append({
            "artifact_id": entry["artifact_id"],
            "binary_sha256": entry["binary_sha256"],
            "executable_path": str(path),
            "ldd": ldd,
            "resolved_shared_library_paths": resolved,
        })
    binaries.sort(key=lambda item: item["artifact_id"])

    driver = command_record(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        executor, which,
    )
    gpu_identity = command_record(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        executor, which,
    )
    nvcc = command_record(["nvcc", "--version"], executor, which)
    nvc = command_record(["nvc", "--version"], executor, which)
    nvcxx = command_record(["nvc++", "--version"], executor, which)
    if require_gpu_tools:
        for name, probe in (
            ("NVIDIA driver", driver), ("nvcc", nvcc), ("nvc", nvc),
            ("nvc++", nvcxx),
        ):
            if probe["status"] != "success":
                raise RuntimeEnvironmentError(name + " runtime probe failed")

    library_versions = {
        "fftw3f": _library_probe(("fftw3f",), executor, which),
        "lapacke": _library_probe(("lapacke",), executor, which),
        "onemkl": _library_probe(("mkl-dynamic-lp64-iomp", "mkl-static-lp64-iomp"), executor, which),
        "openblas": _library_probe(("openblas",), executor, which),
    }
    return {
        "binaries": binaries,
        "build_metadata": build_metadata,
        "cpu_library_versions": library_versions,
        "cpu_runtime_environment": cpu_environment,
        "cpu_thread_semantics": {
            "requested_threads": int(cpu_environment["OMP_NUM_THREADS"]),
            "serial_cpu_baseline_effective_threads": 1,
        },
        "cuda": {
            "configured_toolkit_root": configured_cuda_root,
            "configured_toolkit_version": configured_cuda_version,
            "nvcc": nvcc,
            "runtime_version": environment.get(
                "GPU_SUITE_CUDA_RUNTIME_VERSION", configured_cuda_version
            ),
            "runtime_version_source": (
                "environment"
                if environment.get("GPU_SUITE_CUDA_RUNTIME_VERSION")
                else "configured-toolkit"
            ),
        },
        "environment": {
            "LD_LIBRARY_PATH": environment.get("LD_LIBRARY_PATH", ""),
            "PATH": environment.get("PATH", ""),
        },
        "executables_manifest_sha256": manifest_sha256,
        "module_list": _normalized_lines(module_list_path.read_text(encoding="utf-8")),
        "nvidia": {"driver": driver, "gpus": gpu_identity},
        "nvhpc": {
            "NVHPC_CUDA_HOME": environment.get("NVHPC_CUDA_HOME"),
            "compiler_nvc": nvc,
            "compiler_nvcxx": nvcxx,
            "selected_cuda_toolkit": selected_home,
        },
        "resolved_shared_library_paths": sorted(all_resolved_paths),
        "runtime_environment_schema_version": 1,
    }


def exclusive_write(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--build-metadata", required=True, action="append", type=Path)
    parser.add_argument("--module-list", required=True, type=Path)
    parser.add_argument("--cuda-toolkit-root", required=True)
    parser.add_argument("--cuda-toolkit-version", required=True)
    parser.add_argument("--nvhpc-cuda-home", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-ldd", action="store_true")
    parser.add_argument("--require-gpu-tools", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        document = collect_runtime_environment(
            arguments.manifest, arguments.build_metadata, arguments.module_list,
            arguments.cuda_toolkit_root, arguments.cuda_toolkit_version,
            arguments.nvhpc_cuda_home, require_ldd=arguments.require_ldd,
            require_gpu_tools=arguments.require_gpu_tools,
        )
        content = dump_bytes(document)
        exclusive_write(arguments.output, content)
        print(sha256_file(arguments.output))
    except (OSError, ValueError) as error:
        print("runtime environment collection failed: {0}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

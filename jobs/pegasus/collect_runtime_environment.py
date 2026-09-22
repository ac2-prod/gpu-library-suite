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

from gpu_suite.hashing import deterministic_json_sha256, sha256_file  # noqa: E402
from gpu_suite.manifest import validate_build_metadata  # noqa: E402
from gpu_suite.runner import load_manifest  # noqa: E402
from gpu_suite.strict_json import dump_bytes, load  # noqa: E402
from cuda_runtime_probe import (  # noqa: E402
    CUDA_RUNTIME_PROBE_KEYS,
    CudaRuntimeProbe,
    probe_node_cuda_runtime,
)
from job_config import CPU_ENVIRONMENT_KEYS  # noqa: E402


class RuntimeEnvironmentError(ValueError):
    pass


CommandExecutor = Callable[[Sequence[str]], Tuple[int, str]]
HEX_ADDRESS_RE = re.compile(r"\s+\(0x[0-9A-Fa-f]+\)\s*$")
MODULE_ENTRY_RE = re.compile(r"(?<!\S)([0-9]+)\)\s+([^\s]+)")
SAFE_MODULE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+@/-]*$")


def _default_executor(arguments: Sequence[str]) -> Tuple[int, str]:
    completed = subprocess.run(
        list(arguments), text=False, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    return completed.returncode, completed.stdout.decode("utf-8", errors="strict")


def _normalized_lines(value: str) -> List[str]:
    return [line.rstrip() for line in value.replace("\r\n", "\n").splitlines()]


def _canonical_path(value: str) -> str:
    return str(Path(value).resolve(strict=False))


def _command_record_and_raw(
    arguments: Sequence[str], executor: CommandExecutor = _default_executor,
    which: Callable[[str], Optional[str]] = shutil.which,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    executable = which(arguments[0])
    if executable is None:
        record = {
            "arguments": list(arguments),
            "executable": None,
            "output": [],
            "return_code": None,
            "status": "unavailable",
        }
        raw_record = dict(record)
        raw_record.pop("output")
        raw_record["output_text"] = ""
        return record, raw_record
    return_code, output = executor([executable] + list(arguments[1:]))
    record = {
        "arguments": list(arguments),
        "executable": _canonical_path(executable),
        "output": _normalized_lines(output),
        "return_code": return_code,
        "status": "success" if return_code == 0 else "failure",
    }
    raw_record = dict(record)
    raw_record.pop("output")
    raw_record["output_text"] = output
    return record, raw_record


def command_record(
    arguments: Sequence[str], executor: CommandExecutor = _default_executor,
    which: Callable[[str], Optional[str]] = shutil.which,
) -> Dict[str, Any]:
    return _command_record_and_raw(arguments, executor, which)[0]


def normalize_module_list(value: str) -> List[str]:
    """Return module identities in load order, independent of display layout."""

    numbered = []  # type: List[Tuple[int, str]]
    for line in value.replace("\r\n", "\n").splitlines():
        numbered.extend(
            (int(index), module)
            for index, module in MODULE_ENTRY_RE.findall(line)
        )
    if numbered:
        indices = [item[0] for item in numbered]
        if len(indices) != len(set(indices)) or set(indices) != set(
            range(1, len(indices) + 1)
        ):
            raise RuntimeEnvironmentError(
                "module list has duplicate or non-contiguous indices"
            )
        modules = [module for _, module in sorted(numbered)]
    else:
        ignored = {
            "Currently Loaded Modulefiles:",
            "No Modulefiles Currently Loaded.",
            "Key:",
        }
        modules = []
        for line in value.replace("\r\n", "\n").splitlines():
            text = line.strip()
            if not text or text in ignored or text.startswith("<module-tag>"):
                continue
            if any(character.isspace() for character in text):
                raise RuntimeEnvironmentError(
                    "unrecognized module list display line"
                )
            modules.append(text)
    if not modules:
        raise RuntimeEnvironmentError("module list contains no module identities")
    if len(modules) != len(set(modules)):
        raise RuntimeEnvironmentError("module list contains duplicate identities")
    if any(SAFE_MODULE_RE.fullmatch(module) is None for module in modules):
        raise RuntimeEnvironmentError("module list contains an unsafe identity")
    return modules


def normalize_search_path(value: str, name: str) -> str:
    """Canonicalize path components and remove later duplicate entries."""

    if value == "":
        return ""
    normalized = []
    seen = set()
    for component in value.split(":"):
        if component == "" or not Path(component).is_absolute():
            raise RuntimeEnvironmentError(
                name + " contains an empty or relative component"
            )
        canonical = _canonical_path(component)
        if canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)
    return ":".join(normalized)


def _canonicalize_ldd_line(line: str) -> str:
    text = HEX_ADDRESS_RE.sub("", line).strip()
    if text == "":
        return ""
    if "=>" in text:
        soname, target = (part.strip() for part in text.split("=>", 1))
        if target == "not found":
            return soname + " => not found"
        if target.startswith("/"):
            target = _canonical_path(target)
        return soname + " => " + target
    if text.startswith("/"):
        return _canonical_path(text)
    return text


def normalize_ldd_output(lines: Sequence[str]) -> List[str]:
    """Return deterministic dependency identity without ASLR addresses."""

    return sorted(
        canonical
        for canonical in (_canonicalize_ldd_line(line) for line in lines)
        if canonical
    )


def resolved_library_paths(lines: Sequence[str]) -> List[str]:
    paths = set()
    for line in lines:
        text = HEX_ADDRESS_RE.sub("", line).strip()
        if "=>" in text:
            candidate = text.split("=>", 1)[1].strip()
        else:
            candidate = text.split(" ", 1)[0]
        if candidate.startswith("/"):
            paths.add(_canonical_path(candidate))
    return sorted(paths)


def _library_probe(
    package_names: Sequence[str], executor: CommandExecutor,
    which: Callable[[str], Optional[str]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    probes = []
    raw_probes = []
    for name in package_names:
        probe, raw_probe = _command_record_and_raw(
            ["pkg-config", "--modversion", name], executor, which
        )
        stable_probe = dict(probe)
        stable_probe.pop("output")
        probes.append(stable_probe)
        raw_probes.append(raw_probe)
    successful = [probe for probe in probes if probe["status"] == "success"]
    raw_successful = [
        raw for probe, raw in zip(probes, raw_probes)
        if probe["status"] == "success"
    ]
    identity = {
        "probes": probes,
        "status": "success" if successful else "unavailable",
        "version": (
            _normalized_lines(raw_successful[0]["output_text"])[0]
            if raw_successful
            and _normalized_lines(raw_successful[0]["output_text"])
            else None
        ),
    }
    return identity, {"probes": raw_probes}


def collect_runtime_environment_documents(
    manifest_path: Path, build_metadata_paths: Sequence[Path],
    module_list_path: Optional[Path],
    configured_cuda_root: str, configured_cuda_version: str,
    nvhpc_cuda_home: str, environment: Mapping[str, str] = os.environ,
    executor: CommandExecutor = _default_executor,
    which: Callable[[str], Optional[str]] = shutil.which,
    require_ldd: bool = False, require_gpu_tools: bool = False,
    require_runtime_compilers: bool = False,
    cuda_runtime_probe: CudaRuntimeProbe = probe_node_cuda_runtime,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manifest, manifest_sha256 = load_manifest(manifest_path)
    cpu_environment = {}  # type: Dict[str, str]
    for name in CPU_ENVIRONMENT_KEYS:
        value = environment.get(name)
        if value is None or value == "":
            raise RuntimeEnvironmentError("missing CPU runtime variable " + name)
        cpu_environment[name] = value
    configured_root = _canonical_path(configured_cuda_root)
    configured_nvhpc_home = _canonical_path(nvhpc_cuda_home)
    selected_home = _canonical_path(
        environment.get("NVHPC_CUDA_HOME", nvhpc_cuda_home)
    )
    if (
        selected_home != configured_nvhpc_home
        or selected_home != configured_root
    ):
        raise RuntimeEnvironmentError("runtime CUDA Toolkit selection is inconsistent")

    # None is an explicit declaration that no module system is used, not a
    # failed/empty module command disguised as successful discovery.
    raw_module_list = (
        module_list_path.read_bytes().decode("utf-8", errors="strict")
        if module_list_path is not None else None
    )
    modules = normalize_module_list(raw_module_list) if raw_module_list is not None else []

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
                or Path(metadata_root).resolve() != Path(configured_root).resolve()
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
    raw_binary_evidence = []
    all_resolved_paths = set()
    for entry in manifest["entries"]:
        path = Path(entry["executable_path"]).resolve()
        if not path.is_file() or sha256_file(path) != entry["binary_sha256"]:
            raise RuntimeEnvironmentError(
                "manifest binary is absent or changed: {0}".format(path)
            )
        ldd, raw_ldd = _command_record_and_raw(
            ["ldd", str(path)], executor, which
        )
        ldd["output"] = normalize_ldd_output(ldd["output"])
        if any("=> not found" in line for line in ldd["output"]):
            raise RuntimeEnvironmentError(
                "ldd reported an unresolved library for " + str(path)
            )
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
        raw_binary_evidence.append({
            "artifact_id": entry["artifact_id"],
            "binary_sha256": entry["binary_sha256"],
            "executable_path": str(path),
            "ldd": raw_ldd,
        })
    binaries.sort(key=lambda item: item["artifact_id"])
    raw_binary_evidence.sort(key=lambda item: item["artifact_id"])

    driver, raw_driver = _command_record_and_raw(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        executor, which,
    )
    gpu_identity, raw_gpu_identity = _command_record_and_raw(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        executor, which,
    )
    nvcc, raw_nvcc = _command_record_and_raw(
        ["nvcc", "--version"], executor, which
    )
    nvc, raw_nvc = _command_record_and_raw(
        ["nvc", "--version"], executor, which
    )
    nvcxx, raw_nvcxx = _command_record_and_raw(
        ["nvc++", "--version"], executor, which
    )
    cuda_runtime_identity = dict(cuda_runtime_probe())
    if set(cuda_runtime_identity) != CUDA_RUNTIME_PROBE_KEYS:
        raise RuntimeEnvironmentError("invalid CUDA runtime probe result")
    if cuda_runtime_identity["query_status"] not in {
        "success", "failure", "unavailable",
    }:
        raise RuntimeEnvironmentError("invalid CUDA runtime probe status")
    if require_gpu_tools:
        for name, probe in (
            ("NVIDIA driver", driver), ("node GPU identity", gpu_identity),
            ("CUDA Driver API/Runtime", cuda_runtime_identity),
        ):
            status = probe.get("status", probe.get("query_status"))
            if status != "success":
                raise RuntimeEnvironmentError(name + " runtime probe failed")
    if require_runtime_compilers:
        for name, probe in (("nvcc", nvcc), ("nvc", nvc), ("nvc++", nvcxx)):
            if probe["status"] != "success":
                raise RuntimeEnvironmentError(
                    name + " compiler probe failed under explicit site policy"
                )

    library_versions = {}
    raw_library_probes = {}
    for name, packages in (
        ("fftw3f", ("fftw3f",)),
        ("lapacke", ("lapacke",)),
        ("onemkl", ("mkl-dynamic-lp64-iomp", "mkl-static-lp64-iomp")),
        ("openblas", ("openblas",)),
    ):
        identity, raw_probes = _library_probe(packages, executor, which)
        library_versions[name] = identity
        raw_library_probes[name] = raw_probes

    loaded_library = cuda_runtime_identity["loaded_library"]
    stable_cuda_runtime_identity = {
        "cuda_driver_api_version": cuda_runtime_identity[
            "cuda_driver_api_version"
        ],
        "cuda_runtime_version": cuda_runtime_identity["cuda_runtime_version"],
        "loaded_library": (
            _canonical_path(loaded_library)
            if isinstance(loaded_library, str) and loaded_library else None
        ),
        "query_status": cuda_runtime_identity["query_status"],
    }
    document = {
        "binaries": binaries,
        "build_metadata": build_metadata,
        "cpu_library_versions": library_versions,
        "cpu_runtime_environment": cpu_environment,
        "cpu_thread_semantics": {
            "requested_threads": int(cpu_environment["OMP_NUM_THREADS"]),
            "serial_cpu_baseline_effective_threads": 1,
        },
        "cuda": {
            "compiler_probe_policy": (
                "required-site-policy" if require_runtime_compilers
                else "optional-metadata"
            ),
            "configured_toolkit_root": configured_root,
            "configured_toolkit_version": configured_cuda_version,
            "nvcc": nvcc,
            "runtime_probe": stable_cuda_runtime_identity,
            "runtime_version": cuda_runtime_identity["cuda_runtime_version"],
            "runtime_version_source": (
                "cudaRuntimeGetVersion"
                if cuda_runtime_identity["query_status"] == "success"
                else "unavailable"
            ),
        },
        "environment": {
            "LD_LIBRARY_PATH": normalize_search_path(
                environment.get("LD_LIBRARY_PATH", ""), "LD_LIBRARY_PATH"
            ),
            "PATH": normalize_search_path(
                environment.get("PATH", ""), "PATH"
            ),
        },
        "executables_manifest_sha256": manifest_sha256,
        "module_list": modules,
        "nvidia": {"driver": driver},
        "nvhpc": {
            "NVHPC_CUDA_HOME": selected_home,
            "compiler_nvc": nvc,
            "compiler_nvcxx": nvcxx,
            "compiler_probe_policy": (
                "required-site-policy" if require_runtime_compilers
                else "optional-metadata"
            ),
            "selected_cuda_toolkit": selected_home,
        },
        "resolved_shared_library_paths": sorted(all_resolved_paths),
        "runtime_environment_schema_version": 1,
    }
    if module_list_path is None:
        document["module_system"] = {
            "source": "user-specified", "status": "not-used",
        }
    evidence = {
        "binaries": raw_binary_evidence,
        "command_probes": {
            "cpu_libraries": raw_library_probes,
            "cuda_nvcc": raw_nvcc,
            "nvidia_driver": raw_driver,
            "nvidia_gpu_identity": raw_gpu_identity,
            "nvhpc_nvc": raw_nvc,
            "nvhpc_nvcxx": raw_nvcxx,
        },
        "cuda_runtime_probe": dict(cuda_runtime_identity),
        "executables_manifest_sha256": manifest_sha256,
        "module_list_output": raw_module_list,
        "module_list_sha256": (
            sha256_file(module_list_path) if module_list_path is not None else None
        ),
        "runtime_environment_evidence_schema_version": 1,
        "runtime_environment_sha256": deterministic_json_sha256(document),
    }
    if module_list_path is None:
        evidence["module_system"] = dict(document["module_system"])
    return document, evidence


def collect_runtime_environment(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Compatibility wrapper returning only the canonical identity document."""

    return collect_runtime_environment_documents(*args, **kwargs)[0]


def exclusive_write(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--build-metadata", required=True, action="append", type=Path)
    modules = parser.add_mutually_exclusive_group(required=True)
    modules.add_argument("--module-list", type=Path)
    modules.add_argument(
        "--no-module-system", action="store_true",
        help="Explicitly record that no module system is used; other probes remain enforced",
    )
    parser.add_argument("--cuda-toolkit-root", required=True)
    parser.add_argument("--cuda-toolkit-version", required=True)
    parser.add_argument("--nvhpc-cuda-home", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evidence-output", required=True, type=Path)
    parser.add_argument("--require-ldd", action="store_true")
    parser.add_argument("--require-gpu-tools", action="store_true")
    parser.add_argument("--require-runtime-compilers", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        if arguments.output == arguments.evidence_output:
            raise RuntimeEnvironmentError(
                "runtime identity and evidence outputs must be distinct"
            )
        document, evidence = collect_runtime_environment_documents(
            arguments.manifest, arguments.build_metadata, arguments.module_list,
            arguments.cuda_toolkit_root, arguments.cuda_toolkit_version,
            arguments.nvhpc_cuda_home, require_ldd=arguments.require_ldd,
            require_gpu_tools=arguments.require_gpu_tools,
            require_runtime_compilers=arguments.require_runtime_compilers,
        )
        content = dump_bytes(document)
        exclusive_write(arguments.output, content)
        exclusive_write(arguments.evidence_output, dump_bytes(evidence))
        print(sha256_file(arguments.output))
    except (OSError, ValueError) as error:
        print("runtime environment collection failed: {0}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

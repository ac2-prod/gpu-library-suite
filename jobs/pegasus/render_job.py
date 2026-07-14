#!/usr/bin/env python3
"""Render a Pegasus PBS job without submitting or executing it."""

import argparse
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from gpu_suite.runner import validate_execution_context, validate_sha256  # noqa: E402
from job_config import CPU_ENVIRONMENT_KEYS, load_pegasus_config  # noqa: E402


class RenderError(ValueError):
    pass


PLACEHOLDER_RE = re.compile(r"@[A-Z][A-Z0-9_]*@")
JOB_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")


def _quote(value: Any) -> str:
    return shlex.quote(str(value))


def _existing_file(path: Path, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise RenderError(name + " must be an existing regular file")
    return resolved


def _metadata_arguments(paths: Sequence[Path]) -> str:
    separator = " " + "\\" + "\n  "
    return separator.join("--build-metadata " + _quote(path) for path in paths)


def render_job(
    template_path: Path, pegasus_config_path: Path, repository_root: Path,
    benchmark_config_path: Path, manifest_path: Path,
    build_metadata_paths: Sequence[Path], run_id: str, wave: int,
    system_label: str, job_name: str, git_diff_sha256: Optional[str] = None,
    source_snapshot_sha256: Optional[str] = None,
) -> str:
    config = load_pegasus_config(pegasus_config_path)
    repository = repository_root.resolve()
    if not repository.is_dir() or repository.is_symlink():
        raise RenderError("repository root must be an existing real directory")
    benchmark_config = _existing_file(benchmark_config_path, "benchmark config")
    manifest = _existing_file(manifest_path, "manifest")
    metadata = [
        _existing_file(path, "build metadata") for path in build_metadata_paths
    ]
    if not metadata:
        raise RenderError("at least one build metadata document is required")
    validate_execution_context(run_id, system_label, "render-host", wave, 0, 0)
    if JOB_NAME_RE.fullmatch(job_name) is None:
        raise RenderError("unsafe PBS job name")
    if git_diff_sha256 is not None and source_snapshot_sha256 is not None:
        raise RenderError("select one authoritative dirty-source hash")
    dirty_arguments = ""
    if git_diff_sha256 is not None:
        validate_sha256(git_diff_sha256, "Git diff SHA-256")
        dirty_arguments = "--git-diff-sha256 " + _quote(git_diff_sha256)
    if source_snapshot_sha256 is not None:
        validate_sha256(source_snapshot_sha256, "source snapshot SHA-256")
        dirty_arguments = "--source-snapshot-sha256 " + _quote(source_snapshot_sha256)
    module_lines = []
    if config["module_purge"]:
        module_lines.append("module purge")
    module_lines.extend(
        "module load " + _quote(module)
        for module in config["benchmark_runtime_modules"]
    )
    module_lines.append('module load "openmpi/${NQSV_MPI_VER}"')
    cpu_lines = [
        "export {0}={1}".format(name, _quote(config["cpu_runtime_environment"][name]))
        for name in CPU_ENVIRONMENT_KEYS
    ]
    cpu_lines.extend((
        "export GPU_SUITE_CPU_THREADS_REQUESTED={0}".format(config["cpu_threads"]),
        "export GPU_SUITE_SERIAL_CPU_THREADS_EFFECTIVE=1",
    ))
    environment_lines = [
        "export {0}={1}".format(name, _quote(value))
        for name, value in sorted(config["environment_overrides"].items())
    ]
    values = {
        "@ACCOUNT@": config["account"],
        "@BUILD_METADATA_ARGUMENTS@": _metadata_arguments(metadata),
        "@CPU_ENVIRONMENT@": "\n".join(cpu_lines),
        "@CPU_THREADS@": str(config["cpu_threads"]),
        "@CUDA_TOOLKIT_ROOT@": _quote(config["cuda_toolkit_root"]),
        "@CUDA_TOOLKIT_VERSION@": _quote(config["cuda_toolkit_version"]),
        "@DIRTY_SOURCE_ARGUMENTS@": dirty_arguments,
        "@ENVIRONMENT_OVERRIDES@": "\n".join(environment_lines),
        "@JOB_NAME@": job_name,
        "@LOCAL_SCRATCH_ROOT@": _quote(config["local_scratch_root"]),
        "@MODULE_SETUP@": "\n".join(module_lines),
        "@MPI_VERSION@": config["mpi_version"],
        "@NODE_COUNT@": str(config["nodes"]),
        "@NODES@": str(config["nodes"]),
        "@NVHPC_CUDA_HOME@": _quote(config["nvhpc_cuda_home"]),
        "@OPENMPI_OPTIONS@": " ".join(_quote(value) for value in config["openmpi_options"]),
        "@PBS_STDERR@": config["pbs_stderr"],
        "@PBS_STDOUT@": config["pbs_stdout"],
        "@QUEUE@": config["queue"],
        "@REPOSITORY_ROOT@": _quote(repository),
        "@RESULT_ROOT@": _quote(config["shared_result_root"]),
        "@RUN_ID@": _quote(run_id),
        "@SOURCE_CONFIG@": _quote(benchmark_config),
        "@SOURCE_MANIFEST@": _quote(manifest),
        "@SYSTEM_LABEL@": _quote(system_label),
        "@WALLTIME@": config["walltime"],
        "@WAVE@": str(wave),
    }
    rendered = template_path.read_text(encoding="utf-8")
    for placeholder, value in values.items():
        rendered = rendered.replace(placeholder, value)
    remaining = PLACEHOLDER_RE.findall(rendered)
    if remaining:
        raise RenderError("unresolved template placeholders: {0}".format(remaining))
    if not rendered.startswith("#!/bin/bash\n"):
        raise RenderError("rendered PBS script must begin with #!/bin/bash")
    return rendered


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--template", type=Path,
                        default=Path(__file__).with_name("run_benchmarks.pbs.in"))
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--benchmark-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--build-metadata", required=True, action="append", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--wave", required=True, type=int)
    parser.add_argument("--system-label", required=True)
    parser.add_argument("--job-name", default="gpu-library-suite")
    parser.add_argument("--git-diff-sha256")
    parser.add_argument("--source-snapshot-sha256")
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        rendered = render_job(
            arguments.template, arguments.config, arguments.repository_root,
            arguments.benchmark_config, arguments.manifest,
            arguments.build_metadata, arguments.run_id, arguments.wave,
            arguments.system_label, arguments.job_name,
            arguments.git_diff_sha256, arguments.source_snapshot_sha256,
        )
        with arguments.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    except (OSError, ValueError) as error:
        print("PBS rendering failed: {0}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

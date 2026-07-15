import copy
from pathlib import Path

from gpu_suite.hashing import sha256_file
from gpu_suite.manifest import artifact_id
from gpu_suite.strict_json import dump_bytes

from support import ROOT, pilot_config


CPU_ENVIRONMENT = {
    "OMP_NUM_THREADS": "48",
    "OMP_PROC_BIND": "spread",
    "OMP_PLACES": "cores",
    "OMP_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "48",
    "MKL_DYNAMIC": "FALSE",
    "MKL_THREADING_LAYER": "INTEL",
    "OPENBLAS_NUM_THREADS": "48",
}


def pegasus_config(directory: Path):
    return {
        "pegasus_config_schema_version": 1,
        "account": "ac2-test",
        "queue": "debug",
        "nodes": 2,
        "walltime": "00:30:00",
        "mpi_version": "4.1-test",
        "cpu_threads": 48,
        "cpu_cuda_build_modules": ["cpu-build/1", "cuda-build/1"],
        "openacc_build_modules": ["openacc-build/1"],
        "benchmark_runtime_modules": ["runtime/1"],
        "module_purge": True,
        "cmake_generator": "Unix Makefiles",
        "build_type": "Release",
        "build_parallelism": 4,
        "cuda_architectures": "test-architecture",
        "cuda_toolkit_root": str(directory / "cuda"),
        "cuda_toolkit_version": "13.0.88",
        "nvhpc_cuda_home": str(directory / "cuda"),
        "nvhpc_gpu_target": "test-gpu-target",
        "shared_result_root": str(directory / "results"),
        "local_scratch_root": str(directory / "scratch"),
        "pbs_stdout": str(directory / "pbs" / "job.out"),
        "pbs_stderr": str(directory / "pbs" / "job.err"),
        "openmpi_options": ["--mca", "btl", "self,vader,tcp"],
        "require_runtime_compilers": False,
        "environment_overrides": {"GPU_SUITE_TEST_OVERRIDE": "enabled"},
        "cpu_runtime_environment": dict(CPU_ENVIRONMENT),
    }


def write_runtime_evidence(
    path: Path, runtime_path: Path, manifest_path: Path, **extra
):
    document = {
        "binaries": [],
        "executables_manifest_sha256": sha256_file(manifest_path),
        "module_list_output": "runtime/1\n",
        "module_list_sha256": "0" * 64,
        "runtime_environment_evidence_schema_version": 1,
        "runtime_environment_sha256": sha256_file(runtime_path),
    }
    document.update(extra)
    path.write_bytes(dump_bytes(document))
    return path


def write_campaign_inputs(directory: Path, dirty=False, config=None):
    executable = directory / "fft_cpu_bench"
    executable.write_bytes(b"fixture executable\n")
    metadata_document = {
        "build_profile": "cpu-cuda",
        "build_type": "Release",
        "c": {
            "compiler": "TestCompiler",
            "global_configure_flags": "-O3",
            "compiler_version": "1.0",
        },
        "git_metadata_available": True,
        "git_commit": "abc",
        "git_dirty": dirty,
    }
    metadata_path = directory / "build-metadata.json"
    metadata_path.write_bytes(dump_bytes(metadata_document))
    entry = {
        "artifact_id": "",
        "target_name": "fft_cpu_bench",
        "build_profile": "cpu-cuda",
        "backend_variant": "fftw3f-serial+threaded",
        "executable_path": str(executable),
        "library": "cufft",
        "implementation": "cpu",
        "executable_role": "benchmark",
        "build_type": "Release",
        "binary_sha256": sha256_file(executable),
        "build_metadata_sha256": sha256_file(metadata_path),
        "compiler": "TestCompiler",
        "compiler_language": "c",
        "compiler_version": "1.0",
        "global_configure_flags": "-O3",
        "git_metadata_available": True,
        "git_commit": "abc",
        "git_dirty": dirty,
        "supported_cpu_backends": ["cpu-fftw-threaded", "cpu-fftw-serial"],
    }
    entry["artifact_id"] = artifact_id(entry)
    manifest_path = directory / "executables.json"
    manifest_path.write_bytes(dump_bytes({
        "build_metadata_sha256s": [sha256_file(metadata_path)],
        "entries": [entry],
        "manifest_schema_version": 1,
    }))
    selected_config = copy.deepcopy(config if config is not None else pilot_config())
    config_path = directory / "effective-input.json"
    config_path.write_bytes(dump_bytes(selected_config))
    runtime_document = {
        "build_metadata": [{
            "metadata": metadata_document,
            "sha256": sha256_file(metadata_path),
        }],
        "cpu_runtime_environment": dict(CPU_ENVIRONMENT),
        "executables_manifest_sha256": sha256_file(manifest_path),
        "runtime_environment_schema_version": 1,
        "test_runtime": "one",
    }
    runtime_path = directory / "runtime-environment-input.json"
    runtime_path.write_bytes(dump_bytes(runtime_document))
    evidence_path = write_runtime_evidence(
        directory / "runtime-environment-evidence-input.json",
        runtime_path,
        manifest_path,
    )
    return {
        "config": config_path,
        "executable": executable,
        "manifest": manifest_path,
        "metadata": metadata_path,
        "runtime": runtime_path,
        "runtime_evidence": evidence_path,
    }

import copy
from pathlib import Path

from gpu_suite.config import load_config
from gpu_suite.ordering import implementation_order
from gpu_suite.runner import normalized_parameters, problem_sizes
from gpu_suite.schema import validate_raw_result


ROOT = Path(__file__).resolve().parents[2]
ZERO_HASH = "0" * 64


def pilot_config():
    return copy.deepcopy(load_config(ROOT / "configs" / "pilot.json"))


def raw_success(
    benchmark="cufft", implementation="cpu", cpu_backend=None, wave=0,
    node_index=0, hostname="node0", elapsed=1.0, trial=0, series_role="primary",
):
    config = pilot_config()
    definition = config["benchmarks"][benchmark]
    case = definition["cases"][0]
    parameters = normalized_parameters(benchmark, case["parameters"])
    if benchmark == "curand":
        parameters = dict(parameters)
        if implementation == "cpu":
            parameters.update({
                "cpu_engine": "std::mt19937_64",
                "distribution_interval": "[0,1)",
                "verification_sample_count": parameters["size"],
            })
        else:
            parameters.update({
                "distribution_interval": "(0,1]",
                "generator_algorithm": "CURAND_RNG_PSEUDO_DEFAULT",
                "verification_sample_count": parameters["size"],
            })
    primary_size, secondary_size = problem_sizes(benchmark, case["parameters"])
    if implementation == "cpu":
        if cpu_backend is None:
            cpu_backend = definition["default_speedup_cpu_backend"]
        matching = [
            item for item in definition["series"]
            if item["implementation"] == "cpu" and item["cpu_backend"] == cpu_backend
        ]
        if not matching:
            raise ValueError("CPU series is absent from configuration")
        series = matching[0]
        effective_threads = series["cpu_threads_effective"]
        cpu_parallelism = series["cpu_parallelism"]
    else:
        series = next(
            item for item in definition["series"]
            if item["implementation"] == implementation
        )
        cpu_backend = None
        effective_threads = None
        cpu_parallelism = None
    scope = case["scopes"]["compute"]
    record = {
        "result_schema_version": 1,
        "run_id": "run-1",
        "record_timestamp": "2026-07-14T00:00:01.000Z",
        "measurement_start_timestamp": "2026-07-14T00:00:00.000Z",
        "measurement_end_timestamp": "2026-07-14T00:00:01.000Z",
        "system_label": "local",
        "wave": wave,
        "node_index": node_index,
        "hostname": hostname,
        "block_id": "run-1|{0}|{1}".format(wave, hostname),
        "scheduler": None,
        "scheduler_job_id": None,
        "implementation_order": list(implementation_order(node_index, wave)),
        "benchmark": benchmark,
        "implementation": implementation,
        "scope": "compute",
        "problem_size": primary_size,
        "secondary_size": secondary_size,
        "parameters": parameters,
        "precision": definition["precision"],
        "cpu_backend": cpu_backend,
        "cpu_backend_role": series["cpu_backend_role"],
        "series_role": series_role,
        "cpu_threads_requested": 48 if implementation == "cpu" else None,
        "cpu_threads_effective": effective_threads,
        "cpu_parallelism": cpu_parallelism,
        "warmup": scope["warmup"],
        "repeat": scope["repeat"],
        "trial": trial,
        "attempted": True,
        "failure_origin": None,
        "elapsed_total_sec": elapsed * scope["repeat"],
        "elapsed_sec": elapsed,
        "clock_id": "CLOCK_MONOTONIC",
        "clock_resolution_sec": 1e-9,
        "verification_metrics": {},
        "verification_thresholds": {},
        "verification_primary_metric": None,
        "verification_status": "pass",
        "getrf_info": None,
        "getrs_info": None,
        "device_id": None if implementation == "cpu" else 0,
        "gpu_name": None if implementation == "cpu" else "Test GPU",
        "gpu_uuid": (
            None if implementation == "cpu"
            else "GPU-11111111-1111-1111-1111-111111111111"
        ),
        "cuda_driver_version": None if implementation == "cpu" else "12.8.0",
        "compiler": "TestCompiler",
        "compiler_version": "1.0",
        "global_configure_flags": "-O3",
        "library_name": cpu_backend or benchmark,
        "library_version": None if implementation == "cpu" else "12080",
        "cuda_runtime_version": None if implementation == "cpu" else "12.8.0",
        "git_metadata_available": True,
        "git_commit": "abc",
        "git_dirty": False,
        "git_diff_sha256": None,
        "source_snapshot_sha256": None,
        "config_sha256": ZERO_HASH,
        "runtime_environment_sha256": ZERO_HASH,
        "binary_sha256": ZERO_HASH,
        "exit_code": 0,
        "status": "success",
        "message": "",
    }
    return validate_raw_result(record)

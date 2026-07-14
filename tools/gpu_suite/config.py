"""Strict configuration validation shared by runner and reporting tools."""

import math
from pathlib import Path
from typing import Any, Dict, Mapping, Set, Tuple, Union

from .strict_json import dumps, load


class ConfigError(ValueError):
    pass


BENCHMARKS = ("cufft", "cublas", "cusparse", "cusolver", "curand", "thrust")
TOP_LEVEL_KEYS = {
    "config_schema_version",
    "run_mode",
    "output_format",
    "continue_on_failure",
    "cpu_threads",
    "benchmarks",
}
BENCHMARK_KEYS = {
    "enabled",
    "precision",
    "series",
    "default_speedup_cpu_backend",
    "verification",
    "cases",
}
SERIES_KEYS = {
    "implementation",
    "cpu_backend",
    "cpu_backend_role",
    "series_role",
}
PARAMETER_KEYS = {
    "cufft": {"nfft", "batch", "transform"},
    "cublas": {"size", "alpha", "beta"},
    "cusparse": {"size", "alpha", "beta"},
    "cusolver": {"size", "nrhs"},
    "curand": {"size", "generator", "distribution", "seed", "offset", "order"},
    "thrust": {"size", "operation"},
}
VERIFICATION_KEYS = {
    "cufft": {"abs_tolerance", "rel_tolerance"},
    "cublas": {"abs_tolerance", "rel_tolerance"},
    "cusparse": {"abs_tolerance", "rel_tolerance"},
    "cusolver": {"abs_tolerance", "rel_tolerance"},
    "curand": {
        "sigma_multiplier",
        "expected_mean",
        "expected_second_central_moment",
    },
    "thrust": {"abs_tolerance", "rel_tolerance"},
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _exact_keys(value: Mapping[str, Any], expected: Set[str], context: str) -> None:
    actual = set(value.keys())
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    _require(not missing, "{0} missing keys: {1}".format(context, sorted(missing)))
    _require(not extra, "{0} unknown keys: {1}".format(context, sorted(extra)))


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_parameters(benchmark: str, parameters: Mapping[str, Any]) -> None:
    _exact_keys(parameters, PARAMETER_KEYS[benchmark], benchmark + " parameters")
    primary_name = "nfft" if benchmark == "cufft" else "size"
    _require(_positive_integer(parameters[primary_name]), "invalid primary size")
    for name in ("batch", "nrhs", "seed"):
        if name in parameters:
            _require(_positive_integer(parameters[name]), "invalid " + name)
    if "offset" in parameters:
        _require(
            isinstance(parameters["offset"], int)
            and not isinstance(parameters["offset"], bool)
            and parameters["offset"] >= 0,
            "invalid offset",
        )
    for name in ("alpha", "beta"):
        if name in parameters:
            _require(_finite_number(parameters[name]), "invalid " + name)
    for name in ("transform", "generator", "distribution", "order", "operation"):
        if name in parameters:
            _require(
                isinstance(parameters[name], str) and parameters[name] != "",
                "invalid " + name,
            )
    if benchmark == "cusparse":
        root = math.isqrt(parameters["size"])
        _require(root * root == parameters["size"], "cuSPARSE size must be square")


def _validate_verification(benchmark: str, verification: Mapping[str, Any]) -> None:
    _exact_keys(
        verification, VERIFICATION_KEYS[benchmark], benchmark + " verification"
    )
    for name, value in verification.items():
        _require(_finite_number(value) and float(value) >= 0.0, "invalid " + name)
    if benchmark == "curand":
        _require(verification["sigma_multiplier"] > 0.0, "invalid sigma multiplier")


def _validate_series(benchmark: str, definition: Mapping[str, Any]) -> None:
    series = definition["series"]
    _require(isinstance(series, list) and series, benchmark + " requires series")
    seen = set()  # type: Set[Tuple[str, Any]]
    primary_implementations = {"cpu": 0, "cuda": 0, "openacc": 0}
    primary_production_cpu = []
    for index, item in enumerate(series):
        _require(isinstance(item, Mapping), "series entry must be an object")
        _exact_keys(item, SERIES_KEYS, "series entry")
        implementation = item["implementation"]
        _require(implementation in primary_implementations, "invalid implementation")
        _require(item["series_role"] in {"primary", "auxiliary"}, "invalid series role")
        if implementation == "cpu":
            _require(
                isinstance(item["cpu_backend"], str) and item["cpu_backend"] != "",
                "CPU series requires a backend",
            )
            _require(
                item["cpu_backend_role"] in {"production", "reference"},
                "CPU series requires a backend role",
            )
        else:
            _require(
                item["cpu_backend"] is None and item["cpu_backend_role"] is None,
                "GPU series cannot name a CPU backend",
            )
        key = (implementation, item["cpu_backend"])
        _require(key not in seen, "duplicate configured series")
        seen.add(key)
        if item["series_role"] == "primary":
            primary_implementations[implementation] += 1
            if implementation == "cpu" and item["cpu_backend_role"] == "production":
                primary_production_cpu.append(item["cpu_backend"])
    _require(
        primary_implementations == {"cpu": 1, "cuda": 1, "openacc": 1},
        benchmark + " requires one primary CPU/CUDA/OpenACC series",
    )
    _require(
        len(primary_production_cpu) == 1,
        benchmark + " requires one primary production CPU backend",
    )
    _require(
        definition["default_speedup_cpu_backend"] == primary_production_cpu[0],
        benchmark + " speedup backend must be the primary production CPU backend",
    )


def validate_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ConfigError("configuration must be an object")
    _exact_keys(config, TOP_LEVEL_KEYS, "configuration")
    if config.get("config_schema_version") != 1:
        raise ConfigError("unsupported config_schema_version")
    _require(config["run_mode"] in {"smoke", "pilot", "production"}, "invalid run_mode")
    _require(config["output_format"] in {"jsonl", "csv"}, "invalid output format")
    _require(isinstance(config["continue_on_failure"], bool), "invalid failure policy")
    _require(_positive_integer(config["cpu_threads"]), "invalid CPU thread count")
    benchmarks = config.get("benchmarks")
    if not isinstance(benchmarks, Mapping):
        raise ConfigError("benchmarks must be an object")
    _require(set(benchmarks.keys()) == set(BENCHMARKS), "configuration requires six benchmarks")
    for benchmark in BENCHMARKS:
        definition = benchmarks[benchmark]
        if not isinstance(definition, Mapping):
            raise ConfigError("benchmark {0} must be an object".format(benchmark))
        _exact_keys(definition, BENCHMARK_KEYS, benchmark)
        _require(isinstance(definition["enabled"], bool), "enabled must be boolean")
        _require(
            isinstance(definition["precision"], str) and definition["precision"] != "",
            "invalid precision",
        )
        _require(
            isinstance(definition["default_speedup_cpu_backend"], str)
            and definition["default_speedup_cpu_backend"] != "",
            "invalid speedup backend",
        )
        _require(isinstance(definition["verification"], Mapping), "invalid verification")
        _validate_verification(benchmark, definition["verification"])
        _validate_series(benchmark, definition)
        cases = definition.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ConfigError("benchmark {0} requires cases".format(benchmark))
        signatures = set()  # type: Set[str]
        for case in cases:
            if not isinstance(case, Mapping):
                raise ConfigError("case must be an object")
            _exact_keys(case, {"parameters", "scopes"}, benchmark + " case")
            if not isinstance(case.get("parameters"), Mapping):
                raise ConfigError("case parameters must be an object")
            _validate_parameters(benchmark, case["parameters"])
            signature = dumps(case["parameters"])
            _require(signature not in signatures, "duplicate normalized case")
            signatures.add(signature)
            scopes = case.get("scopes")
            if not isinstance(scopes, Mapping) or set(scopes.keys()) != {
                "compute",
                "end-to-end",
            }:
                raise ConfigError("case requires compute and end-to-end scopes")
            for scope_name, scope in scopes.items():
                if not isinstance(scope, Mapping) or set(scope.keys()) != {
                    "warmup",
                    "repeat",
                    "trials",
                }:
                    raise ConfigError("scope settings are incomplete")
                for name in ("warmup", "repeat", "trials"):
                    value = scope[name]
                    if not isinstance(value, int) or isinstance(value, bool):
                        raise ConfigError("scope count must be an integer")
                    minimum = 0 if name == "warmup" else 1
                    if value < minimum:
                        raise ConfigError("invalid scope count")
                if benchmark == "cusolver" and scope["repeat"] != 1:
                    raise ConfigError("cuSOLVER repeat must be one in every scope")
    return dict(config)


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    return validate_config(load(path))

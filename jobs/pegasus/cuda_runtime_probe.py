#!/usr/bin/env python3
"""Query node-local CUDA Driver API and Runtime versions via libcudart."""

import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping


CudaRuntimeProbe = Callable[[], Mapping[str, Any]]
CUDA_RUNTIME_PROBE_KEYS = {
    "cuda_driver_api_version",
    "cuda_runtime_version",
    "diagnostic",
    "loaded_library",
    "query_status",
}


def _format_cuda_version(value: int) -> str:
    return "{0}.{1}.{2}".format(
        value // 1000, (value % 1000) // 10, value % 10,
    )


def probe_node_cuda_runtime() -> Dict[str, Any]:
    """Load node-local libcudart and call its version APIs."""

    candidates = []
    for directory in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if not directory:
            continue
        try:
            candidates.extend(
                str(path)
                for path in sorted(Path(directory).glob("libcudart.so*"))
                if path.is_file()
            )
        except OSError:
            continue
    located = ctypes.util.find_library("cudart")
    if located:
        candidates.append(located)
    candidates.append("libcudart.so")

    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)

    runtime = None
    load_errors = []
    loaded_library = None
    for candidate in unique_candidates:
        try:
            runtime = ctypes.CDLL(candidate)
            loaded_library = candidate
            break
        except OSError as error:
            load_errors.append("{0}: {1}".format(candidate, error))
    if runtime is None:
        return {
            "cuda_driver_api_version": None,
            "cuda_runtime_version": None,
            "diagnostic": "; ".join(load_errors) or "libcudart was not found",
            "loaded_library": None,
            "query_status": "unavailable",
        }

    try:
        driver_function = runtime.cudaDriverGetVersion
        runtime_function = runtime.cudaRuntimeGetVersion
        for function in (driver_function, runtime_function):
            function.argtypes = [ctypes.POINTER(ctypes.c_int)]
            function.restype = ctypes.c_int
        driver_value = ctypes.c_int()
        runtime_value = ctypes.c_int()
        driver_status = int(driver_function(ctypes.byref(driver_value)))
        runtime_status = int(runtime_function(ctypes.byref(runtime_value)))
    except (AttributeError, TypeError, ValueError) as error:
        return {
            "cuda_driver_api_version": None,
            "cuda_runtime_version": None,
            "diagnostic": "CUDA version API lookup failed: {0}".format(error),
            "loaded_library": loaded_library,
            "query_status": "failure",
        }
    if driver_status != 0 or runtime_status != 0:
        return {
            "cuda_driver_api_version": None,
            "cuda_runtime_version": None,
            "diagnostic": (
                "cudaDriverGetVersion/cudaRuntimeGetVersion returned {0}/{1}"
                .format(driver_status, runtime_status)
            ),
            "loaded_library": loaded_library,
            "query_status": "failure",
        }
    return {
        "cuda_driver_api_version": _format_cuda_version(driver_value.value),
        "cuda_runtime_version": _format_cuda_version(runtime_value.value),
        "diagnostic": None,
        "loaded_library": loaded_library,
        "query_status": "success",
    }

"""Validate FFTW threading provenance without loading any library."""

from typing import Any, Mapping


def validate_fftw_threading(metadata: Mapping[str, Any],
                            require_threaded: bool = False) -> None:
    dependencies = metadata.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise ValueError("FFTW metadata lacks resolved dependencies")
    threading = metadata.get("fftw_threading")
    if "fftw_threading" not in metadata:
        # Saved pre-extension AMD artifacts supported only the pthread library.
        # Preserve those records; never infer a different provider for them.
        if require_threaded and "fftw3f_threads" not in dependencies:
            raise ValueError("legacy threaded FFTW metadata lacks fftw3f_threads")
        return
    fields = {"requested", "backend", "compile_options", "link_options",
              "dependency_names"}
    if not isinstance(threading, Mapping) or set(threading) != fields:
        raise ValueError("invalid FFTW threading metadata fields")
    requested, backend = threading["requested"], threading["backend"]
    if requested not in ("AUTO", "PTHREAD", "OPENMP", "COMBINED", "NONE"):
        raise ValueError("invalid requested FFTW threading mode")
    if backend not in ("none", "pthread", "openmp", "combined"):
        raise ValueError("invalid selected FFTW threading backend")
    if backend != "none" and requested not in ("AUTO", backend.upper()):
        raise ValueError("requested/selected FFTW threading modes differ")
    if require_threaded and backend == "none":
        raise ValueError("threaded FFTW artifact has no verified threading backend")
    for field in ("compile_options", "link_options", "dependency_names"):
        values = threading[field]
        if not isinstance(values, list) or any(
                not isinstance(value, str) or not value or "\x00" in value
                for value in values):
            raise ValueError("invalid FFTW threading " + field)
    names = threading["dependency_names"]
    if len(names) != len(set(names)):
        raise ValueError("duplicate FFTW threading dependency names")
    actual = {name for name in dependencies if name.startswith("fftw")}
    expected = {"fftw3f"} if "fftw3f" in dependencies else set()
    if backend != "none" and not expected:
        raise ValueError("threading backend lacks base FFTW dependency")
    if backend == "pthread":
        expected.add("fftw3f_threads")
    elif backend == "openmp":
        expected.add("fftw3f_omp")
    if backend != "none":
        runtime_prefix = "fftw_openmp_runtime_" if backend == "openmp" else "fftw_pthread_runtime_"
        runtimes = {name for name in actual if name.startswith(runtime_prefix)}
        if runtimes != {runtime_prefix + str(index)
                        for index in range(len(runtimes))}:
            raise ValueError("invalid FFTW threading runtime dependency names")
        expected.update(runtimes)
        if backend == "openmp" and not runtimes and not (threading["compile_options"] or threading["link_options"]):
            raise ValueError("FFTW OpenMP backend lacks runtime link evidence")
    if actual != expected or set(names) != expected:
        raise ValueError("FFTW threading backend/dependency identities differ")

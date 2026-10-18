#!/usr/bin/env python3
"""Build all six AMD families inside one approved PBS allocation, without running them."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time

sys.dont_write_bytecode = True
PROFILES = ("cpu", "hip", "openmp")
FAMILIES = {"rocblas": "blas", "rocfft": "fft", "rocsparse": "sparse",
            "rocsolver": "solver", "rocrand": "rand", "rocthrust": "reduce"}
ENVIRONMENT = ("PATH", "HOME", "LOADEDMODULES", "MODULEPATH", "AOCL_ROOT", "MKLROOT",
               "AOCLSPARSE_NUM_THREADS",
               "ROCM_PATH", "HIP_PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH",
               "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "CMAKE_PREFIX_PATH",
               "CMAKE_INCLUDE_PATH", "CMAKE_LIBRARY_PATH", "FFTW3f_ROOT", "FFTW3f_THREADING",
               "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "HIP_VISIBLE_DEVICES",
               "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
CPU_PATH_OPTIONS = {
    "GPU_SUITE_GENERIC_" + provider + "_" + kind
    for provider in ("CBLAS", "LAPACKE")
    for kind in ("INCLUDE_DIR", "LIBRARY", "EXTRA_LIBRARIES")
} | {"GPU_SUITE_AOCL_SPARSE_" + kind
     for kind in ("INCLUDE_DIR", "LIBRARY", "EXTRA_LIBRARIES")}
AOCL_PROVIDERS = {
    "CBLAS": ("blas_cpu", "GPU_SUITE_GENERIC_CBLAS_", "CBLAS", "cblas", "cblas.h"),
    "LAPACKE": ("solver_cpu", "GPU_SUITE_GENERIC_LAPACKE_", "LAPACKE", "lapacke", "lapacke.h"),
    "AOCLSPARSE": ("sparse_cpu", "GPU_SUITE_AOCL_SPARSE_", "AOCLSparse", "aocl_sparse", "aoclsparse.h"),
}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Non-standard JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=unique, parse_constant=invalid)


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "resolved_path": str(path.resolve(strict=True)),
            "sha256": digest.hexdigest()}


def source_snapshot_sha256(root, relative_paths):
    """Bootstrap v1 snapshot without importing unverified repository Python.

    Keep byte-compatible with tools/gpu_suite/hashing.py. A regression compares
    both implementations, including executable bits, paths and file content.
    """
    repository = Path(root).resolve()
    supplied = list(relative_paths)
    if not repository.is_dir() or len(supplied) != len(set(supplied)):
        raise ValueError("Invalid source snapshot root or duplicate paths")
    normalized = []
    for value in supplied:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ValueError("Invalid snapshot path")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in {"", "."}:
            raise ValueError("Snapshot path escapes source")
        normalized.append(relative.as_posix())
    if len(normalized) != len(set(normalized)):
        raise ValueError("Snapshot paths normalize to duplicates")
    digest = hashlib.sha256(b"gpu-library-suite-source-snapshot-v1\0")
    digest.update(len(normalized).to_bytes(8, "big"))
    for relative in sorted(normalized):
        path = repository
        for component in Path(relative).parts:
            path = path / component
            if path.is_symlink():
                raise ValueError("Snapshot contains a symbolic link: " + relative)
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Snapshot contains a non-regular file: " + relative)
        content, size = hashlib.sha256(), 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                content.update(block)
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or size != after.st_size:
            raise ValueError("Source changed while hashing: " + relative)
        path_bytes = relative.encode("utf-8", errors="strict")
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(bytes((1 if before.st_mode & 0o111 else 0,)))
        digest.update(size.to_bytes(8, "big"))
        digest.update(content.digest())
    return digest.hexdigest()


def slots(benchmarks_only=False):
    result = []
    for profile in PROFILES:
        for library, stem in FAMILIES.items():
            stem = (stem + "_cpu" if profile == "cpu" else stem + "_gpu"
                    if profile == "hip" else "openmp_" + library)
            for role in (("benchmark",) if benchmarks_only else ("example", "benchmark")):
                result.append({"library": library, "profile": profile, "role": role,
                               "target": stem + ("_bench" if role == "benchmark" else ""),
                               "status": "NOT_ATTEMPTED", "compile_status": "NOT_ATTEMPTED",
                               "reason": "Build has not started"})
    return result


def cpu_options(args):
    if args.aocl_cpu_candidates and (args.cpu_blas_backend != "GENERIC_CBLAS"
            or args.cpu_lapack_backend != "GENERIC_LAPACKE" or args.cpu_cmake_option):
        raise ValueError("AOCL candidates require both explicit GENERIC backends and no manual CPU path options")
    result = {"GPU_SUITE_CPU_BLAS_BACKEND": args.cpu_blas_backend,
              "GPU_SUITE_CPU_LAPACK_BACKEND": args.cpu_lapack_backend,
              "GPU_SUITE_CPU_SPARSE_BACKEND": args.cpu_sparse_backend,
              "GPU_SUITE_CPU_THRUST_BACKEND": "STL"}
    for value in args.cpu_cmake_option:
        key, separator, content = value.partition("=")
        if not separator or key not in CPU_PATH_OPTIONS or key in result or not content:
            raise ValueError("Invalid or duplicate explicit CPU CMake option: " + key)
        if key.startswith("GPU_SUITE_AOCL_SPARSE_"):
            if args.cpu_sparse_backend != "AOCLSPARSE":
                raise ValueError("AOCL Sparse paths require the corresponding explicit AOCLSPARSE provider")
        else:
            provider = "CBLAS" if "_CBLAS_" in key else "LAPACKE"
            selected = args.cpu_blas_backend if provider == "CBLAS" else args.cpu_lapack_backend
            if selected != "GENERIC_" + provider:
                raise ValueError("Generic paths require the corresponding explicit GENERIC provider")
        paths = content.split(";") if key.endswith("EXTRA_LIBRARIES") else [content]
        for text in paths:
            path = Path(text)
            if not text or any(c in text for c in "\n\r\x00;") or not path.is_absolute():
                raise ValueError("CPU dependency paths must be absolute")
            if key.endswith("INCLUDE_DIR"):
                if not path.is_dir():
                    raise ValueError("Explicit CPU include directory is unavailable")
            elif not path.is_file():
                raise ValueError("Explicit CPU library is unavailable")
        result[key] = content
    return result


def select_aocl_candidates(env, disabled_root, deadline, clock=time.monotonic, sparse_backend="ONEMKL"):
    """Opt-in, finite shared-library recipes; API validation remains in CMake.

    No version glob, static/shared preference, library permutation or alternate
    installation search is performed. Distinct files are ambiguous even when
    their hashes match. Only aliases resolving to the same file are deduplicated.
    """
    report = {"mode": "AOCL_SHARED_CANDIDATES", "aocl_root": env.get("AOCL_ROOT"),
              "recursive_search": False, "api_validation": "DEFERRED_TO_CMAKE_COMPILE_LINK",
              "abi_validation": "NOT_VERIFIED_BUILD_ONLY", "explicit_ilp64_rejected": True,
              "providers": {}, "candidates": {}, "cmake_options": {}}
    root = Path(env.get("AOCL_ROOT", ""))
    root_error = ""
    try:
        if not root.is_absolute() or not root.is_dir():
            raise ValueError("AOCL_ROOT is not an existing absolute directory")
        root = root.resolve(strict=True)
        if any("ilp64" in part.lower() for part in root.parts):
            raise ValueError("AOCL_ROOT identifies an ILP64 layout")
        report["resolved_aocl_root"] = str(root)
    except (OSError, ValueError) as error:
        root_error = str(error)
    total_bytes = 0

    def collect(label, directories, names):
        nonlocal total_bytes
        identities, errors, observed = {}, [], []
        if root_error:
            errors.append(root_error)
        else:
            for directory in directories:
                for name in names:
                    path = root / directory / name
                    try:
                        if clock() >= deadline:
                            raise ValueError("AOCL selection deadline exhausted")
                        if not os.path.lexists(path):
                            continue
                        resolved_directory = path.parent.resolve(strict=True)
                        if root not in resolved_directory.parents or any("ilp64" in part.lower() for part in resolved_directory.relative_to(root).parts):
                            raise ValueError("Candidate directory escapes AOCL LP64 scope: " + str(path.parent))
                        resolved = path.resolve(strict=True)
                        if root not in resolved.parents:
                            raise ValueError("Candidate resolves outside AOCL_ROOT: " + str(path))
                        if any("ilp64" in part.lower() for part in resolved.relative_to(root).parts):
                            raise ValueError("Candidate resolves to ILP64: " + str(path))
                        info = resolved.stat()
                        if not stat.S_ISREG(info.st_mode):
                            raise ValueError("Candidate is not a regular file: " + str(path))
                        identity = identities.get(str(resolved))
                        if identity is None:
                            if info.st_size > 128 * 1024 * 1024 or total_bytes + info.st_size > 512 * 1024 * 1024:
                                raise ValueError("AOCL candidate hash size limit: " + str(path))
                            identity = file_identity(resolved)
                            identity["observed_layout"] = str(resolved.parent.relative_to(root))
                            total_bytes += info.st_size
                            identity["aliases"] = []
                            identities[str(resolved)] = identity
                        identity["aliases"].append(str(path))
                        observed.append(str(path))
                    except (OSError, ValueError) as error:
                        errors.append(str(error))
        found = list(identities.values())
        if len(found) != 1:
            errors.append(("Missing" if not found else "Ambiguous") + " AOCL " + label + " candidates")
        report["candidates"][label] = {"files": found, "observed_paths": observed, "rejection_reasons": errors}
        return found[0] if not errors else None, errors

    cblas_header, cblas_errors = collect("cblas_header", ("include", "include_LP64"), ("cblas.h",))
    blis, blis_errors = collect("blis", ("lib", "lib_LP64"), ("libblis.so",))
    lapacke_header, lapacke_errors = collect("lapacke_header", ("include", "include_LP64"), ("lapacke.h",))
    lapacke, lapacke_library_errors = collect("lapacke_library", ("lib", "lib_LP64"), ("libflame.so", "liblapacke.so"))
    recipes = [
            ("CBLAS", cblas_header, [blis], cblas_errors + blis_errors),
            ("LAPACKE", lapacke_header, [lapacke, blis], lapacke_errors + lapacke_library_errors + blis_errors)]
    if sparse_backend == "AOCLSPARSE":
        sparse_header, header_errors = collect("aocl_sparse_header", ("include", "include_LP64"), ("aoclsparse.h",))
        sparse_library, library_errors = collect("aocl_sparse_library", ("lib", "lib_LP64"), ("libaoclsparse.so",))
        recipes.append(("AOCLSPARSE", sparse_header, [sparse_library], header_errors + library_errors))
    for provider, header, libraries, errors in recipes:
        _, prefix, _, _, header_name = AOCL_PROVIDERS[provider]
        record = {"status": "REJECTED" if errors else "CANDIDATE_SELECTED",
                  "rejection_reasons": errors, "header": header,
                  "ordered_libraries": [] if errors else libraries}
        if errors:
            # A non-NOTFOUND cache value suppresses CMake find_path/find_library.
            # Both paths stay nonexistent: missing candidates cannot search or
            # accidentally select a system/oneMKL/OpenBLAS provider.
            unavailable = Path(disabled_root) / provider
            if os.path.lexists(unavailable):
                raise ValueError("AOCL rejection sentinel already exists")
            include, library, extras = str(unavailable), str(unavailable / "unavailable.so"), ""
        else:
            include = str(Path(header["aliases"][0]).parent.resolve(strict=True))
            library = libraries[0]["resolved_path"]
            extras = ";".join(item["resolved_path"] for item in libraries[1:])
            record["consumed_header_path"] = str(Path(include) / header_name)
        record["cmake_options"] = {prefix + "INCLUDE_DIR": include,
                                    prefix + "LIBRARY": library,
                                    prefix + "EXTRA_LIBRARIES": extras}
        report["providers"][provider] = record
        report["cmake_options"].update(record["cmake_options"])
    report["hashed_bytes"] = total_bytes
    return report


def validate_aocl_selection(selection, metadata=None, names=()):
    """Reject changed inputs or substitution; never treat filename as API proof."""
    result = {}
    for provider, record in selection["providers"].items():
        target, prefix, interface_name, dependency_name, _ = AOCL_PROVIDERS[provider]
        enabled = target in names or target + "_bench" in names
        if record["status"] == "REJECTED":
            if enabled:
                raise ValueError("Rejected AOCL provider unexpectedly enabled: " + provider)
            for key, value in record["cmake_options"].items():
                if value and os.path.lexists(value):
                    raise ValueError("AOCL rejection sentinel now exists: " + key)
            result[provider] = {"status": "REJECTED", "reasons": record["rejection_reasons"]}
            continue
        for identity in [record["header"]] + record["ordered_libraries"]:
            current = file_identity(identity["resolved_path"])
            if any(current[key] != identity[key] for key in ("resolved_path", "sha256")):
                raise ValueError("Selected AOCL input changed: " + identity["resolved_path"])
        header = file_identity(record["consumed_header_path"])
        if any(header[key] != record["header"][key] for key in ("resolved_path", "sha256")):
            raise ValueError("Selected AOCL consumed header changed: " + record["consumed_header_path"])
        result[provider] = {"status": "API_COMPILE_LINK_PASSED" if enabled else "NOT_ENABLED_BY_CMAKE"}
        if metadata is None or not enabled:
            continue
        inputs = metadata["amd_build_inputs"]
        if inputs.get(interface_name + "_INTERFACE_INCLUDE_DIRECTORIES") != record["cmake_options"][prefix + "INCLUDE_DIR"]:
            raise ValueError("AOCL selected header directory differs from build metadata: " + provider)
        if provider == "AOCLSPARSE" and metadata["dependencies"].get("aocl_sparse_header") != {
                "path": header["resolved_path"], "sha256": header["sha256"]}:
            raise ValueError("AOCL selected header identity differs from build metadata")
        ordered = [item["resolved_path"] for item in record["ordered_libraries"]]
        actual = inputs.get(interface_name + "_INTERFACE_LINK_LIBRARIES", "").split(";")
        if actual[:len(ordered)] != ordered:
            raise ValueError("AOCL selected link order differs from build metadata: " + provider)
        for index, identity in enumerate(record["ordered_libraries"]):
            key = dependency_name + ("_link_" + str(index) if index else "")
            if metadata["dependencies"].get(key) != {"path": identity["resolved_path"], "sha256": identity["sha256"]}:
                raise ValueError("AOCL selected dependency differs from build metadata: " + key)
    return result


def validate(args, env):
    if env.get("GPU_SUITE_SIRIUS_AMD_BUILD_APPROVED") != "yes":
        raise ValueError("Explicit AMD allocated build approval is required")
    if not re.fullmatch(r"[0-9]+(?:\.[A-Za-z0-9_.-]+)?", env.get("PBS_JOBID", "")):
        raise ValueError("A valid PBS_JOBID is required; login-node builds are prohibited")
    if env.get("PBS_ENVIRONMENT") not in (None, "PBS_BATCH"):
        raise ValueError("Only a PBS batch allocation is allowed")
    for path in (args.source_root, args.source_root / "runs", args.run_root):
        if not path.is_absolute() or not path.is_dir() or path.is_symlink() or path.resolve() != path:
            raise ValueError("Source/runs/run roots must be canonical existing directories")
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", args.run_id)
            or ".." in args.run_id or args.run_root != args.source_root / "runs" / args.run_id):
        raise ValueError("Run root must be SOURCE/runs/RUN_ID")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_source_sha256):
        raise ValueError("The approved source snapshot SHA-256 is required")
    if args.clean_benchmarks:
        if not re.fullmatch(r"[0-9a-f]{40}", args.expected_git_commit or ""):
            raise ValueError("Clean benchmark mode requires the approved Git commit")
        if args.rocm_version_file is None or not args.rocm_version_file.is_absolute():
            raise ValueError("Clean benchmark mode requires an explicit ROCm version file")
    elif args.expected_git_commit or args.rocm_version_file:
        raise ValueError("Production preparation options require --clean-benchmarks")
    if not re.fullmatch(r"gfx[0-9a-z]+(?::[a-z0-9]+[+-])*", args.gpu_arch):
        raise ValueError("An explicit approved GPU architecture is required")
    for path in (args.cpu_c_compiler, args.cpu_cxx_compiler, args.amd_cxx_compiler):
        if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError("Explicit executable compiler paths are required")
    if args.amd_cxx_compiler.name == "hipcc":
        raise ValueError("Native HIP must use the Clang driver, not hipcc")
    if not args.rocm_root.is_absolute() or not args.rocm_root.is_dir():
        raise ValueError("An observed absolute ROCm root is required")
    if not (30 <= args.timeout_seconds <= 86400 and 1 <= args.configure_timeout <= args.timeout_seconds
            and 1 <= args.build_timeout <= args.timeout_seconds):
        raise ValueError("Invalid job-local command/time budget")
    cpu_options(args)


def cpu_hints(env):
    groups = {"CMAKE_INCLUDE_PATH": ("CMAKE_INCLUDE_PATH", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"),
              "CMAKE_LIBRARY_PATH": ("CMAKE_LIBRARY_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH"),
              "CMAKE_PREFIX_PATH": ("CMAKE_PREFIX_PATH", "AOCL_ROOT", "AOCL_HOME", "AOCL_INSTALL_DIR")}
    result = {}
    for key, names in groups.items():
        paths = []
        for name in names:
            for path in re.split(r"[:;]", env.get(name, "")):
                if path and path not in paths:
                    if not Path(path).is_absolute():
                        raise ValueError("Relative environment search path: " + name)
                    paths.append(path)
        if paths:
            result[key] = ";".join(paths)
    if env.get("FFTW3f_ROOT"):
        if not Path(env["FFTW3f_ROOT"]).is_absolute():
            raise ValueError("FFTW3f_ROOT must be absolute")
        result["FFTW3f_ROOT"] = env["FFTW3f_ROOT"]
    result["FFTW3f_THREADING"] = "OPENMP"
    return result


def configure_argv(args, cmake, build, profile, env, selected_cpu_options=None):
    flags = {"CMAKE_BUILD_TYPE": "Release", "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
             "GPU_SUITE_VENDOR": "amd", "GPU_SUITE_AMD_LIBRARIES": ";".join(FAMILIES),
             "GPU_SUITE_SOURCE_LANGUAGE": "c-cpp", "GPU_SUITE_BUILD_CUDA": "OFF",
             "GPU_SUITE_BUILD_OPENACC": "OFF", "GPU_SUITE_BUILD_TESTS": "OFF",
             "GPU_SUITE_BUILD_EXAMPLES": "OFF" if args.clean_benchmarks else "ON",
             "GPU_SUITE_BUILD_BENCHMARKS": "ON",
             "Python3_EXECUTABLE": sys.executable,
             "CMAKE_C_COMPILER": str(args.cpu_c_compiler),
             "CMAKE_CXX_COMPILER": str(args.amd_cxx_compiler if profile == "openmp" else args.cpu_cxx_compiler)}
    flags.update(cpu_options(args) if selected_cpu_options is None else selected_cpu_options)
    for item in PROFILES:
        flags["GPU_SUITE_BUILD_" + item.upper()] = "ON" if item == profile else "OFF"
    if profile == "cpu":
        flags.update(cpu_hints(env))
    else:
        flags.update(GPU_SUITE_ROCM_ROOT=str(args.rocm_root), CMAKE_HIP_COMPILER=str(args.amd_cxx_compiler),
                     CMAKE_HIP_ARCHITECTURES=args.gpu_arch)
        if profile == "openmp":
            flags["GPU_SUITE_AMD_GPU_TARGET"] = args.gpu_arch
    return [str(cmake), "-S", str(args.source_root), "-B", str(build)] + ["-D" + k + "=" + v for k, v in flags.items()]


def provider_inventory(env, deadline, clock=time.monotonic):
    """Bounded direct-directory inventory; files are never loaded or executed."""
    result = {"roots": {}, "directories": [], "candidates": [], "limitations": [],
              "automatic_provider_selection": False, "recursive_search": False}
    total_bytes, matched = 0, 0
    for variable in ("AOCL_ROOT", "MKLROOT"):
        value = env.get(variable)
        result["roots"][variable] = value
        if not value:
            continue
        root = Path(value)
        if not root.is_absolute() or not root.is_dir():
            result["limitations"].append(variable + " is not an existing absolute directory")
            continue
        suffixes = ("include", "include_LP64", "lib", "lib_LP64") if variable == "AOCL_ROOT" else ("include", "lib", "lib/intel64", "lib/intel64_lin")
        for suffix in suffixes:
            directory = root / suffix
            item = {"path": str(directory), "exists": directory.is_dir()}
            result["directories"].append(item)
            if not item["exists"]:
                continue
            with os.scandir(directory) as entries:
                for number, entry in enumerate(entries):
                    if number >= 4096 or clock() >= deadline:
                        result["limitations"].append("Directory/time limit reached: " + str(directory))
                        break
                    name = entry.name.lower()
                    if not (name in {"cblas.h", "blis.h", "lapacke.h", "mkl_cblas.h", "mkl_lapacke.h", "mkl_spblas.h"}
                            or name.startswith(("libcblas", "libblis", "liblapacke", "libflame", "libmkl_rt")) or "sparse" in name):
                        continue
                    matched += 1
                    if matched > 256:
                        result["limitations"].append("Candidate limit reached")
                        return result
                    path = Path(entry.path)
                    row = {"path": str(path), "resolved_path": str(path.resolve()), "symlink": path.is_symlink()}
                    try:
                        info = path.stat()
                        row.update(type="regular" if stat.S_ISREG(info.st_mode) else "other", size=info.st_size)
                        if stat.S_ISREG(info.st_mode):
                            if info.st_size > 128 * 1024 * 1024 or total_bytes + info.st_size > 512 * 1024 * 1024:
                                row["hash_status"] = "NOT_HASHED_SIZE_LIMIT"
                                result["limitations"].append("Hash size limit: " + str(path))
                            else:
                                row.update(file_identity(path), hash_status="HASHED")
                                total_bytes += info.st_size
                    except OSError as error:
                        row["error"] = str(error)
                    result["candidates"].append(row)
    result["candidates"].sort(key=lambda value: value["path"])
    result["hashed_bytes"] = total_bytes
    return result


class CommandFailure(ValueError):
    pass


class Build:
    def __init__(self, args, env=None, popen=None, clock=time.monotonic):
        self.args, self.clock = args, clock
        self.env = dict(os.environ if env is None else env)
        validate(args, self.env)
        self.root = args.run_root / "amd-build"
        self.root.mkdir()  # One attempt, including failure: never overwrite/retry.
        for name in ("logs", "tmp", "cache"):
            (self.root / name).mkdir()
        self.popen = subprocess.Popen if popen is None else popen
        self.deadline = clock() + args.timeout_seconds
        self.commands, self.slots, self.profiles = [], slots(args.clean_benchmarks), {}
        self.cpu_flags, self.aocl_selection = cpu_options(args), None
        self.env.update(PYTHONDONTWRITEBYTECODE="1", GIT_OPTIONAL_LOCKS="0",
                        TMPDIR=str(self.root / "tmp"), XDG_CACHE_HOME=str(self.root / "cache"))
        count = int(self.env.get("GIT_CONFIG_COUNT", "0"))
        for key in ("core.fsmonitor", "core.untrackedCache"):
            self.env["GIT_CONFIG_KEY_" + str(count)] = key
            self.env["GIT_CONFIG_VALUE_" + str(count)] = "false"
            count += 1
        self.env["GIT_CONFIG_COUNT"] = str(count)
        self.env.pop("GIT_EXTERNAL_DIFF", None)

    def run(self, stage, argv, timeout=30, env=None):
        prefix = self.root / "logs" / ("%03d-%s" % (len(self.commands) + 1, stage))
        item = {"stage": stage, "argv": [str(v) for v in argv], "started": utc(), "exit_status": None,
                "stdout": str(prefix) + ".stdout", "stderr": str(prefix) + ".stderr", "timeout_seconds": timeout,
                "process_started": False, "timed_out": False}
        self.commands.append(item)
        try:
            with open(item["stdout"], "xb") as out, open(item["stderr"], "xb") as err:
                remaining = self.deadline - self.clock()
                if remaining <= 0:
                    raise CommandFailure("Global build deadline exhausted; command not started")
                item["effective_timeout_seconds"] = min(timeout, remaining)
                process = self.popen(item["argv"], cwd=str(self.root), env=self.env if env is None else env,
                                     stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
                item["process_started"] = True
                try:
                    item["exit_status"] = process.wait(timeout=item["effective_timeout_seconds"])
                except subprocess.TimeoutExpired:
                    item["timed_out"] = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise CommandFailure("Command timeout; owned child stopped, no retry")
        except (OSError, ValueError) as error:
            item["error"] = str(error)
        finally:
            item["ended"] = utc()
            for stream in ("stdout", "stderr"):
                if Path(item[stream]).is_file():
                    item[stream + "_sha256"] = file_identity(item[stream])["sha256"]
            write_json(str(prefix) + ".command.json", item)
        if item["exit_status"] != 0 or "error" in item:
            raise CommandFailure("Command failed: " + stage + ": " + item.get("error", str(item["exit_status"])))
        return Path(item["stdout"]).read_text(encoding="utf-8", errors="strict")


def source_evidence(job):
    git = shutil.which("git", path=job.env.get("PATH"))
    if not git:
        raise ValueError("git is unavailable")
    prefix = [git, "-c", "core.fsmonitor=false", "-c", "core.untrackedCache=false", "-C", str(job.args.source_root)]
    root = job.run("git-root", prefix + ["rev-parse", "--show-toplevel"])
    if Path(root.strip()).resolve() != job.args.source_root:
        raise ValueError("Source is not the expected Git root")
    head = job.run("git-head", prefix + ["rev-parse", "HEAD"]).strip()
    status = job.run("git-status", prefix + ["status", "--short", "--branch", "--untracked-files=all"])
    if job.args.clean_benchmarks and (head != job.args.expected_git_commit
            or any(line and not line.startswith("## ") for line in status.splitlines())):
        raise ValueError("Production source must be clean and match the approved Git commit")
    job.run("git-diff-check", prefix + ["diff", "--no-ext-diff", "--check"])
    inventory = job.run("git-inventory", prefix + ["ls-files", "-z", "--cached", "--others", "--exclude-standard"])
    paths = sorted({p for p in inventory.split("\0") if p and Path(p).parts[0] != "runs"
                    and not any(v.startswith(".fuse_hidden") for v in Path(p).parts)
                    and ((job.args.source_root / p).exists() or (job.args.source_root / p).is_symlink())})
    return {"git_head": head, "git_status": status, "paths": paths,
            "source_snapshot_sha256": source_snapshot_sha256(job.args.source_root, paths),
            "exclusions": ["runs/", ".fuse_hidden* in any path component"]}


def profile_targets(job, profile, build, configure_output):
    expected = {row["target"]: row for row in job.slots if row["profile"] == profile}
    descriptor_path = build / "partial-manifest-input.json"
    descriptor = load_json(descriptor_path) if descriptor_path.is_file() else {
        "build_metadata_path": str(build / "build-metadata.json"), "entries": []}
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("entries"), list):
        raise ValueError("Invalid configured target descriptor")
    if descriptor.get("build_metadata_path") != str(build / "build-metadata.json"):
        raise ValueError("Unexpected build metadata path")
    names = []
    for entry in descriptor["entries"]:
        if not isinstance(entry, dict):
            raise ValueError("Invalid configured target entry")
        name = entry.get("target_name")
        if name not in expected or name in names:
            raise ValueError("Unexpected or duplicate configured target")
        row = expected[name]
        if any(entry.get(key) != value for key, value in (
                ("library", row["library"]), ("implementation", profile), ("build_profile", profile),
                ("executable_role", row["role"]),
                ("executable_path", str(build / "amd/c-cpp" / row["library"] / name)))):
            raise ValueError("Configured target identity mismatch")
        names.append(name)
    for library in FAMILIES:
        rows = [row for row in expected.values() if row["library"] == library]
        present = [row for row in rows if row["target"] in names]
        if 0 < len(present) < len(rows):
            raise ValueError("Only one of the required example/benchmark pair was configured")
        if not present:
            pattern = r"^-- " + library + r" " + profile + r" targets: DISABLED \((.+)\)\s*$"
            reasons = re.findall(pattern, configure_output, re.MULTILINE | re.IGNORECASE)
            if len(reasons) != 1 or not reasons[0].strip():
                raise ValueError("Missing target pair lacks exactly one dependency reason: " + library)
            for row in rows:
                row.update(status="NOT_BUILT_DEPENDENCY", reason=reasons[0])
                provider = {"rocblas": "CBLAS", "rocsolver": "LAPACKE", "rocsparse": "AOCLSPARSE"}.get(library)
                if profile == "cpu" and job.aocl_selection and provider in job.aocl_selection["providers"]:
                    rejected = job.aocl_selection["providers"][provider]["rejection_reasons"]
                    if rejected:
                        row["reason"] += "; AOCL selection: " + "; ".join(rejected)
    return sorted(names)


def verify_artifacts(job, profile, build, names):
    from gpu_suite.manifest import validate_entry
    from gpu_suite.runner import validate_manifest_artifacts
    manifest = load_json(build / "partial-manifest.json")
    metadata = load_json(build / "build-metadata.json")
    if not isinstance(manifest, dict) or not isinstance(metadata, dict):
        raise ValueError("Manifest and build metadata must be objects")
    expected = {row["target"]: row for row in job.slots if row["profile"] == profile}
    if metadata.get("vendor") != "amd" or metadata.get("build_profile") != profile or metadata.get("build_type") != "Release":
        raise ValueError("Build metadata identifies a different build")
    if metadata.get("git_commit") != job.source_before["git_head"] or metadata.get("git_metadata_available") is not True:
        raise ValueError("Build metadata Git identity mismatch")
    if job.args.clean_benchmarks and metadata.get("git_dirty") is not False:
        raise ValueError("Production build metadata must record clean source")
    compilers = {"c": job.args.cpu_c_compiler,
                 "cxx": job.args.amd_cxx_compiler if profile == "openmp" else job.args.cpu_cxx_compiler}
    if profile != "cpu":
        compilers["hip"] = job.args.amd_cxx_compiler
    for language, compiler in compilers.items():
        if Path(metadata[language]["compiler_path"]).resolve() != compiler.resolve():
            raise ValueError("Build metadata compiler mismatch: " + language)
    inputs = metadata.get("amd_build_inputs", {})
    if not isinstance(inputs, dict) or not isinstance(metadata.get("dependencies"), dict):
        raise ValueError("AMD inputs and dependency metadata must be objects")
    if set(inputs.get("GPU_SUITE_AMD_LIBRARIES", "").split(";")) != set(FAMILIES):
        raise ValueError("Build metadata does not select exactly six AMD families")
    for key, value in cpu_options(job.args).items():
        if key.endswith("BACKEND") and inputs.get(key) != value:
            raise ValueError("CPU provider selection changed: " + key)
    if profile != "cpu":
        if metadata["hip"].get("architectures") != job.args.gpu_arch or Path(metadata["rocm"]["root"]).resolve() != job.args.rocm_root.resolve():
            raise ValueError("GPU target or ROCm root mismatch")
        if profile == "openmp" and metadata["openmp"].get("gpu_target") != job.args.gpu_arch:
            raise ValueError("OpenMP GPU target mismatch")
    if profile == "cpu" and "fft_cpu_bench" in names and metadata.get("fftw_threading", {}).get("backend") != "openmp":
        raise ValueError("Requested OpenMP FFTW support is absent; no serial fallback")
    dependencies = {}
    for name, dependency in metadata.get("dependencies", {}).items():
        identity = file_identity(dependency["path"])
        if identity["sha256"] != dependency["sha256"]:
            raise ValueError("Configured dependency hash mismatch: " + name)
        dependencies[name] = identity
    required_dependencies = set()
    for row in expected.values():
        if row["target"] not in names:
            continue
        if profile != "cpu":
            required_dependencies.update(("hip", row["library"]))
            if row["library"] == "rocsolver":
                required_dependencies.add("rocblas")
            if row["library"] == "rocthrust":
                required_dependencies.add("rocprim")
        else:
            sparse_dependencies = {"aocl_sparse", "aocl_sparse_header"} if job.args.cpu_sparse_backend == "AOCLSPARSE" else {"mkl_sparse"}
            required_dependencies.update({"rocblas": {"cblas"}, "rocfft": {"fftw3f", "fftw3f_omp"},
                                          "rocsolver": {"lapacke"}, "rocsparse": sparse_dependencies}.get(row["library"], set()))
    if not required_dependencies.issubset(dependencies):
        raise ValueError("Required configured dependency identities are missing")
    if profile == "cpu" and job.args.cpu_sparse_backend == "AOCLSPARSE" and any(name in names for name in ("sparse_cpu", "sparse_cpu_bench")):
        # Manual ordered inputs are an explicit contract too. Merely recording
        # an unrelated but internally consistent dependency is not sufficient.
        prefix = "GPU_SUITE_AOCL_SPARSE_"
        include = job.cpu_flags.get(prefix + "INCLUDE_DIR")
        if include and inputs.get("AOCLSparse_INTERFACE_INCLUDE_DIRECTORIES") != include:
            raise ValueError("Explicit AOCL Sparse header directory differs from build metadata")
        actual = inputs.get("AOCLSparse_INTERFACE_LINK_LIBRARIES", "").split(";")
        library = job.cpu_flags.get(prefix + "LIBRARY")
        if library and actual[:1] != [library]:
            raise ValueError("Explicit AOCL Sparse primary link order differs from build metadata")
        extras = job.cpu_flags.get(prefix + "EXTRA_LIBRARIES", "")
        expected_extras = extras.split(";") if extras else []
        if actual[1:1 + len(expected_extras)] != expected_extras:
            raise ValueError("Explicit AOCL Sparse extra link order differs from build metadata")
    if profile == "openmp" and any(name in names for name in ("openmp_rocthrust", "openmp_rocthrust_bench")):
        interop = metadata.get("rocthrust_interop") or {}
        if not isinstance(interop, dict):
            raise ValueError("Invalid rocThrust interop metadata")
        wrapper = "amd/c-cpp/rocthrust/examples/rocthrust_wrapper.hip"
        if (interop.get("source") != wrapper or interop.get("source_sha256") != file_identity(job.args.source_root / wrapper)["sha256"]
                or interop.get("compiler_language") != "hip" or interop.get("caller_language") != "cxx"
                or Path(interop.get("compiler_path", "")).resolve() != job.args.amd_cxx_compiler.resolve()
                or interop.get("compile_flags") != metadata["hip"].get("global_configure_flags")
                or interop.get("caller_compile_flags") != metadata["openmp"].get("compile_flags")
                or interop.get("link_flags") != metadata["openmp"].get("link_flags")
                or interop.get("linker_language") != "cxx"
                or Path(interop.get("linker_path", "")).resolve() != job.args.amd_cxx_compiler.resolve()):
            raise ValueError("rocThrust HIP/OpenMP translation-unit provenance mismatch")
    metadata_hash = file_identity(build / "build-metadata.json")["sha256"]
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Invalid manifest entries")
    for entry in entries:
        validate_entry(entry)
    if (manifest.get("build_metadata_sha256") != metadata_hash or len(entries) != len(names)
            or {entry.get("target_name") for entry in entries} != set(names)):
        raise ValueError("Manifest target set or metadata hash mismatch")
    artifacts = {}
    for entry in entries:
        name, row = entry["target_name"], expected[entry["target_name"]]
        path = build / "amd/c-cpp" / row["library"] / name
        if (path.is_symlink() or not path.is_file() or path.resolve() != path
                or any(entry.get(key) != value for key, value in (("library", row["library"]),
                    ("implementation", profile), ("build_profile", profile), ("executable_role", row["role"]),
                    ("executable_path", str(path)), ("build_metadata_sha256", metadata_hash)))):
            raise ValueError("Manifest artifact identity mismatch: " + name)
        identity = file_identity(path)
        language = entry.get("compiler_language")
        expected_language = "hip" if profile == "hip" else "cxx" if profile == "openmp" or row["library"] in {"rocrand", "rocthrust"} else "c"
        if (language != expected_language or entry.get("compiler") != metadata[language]["compiler"]
                or entry.get("compiler_version") != metadata[language]["compiler_version"]
                or entry.get("global_configure_flags") != metadata[language].get("global_configure_flags")):
            raise ValueError("Manifest compiler identity mismatch: " + name)
        if entry.get("binary_sha256") != identity["sha256"]:
            raise ValueError("Built binary hash mismatch: " + name)
        artifacts[name] = {"identity": identity, "manifest_entry": entry}
    # Reuse the suite's read-only manifest/dependency/source checks, not its runner.
    validate_manifest_artifacts(manifest, {metadata_hash: metadata},
                                "production" if job.args.clean_benchmarks else "smoke",
                                not job.args.clean_benchmarks)
    aocl_validation = validate_aocl_selection(job.aocl_selection, metadata, names) if profile == "cpu" and job.aocl_selection else None
    return {"artifacts": artifacts, "dependencies": dependencies, "build_metadata": metadata,
            "aocl_cpu_provider_validation": aocl_validation,
            "files": {name: file_identity(build / name) for name in
                      ("partial-manifest-input.json", "built-manifest-input.json", "partial-manifest.json", "build-metadata.json", "CMakeCache.txt", "compile_commands.json")}}


def build_profile(job, profile, build, names, cmake):
    """Attempt independent targets once, then certify only the successful subset.

    Never invoke gpu_suite_partial_manifest: its dependencies include failed or
    unattempted targets. The existing generator consumes a separate filtered
    descriptor without compiling or executing any artifact.
    """
    rows = {row["target"]: row for row in job.slots if row["profile"] == profile}
    result = job.profiles[profile]
    deadline = min(job.deadline, job.clock() + job.args.build_timeout)
    pinned = {name: file_identity(build / name) for name in
              ("partial-manifest-input.json", "build-metadata.json")}
    result.update(configured_file_identities=pinned, dependency_builds={}, target_builds={})
    compiled = []
    phase = "PROFILE_PROVENANCE"

    def unchanged():
        for name, identity in pinned.items():
            if file_identity(build / name) != identity:
                raise ValueError("Configured provenance changed during build: " + name)

    def attempt(target):
        remaining = deadline - job.clock()
        if remaining <= 0:
            return {"compile_status": "NOT_ATTEMPTED", "failure_stage": "BUILD_BUDGET",
                    "reason": "Shared profile/global build deadline exhausted; command not started"}
        record = {"command_index": len(job.commands) + 1}
        try:
            job.run("build-" + profile + "-" + target,
                    [cmake, "--build", build, "--parallel", "1", "--verbose", "--target", target], remaining)
            record.update(compile_status="SUCCEEDED", reason="Compile/link command exited zero; not yet verified")
        except CommandFailure as error:
            command = job.commands[record["command_index"] - 1]
            state = "TIMED_OUT" if command["timed_out"] else "FAILED" if command["process_started"] else "NOT_ATTEMPTED"
            record.update(compile_status=state, failure_stage="COMPILE_LINK_TIMEOUT" if command["timed_out"] else "COMPILE_LINK",
                          reason=str(error))
        return record

    try:
        for name in names:
            row = rows[name]
            dependencies = ["gpu_suite_common"] if row["role"] == "benchmark" else []
            if profile == "openmp" and row["library"] == "rocthrust":
                dependencies.append("rocthrust_openmp_wrapper")
            blocked = None
            for dependency in dependencies:
                if dependency not in result["dependency_builds"]:
                    result["dependency_builds"][dependency] = attempt(dependency)
                    unchanged()
                if result["dependency_builds"][dependency]["compile_status"] != "SUCCEEDED":
                    blocked = dependency
                    break
            if blocked:
                dependency_record = result["dependency_builds"][blocked]
                row.update(failure_stage="BUILD_BUDGET" if dependency_record["compile_status"] == "NOT_ATTEMPTED" else "SHARED_DEPENDENCY",
                           reason="Shared dependency did not build; no retry: " + blocked + ": " + dependency_record["reason"])
                result["target_builds"][name] = {"compile_status": "NOT_ATTEMPTED", "blocked_by": blocked,
                                                "failure_stage": row["failure_stage"], "reason": row["reason"]}
                continue
            record = attempt(name)
            result["target_builds"][name] = record
            row.update(record)
            if record["compile_status"] == "SUCCEEDED":
                compiled.append(name)
                # A zero compile/link exit alone never establishes BUILT.
                row.update(status="FAILED", failure_stage="MANIFEST_PENDING", reason="Compiled; artifact identity is not yet verified")
            elif record["compile_status"] != "NOT_ATTEMPTED":
                row["status"] = "FAILED"
            unchanged()
        if not compiled:
            return
        unchanged()
        phase = "MANIFEST_GENERATION"
        descriptor = load_json(build / "partial-manifest-input.json")
        filtered = dict(descriptor, entries=[entry for entry in descriptor["entries"] if entry["target_name"] in compiled])
        selected = build / "built-manifest-input.json"
        manifest = build / "partial-manifest.json"
        if os.path.lexists(selected) or os.path.lexists(manifest):
            raise ValueError("Refusing existing subset manifest input or output")
        write_json(selected, filtered)
        remaining = deadline - job.clock()
        if remaining <= 0:
            raise ValueError("Shared profile/global build deadline exhausted before manifest generation")
        result["manifest_command_index"] = len(job.commands) + 1
        for name in compiled:
            rows[name]["failure_stage"] = "MANIFEST_GENERATION"
        job.run("manifest-" + profile,
                [sys.executable, "-B", job.args.source_root / "tools/generate_partial_manifest.py", selected, manifest], remaining)
        phase = "PROFILE_PROVENANCE"
        unchanged()
        phase = "ARTIFACT_VERIFICATION"
        if job.clock() >= deadline:
            raise ValueError("Shared profile/global build deadline exhausted before artifact verification")
        for name in compiled:
            rows[name]["failure_stage"] = "ARTIFACT_VERIFICATION"
        verified = verify_artifacts(job, profile, build, compiled)
        unchanged()
        if job.clock() >= deadline:
            raise ValueError("Shared profile/global build deadline exhausted during artifact verification")
        result.update(verified)
        for name in compiled:
            rows[name].update(status="BUILT", failure_stage=None,
                              reason="Compile/link and subset artifact identity verified; executable not run")
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["error"] = str(error)
        for name in names:
            row = rows[name]
            if row["compile_status"] == "SUCCEEDED":
                row.update(status="FAILED", failure_stage=phase, reason=str(error))
            elif row["status"] == "NOT_ATTEMPTED" and "failure_stage" not in row:
                row.update(failure_stage="PROFILE_PROVENANCE", reason="Profile stopped before target: " + str(error))


def execute(job):
    job.source_before = source_evidence(job)
    write_json(job.root / "source-before.json", job.source_before)
    if job.source_before["source_snapshot_sha256"] != job.args.expected_source_sha256:
        raise ValueError("Source snapshot differs from the approved snapshot; no configure/build")
    # Only the hash-approved source may supply the read-only shared validators.
    sys.path.insert(0, str(job.args.source_root / "tools"))
    try:
        cmake = shutil.which("cmake", path=job.env.get("PATH"))
        if not cmake:
            raise ValueError("cmake is unavailable")
        if job.args.clean_benchmarks:
            sys.path.insert(0, str(job.args.source_root / "jobs/sirius"))
            import amd_prepare
            # Fail before any compiler call if this is not the planned full-host
            # allocation or required preparation tools/evidence are unavailable.
            job.preparation_inputs = amd_prepare.preflight(job)
        if job.args.aocl_cpu_candidates:
            job.aocl_selection = select_aocl_candidates(job.env, job.root / "unavailable-aocl-candidates",
                                                        min(job.deadline, job.clock() + 30), job.clock,
                                                        sparse_backend=job.args.cpu_sparse_backend)
            job.cpu_flags.update(job.aocl_selection["cmake_options"])
            write_json(job.root / "cpu-provider-selection.json", job.aocl_selection)
        tools = {name: file_identity(path) for name, path in (("cmake", cmake), ("cpu_c", job.args.cpu_c_compiler),
                 ("cpu_cxx", job.args.cpu_cxx_compiler), ("amd_cxx", job.args.amd_cxx_compiler), ("python", sys.executable))}
        write_json(job.root / "toolchain.json", {"tools": tools, "environment": {name: job.env.get(name) for name in ENVIRONMENT},
                   "cpu_options": job.cpu_flags, "effective_cpu_hints": cpu_hints(job.env),
                   "gpu_architecture": job.args.gpu_arch})
        for name, identity in tools.items():
            job.run(name + "-version", [identity["path"], "--version"])
        write_json(job.root / "cpu-provider-inventory.json", provider_inventory(job.env, min(job.deadline, job.clock() + 30), job.clock))
        for profile in PROFILES:
            build = job.root / ("build-" + profile)
            profile_rows = [row for row in job.slots if row["profile"] == profile]
            if job.clock() >= job.deadline:
                for row in job.slots:
                    if row["status"] == "NOT_ATTEMPTED":
                        row["reason"] = "Global build deadline exhausted before this profile"
                break
            current = source_evidence(job)
            write_json(job.root / ("source-before-" + profile + ".json"), current)
            if any(current.get(key) != job.source_before.get(key) for key in ("git_head", "paths", "source_snapshot_sha256")):
                raise ValueError("Source changed before " + profile + "; no further builds")
            try:
                if os.path.lexists(build):
                    raise ValueError("Refusing existing profile build tree")
                if profile == "cpu" and job.aocl_selection:
                    validate_aocl_selection(job.aocl_selection)
                output = job.run("configure-" + profile, configure_argv(job.args, cmake, build, profile, job.env, job.cpu_flags), job.args.configure_timeout)
                names = profile_targets(job, profile, build, output)
                if profile == "cpu" and job.aocl_selection:
                    validate_aocl_selection(job.aocl_selection, names=names)
                job.profiles[profile] = {"configured_targets": names}
                if names:
                    build_profile(job, profile, build, names, cmake)
            except (OSError, ValueError, KeyError, TypeError) as error:
                job.profiles[profile] = {"error": str(error)}
                for row in profile_rows:
                    if row["status"] != "NOT_BUILT_DEPENDENCY":
                        row.update(status="FAILED", failure_stage="CONFIGURE", reason=str(error))
        # The same HIP/runtime dependency must identify the same bytes in both
        # GPU build profiles, even though each artifact has its own compiler hash.
        from gpu_suite.runner import validate_manifest_artifacts
        combined, metadata_by_hash = [], {}
        for profile in job.profiles.values():
            if "artifacts" in profile:
                combined.extend(item["manifest_entry"] for item in profile["artifacts"].values())
                metadata_by_hash[profile["files"]["build-metadata.json"]["sha256"]] = profile["build_metadata"]
        validate_manifest_artifacts({"entries": combined}, metadata_by_hash,
                                    "production" if job.args.clean_benchmarks else "smoke",
                                    not job.args.clean_benchmarks)
    finally:
        after = source_evidence(job)
        write_json(job.root / "source-after.json", after)
        if any(after.get(key) != job.source_before.get(key) for key in ("git_head", "paths", "source_snapshot_sha256")):
            raise ValueError("Source changed during build; all produced artifacts are unusable")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "run-root", "cpu-c-compiler", "cpu-cxx-compiler", "amd-cxx-compiler", "rocm-root"):
        result.add_argument("--" + name, type=Path, required=True)
    for name in ("run-id", "expected-source-sha256", "gpu-arch"):
        result.add_argument("--" + name, required=True)
    result.add_argument("--timeout-seconds", type=int, default=1500)
    result.add_argument("--configure-timeout", type=int, default=180)
    result.add_argument("--build-timeout", type=int, default=600)
    result.add_argument("--clean-benchmarks", action="store_true",
                        help="Build only 18 clean production benchmarks, then check CPU/APU mapping and collect runtime inputs in this allocation")
    result.add_argument("--expected-git-commit")
    result.add_argument("--rocm-version-file", type=Path)
    result.add_argument("--cpu-blas-backend", choices=("ONEMKL", "OPENBLAS", "GENERIC_CBLAS"), default="ONEMKL")
    result.add_argument("--cpu-lapack-backend", choices=("ONEMKL", "OPENBLAS", "GENERIC_LAPACKE"), default="ONEMKL")
    result.add_argument("--cpu-sparse-backend", choices=("ONEMKL", "AOCLSPARSE"), default="ONEMKL")
    result.add_argument("--cpu-cmake-option", action="append", default=[])
    result.add_argument("--aocl-cpu-candidates", action="store_true",
                        help="Opt in to bounded AOCL shared-library candidates for both explicit GENERIC CPU backends and, when selected, AOCLSPARSE; ABI remains unverified")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    job = None
    summary = {"run_id": args.run_id, "started": utc(), "status": "FAILED", "artifact_execution_count": 0,
               "gpu_query_count": 0, "scheduler_call_count": 0, "ssh_call_count": 0,
               "job_id": os.environ.get("PBS_JOBID"), "hostname": socket.gethostname(),
               "user_id": os.getuid(), "HOME": os.environ.get("HOME"), "PATH": os.environ.get("PATH")}
    try:
        os.umask(0o077)
        validate(args, os.environ)
        job = Build(args)
        execute(job)
        counts = Counter(row["status"] for row in job.slots)
        summary["status"] = "BUILT" if counts.get("BUILT") == len(job.slots) else "INCOMPLETE" if set(counts) <= {"BUILT", "NOT_BUILT_DEPENDENCY"} else "FAILED"
    except (OSError, ValueError, KeyError, TypeError) as error:
        summary["error"] = str(error)
        if job:
            for row in job.slots:
                if row["status"] == "BUILT":
                    row.update(status="FAILED", failure_stage="FINAL_PROVENANCE", reason=str(error))
                elif row["status"] == "NOT_ATTEMPTED":
                    row["reason"] = str(error)
    summary["build_status"] = summary["status"]
    if args.clean_benchmarks:
        summary["preparation_status"] = "NOT_RUN"
        if summary["build_status"] == "BUILT":
            try:
                import amd_prepare
                summary["preparation"] = amd_prepare.prepare(job)
                summary.update(preparation_status="COMPLETE", status="PREPARED")
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                summary.update(preparation_status="FAILED", status="PREPARATION_FAILED", error=str(error))
        # These are real device-runtime checks, never benchmark executions.
        summary["gpu_query_count"] = None
        summary["device_check_commands"] = [r for r in job.commands if r["stage"] == "placement-mpi"] if job else []
    summary.update(ended=utc(), slots=job.slots if job else slots(args.clean_benchmarks),
                   mode="clean-benchmarks" if args.clean_benchmarks else "all",
                   commands=job.commands if job else [], profiles=job.profiles if job else {},
                   aocl_cpu_provider_selection=job.aocl_selection if job else None)
    summary["counts"] = dict(Counter(row["status"] for row in summary["slots"]))
    if job:
        write_json(job.root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    return 0 if summary["status"] in {"BUILT", "PREPARED"} else 1


if __name__ == "__main__":
    sys.exit(main())

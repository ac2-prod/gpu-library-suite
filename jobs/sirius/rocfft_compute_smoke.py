#!/usr/bin/env python3
"""One allocated Sirius job; no scheduler, network, retry, or source mutation."""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import os
import pwd
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True  # Imports must not create __pycache__ in source.

FFTW_HINT_VARIABLES = (
    "FFTW_ROOT", "FFTW3f_ROOT", "FFTW3F_ROOT", "FFTW3_ROOT",
    "CMAKE_PREFIX_PATH", "CMAKE_INCLUDE_PATH", "CMAKE_LIBRARY_PATH",
    "PKG_CONFIG_PATH", "PKG_CONFIG_LIBDIR",
)
CPU_COMPILERS = {
    "c": ("CPU_CC", ("gcc", "cc", "clang"), "cpu-c-version"),
    "cxx": ("CPU_CXX", ("g++", "c++", "clang++"), "cpu-cxx-version"),
}


def validate_run_location(source, output, run_id=None):
    """Keep every new run visible through the repository's runs/ mount."""
    source, output = Path(source), Path(output)
    if not source.is_absolute() or not output.is_absolute():
        raise ValueError("Source/run roots must be absolute paths")
    if source.is_symlink() or output.is_symlink() or (source / "runs").is_symlink():
        raise ValueError("Source/runs/output directories must not be symlinks")
    if source.resolve(strict=True) != source or output.resolve() != output:
        raise ValueError("Source/run roots must use canonical directory paths")
    if output.parent != source / "runs":
        raise ValueError("Run root must be a direct child of SOURCE_ROOT/runs")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", output.name) or ".." in output.name:
        raise ValueError("Invalid run directory name")
    if run_id is not None and output.name != run_id:
        raise ValueError("Run directory must match the run ID")
    return source, output


def snapshot_paths(paths):
    """Exclude outputs and FUSE tombstones before reading any file bytes."""
    return sorted({path for path in paths if path and Path(path).parts[0] != "runs"
                   and not any(part.startswith(".fuse_hidden") for part in Path(path).parts)})


def tool_identity(path):
    # Realpath is evidence only: invoking it can change compiler-driver behavior.
    return {"invocation": path, "resolved_path": str(Path(path).resolve(strict=True)),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def fftw_cmake_flags(environment):
    flags = ["-D" + name + "=" + environment[name] for name in FFTW_HINT_VARIABLES[:4]
             if environment.get(name)]
    if "FFTW3f_THREADING" in environment:
        selection = environment["FFTW3f_THREADING"]
        if selection not in {"AUTO", "PTHREAD", "OPENMP", "COMBINED", "NONE"}:
            raise ValueError("Invalid FFTW3f_THREADING selection")
        flags.append("-DFFTW3f_THREADING=" + selection)
    return flags


def cpu_thread_allocation(requested, allocated, environment=None):
    """Bound an explicit thread request by the verified allocation and affinity."""
    if type(requested) is not int or requested < 1:
        raise ValueError("CPU threads must be a positive integer")
    if type(allocated) is not int or allocated < 1:
        raise ValueError("A verified CPU allocation is required")
    environment = os.environ if environment is None else environment
    openmp_limit = None
    if "OMP_THREAD_LIMIT" in environment:
        value = environment["OMP_THREAD_LIMIT"]
        if not re.fullmatch(r"[0-9]+", value) or int(value) < 1:
            raise ValueError("OMP_THREAD_LIMIT must be a positive integer when set")
        openmp_limit = int(value)
        if requested > openmp_limit:
            raise ValueError("OMP_THREAD_LIMIT is below the requested CPU thread count")
    affinity_count = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except OSError as error:
            raise ValueError("Cannot confirm process CPU affinity: " + str(error)) from error
        if affinity_count < 1:
            raise ValueError("Process CPU affinity is empty")
    limit = min(allocated, affinity_count) if affinity_count is not None else allocated
    if requested > limit:
        raise ValueError("CPU thread request exceeds verified allocation or affinity")
    return {"requested_threads": requested, "effective_threads": requested,
            "allocated_cpus": allocated, "affinity_cpu_count": affinity_count,
            "confirmed_thread_limit": limit, "openmp_thread_limit": openmp_limit}


def smoke_config(source, cpu_threads):
    from gpu_suite.strict_json import load
    from gpu_suite.config import validate_config
    name = "rocfft-smoke.json" if cpu_threads == 1 else "rocfft-threaded-smoke.json"
    config = load(source / "jobs/sirius" / name)
    config.update(cpu_threads=cpu_threads)
    if config["continue_on_failure"] or config["run_mode"] != "smoke":
        raise ValueError("Canonical smoke must stop on failure and retain smoke mode")
    cpu = [item for item in config["benchmarks"]["rocfft"]["series"]
           if item["implementation"] == "cpu"]
    expected = "cpu-fftw-serial" if cpu_threads == 1 else "cpu-fftw-threaded"
    if len(cpu) != 1 or cpu[0]["cpu_backend"] != expected:
        raise ValueError("Smoke configuration must have exactly the requested CPU backend")
    cpu[0]["cpu_threads_effective"] = cpu_threads
    validate_config(config)
    return config


def campaign_threads(cpu_threads, include_serial=False):
    if include_serial and cpu_threads < 2:
        raise ValueError("--include-serial-campaign requires --cpu-threads greater than 1")
    return [("serial", 1), ("threaded", cpu_threads)] if include_serial else [(None, cpu_threads)]


def require_all_profiles(successful, failures):
    if set(successful) != {"cpu", "hip", "openmp"} or failures:
        raise ValueError("All CPU/HIP/OpenMP build, dependency and example checks must pass; no benchmark will run")


def validate_smoke_rows(rows, config, manifest, config_hash, runtime_hash, run_id, source_hash,
                        scheduler, job_id):
    """Require the full canonical matrix and numerical evidence, not just exit 0."""
    from gpu_suite.validation import validate_campaign
    from gpu_suite.runner import build_schedule, validate_subprocess_record, core_raw_parameters
    from gpu_suite.ordering import implementation_order
    from gpu_suite.strict_json import dumps
    report = validate_campaign(rows, config, config_hash, manifest)
    definition = config["benchmarks"]["rocfft"]
    expected_count = len(definition["series"]) * sum(
        scope["trials"] for case in definition["cases"] for scope in case["scopes"].values())
    if (report["validation_status"] != "pass" or report["record_count"] != expected_count
            or report["block_count"] != 1 or report["wave_count"] != 1):
        raise ValueError("Smoke requires the complete successful canonical raw-result matrix")
    first = rows[0]
    context = dict(config=config, cpu_threads=config["cpu_threads"], device=0,
                   hostname=first["hostname"], node_index=0, run_id=run_id, wave=0,
                   scheduler=scheduler, scheduler_job_id=job_id,
                   system_label="Sirius", implementation_order=implementation_order(0, 0, "amd"))
    schedule = build_schedule(config, manifest, context)
    expected = {(item["series"]["implementation"], item["scope"],
                 dumps(item["parameters"])): item for item in schedule}
    for row in rows:
        if row["status"] != "success" or row["verification_status"] != "pass":
            raise ValueError("Every smoke row must succeed with numerical verification; skips are not passes")
        key = (row["implementation"], row["scope"], dumps(core_raw_parameters("rocfft", row["parameters"])))
        if key not in expected:
            raise ValueError("Unexpected smoke invocation")
        validate_subprocess_record(row, expected[key], context, config_hash, runtime_hash,
                                   row["git_diff_sha256"], source_hash)
        for metric, tolerance in (("dc_relative_error", "rel_tolerance"),
                                  ("non_dc_max_abs_error", "abs_tolerance")):
            value = row["verification_metrics"][metric]
            if value is None or not 0 <= value <= definition["verification"][tolerance]:
                raise ValueError("Smoke numerical metric violates configured tolerance: " + metric)
    return report


def execute_campaigns(job, successful, runtime, source_hash, manifest, python):
    """Run separate serial/threaded and JSONL/CSV campaigns with fixed provenance."""
    from gpu_suite.hashing import sha256_file
    from gpu_suite.strict_json import load, dump_bytes
    from gpu_suite.results_io import load_raw_results
    require_all_profiles(successful, job.failures)
    source = job.args.source_root
    original_env = job.env.copy()
    try:
        for label, threads in campaign_threads(job.args.cpu_threads, getattr(job.args, "include_serial_campaign", False)):
            root = job.root if label is None else job.root / label
            if label is not None:
                root.mkdir()
            job.env.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE")
            allocation = cpu_thread_allocation(threads, job.args.allocated_cpus, job.env)
            config = smoke_config(source, threads)
            for fmt in ("jsonl", "csv"):
                suffix = fmt if label is None else label + "-" + fmt
                run_id = job.args.run_id + "-" + suffix
                document = dict(config, output_format=fmt)
                config_path, raw = root / ("config-" + fmt + ".json"), root / ("raw." + fmt)
                with config_path.open("xb") as stream:
                    stream.write(dump_bytes(document))
                config_hash = sha256_file(config_path)
                campaign_runtime = copy.deepcopy(runtime)
                campaign_runtime.update(run_id=run_id, config_sha256=config_hash,
                                        cpu_thread_allocation=allocation)
                campaign_runtime["environment"].update(OMP_NUM_THREADS=job.env["OMP_NUM_THREADS"],
                                                       OMP_DYNAMIC=job.env["OMP_DYNAMIC"])
                runtime_bytes = dump_bytes(campaign_runtime)
                with (root / ("runtime-environment-" + fmt + ".json")).open("xb") as stream:
                    stream.write(runtime_bytes)
                runtime_hash = hashlib.sha256(runtime_bytes).hexdigest()
                argv = [python, source / "tools/run_suite.py", "--config", config_path, "--manifest", manifest,
                    "--run-id", run_id, "--system-label", "Sirius", "--wave", "0", "--node-index", "0",
                    "--device", "0", "--source-snapshot-sha256", source_hash, "--runtime-environment-sha256", runtime_hash,
                    "--scheduler", job.args.scheduler, "--scheduler-job-id", job.args.job_id, "--output", raw]
                for build in successful.values():
                    argv += ["--build-metadata", build / "build-metadata.json"]
                code, _ = job.run("benchmark-" + suffix, argv, timeout=180)
                validation_code, _ = job.run("validator-" + suffix, [python, source / "tools/validate_results.py",
                    "--config", config_path, "--manifest", manifest,
                    "--report-output", root / ("validation-" + fmt + ".json"), raw])
                try:
                    rows = load_raw_results(raw)
                    report = validate_smoke_rows(rows, document, load(manifest), config_hash,
                                                runtime_hash, run_id, source_hash,
                                                job.args.scheduler, job.args.job_id)
                    write_json(root / ("strict-validation-" + fmt + ".json"), report)
                    write_json(root / ("numerical-evidence-" + fmt + ".json"), [{key: row[key] for key in
                        ("implementation", "cpu_backend", "cpu_threads_effective", "scope", "trial", "status",
                         "verification_status", "verification_metrics", "verification_thresholds", "gpu_name",
                         "device_id", "parameters")} for row in rows])
                except (OSError, ValueError, KeyError, TypeError) as error:
                    job.fail("raw-verification-" + suffix, detail=str(error))
                if code or validation_code or job.failures:
                    return
    finally:
        job.env.clear()
        job.env.update(original_env)


def require_threaded_capability(manifest, metadata, cpu_threads):
    if cpu_threads == 1:
        return
    entries = [entry for entry in manifest["entries"]
               if entry.get("target_name") == "fft_cpu_bench"
               and entry.get("implementation") == "cpu" and entry.get("library") == "rocfft"]
    if len(entries) != 1 or "cpu-fftw-threaded" not in entries[0].get("supported_cpu_backends", []):
        raise ValueError("Threaded CPU smoke requires a manifest-verified cpu-fftw-threaded artifact; no serial fallback")
    threading = metadata.get("fftw_threading", {})
    if threading.get("backend") not in {"pthread", "openmp", "combined"}:
        raise ValueError("Threaded CPU smoke requires explicit FFTW threading provenance")
    from gpu_suite.fftw_metadata import validate_fftw_threading
    validate_fftw_threading(metadata, require_threaded=True)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


class Job:
    def __init__(self, args, root):
        self.args, self.root = args, root
        self.cpu_allocation = cpu_thread_allocation(args.cpu_threads, args.allocated_cpus)
        self.logs = root / "logs"
        self.logs.mkdir()
        self.failures, self.commands = [], []
        self.deadline = time.monotonic() + 12 * 60
        self.env = os.environ.copy()
        self.env.update(PYTHONDONTWRITEBYTECODE="1", GIT_OPTIONAL_LOCKS="0",
                        OMP_NUM_THREADS=str(args.cpu_threads), OMP_TARGET_OFFLOAD="MANDATORY",
                        OMP_DEFAULT_DEVICE="0", OMP_DYNAMIC="FALSE",
                        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        # Also constrain Git reads performed indirectly by CMake/manifest tools.
        count = int(self.env.get("GIT_CONFIG_COUNT", "0"))
        for key, value in (("core.fsmonitor", "false"), ("core.untrackedCache", "false")):
            self.env["GIT_CONFIG_KEY_" + str(count)] = key
            self.env["GIT_CONFIG_VALUE_" + str(count)] = value
            count += 1
        self.env["GIT_CONFIG_COUNT"] = str(count)
        self.env.pop("GIT_EXTERNAL_DIFF", None)
        for name in ("tmp", "cache"):
            (root / name).mkdir()
        self.env.update(TMPDIR=str(root / "tmp"), XDG_CACHE_HOME=str(root / "cache"),
                        ROCFFT_RTC_CACHE_PATH=str(root / "cache" / "rocfft.db"),
                        PYTHONPATH=str(args.source_root / "tools"))
        write_json(root / "allocation-start.json", {"hostname": socket.gethostname(), "utc": utc(),
            "user": pwd.getpwuid(os.getuid()).pw_name,
            "effective_user": pwd.getpwuid(os.geteuid()).pw_name,
            "HOME": self.env.get("HOME"), "PATH": self.env.get("PATH"),
            "scheduler": args.scheduler, "job_id": args.job_id, "allocated_cpus": args.allocated_cpus,
            "allocated_gpus": args.allocated_gpus, "host_cpu_count": os.cpu_count(), "run_id": args.run_id,
            "cpu_thread_allocation": self.cpu_allocation,
            "visibility": {key: self.env.get(key) for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")}})

    def fail(self, stage, code=1, detail=""):
        self.failures.append({"stage": stage, "exit_status": int(code), "detail": detail})

    def run(self, stage, argv, timeout=180, required=True):
        argv = [str(value) for value in argv]
        prefix = self.logs / ("%03d-%s" % (len(self.commands) + 1, stage))
        item = {"stage": stage, "argv": argv, "started": utc(), "stdout": str(prefix) + ".stdout",
                "stderr": str(prefix) + ".stderr", "exit_status": None}
        self.commands.append(item)
        code = 125
        with open(item["stdout"], "xb") as out, open(item["stderr"], "xb") as err:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                err.write(b"Job-local time budget exhausted; command not started.\n")
            else:
                try:
                    process = subprocess.Popen(argv, stdout=out, stderr=err, env=self.env,
                                               cwd=str(self.root), start_new_session=True)
                    try:
                        code = process.wait(timeout=min(timeout, remaining))
                    except subprocess.TimeoutExpired:
                        # Only this helper's own newly created child process group.
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        err.write(b"Command exceeded the job-local timeout.\n")
                        code = 124
                except OSError as error:
                    err.write((str(error) + "\n").encode())
                    code = 127
        item.update(exit_status=code, ended=utc())
        write_json(Path(str(prefix) + ".command.json"), item)
        if required and code:
            self.fail(stage, code)
        text = Path(item["stdout"]).read_text(encoding="utf-8", errors="replace")
        return code, text

    def executable(self, name):
        found = shutil.which(name, path=self.env.get("PATH"))
        # Preserve argv[0], including clang++ and AMD's multicall driver names.
        return os.path.abspath(found) if found and Path(found).is_file() else None

    def tool(self, names, stage):
        for name in names:
            path = self.executable(name)
            if path:
                code, version = self.run(stage, [path, "--version"], timeout=20, required=False)
                if code == 0:
                    return path, version
        raise ValueError("No working " + stage)


def load_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_compiler(job, language):
    """Honor an explicit module-selected driver, or prefer the GNU PATH driver."""
    variable, candidates, stage = CPU_COMPILERS[language]
    if variable not in job.env:
        return job.tool(candidates, stage)
    # An explicit selection is a single executable path, never a shell command.
    # Do not resolve symlinks: argv[0] can select the C++ compiler driver mode.
    invocation = job.env[variable]
    path = Path(invocation)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(variable + " must name an absolute executable file")
    code, version = job.run(stage, [invocation, "--version"], timeout=20, required=False)
    if code:
        raise ValueError(variable + " compiler --version failed; no fallback")
    return invocation, version


def architecture(job, rocm_root):
    # Do not change scheduler-provided GPU visibility or select a physical GPU.
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        value = job.env.get(name)
        if value is not None:
            entries = [part.strip() for part in value.split(",") if part.strip()]
            if len(entries) != 1 or entries[0] in {"-1", "all"}:
                raise ValueError("GPU visibility is not a single device: " + name)
    candidates = []
    for name in ("amdgpu-arch", "offload-arch", "rocminfo"):
        path = job.executable(name)
        for candidate in ([Path(path)] if path else []) + [rocm_root / "bin" / name, rocm_root / "llvm" / "bin" / name]:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                pair = (str(candidate.absolute()), name)
                if pair not in candidates:
                    candidates.append(pair)
    for path, name in candidates:
        code, output = job.run("gpu-architecture-" + name, [path], timeout=45, required=False)
        if code:
            continue
        if name == "rocminfo":
            targets = re.findall(r"^\s*Name:\s+(gfx[0-9a-z]+(?:\:[a-z0-9]+[+-])*)\s*$", output, re.M)
        else:
            targets = [line.strip() for line in output.splitlines() if re.fullmatch(r"gfx[0-9a-z]+(?:\:[a-z0-9]+[+-])*", line.strip())]
        if len(targets) == 1:
            return targets[0]
        if targets:
            raise ValueError("Architecture query exposes multiple GPU agents; no implicit device masking")
    raise ValueError("Cannot verify one compute-node GPU architecture")


def find_rocm(job):
    roots = []
    for name in ("ROCM_PATH", "HIP_PATH"):
        if job.env.get(name):
            roots.append(Path(job.env[name]))
    hipconfig = job.executable("hipconfig")
    if hipconfig:
        job.run("hipconfig-version", [hipconfig, "--version"], timeout=20, required=False)
        code, output = job.run("hipconfig-path", [hipconfig, "--path"], timeout=20, required=False)
        if code == 0 and output.strip().startswith("/") and "\n" not in output.strip():
            roots.append(Path(output.strip()))
    for name in ("hipcc", "amdclang++", "clang++"):
        path = job.executable(name)
        if path:
            job.run(name.replace("+", "x") + "-version", [path, "--version"], timeout=20, required=False)
            roots.extend(list(Path(path).parents)[:3])
            roots.extend(list(Path(path).resolve().parents)[:4])
    seen = set()
    for root in roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        if ((root / "include/hip/hip_runtime_api.h").is_file() and
                (root / "include/rocfft/rocfft.h").is_file()):
            return root
    raise ValueError("No observed ROCm prefix has both HIP and rocFFT headers")


def amd_compiler(job, root):
    candidates = [job.executable("amdclang++"), job.executable("clang++")]
    candidates += [str(root / part) for part in ("bin/amdclang++", "llvm/bin/clang++", "bin/clang++")]
    seen = set()
    for path in candidates:
        if not path or not Path(path).is_file() or not os.access(path, os.X_OK):
            continue
        path = os.path.abspath(path)
        if path in seen or Path(path).name == "hipcc":
            continue
        seen.add(path)
        code, version = job.run("amd-cxx-version", [path, "--version"], timeout=20, required=False)
        if code == 0 and "AMD clang" in version:
            return path
    raise ValueError("No observed AMD Clang C++ executable; hipcc is not used as native HIP compiler")


def execute(job):
    args, root, source = job.args, job.root, job.args.source_root
    sys.path.insert(0, str(source / "tools"))
    from gpu_suite.hashing import source_snapshot_sha256, sha256_file
    from gpu_suite.strict_json import load
    from gpu_suite.scheduler import validate_scheduler_identity
    validate_scheduler_identity(args.scheduler, args.job_id)
    evidence_helper = load_helper("stage31_evidence", source / "jobs/sirius/rocfft_smoke_evidence.py")
    git = job.executable("git")
    if not git:
        raise ValueError("git is unavailable")
    git_prefix = [git, "-c", "core.fsmonitor=false", "-c", "core.untrackedCache=false", "-C", source]
    def source_paths():
        code, _ = job.run("git-source-paths", git_prefix + ["ls-files", "-z", "--cached", "--others", "--exclude-standard"])
        if code:
            raise ValueError("Source path inventory failed")
        content = Path(job.commands[-1]["stdout"]).read_bytes().decode("utf-8", errors="strict")
        all_paths = content.rstrip("\0").split("\0")
        # Never read/hash a tombstone or this/another run's output as source.
        return [path for path in snapshot_paths(all_paths)
                if (source / path).exists() or (source / path).is_symlink()]
    code, git_root = job.run("git-root", git_prefix + ["rev-parse", "--show-toplevel"])
    if code or Path(git_root.strip()).resolve() != source:
        raise ValueError("remote source is not the expected Git root")
    for label, tail in (("git-head", ["rev-parse", "HEAD"]),
                        ("git-status", ["-c", "core.fsmonitor=false", "-c", "core.untrackedCache=false", "status", "--short", "--branch", "--untracked-files=all"]),
                        ("git-diff-check", ["diff", "--no-ext-diff", "--check"])):
        code, _ = job.run(label, git_prefix + tail)
        if code:
            raise ValueError(label + " failed")
    paths = source_paths()
    source_hash = source_snapshot_sha256(source, paths)
    write_json(root / "source-identity.json", {"sha256": source_hash, "paths": paths,
               "exclusions": ["runs/", ".fuse_hidden* in any path component"],
               "kind": "dirty preliminary smoke; not a measurement release"})
    cmake, _ = job.tool(("cmake",), "cmake-version")
    cpu_cc, _ = cpu_compiler(job, "c")
    cpu_cxx, _ = cpu_compiler(job, "cxx")
    python = str(Path(sys.executable).resolve())
    job.run("python-version", [python, "--version"], timeout=20)
    rocm = compiler = arch = None
    try:
        rocm = find_rocm(job)
        compiler = amd_compiler(job, rocm)
        arch = architecture(job, rocm)
    except (OSError, ValueError) as error:
        job.fail("GPU_TOOLCHAIN_OR_ALLOCATION", detail=str(error))
    write_json(root / "toolchain.json", {
        "tools": {name: tool_identity(path) for name, path in (
            ("cmake", cmake), ("cpu_cc", cpu_cc), ("cpu_cxx", cpu_cxx),
            ("python", python), ("amd_cxx", compiler)) if path},
        "fftw_search_environment": {key: job.env.get(key) for key in FFTW_HINT_VARIABLES},
        "fftw_threading_requested": job.env.get("FFTW3f_THREADING", "AUTO"),
        "cpu_compiler_selection": {language: {
            "environment_variable": variable, "requested": job.env.get(variable),
            "default_candidates": list(candidates),
        } for language, (variable, candidates, _) in CPU_COMPILERS.items()},
    })
    write_json(root / "allocation.json", {"hostname": socket.gethostname(), "utc": utc(),
        "scheduler": args.scheduler, "job_id": args.job_id, "allocated_cpus": args.allocated_cpus,
        "allocated_gpus": args.allocated_gpus, "host_cpu_count": os.cpu_count(), "run_id": args.run_id,
        "cpu_thread_allocation": job.cpu_allocation,
        "gpu_architecture": arch, "hip_device": 0, "openmp_device": 0,
        "visibility": {key: job.env.get(key) for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")},
        "rocm_root": str(rocm) if rocm else None, "hip_compiler": compiler, "openmp_compiler": compiler})
    common = [cmake, "-S", source, "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
              "-DGPU_SUITE_VENDOR=amd", "-DGPU_SUITE_AMD_LIBRARIES=rocfft", "-DGPU_SUITE_SOURCE_LANGUAGE=c-cpp",
              "-DGPU_SUITE_BUILD_CUDA=OFF", "-DGPU_SUITE_BUILD_OPENACC=OFF", "-DGPU_SUITE_BUILD_TESTS=OFF",
              "-DGPU_SUITE_BUILD_EXAMPLES=ON", "-DGPU_SUITE_BUILD_BENCHMARKS=ON",
              "-DPython3_EXECUTABLE=" + python, "-DCMAKE_C_COMPILER=" + cpu_cc,
              "-DCMAKE_CXX_COMPILER=" + cpu_cxx]
    successful, runtime = {}, {"run_id": args.run_id, "gpu_architecture": arch, "build_metadata": {},
        "cpu_thread_allocation": job.cpu_allocation,
        "resolution": {}, "resolved_library_sha256": {}, "linker_input_bindings": {},
        "environment": {key: job.env.get(key) for key in
        ("ROCM_PATH", "HIP_PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
         "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "OMP_DEFAULT_DEVICE", "OMP_TARGET_OFFLOAD", "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_THREAD_LIMIT")}}
    for profile, stem in (("cpu", "fft_cpu"), ("hip", "fft_gpu"), ("openmp", "openmp_rocfft")):
        if profile != "cpu" and not all((rocm, compiler, arch)):
            job.fail(profile.upper() + "_BLOCKED_TOOLCHAIN_OR_ALLOCATION")
            continue
        build = root / ("build-" + profile)
        flags = ["-B", build, "-DGPU_SUITE_BUILD_CPU=" + ("ON" if profile == "cpu" else "OFF"),
                 "-DGPU_SUITE_BUILD_HIP=" + ("ON" if profile == "hip" else "OFF"),
                 "-DGPU_SUITE_BUILD_OPENMP=" + ("ON" if profile == "openmp" else "OFF")]
        if profile == "cpu":
            flags += fftw_cmake_flags(job.env)
        else:
            flags.append("-DGPU_SUITE_ROCM_ROOT=" + str(rocm))
            if profile == "hip":
                flags += ["-DCMAKE_HIP_COMPILER=" + compiler, "-DCMAKE_HIP_ARCHITECTURES=" + arch]
            else:
                flags += ["-DCMAKE_CXX_COMPILER=" + compiler, "-DGPU_SUITE_AMD_GPU_TARGET=" + arch]
        code, output = job.run("configure-" + profile, common + flags)
        if code:
            if profile == "cpu" and args.cpu_threads > 1:
                raise ValueError("Threaded CPU configure failed; no serial fallback")
            continue
        target_input = build / "partial-manifest-input.json"
        target_names = {entry["target_name"] for entry in load(target_input)["entries"]} if target_input.is_file() else set()
        if not {stem, stem + "_bench"}.issubset(target_names):
            status = "CPU_BLOCKED_DEPENDENCY" if profile == "cpu" and "rocFFT CPU targets: DISABLED" in output else profile.upper() + "_TARGETS_UNAVAILABLE"
            job.fail(status, detail="See configure log and real compile/link checks")
            if profile == "cpu" and args.cpu_threads > 1:
                raise ValueError("Threaded CPU targets unavailable; no serial fallback")
            continue
        code, _ = job.run("build-" + profile, [cmake, "--build", build, "--parallel", "1", "--verbose",
            "--target", stem, stem + "_bench", "gpu_suite_partial_manifest"], timeout=300)
        if code:
            if profile == "cpu" and args.cpu_threads > 1:
                raise ValueError("Threaded CPU build failed; no serial fallback")
            continue
        if profile == "cpu":
            require_threaded_capability(load(build / "partial-manifest.json"),
                                        load(build / "build-metadata.json"), args.cpu_threads)
        binary = build / "amd/c-cpp/rocfft" / (stem + "_bench")
        ldd = job.executable("ldd")
        if not ldd:
            job.fail("resolved-" + profile, detail="ldd unavailable")
            continue
        code, output = job.run("resolved-" + profile, [ldd, binary], timeout=30)
        if code:
            continue
        try:
            metadata = load(build / "build-metadata.json")
            normalized, libraries = evidence_helper.resolved_libraries(output)
            bindings = evidence_helper.check_configured_dependencies(metadata, libraries,
                build / "amd/c-cpp/rocfft/CMakeFiles" / (stem + "_bench.dir") / "link.d")
        except (OSError, ValueError, KeyError, TypeError) as error:
            job.fail("dependencies-" + profile, detail=str(error))
            continue
        runtime["build_metadata"][profile] = metadata
        runtime["resolution"][profile] = normalized
        runtime["resolved_library_sha256"][profile] = libraries
        runtime["linker_input_bindings"][profile] = bindings
        previous_threads = job.env["OMP_NUM_THREADS"]
        if profile == "cpu":
            job.env["OMP_NUM_THREADS"] = "1"  # The teaching example remains serial.
        try:
            code, _ = job.run("example-" + profile, [build / "amd/c-cpp/rocfft" / stem], timeout=120)
        finally:
            job.env["OMP_NUM_THREADS"] = previous_threads
        if code == 0:
            successful[profile] = build
    write_json(root / "profile-status.json", {"eligible_benchmark_profiles": sorted(successful), "failures": job.failures})
    require_all_profiles(successful, job.failures)
    current_paths = source_paths()
    if set(current_paths) != set(paths) or source_snapshot_sha256(source, current_paths) != source_hash:
        raise ValueError("Source changed during the build; no benchmark will run")
    manifest = root / "executables.json"
    code, _ = job.run("merge-manifests", [python, source / "tools/merge_manifests.py", "--output", manifest] +
                      [build / "partial-manifest.json" for build in successful.values()])
    if code:
        return
    execute_campaigns(job, successful, runtime, source_hash, manifest, python)
    evidence_roots = [root] + [root / name for name in ("serial", "threaded") if (root / name).is_dir()]
    write_json(root / "artifact-sha256.json", {str(path.relative_to(root)): sha256_file(path)
               for directory in evidence_roots for path in directory.iterdir() if path.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "run-root", "run-id", "scheduler", "job-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--allocated-cpus", required=True, type=int)
    parser.add_argument("--allocated-gpus", required=True, type=int)
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="Explicit CPU smoke thread count; 1 keeps the serial default")
    parser.add_argument("--include-serial-campaign", action="store_true",
                        help="Also run the canonical serial CPU/HIP/OpenMP matrix in a separate campaign")
    args = parser.parse_args()
    job = None
    try:
        if os.environ.get("GPU_SUITE_ALLOCATED_COMPUTE_NODE") != "yes" or os.environ.get("GPU_SUITE_SITE_BINDING_VERIFIED") != "yes":
            raise ValueError("Site-confirmed allocated compute-node gate is required")
        if args.allocated_cpus < 1 or args.allocated_gpus != 1:
            raise ValueError("Verified allocation must contain at least one CPU and exactly one GPU")
        cpu_thread_allocation(args.cpu_threads, args.allocated_cpus)
        campaign_threads(args.cpu_threads, args.include_serial_campaign)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", args.run_id) or ".." in args.run_id:
            raise ValueError("Invalid run ID")
        source_path, run_path = Path(args.source_root), Path(args.run_root)
        if not source_path.is_absolute() or not run_path.is_absolute() or source_path.is_symlink() or run_path.is_symlink():
            raise ValueError("Source/run roots must be absolute real directories")
        args.source_root, args.run_root = validate_run_location(source_path, run_path, args.run_id)
        if not args.source_root.is_dir() or not args.run_root.is_dir():
            raise ValueError("Source/run roots must exist")
        os.umask(0o077)
        root = args.run_root / "compute"
        root.mkdir()  # Refuse previous work; no overwrite, deletion, or retry.
        job = Job(args, root)
        execute(job)
    except Exception as error:
        if job:
            job.fail("PREPARATION_OR_INTERNAL", detail=str(error))
        else:
            print("GPU_SUITE_SIRIUS_ROCFFT_SMOKE FAIL %s ALLOCATION_GATE 2" % args.run_id, flush=True)
            print(str(error), file=sys.stderr)
            return 2
    write_json(job.root / "job-summary.json", {"run_id": args.run_id, "ended": utc(),
               "failures": job.failures, "commands": job.commands})
    if job.failures:
        first = job.failures[0]
        print("GPU_SUITE_SIRIUS_ROCFFT_SMOKE FAIL %s %s %s" % (args.run_id, first["stage"], first["exit_status"]), flush=True)
        return 1
    print("GPU_SUITE_SIRIUS_ROCFFT_SMOKE PASS " + args.run_id, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

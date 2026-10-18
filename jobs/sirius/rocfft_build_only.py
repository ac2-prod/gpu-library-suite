#!/usr/bin/env python3
"""Build one approved Sirius rocFFT profile; never execute its artifacts."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

sys.dont_write_bytecode = True
STEMS = {"cpu": "fft_cpu", "hip": "fft_gpu", "openmp": "openmp_rocfft"}
ENVIRONMENT = (
    "PATH", "LOADEDMODULES", "MODULEPATH", "ROCM_PATH", "HIP_PATH",
    "FFTW_ROOT", "FFTW3f_ROOT", "FFTW3F_ROOT", "FFTW3_ROOT",
    "AOCL_ROOT", "AOCL_HOME", "AOCL_INSTALL_DIR", "CMAKE_PREFIX_PATH",
    "CMAKE_INCLUDE_PATH", "CMAKE_LIBRARY_PATH", "CPATH", "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH", "PKG_CONFIG_PATH",
    "PKG_CONFIG_LIBDIR", "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS",
    "FFTW3f_THREADING",
)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "resolved_path": str(path.resolve(strict=True)),
            "sha256": digest.hexdigest()}


def validate(args):
    source, run = args.source_root, args.run_root
    for path in (source, source / "runs", run):
        if not path.is_absolute() or not path.is_dir() or path.is_symlink() or path.resolve() != path:
            raise ValueError("Source/runs/run roots must be existing canonical directories")
    if run.parent != source / "runs" or run.name != args.run_id:
        raise ValueError("Run root must be SOURCE/runs/RUN_ID")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", args.run_id) or ".." in args.run_id:
        raise ValueError("Invalid run ID")
    for compiler in (args.c_compiler, args.cxx_compiler):
        if not compiler.is_absolute() or not compiler.is_file() or not os.access(compiler, os.X_OK):
            raise ValueError("Explicit executable compiler paths are required")
    if args.profile != "cpu":
        if not args.gpu_arch or not re.fullmatch(r"gfx[0-9a-z]+(?::[a-z0-9]+[+-])*", args.gpu_arch):
            raise ValueError("GPU builds require an explicitly approved architecture")
        if args.cxx_compiler.name == "hipcc":
            raise ValueError("Native HIP requires a Clang compiler, not hipcc")
        if not args.rocm_root or not args.rocm_root.is_absolute() or not args.rocm_root.is_dir():
            raise ValueError("An observed absolute ROCm prefix is required")
    elif args.gpu_arch or args.rocm_root:
        raise ValueError("CPU build must not receive GPU toolchain settings")


def cpu_hints(env):
    """Translate only loaded-environment paths; do not search the filesystem."""
    groups = {
        "CMAKE_INCLUDE_PATH": ("CMAKE_INCLUDE_PATH", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"),
        "CMAKE_LIBRARY_PATH": ("CMAKE_LIBRARY_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH"),
        "CMAKE_PREFIX_PATH": ("CMAKE_PREFIX_PATH", "AOCL_ROOT", "AOCL_HOME", "AOCL_INSTALL_DIR"),
    }
    result = {}
    for target, names in groups.items():
        paths = []
        for name in names:
            for path in re.split(r"[:;]", env.get(name, "")):
                if path and path not in paths:
                    if not Path(path).is_absolute():
                        raise ValueError("Loaded environment has a relative path in " + name)
                    paths.append(path)
        if paths:
            result[target] = ";".join(paths)
    result.update({name: env[name] for name in ENVIRONMENT[5:9] if env.get(name)})
    if "FFTW3f_THREADING" in env:
        if env["FFTW3f_THREADING"] not in {"AUTO", "PTHREAD", "OPENMP", "COMBINED", "NONE"}:
            raise ValueError("Invalid FFTW3f_THREADING selection")
        result["FFTW3f_THREADING"] = env["FFTW3f_THREADING"]
    return result


def configure_argv(args, cmake, build, env):
    flags = {"CMAKE_BUILD_TYPE": "Release", "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
             "GPU_SUITE_VENDOR": "amd", "GPU_SUITE_AMD_LIBRARIES": "rocfft", "GPU_SUITE_SOURCE_LANGUAGE": "c-cpp",
             "GPU_SUITE_BUILD_CUDA": "OFF", "GPU_SUITE_BUILD_OPENACC": "OFF",
             "GPU_SUITE_BUILD_TESTS": "OFF", "GPU_SUITE_BUILD_EXAMPLES": "ON",
             "GPU_SUITE_BUILD_BENCHMARKS": "ON", "Python3_EXECUTABLE": sys.executable,
             "CMAKE_C_COMPILER": str(args.c_compiler), "CMAKE_CXX_COMPILER": str(args.cxx_compiler)}
    for profile in STEMS:
        flags["GPU_SUITE_BUILD_" + profile.upper()] = "ON" if profile == args.profile else "OFF"
    if args.profile == "cpu":
        flags.update(cpu_hints(env))
    else:
        flags["GPU_SUITE_ROCM_ROOT"] = str(args.rocm_root)
        if args.profile == "hip":
            flags.update(CMAKE_HIP_COMPILER=str(args.cxx_compiler), CMAKE_HIP_ARCHITECTURES=args.gpu_arch)
        else:
            flags["GPU_SUITE_AMD_GPU_TARGET"] = args.gpu_arch
    return [cmake, "-S", str(args.source_root), "-B", str(build)] + ["-D" + key + "=" + value for key, value in flags.items()]


class Build:
    def __init__(self, args):
        self.args = args
        self.build = args.run_root / ("build-" + args.profile)
        self.root = args.run_root / ("build-only-" + args.profile)
        if self.build.exists() or self.build.is_symlink():
            raise ValueError("Refusing an existing build directory")
        self.root.mkdir()  # Exclusive attempt; no retry or overwrite.
        for name in ("logs", "tmp", "cache"):
            (self.root / name).mkdir()
        self.commands = []
        self.env = os.environ.copy()
        self.env.update(PYTHONDONTWRITEBYTECODE="1", GIT_OPTIONAL_LOCKS="0",
                        TMPDIR=str(self.root / "tmp"), XDG_CACHE_HOME=str(self.root / "cache"))
        # These options also apply to Git called indirectly by CMake metadata tools.
        count = int(self.env.get("GIT_CONFIG_COUNT", "0"))
        for key in ("core.fsmonitor", "core.untrackedCache"):
            self.env["GIT_CONFIG_KEY_" + str(count)] = key
            self.env["GIT_CONFIG_VALUE_" + str(count)] = "false"
            count += 1
        self.env["GIT_CONFIG_COUNT"] = str(count)
        self.env.pop("GIT_EXTERNAL_DIFF", None)

    def run(self, stage, argv, timeout=180):
        prefix = self.root / "logs" / ("%02d-%s" % (len(self.commands) + 1, stage))
        item = {"stage": stage, "argv": [str(v) for v in argv], "started": utc(),
                "stdout": str(prefix) + ".stdout", "stderr": str(prefix) + ".stderr", "exit_status": None}
        self.commands.append(item)
        try:
            with open(item["stdout"], "xb") as out, open(item["stderr"], "xb") as err:
                process = subprocess.Popen(item["argv"], cwd=self.root, env=self.env,
                                           stdout=out, stderr=err, start_new_session=True)
                try:
                    item["exit_status"] = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    item["failure"] = "timeout; command was not retried"
        except OSError as error:
            item["failure"] = str(error)
        finally:
            item["ended"] = utc()
            write_json(Path(str(prefix) + ".command.json"), item)
        if item["exit_status"] != 0 or "failure" in item:
            raise ValueError("Command failed: " + stage)
        return Path(item["stdout"]).read_text(encoding="utf-8", errors="strict")


def source_evidence(job):
    from gpu_suite.hashing import source_snapshot_sha256
    git = shutil.which("git", path=job.env.get("PATH"))
    if not git:
        raise ValueError("git is unavailable")
    prefix = [git, "-c", "core.fsmonitor=false", "-c", "core.untrackedCache=false", "-C", job.args.source_root]
    root = job.run("git-root", prefix + ["rev-parse", "--show-toplevel"])
    if Path(root.strip()).resolve() != job.args.source_root:
        raise ValueError("Source is not the expected Git root")
    head = job.run("git-head", prefix + ["rev-parse", "HEAD"]).strip()
    status = job.run("git-status", prefix + ["status", "--short", "--branch", "--untracked-files=all"])
    job.run("git-diff-check", prefix + ["diff", "--no-ext-diff", "--check"])
    inventory = job.run("git-inventory", prefix + ["ls-files", "-z", "--cached", "--others", "--exclude-standard"])
    paths = sorted({p for p in inventory.split("\0") if p and Path(p).parts[0] != "runs"
                    and not any(v.startswith(".fuse_hidden") for v in Path(p).parts)
                    and (job.args.source_root / p).exists()})
    return {"git_head": head, "git_status": status, "paths": paths,
            "source_snapshot_sha256": source_snapshot_sha256(job.args.source_root, paths),
            "exclusions": ["runs/", ".fuse_hidden* in any path component"]}


def verify_artifacts(job):
    from gpu_suite.strict_json import load
    build = job.build
    manifest = load(build / "partial-manifest.json")
    metadata = load(build / "build-metadata.json")
    if metadata.get("vendor") != "amd" or metadata.get("build_profile") != job.args.profile:
        raise ValueError("Build metadata identifies a different profile")
    for language, compiler in (("c", job.args.c_compiler), ("cxx", job.args.cxx_compiler)):
        if Path(metadata[language]["compiler_path"]).resolve() != compiler.resolve():
            raise ValueError("Build metadata identifies a different compiler")
    expected = {STEMS[job.args.profile], STEMS[job.args.profile] + "_bench"}
    entries = manifest["entries"]
    if len(entries) != 2 or {e["target_name"] for e in entries} != expected:
        raise ValueError("Manifest does not contain exactly the requested rocFFT artifacts")
    metadata_hash = file_identity(build / "build-metadata.json")["sha256"]
    if manifest["build_metadata_sha256"] != metadata_hash:
        raise ValueError("Manifest metadata hash mismatch")
    binaries = {}
    for entry in entries:
        if entry.get("implementation") != job.args.profile or entry.get("library") != "rocfft":
            raise ValueError("Manifest identifies a different implementation")
        path = build / "amd/c-cpp/rocfft" / entry["target_name"]
        if entry["executable_path"] != str(path) or path.is_symlink() or not path.is_file():
            raise ValueError("Manifest executable path is unexpected")
        identity = file_identity(path)
        if entry["binary_sha256"] != identity["sha256"] or entry["build_metadata_sha256"] != metadata_hash:
            raise ValueError("Built artifact hash mismatch")
        binaries[entry["target_name"]] = identity
    cache = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if "=" in line and not line.startswith(("#", "//")):
            key, value = line.split("=", 1)
            if key.split(":", 1)[0] in {"FFTW3f_INCLUDE_DIR", "FFTW3f_LIBRARY", "FFTW3f_THREADS_LIBRARY",
                                        "FFTW3f_OPENMP_LIBRARY", "FFTW3f_THREADING",
                                        "FFTW3f_PTHREAD_LINKS", "FFTW3f_OPENMP_LINKS", "FFTW3f_COMBINED_LINKS"}:
                cache[key.split(":", 1)[0]] = value
    selected = {}
    selected_dependencies = {}
    if job.args.profile == "cpu":
        for key in ("FFTW3f_INCLUDE_DIR", "FFTW3f_LIBRARY"):
            path = Path(cache.get(key, ""))
            if key.endswith("INCLUDE_DIR"):
                path = path / "fftw3.h"
            if not path.is_absolute() or not path.is_file():
                raise ValueError("Selected FFTW header/library is unavailable: " + key)
            selected[key] = file_identity(path)
        for name in metadata.get("fftw_threading", {}).get("dependency_names", []):
            dependency = metadata.get("dependencies", {}).get(name)
            if not isinstance(dependency, dict):
                raise ValueError("Selected FFTW threading dependency is missing: " + name)
            path = Path(dependency.get("path", ""))
            if not path.is_absolute() or not path.is_file():
                raise ValueError("Selected FFTW threading dependency is unavailable: " + name)
            identity = file_identity(path)
            if identity["sha256"] != dependency.get("sha256"):
                raise ValueError("Selected FFTW threading dependency hash mismatch: " + name)
            selected_dependencies[name] = identity
    return {"binaries": binaries, "build_metadata": metadata, "fftw_cache": cache,
            "selected_fftw": selected, "selected_fftw_dependencies": selected_dependencies,
            "files": {name: file_identity(build / name) for name in
            ("partial-manifest.json", "build-metadata.json", "CMakeCache.txt", "compile_commands.json")}}


def execute(job):
    from gpu_suite.strict_json import load
    before = source_evidence(job)
    write_json(job.root / "source-before.json", before)
    try:
        cmake = shutil.which("cmake", path=job.env.get("PATH"))
        if not cmake:
            raise ValueError("cmake is unavailable")
        tools = {name: file_identity(path) for name, path in
                 (("cmake", cmake), ("c", job.args.c_compiler), ("cxx", job.args.cxx_compiler), ("python", sys.executable))}
        write_json(job.root / "toolchain.json", {"tools": tools,
            "environment": {name: job.env.get(name) for name in ENVIRONMENT},
            "effective_cpu_hints": cpu_hints(job.env) if job.args.profile == "cpu" else {}})
        for name, item in tools.items():
            job.run(name + "-version", [item["path"], "--version"], timeout=30)
        job.run("configure", configure_argv(job.args, cmake, job.build, job.env))
        names = {entry["target_name"] for entry in load(job.build / "partial-manifest-input.json")["entries"]}
        stem = STEMS[job.args.profile]
        if not {stem, stem + "_bench"}.issubset(names):
            raise ValueError("Configure disabled required rocFFT targets; inspect dependency diagnostics")
        job.run("build", [cmake, "--build", job.build, "--parallel", "1", "--verbose", "--target",
                          stem, stem + "_bench", "gpu_suite_partial_manifest"], timeout=600)
        return verify_artifacts(job)
    finally:
        after = source_evidence(job)
        write_json(job.root / "source-after.json", after)
        # Git status also lists FUSE tombstones, which may disappear when an
        # unrelated editor closes a file. Compare only the source identity;
        # retain the full before/after status as evidence.
        identity_keys = ("git_head", "paths", "source_snapshot_sha256")
        if any(after.get(key) != before.get(key) for key in identity_keys):
            raise ValueError("Source changed during build; artifacts must not be used")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "run-root", "c-compiler", "cxx-compiler"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=tuple(STEMS), required=True)
    parser.add_argument("--gpu-arch")
    parser.add_argument("--rocm-root", type=Path)
    args = parser.parse_args()
    job = None
    summary = {"run_id": args.run_id, "profile": args.profile, "status": "FAILED", "started": utc(),
               "artifact_execution_count": 0, "scheduler_call_count": 0}
    try:
        if os.environ.get("GPU_SUITE_SIRIUS_BUILD_ONLY_APPROVED") != "yes":
            raise ValueError("Explicit Sirius build-only approval gate is required")
        validate(args)
        sys.path.insert(0, str(args.source_root / "tools"))
        os.umask(0o077)
        job = Build(args)
        summary["artifacts"] = execute(job)
        summary["status"] = "BUILT"
    except (OSError, ValueError, KeyError, TypeError) as error:
        summary["error"] = str(error)
    summary.update(ended=utc(), commands=job.commands if job else [])
    if job:
        write_json(job.root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    return 0 if summary["status"] == "BUILT" else 1


if __name__ == "__main__":
    sys.exit(main())

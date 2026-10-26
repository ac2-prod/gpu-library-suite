#!/usr/bin/env python3
"""Build the pinned hipFORT dependency offline inside an approved PBS job."""
import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import amd_build

MODULES = ("hipfort", "hipfort_types", "hipfort_rocfft", "hipfort_rocblas",
           "hipfort_rocsolver", "hipfort_rocrand")


def extract_source(archive, destination, name, record):
    """Only bounded regular files/directories from the hash-locked archive."""
    if archive.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Source archive exceeds size bound")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Source archive hash mismatch: " + name)
    prefix = name + "-" + record["commit"]
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        if len(members) > 10000 or sum(m.size for m in members) > 64 * 1024 * 1024:
            raise ValueError("Expanded source exceeds bound")
        seen = set()
        for member in members:
            parts = PurePosixPath(member.name).parts
            if (not parts or parts[0] != prefix or ".." in parts or member.name.startswith("/")
                    or not (member.isfile() or member.isdir()) or member.name in seen):
                raise ValueError("Unsafe or duplicate archive member: " + member.name)
            seen.add(member.name)
        destination.mkdir()  # An existing build/source tree is never reused.
        for member in members:
            path = destination.joinpath(*PurePosixPath(member.name).parts[1:])
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as src, path.open("xb") as dst:
                    dst.write(src.read())
                path.chmod(0o700 if member.mode & 0o111 else 0o600)


def configure_argv(cmake, root, rocm, gcc):
    gcc_flag = amd_build.gcc_toolchain_flag(gcc)
    flags = {
        "CMAKE_BUILD_TYPE": "Release", "CMAKE_Fortran_COMPILER": rocm / "bin/amdflang",
        "CMAKE_C_COMPILER": rocm / "bin/amdclang", "ROCM_PATH": rocm,
        "CMAKE_PREFIX_PATH": rocm, "CMAKE_INSTALL_PREFIX": root / "install",
        "CMAKE_INSTALL_LIBDIR": "lib", "CMAKE_INSTALL_INCLUDEDIR": "include",
        "ROCmCMakeBuildTools_DIR": root / "rocm-cmake/share/rocmcmakebuildtools/cmake",
        # Makes the first upstream QUIET lookup fatal if unavailable, before
        # its otherwise automatic download of the moving master branch.
        "CMAKE_REQUIRE_FIND_PACKAGE_ROCmCMakeBuildTools": "ON",
        "HIPFORT_MULTITOOLCHAIN_LAYOUT": "ON", "HIPFORT_BUILD_AMDGCN": "ON",
        "HIPFORT_BUILD_NVPTX": "OFF", "HIPFORT_USE_FPOINTER_INTERFACES": "ON",
        "HIPFORT_ASSUMED_RANK": "OFF", "BUILD_TESTING": "OFF", "HIPFORT_EXTENDED_TESTS": "OFF",
        "BUILD_NATIVE": "OFF", "HIPFORT_CODE_COVERAGE": "OFF",
        # Avoid hipFORT's compiler-dependent Release/LTO flag probing. These
        # bindings do not introduce a different numerical mode for Fortran.
        "CMAKE_C_FLAGS": gcc_flag,
        "CMAKE_Fortran_FLAGS": gcc_flag + " -fno-fast-math -ffp-contract=off",
        "CMAKE_Fortran_FLAGS_RELEASE": "-O3", "CMAKE_Fortran_FLAGS_TESTING": "-O0",
        "CMAKE_Fortran_FLAGS_DEBUG": "-O0 -g",
    }
    return [str(cmake), "-S", str(root / "hipfort"), "-B", str(root / "build")] + [
        "-D" + key + "=" + str(value) for key, value in flags.items()]


def build(args, *, environment=None, run=subprocess.run):
    env = dict(os.environ if environment is None else environment)
    if (env.get("GPU_SUITE_SIRIUS_AMD_BUILD_APPROVED") != "yes"
            or env.get("PBS_ENVIRONMENT") != "PBS_BATCH" or not env.get("PBS_JOBID")):
        raise ValueError("Explicit approval and PBS batch allocation required")
    if not 30 <= args.timeout_seconds <= 1800:
        raise ValueError("Invalid dependency build budget")
    if not all(p.is_absolute() for p in (args.archives, args.output, args.rocm_root, args.cmake, args.lock)):
        raise ValueError("Absolute dependency paths required")
    lock = amd_build.load_json(args.lock)
    if lock.get("schema_version") != 1 or lock.get("rocm_version") != "10.1.0" or set(lock.get("sources", {})) != {"hipfort", "rocm-cmake"}:
        raise ValueError("Unexpected source lock")
    if (args.rocm_root / ".info/version").read_text().strip() != lock["rocm_version"]:
        raise ValueError("Selected ROCm version differs from source lock")
    compilers = [args.rocm_root / "bin" / name for name in ("amdclang", "amdflang")]
    gcc = Path(env.get("CPU_CC", ""))
    amd_build.gcc_toolchain_flag(gcc)  # Module helper supplies the validated GCC driver.
    if not all(p.is_file() and os.access(p, os.X_OK) for p in [args.cmake] + compilers):
        raise ValueError("Selected compiler or CMake is unavailable")
    if args.output.is_symlink():
        raise ValueError("Dependency output must not be a symlink")
    args.output.mkdir()  # Never overwrite/retry a partial dependency build.
    (args.output / "logs").mkdir()
    record = dict(status="FAILED", started=amd_build.utc(), sources=lock,
                  lock=amd_build.file_identity(args.lock), job_id=env["PBS_JOBID"], commands=[],
                  tools={p.name: amd_build.file_identity(p) for p in [args.cmake] + compilers + [gcc]},
                  environment={k: env.get(k) for k in ("PATH", "LD_LIBRARY_PATH", "LOADEDMODULES", "ROCM_PATH", "HIP_PATH", "CPU_CC")})
    deadline = time.monotonic() + args.timeout_seconds
    try:
        for name, item in lock["sources"].items():
            if Path(item["archive"]).name != item["archive"]:
                raise ValueError("Source archive must be a basename")
            extract_source(args.archives / item["archive"], args.output / name, name, item)
        config = args.output / "rocm-cmake/share/rocmcmakebuildtools/cmake/ROCmCMakeBuildToolsConfig.cmake"
        if not config.is_file():
            raise ValueError("Pinned offline ROCmCMakeBuildTools config missing")
        commands = [[str(p), "--version"] for p in [args.cmake] + compilers + [gcc]] + [
            configure_argv(args.cmake, args.output, args.rocm_root, gcc),
            [str(args.cmake), "--build", str(args.output / "build"), "--parallel", "4"],
            [str(args.cmake), "--install", str(args.output / "build")]]
        for index, argv in enumerate(commands):
            prefix = args.output / "logs" / ("%02d" % index)
            item = dict(argv=argv, started=amd_build.utc(), exit_code=None,
                        stdout=str(prefix) + ".stdout", stderr=str(prefix) + ".stderr")
            record["commands"].append(item)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Dependency build budget exhausted")
            try:
                with open(item["stdout"], "xb") as out, open(item["stderr"], "xb") as err:
                    result = run(argv, cwd=args.output, env=env, stdin=subprocess.DEVNULL,
                                 stdout=out, stderr=err, timeout=remaining, check=False)
                    item["exit_code"] = result.returncode
                if result.returncode != 0:
                    raise ValueError("Dependency command failed: " + str(index))
            finally:
                item["ended"] = amd_build.utc()
        modules = args.output / "install/include/fortran/amdflang/hipfort/amdgcn"
        archive = args.output / "install/lib/fortran/amdflang/libhipfort-amdgcn.a"
        paths = [archive] + [modules / (name + ".mod") for name in MODULES] + [
            args.output / "build/CMakeCache.txt", args.output / "build/install_manifest.txt"]
        record.update(outputs=[amd_build.file_identity(p) for p in paths],
                      module_directory=str(modules), library=str(archive), status="BUILT",
                      compatibility="UNVERIFIED_UNTIL_SUITE_BUILD_AND_SMOKE")
        return record
    except Exception as error:
        record["error"] = str(error)
        raise
    finally:
        record["ended"] = amd_build.utc()
        amd_build.write_json(args.output / "summary.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archives", "output", "rocm-root", "cmake", "lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    try:
        build(parser.parse_args())
    except (OSError, ValueError, subprocess.SubprocessError, TimeoutError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

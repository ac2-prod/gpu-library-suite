#!/usr/bin/env python3
"""Exercise oneMKL and FFTW discovery through staged external roots."""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def run_configure(arguments, environment):
    completed = subprocess.run(
        arguments, env=environment, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "configure failed:\n{0}\n{1}".format(
                " ".join(str(item) for item in arguments), completed.stdout,
            )
        )


def base_environment():
    environment = os.environ.copy()
    for name in (
        "CMAKE_PREFIX_PATH", "CPATH", "FFTW_ROOT", "LD_LIBRARY_PATH",
        "LIBRARY_PATH", "MKLROOT",
    ):
        environment.pop(name, None)
    return environment


def configure_command(arguments, build, provider, expect_found):
    return [
        arguments.cmake,
        "-S", str(arguments.source),
        "-B", str(build),
        "-G", arguments.generator,
        "-DGPU_SUITE_ROOT={0}".format(arguments.repository_root),
        "-DFIXTURE_PROVIDER={0}".format(provider),
        "-DEXPECT_FOUND={0}".format("ON" if expect_found else "OFF"),
    ]


def copy_headers(source, destination, names):
    destination.mkdir(parents=True)
    for name in names:
        shutil.copy2(str(source / name), str(destination / name))


def copy_library(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination / source.name))


def require_notfound_cache(cache_path, variables):
    cache = cache_path.read_text(encoding="utf-8")
    for variable in variables:
        pattern = r"^{0}:[^=]+={0}-NOTFOUND$".format(re.escape(variable))
        if re.search(pattern, cache, flags=re.MULTILINE) is None:
            raise RuntimeError(
                "{0} was not cached as NOTFOUND".format(variable)
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--fixture-root", required=True, type=Path)
    parser.add_argument("--headers-root", required=True, type=Path)
    parser.add_argument("--mkl-library", required=True, type=Path)
    parser.add_argument("--fftw-library", required=True, type=Path)
    parser.add_argument("--fftw-threads-library", required=True, type=Path)
    arguments = parser.parse_args()

    arguments.fixture_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="discovery-", dir=str(arguments.fixture_root)
    ) as temporary:
        root = Path(temporary)
        environment = base_environment()

        mkl_build = root / "mkl-build"
        run_configure(
            configure_command(arguments, mkl_build, "ONEMKL", False),
            environment,
        )
        require_notfound_cache(
            mkl_build / "CMakeCache.txt",
            (
                "GPU_SUITE_ONEMKL_CBLAS_INCLUDE_DIR",
                "GPU_SUITE_ONEMKL_CBLAS_LIBRARY",
                "GPU_SUITE_ONEMKL_LAPACKE_INCLUDE_DIR",
                "GPU_SUITE_ONEMKL_LAPACKE_LIBRARY",
                "GPU_SUITE_MKL_SPARSE_INCLUDE_DIR",
                "GPU_SUITE_MKL_SPARSE_LIBRARY",
            ),
        )
        mkl_root = root / "mkl-root"
        copy_headers(
            arguments.headers_root / "cpu_libs" / "include",
            mkl_root / "include",
            (
                "cblas.h", "lapacke.h", "mkl_cblas.h", "mkl_lapacke.h",
                "mkl_spblas.h",
            ),
        )
        copy_library(arguments.mkl_library, mkl_root / "lib")
        mkl_environment = dict(environment)
        mkl_environment["MKLROOT"] = str(mkl_root)
        run_configure(
            configure_command(arguments, mkl_build, "ONEMKL", True),
            mkl_environment,
        )

        fftw_root = root / "fftw-root"
        copy_headers(
            arguments.headers_root / "fftw" / "include",
            fftw_root / "include", ("fftw3.h",),
        )
        copy_library(arguments.fftw_library, fftw_root / "lib")
        copy_library(arguments.fftw_threads_library, fftw_root / "lib")

        fftw_root_command = configure_command(
            arguments, root / "fftw-root-build", "FFTW", True,
        )
        fftw_root_command.append("-DFFTW_ROOT={0}".format(fftw_root))
        run_configure(fftw_root_command, environment)

        fftw_prefix_command = configure_command(
            arguments, root / "fftw-prefix-build", "FFTW", True,
        )
        fftw_prefix_command.append(
            "-DCMAKE_PREFIX_PATH={0}".format(fftw_root)
        )
        run_configure(fftw_prefix_command, environment)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""GNU Fortran calls against controlled providers; never MKL/FFTW/GPU evidence."""
import argparse
import os
import subprocess
from pathlib import Path

from gpu_suite.results_io import parse_stdout_prefix
from gpu_suite.runner import load_manifest
from gpu_suite.schema import validate_raw_result


def run(command, environment, expected=0):
    result = subprocess.run([str(value) for value in command], env=environment,
                            capture_output=True, text=True, check=False)
    if result.returncode != expected:
        raise AssertionError(f"{command}\nrc={result.returncode}\n{result.stdout}\n{result.stderr}")
    return result


def main():
    parser = argparse.ArgumentParser()
    for argument in ("source", "output", "compiler", "cmake"):
        parser.add_argument("--" + argument, required=True)
    args = parser.parse_args()
    root, output = Path(args.source), Path(args.output)
    provider, build = output / "provider", output / "build"
    environment = os.environ.copy()
    for key in ("MKLROOT", "FFTW_ROOT", "CMAKE_PREFIX_PATH", "NVCOMPILER_FPU_STATE"):
        environment.pop(key, None)
    run([args.cmake, "-S", root / "tests/fortran/fixtures", "-B", provider], environment)
    run([args.cmake, "--build", provider, "--parallel", "2"], environment)
    environment["MKLROOT"] = str(provider)
    configure = [args.cmake, "-S", root, "-B", build,
                 "-DGPU_SUITE_SOURCE_LANGUAGE=fortran", "-DCMAKE_Fortran_COMPILER=" + args.compiler,
                 "-DCMAKE_BUILD_TYPE=Release", "-DGPU_SUITE_BUILD_CUDA=OFF",
                 "-DGPU_SUITE_BUILD_OPENACC=OFF", "-DGPU_SUITE_BUILD_TESTS=OFF",
                 "-DGPU_SUITE_BUILD_EXAMPLES=OFF", "-DFFTW_ROOT=" + str(provider)]
    run(configure, environment)
    run([args.cmake, "--build", build, "--target", "gpu_suite_partial_manifest", "--parallel", "2"], environment)
    manifest, _ = load_manifest(build / "partial-manifest.json")
    assert len(manifest["entries"]) == 6, manifest
    cases = [
        ("cufft", "fft", ["--size", "16", "--batch", "3"], "2"),
        ("cublas", "blas", ["--m", "3", "--n", "5", "--k", "7", "--alpha", "0.5", "--beta", "-0.5"], "3"),
        ("cusparse", "sparse", ["--nx", "1", "--ny", "4", "--alpha", "0.5", "--beta", "-0.5"], "3"),
        ("cusolver", "solver", ["--size", "8", "--nrhs", "3"], "1"),
    ]
    for library, stem, parameters, repeat in cases:
        for scope in ("compute", "end-to-end"):
            command = [build / "nvidia/fortran" / library / (stem + "_cpu_bench"),
                       *parameters, "--warmup", "2", "--repeat", repeat, "--trials", "3",
                       "--scope", scope, "--verify", "true", "--output", "-", "--format", "jsonl"]
            for injected in (None, "nan", "inf", "-inf"):
                current = dict(environment)
                if injected:
                    current["GPU_SUITE_TEST_NONFINITE"] = injected
                result = run(command, current, expected=1 if injected else 0)
                rows, error = parse_stdout_prefix(result.stdout, "jsonl")
                assert error is None, error
                assert len(rows) == 3
                for row in rows:
                    validate_raw_result(row)
                    assert "SYNTHETIC" in row["library_version"]
                    assert row["parameters"]["source_language"] == "fortran"
                    assert row["verification_status"] == ("nonfinite" if injected else "pass"), row
                assert rows[0]["verification_metrics"] == rows[1]["verification_metrics"]
                assert rows[1]["verification_metrics"] == rows[2]["verification_metrics"]
            if library == "cusolver":
                for key, expected_infos in (("GPU_SUITE_TEST_GETRF_INFO", (2, -1)),
                                             ("GPU_SUITE_TEST_GETRS_INFO", (0, 2))):
                    current = dict(environment, **{key: "2"})
                    result = run(command, current, expected=1)
                    rows, error = parse_stdout_prefix(result.stdout, "jsonl")
                    assert error is None
                    assert all((row["getrf_info"],row["getrs_info"]) == expected_infos for row in rows)
            if library == "cufft":
                threaded = command + ["--cpu-backend", "cpu-fftw-threaded", "--cpu-threads", "2",
                                      "--cpu-threads-effective", "2", "--cpu-parallelism", "threaded"]
                rows, error = parse_stdout_prefix(run(threaded, environment).stdout, "jsonl")
                assert error is None
                assert all(row["cpu_backend"] == "cpu-fftw-threaded" for row in rows)

    # A wrong external provider must disable only its targets; no fallback.
    missing = list(configure)
    missing[missing.index("-B") + 1] = output / "missing-mkl-symbols"
    missing.append("-DGPU_SUITE_FORTRAN_MKL_LIBRARY=" + str(provider / "lib/libfftw3f.a"))
    report = run(missing, environment).stdout
    assert "Fortran cublas cpu: DISABLED" in report, report
    assert "Fortran curand cpu: ENABLED" in report, report
    # Explicitly unsafe new Fortran profiles and cross-language build-tree reuse fail.
    unsafe = list(configure)
    unsafe[unsafe.index("-B") + 1] = output / "unsafe-flags"
    unsafe.append("-DCMAKE_Fortran_FLAGS=-ffast-math")
    result = subprocess.run([str(v) for v in unsafe], env=environment, text=True, capture_output=True)
    assert result.returncode != 0 and "Unapproved numerical" in result.stderr
    result = subprocess.run([args.cmake, "-S", str(root), "-B", str(build),
                             "-DGPU_SUITE_SOURCE_LANGUAGE=c-cpp"],
                            env=environment, text=True, capture_output=True)
    assert result.returncode != 0 and "build tree" in result.stderr, result.stderr
    print("PASS: controlled CPU providers, both scopes, restoration, nonfinite/info failures, dependency/flags/language gates")


if __name__ == "__main__":
    main()

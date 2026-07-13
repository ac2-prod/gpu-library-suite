# GPU Library Suite

Portable examples and reproducible benchmarks for GPU-accelerated
libraries on NVIDIA and AMD GPUs, written in C/C++ and Fortran.

## Status

This repository is under active development.

It is currently maintained by the Advanced Computing ACceleration
(AC2) group and is not an official HAIRDESC GitHub repository.

## Scope

This repository is intended to contain:

- examples for NVIDIA GPUs in C/C++;
- examples for NVIDIA GPUs in Fortran;
- examples for AMD GPUs in C/C++;
- examples for AMD GPUs in Fortran;
- reproducible CPU/GPU benchmarks;
- common measurement, validation, and plotting tools; and
- system-specific job scripts for supported supercomputers.

## Repository organization

The top-level structure follows the order:

1. GPU vendor;
2. programming language;
3. GPU library; and
4. source-code purpose.

```text
nvidia/
  c-cpp/
    <library>/
      examples/
      benchmarks/
  fortran/
    <library>/
      examples/
      benchmarks/

amd/
  c-cpp/
    <library>/
      examples/
      benchmarks/
  fortran/
    <library>/
      examples/
      benchmarks/

common/
configs/
tools/
jobs/
```

## Examples and benchmarks

`examples/` contains minimal programs corresponding to the training
materials. Source filenames in this directory should match the
filenames shown in the materials.

`benchmarks/` contains measurement-oriented versions with command-line
options, warm-up runs, repeated trials, result verification, timing,
and machine-readable output.

Portable source code must not contain parameters specific to a
particular supercomputer. Scheduler directives, module settings,
resource requests, launch commands, and system-specific monitoring
belong only under `jobs/`.

## Initial implementation target

The first implementation target is:

```text
nvidia/c-cpp/
```

The initial library set is:
- cuFFT
- cuBLAS
- cuSPARSE
- cuSOLVER
- cuRAND
- Thrust

## License

A project license has not yet been selected. The repository will remain
private until the licensing and publication policy has been confirmed.

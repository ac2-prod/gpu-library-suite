# Build and run a teaching example

[English](BUILDING.md) | [日本語](BUILDING.ja.md)

Build one teaching example without building benchmarks, running a full test suite, or creating a measurement campaign. These recipes target Linux with the required compilers and libraries already installed. They do not install dependencies. On a shared cluster, run inside a compute allocation and follow the site rules, such as [Pegasus](PEGASUS_EXECUTION.md) or [Sirius](../jobs/sirius/README.md).

<a id="select"></a>

## 1. Select the example and environment

Choose a library from the [source map](../README.md#start-here) and run its “Select a minimal build” variable block in Bash. `CPU_OPTIONS` explicitly selects that example’s CPU provider. Run the following from the repository root in the same Bash session. Use only the sections you need, and stop after a failed command.

| Variant | Requirements |
| --- | --- |
| C/C++ CPU | CMake 3.20+, C17 / C++17, and the family’s CPU dependencies |
| Fortran CPU | Above plus GNU Fortran or NVHPC; set `F90_COMPILER=gfortran` or `nvfortran` |
| NVIDIA C/C++ CUDA | NVIDIA GPU/driver, CUDA Toolkit / nvcc, compatible host compiler |
| NVIDIA C/C++ OpenACC | Same CUDA Toolkit plus NVHPC nvc / nvc++ |
| NVIDIA Fortran GPU | NVHPC nvfortran and CUDA Toolkit; Thrust also needs NVCC / CCCL |
| AMD HIP | AMD GPU/driver, ROCm, HIP-capable Clang, CMake 3.21+ |
| AMD OpenMP target | ROCm and compatible AMD Clang; rocThrust also needs HIP Clang and CMake 3.21+ |

Choose compiler paths, GPU targets, and ROCm / CUDA prefixes from your installation. Use a new build tree for another vendor, language, compiler, or execution profile; do not change the compiler in an already configured tree.
```bash
BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-examples.XXXXXX")" || exit 1
COMMON=(-DCMAKE_BUILD_TYPE=Release -DGPU_SUITE_BUILD_EXAMPLES=ON
  -DGPU_SUITE_BUILD_BENCHMARKS=OFF -DGPU_SUITE_BUILD_TESTS=OFF
  "-DGPU_SUITE_VENDOR=$VENDOR" "-DGPU_SUITE_SOURCE_LANGUAGE=$SOURCE_LANGUAGE")
if [ "$VENDOR" = amd ]; then
  COMMON+=("-DGPU_SUITE_AMD_LIBRARIES=$LIBRARY")
fi
```


<a id="cpu"></a>

## 2. CPU only

Build and run only the selected CPU example, without enabling a GPU profile.
```bash
F90_OPTIONS=()
if [ "$SOURCE_LANGUAGE" = fortran ]; then
  F90_OPTIONS=("-DCMAKE_Fortran_COMPILER=${F90_COMPILER:?Set gfortran or nvfortran}")
  unset NVCOMPILER_FPU_STATE
fi
cmake -S . -B "$BUILD_ROOT/cpu" "${COMMON[@]}" "${F90_OPTIONS[@]}" \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=OFF -DGPU_SUITE_BUILD_HIP=OFF \
  -DGPU_SUITE_BUILD_OPENMP=OFF "${CPU_OPTIONS[@]}" &&
cmake --build "$BUILD_ROOT/cpu" --target "$CPU_TARGET" --parallel 2 &&
"$BUILD_ROOT/cpu/$VENDOR/$SOURCE_LANGUAGE/$LIBRARY/$CPU_TARGET"
```


<a id="cuda"></a>

## 3. NVIDIA CUDA / CUDA Fortran

CPU numerical libraries are not needed. Fortran uses the same external Toolkit for NVHPC and NVCC. Choose GCC / G++ versions supported by that Toolkit.
```bash
test "$VENDOR" = nvidia || exit 1
: "${CUDA_TOOLKIT_ROOT:?Set the installed CUDA Toolkit prefix}"
: "${CUDA_ARCHITECTURES:?Set the CMake architecture for your GPU}"
CUDA_F90_OPTIONS=()
if [ "$SOURCE_LANGUAGE" = fortran ]; then
  : "${NVHPC_GPU_TARGET:?Set the NVHPC GPU target}"
  export NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
  unset NVCOMPILER_FPU_STATE
  CUDA_F90_OPTIONS=(-DCMAKE_Fortran_COMPILER=nvfortran
    "-DGPU_SUITE_NVHPC_GPU_TARGET=$NVHPC_GPU_TARGET")
fi
cmake -S . -B "$BUILD_ROOT/cuda" "${COMMON[@]}" "${CUDA_F90_OPTIONS[@]}" \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CUDA_COMPILER="$CUDA_TOOLKIT_ROOT/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=ON \
  -DGPU_SUITE_BUILD_OPENACC=OFF &&
cmake --build "$BUILD_ROOT/cuda" --target "$GPU_TARGET" --parallel 2 &&
"$BUILD_ROOT/cuda/nvidia/$SOURCE_LANGUAGE/$LIBRARY/$GPU_TARGET"
```


<a id="openacc"></a>

## 4. NVIDIA OpenACC

Match the CUDA Toolkit used by NVHPC and CMake. CMake explicitly compiles and links the Fortran Thrust wrapper.
```bash
test "$VENDOR" = nvidia || exit 1
: "${CUDA_TOOLKIT_ROOT:?Set the installed CUDA Toolkit prefix}"
: "${NVHPC_GPU_TARGET:?Set the NVHPC GPU target}"
export NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
ACC_COMPILERS=(-DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++)
if [ "$SOURCE_LANGUAGE" = fortran ]; then
  : "${CUDA_ARCHITECTURES:?Set the CMake CUDA architecture for the Thrust wrapper}"
  unset NVCOMPILER_FPU_STATE
  ACC_COMPILERS=(-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
    -DCMAKE_Fortran_COMPILER=nvfortran
    "-DCMAKE_CUDA_COMPILER=$CUDA_TOOLKIT_ROOT/bin/nvcc"
    "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES")
fi
cmake -S . -B "$BUILD_ROOT/openacc" "${COMMON[@]}" "${ACC_COMPILERS[@]}" \
  -DCUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="$NVHPC_GPU_TARGET" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=ON &&
cmake --build "$BUILD_ROOT/openacc" --target "$DIRECTIVE_TARGET" --parallel 2 &&
"$BUILD_ROOT/openacc/nvidia/$SOURCE_LANGUAGE/$LIBRARY/$DIRECTIVE_TARGET"
```


<a id="hip"></a>

## 5. AMD HIP

Native HIP uses a supported Clang as `CMAKE_HIP_COMPILER`, not the `hipcc` wrapper. Manual compilation may use `hipcc`.
```bash
test "$VENDOR" = amd || exit 1
: "${ROCM_ROOT:?Set the installed ROCm prefix}"
: "${HIP_CLANGXX:?Set the absolute HIP-capable Clang compiler path, not hipcc}"
: "${AMD_GPU_TARGET:?Set the target supported by your GPU and runtime}"
cmake -S . -B "$BUILD_ROOT/hip" "${COMMON[@]}" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=OFF -DGPU_SUITE_BUILD_HIP=ON \
  -DGPU_SUITE_BUILD_OPENMP=OFF -DGPU_SUITE_ROCM_ROOT="$ROCM_ROOT" \
  -DCMAKE_HIP_COMPILER="$HIP_CLANGXX" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_ROOT" \
  -DCMAKE_HIP_ARCHITECTURES="$AMD_GPU_TARGET" &&
cmake --build "$BUILD_ROOT/hip" --target "$GPU_TARGET" --parallel 2 &&
"$BUILD_ROOT/hip/amd/c-cpp/$LIBRARY/$GPU_TARGET"
```


<a id="openmp"></a>

## 6. AMD OpenMP target

OpenMP maps arrays and passes their device addresses to the GPU library. Do not assume USM or omitted copies. The rocThrust OpenMP caller and `rocthrust_wrapper.hip` are linked as separate translation units.
```bash
test "$VENDOR" = amd || exit 1
: "${ROCM_ROOT:?Set the installed ROCm prefix}"
: "${AMD_CLANGXX:?Set the absolute AMD Clang OpenMP C++ compiler path}"
: "${AMD_GPU_TARGET:?Set the target supported by your GPU and runtime}"
WRAPPER_OPTIONS=()
if [ "$LIBRARY" = rocthrust ]; then
  WRAPPER_OPTIONS=("-DCMAKE_HIP_COMPILER=${HIP_CLANGXX:?Set HIP Clang for the wrapper}"
    "-DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROCM_ROOT"
    "-DCMAKE_HIP_ARCHITECTURES=$AMD_GPU_TARGET")
fi
cmake -S . -B "$BUILD_ROOT/openmp" "${COMMON[@]}" "${WRAPPER_OPTIONS[@]}" \
  -DCMAKE_CXX_COMPILER="$AMD_CLANGXX" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=OFF -DGPU_SUITE_BUILD_HIP=OFF \
  -DGPU_SUITE_BUILD_OPENMP=ON -DGPU_SUITE_ROCM_ROOT="$ROCM_ROOT" \
  -DGPU_SUITE_AMD_GPU_TARGET="$AMD_GPU_TARGET" &&
cmake --build "$BUILD_ROOT/openmp" --target "$DIRECTIVE_TARGET" --parallel 2 &&
OMP_TARGET_OFFLOAD=MANDATORY \
  "$BUILD_ROOT/openmp/amd/c-cpp/$LIBRARY/$DIRECTIVE_TARGET"
```


<a id="cpu-providers"></a>

## CPU numerical providers

If only CPU dependencies are missing, you can still use the GPU-only recipes. There is no automatic fallback to another CPU provider.

| Operation | Selection / required files |
| --- | --- |
| FFT | FP32 FFTW: `fftw3.h`, `libfftw3f`; Fortran also `fftw3.f03`. `FFTW_ROOT` sets a prefix. Teaching is serial; threaded APIs are for benchmarks. |
| C/C++ BLAS | `GPU_SUITE_CPU_BLAS_BACKEND=ONEMKL`, `OPENBLAS`, `GENERIC_CBLAS` |
| C/C++ Solver | `GPU_SUITE_CPU_LAPACK_BACKEND=ONEMKL`, `OPENBLAS`, `GENERIC_LAPACKE`; LAPACKE API |
| C/C++ Sparse | NVIDIA: `ONEMKL`; AMD: `ONEMKL` / `AOCLSPARSE`. REFERENCE is not a substitute for the teaching target. |
| Fortran BLAS / Sparse / Solver | oneMKL LP64, `mkl_rt`, `mkl_service.h`; Sparse: `mkl_spblas.f90`; `MKLROOT` |
| RAND / Thrust | Serial CPU teaching needs only C++ standard library / Fortran intrinsics. |

For installations outside the search path, add `CMAKE_PREFIX_PATH` or the following inputs to `CPU_OPTIONS`. Keep headers and libraries from the same provider / ABI. AOCL BLIS / libFLAME can use the C/C++ GENERIC providers; when measuring parallel CPU code, select the intended parallel library and its dependencies.

| Provider | CMake inputs |
| --- | --- |
| Generic CBLAS | `GPU_SUITE_GENERIC_CBLAS_INCLUDE_DIR`, `GPU_SUITE_GENERIC_CBLAS_LIBRARY`, `GPU_SUITE_GENERIC_CBLAS_EXTRA_LIBRARIES` |
| Generic LAPACKE | `GPU_SUITE_GENERIC_LAPACKE_INCLUDE_DIR`, `GPU_SUITE_GENERIC_LAPACKE_LIBRARY`, `GPU_SUITE_GENERIC_LAPACKE_EXTRA_LIBRARIES` |
| AOCL Sparse | `GPU_SUITE_AOCL_SPARSE_INCLUDE_DIR`, `GPU_SUITE_AOCL_SPARSE_LIBRARY`, `GPU_SUITE_AOCL_SPARSE_EXTRA_LIBRARIES` |

`*_EXTRA_LIBRARIES` contains semicolon-separated absolute file paths in link order, not a string of flags. Keep the installation’s loader environment available at runtime so the same shared libraries can be resolved. Threaded FFTW discovery uses `FFTW3f_THREADING=AUTO/PTHREAD/OPENMP/COMBINED/NONE`; this does not parallelize the serial teaching example.

<a id="check"></a>

## Check success and common failures

- Successful configure may still disable a target because a dependency is missing. Require the target build to succeed, exit status 0, and the expected result in its README. Fortran examples print `verification PASS`.
- Target not found: read the configure disable reason and provide the required headers, APIs, and libraries. rocSPARSE requires its v2 API.
- Missing shared library at runtime: restore the provider/toolchain loader environment used for the build; do not silently substitute another library.
- OpenMP host fallback / device mapping failure: check the offload compiler/runtime, GPU target, and device visibility. CPU fallback is not successful GPU execution.
- Numerical verification failure: inspect API returns, `info`, layout, initialization, and precision; do not silently relax verification.

This route builds examples only. For `_bench` programs, create a **new** tree with `GPU_SUITE_BUILD_BENCHMARKS=ON` and follow the [measurement guide](PORTABILITY.md#measuring-on-your-own-system). Unchanged source does not establish GPU execution on another installation.

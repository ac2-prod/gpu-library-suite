cmake_minimum_required(VERSION 3.20)

project(gpu_library_suite LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

option(GPU_SUITE_BUILD_EXAMPLES "Build teaching examples" ON)
option(GPU_SUITE_BUILD_BENCHMARKS "Build benchmark executables" ON)
option(GPU_SUITE_BUILD_CPU "Build CPU implementations" ON)
option(GPU_SUITE_BUILD_CUDA "Build direct CUDA implementations" OFF)
option(GPU_SUITE_BUILD_OPENACC "Build OpenACC implementations" OFF)
option(GPU_SUITE_BUILD_TESTS "Build and register local tests" ON)

if(GPU_SUITE_BUILD_OPENACC AND
   (GPU_SUITE_BUILD_CPU OR GPU_SUITE_BUILD_CUDA))
  message(FATAL_ERROR
    "OpenACC build profile requires GPU_SUITE_BUILD_CPU=OFF and GPU_SUITE_BUILD_CUDA=OFF")
endif()

set(GPU_SUITE_CPU_BLAS_BACKEND "ONEMKL" CACHE STRING
    "CPU BLAS provider: ONEMKL, OPENBLAS, or GENERIC_CBLAS")
set_property(CACHE GPU_SUITE_CPU_BLAS_BACKEND PROPERTY STRINGS
             ONEMKL OPENBLAS GENERIC_CBLAS)
set(GPU_SUITE_CPU_SPARSE_BACKEND "ONEMKL" CACHE STRING
    "CPU sparse provider: ONEMKL or REFERENCE")
set_property(CACHE GPU_SUITE_CPU_SPARSE_BACKEND PROPERTY STRINGS
             ONEMKL REFERENCE)
set(GPU_SUITE_CPU_LAPACK_BACKEND "ONEMKL" CACHE STRING
    "CPU LAPACKE provider: ONEMKL, OPENBLAS, or GENERIC_LAPACKE")
set_property(CACHE GPU_SUITE_CPU_LAPACK_BACKEND PROPERTY STRINGS
             ONEMKL OPENBLAS GENERIC_LAPACKE)
set(GPU_SUITE_CPU_THRUST_BACKEND "STL" CACHE STRING
    "CPU transform-reduce provider: STL or OPENMP")
set_property(CACHE GPU_SUITE_CPU_THRUST_BACKEND PROPERTY STRINGS STL OPENMP)
set(GPU_SUITE_NVHPC_GPU_TARGET "" CACHE STRING
    "Externally selected NVHPC GPU target")

function(_gpu_suite_resolve_cuda_toolkit_root output)
  set(_root "")
  if(CUDAToolkit_LIBRARY_ROOT)
    set(_root "${CUDAToolkit_LIBRARY_ROOT}")
  elseif(CUDAToolkit_ROOT)
    set(_root "${CUDAToolkit_ROOT}")
  elseif(CUDAToolkit_BIN_DIR)
    get_filename_component(_root "${CUDAToolkit_BIN_DIR}" DIRECTORY)
  endif()
  if(NOT "${_root}" STREQUAL "")
    file(REAL_PATH "${_root}" _root)
  endif()
  set(${output} "${_root}" PARENT_SCOPE)
endfunction()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GpuSuiteStatus)
include(GpuSuiteBuildMetadata)
include(GpuSuiteTargets)

set(GPU_SUITE_CUDA_LANGUAGE_ENABLED OFF)
set(GPU_SUITE_CUDA_TARGETS_ENABLED OFF)
set(GPU_SUITE_CUDA_DISABLE_REASON "not requested")
set(GPU_SUITE_CUDA_TOOLKIT_ROOT "")

if(GPU_SUITE_BUILD_CUDA)
  set(_gpu_suite_cuda_architectures_explicit OFF)
  if(DEFINED CMAKE_CUDA_ARCHITECTURES AND
     NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    set(_gpu_suite_cuda_architectures_explicit ON)
  endif()

  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(GPU_SUITE_CUDA_LANGUAGE_ENABLED ON)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      _gpu_suite_resolve_cuda_toolkit_root(GPU_SUITE_CUDA_TOOLKIT_ROOT)
    endif()
    if(NOT _gpu_suite_cuda_architectures_explicit)
      set(GPU_SUITE_CUDA_DISABLE_REASON
          "CMAKE_CUDA_ARCHITECTURES was not supplied externally")
    elseif(NOT CUDAToolkit_FOUND)
      set(GPU_SUITE_CUDA_DISABLE_REASON "CUDAToolkit was not found")
    else()
      set(GPU_SUITE_CUDA_TARGETS_ENABLED ON)
      set(GPU_SUITE_CUDA_DISABLE_REASON "")
    endif()
  else()
    set(GPU_SUITE_CUDA_DISABLE_REASON "no CUDA compiler was found")
  endif()
endif()

set(GPU_SUITE_OPENACC_TARGETS_ENABLED OFF)
set(GPU_SUITE_OPENACC_DISABLE_REASON "not requested")
set(GPU_SUITE_OPENACC_COMPILE_OPTIONS "")
set(GPU_SUITE_OPENACC_LINK_OPTIONS "")
if(GPU_SUITE_BUILD_OPENACC)
  find_package(OpenACC COMPONENTS CXX QUIET)
  if(NOT (OpenACC_CXX_FOUND OR TARGET OpenACC::OpenACC_CXX))
    set(GPU_SUITE_OPENACC_DISABLE_REASON "OpenACC CXX support was not found")
  elseif(NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    set(GPU_SUITE_OPENACC_DISABLE_REASON
        "NVIDIA CUDA-library OpenACC targets require nvc++")
  elseif("${GPU_SUITE_NVHPC_GPU_TARGET}" STREQUAL "")
    set(GPU_SUITE_OPENACC_DISABLE_REASON
        "GPU_SUITE_NVHPC_GPU_TARGET was not supplied externally")
  elseif(NOT GPU_SUITE_NVHPC_GPU_TARGET MATCHES
         "^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    set(GPU_SUITE_OPENACC_DISABLE_REASON
        "GPU_SUITE_NVHPC_GPU_TARGET contains unsafe characters")
  else()
    find_package(CUDAToolkit QUIET)
    if(NOT CUDAToolkit_FOUND)
      set(GPU_SUITE_OPENACC_DISABLE_REASON "CUDAToolkit was not found")
    elseif(NOT DEFINED ENV{NVHPC_CUDA_HOME} OR
           "$ENV{NVHPC_CUDA_HOME}" STREQUAL "")
      message(FATAL_ERROR
        "NVHPC_CUDA_HOME must identify the CUDA Toolkit selected by NVHPC")
    else()
      file(REAL_PATH "$ENV{NVHPC_CUDA_HOME}" _gpu_suite_nvhpc_cuda_home)
      _gpu_suite_resolve_cuda_toolkit_root(GPU_SUITE_CUDA_TOOLKIT_ROOT)
      if(NOT "${_gpu_suite_nvhpc_cuda_home}" STREQUAL
             "${GPU_SUITE_CUDA_TOOLKIT_ROOT}")
        message(FATAL_ERROR
          "CUDAToolkit path differs from NVHPC_CUDA_HOME")
      else()
        set(GPU_SUITE_OPENACC_COMPILE_OPTIONS
            "-gpu=${GPU_SUITE_NVHPC_GPU_TARGET}")
        set(GPU_SUITE_OPENACC_LINK_OPTIONS
            "-gpu=${GPU_SUITE_NVHPC_GPU_TARGET}")
        set(GPU_SUITE_OPENACC_TARGETS_ENABLED ON)
        set(GPU_SUITE_OPENACC_DISABLE_REASON "")
      endif()
    endif()
  endif()
endif()

gpu_suite_generate_build_metadata()

add_subdirectory(common/c-cpp)

foreach(_gpu_suite_library cufft cublas cusparse cusolver curand thrust)
  set(_gpu_suite_dir
      "${CMAKE_CURRENT_SOURCE_DIR}/nvidia/c-cpp/${_gpu_suite_library}")
  if(EXISTS "${_gpu_suite_dir}/CMakeLists.txt")
    add_subdirectory("${_gpu_suite_dir}")
  endif()
endforeach()

if(GPU_SUITE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

gpu_suite_add_partial_manifest_target()

gpu_suite_record_status("CPU implementations"
                        "${GPU_SUITE_BUILD_CPU}"
                        "controlled by GPU_SUITE_BUILD_CPU")
gpu_suite_record_status("CUDA implementations"
                        "${GPU_SUITE_CUDA_TARGETS_ENABLED}"
                        "${GPU_SUITE_CUDA_DISABLE_REASON}")
gpu_suite_record_status("OpenACC implementations"
                        "${GPU_SUITE_OPENACC_TARGETS_ENABLED}"
                        "${GPU_SUITE_OPENACC_DISABLE_REASON}")
gpu_suite_print_status()

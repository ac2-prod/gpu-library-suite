include(GpuSuiteCpuDependencies)
include(GpuSuiteProbe)

set(_cpu_enabled FALSE)
set(_cpu_reason "GPU_SUITE_BUILD_CPU is OFF")
if(GPU_SUITE_BUILD_CPU)
  gpu_suite_find_cblas()
  if(GPU_SUITE_CBLAS_FOUND)
    set(_cpu_enabled TRUE)
    set(_cpu_reason "")
    string(TOLOWER "${GPU_SUITE_CBLAS_PROVIDER}" _cpu_variant)
    if(_cpu_variant STREQUAL "generic_cblas")
      set(_cpu_backend cpu-generic-cblas)
    elseif(_cpu_variant STREQUAL "openblas")
      set(_cpu_backend cpu-openblas)
    else()
      set(_cpu_backend cpu-onemkl)
    endif()
  else()
    set(_cpu_reason "${GPU_SUITE_CBLAS_REASON}")
  endif()
endif()
if(_cpu_enabled AND GPU_SUITE_BUILD_EXAMPLES)
  add_executable(blas_cpu examples/blas_cpu.c)
  target_link_libraries(blas_cpu PRIVATE GpuSuite::CBLAS)
  gpu_suite_configure_executable(blas_cpu examples/blas_cpu.c
    LIBRARY cublas IMPLEMENTATION cpu ROLE example BUILD_PROFILE cpu-cuda
    BACKEND_VARIANT "${_cpu_variant}" CPU_BACKENDS "${_cpu_backend}")
endif()
if(_cpu_enabled AND GPU_SUITE_BUILD_BENCHMARKS)
  add_executable(blas_cpu_bench benchmarks/blas_cpu_bench.c)
  target_link_libraries(blas_cpu_bench PRIVATE gpu_suite::common GpuSuite::CBLAS)
  target_compile_definitions(blas_cpu_bench PRIVATE
    "GPU_SUITE_COMPILED_CPU_BACKEND=\"${_cpu_backend}\"")
  gpu_suite_configure_executable(blas_cpu_bench benchmarks/blas_cpu_bench.c
    LIBRARY cublas IMPLEMENTATION cpu ROLE benchmark BUILD_PROFILE cpu-cuda
    BACKEND_VARIANT "${_cpu_variant}" CPU_BACKENDS "${_cpu_backend}")
endif()

set(_probe [=[
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
int main() {
  cublasHandle_t handle;
  const double alpha=1.0, beta=0.0;
  if (cudaFree(0) != cudaSuccess) return 1;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 2;
  (void)cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1,
                    &alpha, 0, 1, 0, 1, &beta, 0, 1);
  return cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS ? 0 : 3;
}
]=])
set(_cuda_enabled FALSE)
set(_cuda_reason "direct CUDA targets are disabled")
if(GPU_SUITE_CUDA_TARGETS_ENABLED AND TARGET CUDA::cudart AND TARGET CUDA::cublas)
  gpu_suite_check_cuda_link(GPU_SUITE_CUBLAS_CUDA_LINKS SOURCE "${_probe}"
    LIBRARIES CUDA::cudart CUDA::cublas)
  if(GPU_SUITE_CUBLAS_CUDA_LINKS)
    set(_cuda_enabled TRUE)
    set(_cuda_reason "")
  else()
    set(_cuda_reason "CUDA Runtime/cublasDgemm symbols did not compile and link")
  endif()
endif()
if(_cuda_enabled AND GPU_SUITE_BUILD_EXAMPLES)
  add_executable(blas_gpu examples/blas_gpu.cu)
  target_link_libraries(blas_gpu PRIVATE CUDA::cudart CUDA::cublas)
  gpu_suite_configure_executable(blas_gpu examples/blas_gpu.cu
    LIBRARY cublas IMPLEMENTATION cuda ROLE example BUILD_PROFILE cpu-cuda
    BACKEND_VARIANT cuda-cublas)
endif()
if(_cuda_enabled AND GPU_SUITE_BUILD_BENCHMARKS)
  add_executable(blas_gpu_bench benchmarks/blas_gpu_bench.cu)
  target_link_libraries(blas_gpu_bench PRIVATE gpu_suite::common CUDA::cudart CUDA::cublas)
  gpu_suite_configure_executable(blas_gpu_bench benchmarks/blas_gpu_bench.cu
    LIBRARY cublas IMPLEMENTATION cuda ROLE benchmark BUILD_PROFILE cpu-cuda
    BACKEND_VARIANT cuda-cublas)
endif()

set(_openacc_enabled FALSE)
set(_openacc_reason "OpenACC targets are disabled")
if(GPU_SUITE_OPENACC_TARGETS_ENABLED AND TARGET OpenACC::OpenACC_CXX AND
   TARGET CUDA::cudart AND TARGET CUDA::cublas)
  gpu_suite_check_cxx_link(GPU_SUITE_CUBLAS_OPENACC_LINKS SOURCE "${_probe}"
    LIBRARIES OpenACC::OpenACC_CXX CUDA::cudart CUDA::cublas)
  if(GPU_SUITE_CUBLAS_OPENACC_LINKS)
    set(_openacc_enabled TRUE)
    set(_openacc_reason "")
  else()
    set(_openacc_reason "nvc++ OpenACC CUDA Runtime/cuBLAS probe failed")
  endif()
endif()
if(_openacc_enabled AND GPU_SUITE_BUILD_EXAMPLES)
  add_executable(openacc_cublas examples/openacc_cublas.cpp)
  target_link_libraries(openacc_cublas PRIVATE OpenACC::OpenACC_CXX CUDA::cudart CUDA::cublas)
  gpu_suite_configure_executable(openacc_cublas examples/openacc_cublas.cpp
    LIBRARY cublas IMPLEMENTATION openacc ROLE example BUILD_PROFILE openacc
    BACKEND_VARIANT nvhpc-cublas)
endif()
if(_openacc_enabled AND GPU_SUITE_BUILD_BENCHMARKS)
  add_executable(openacc_cublas_bench benchmarks/openacc_cublas_bench.cpp)
  target_link_libraries(openacc_cublas_bench PRIVATE gpu_suite::common OpenACC::OpenACC_CXX CUDA::cudart CUDA::cublas)
  gpu_suite_configure_executable(openacc_cublas_bench benchmarks/openacc_cublas_bench.cpp
    LIBRARY cublas IMPLEMENTATION openacc ROLE benchmark BUILD_PROFILE openacc
    BACKEND_VARIANT nvhpc-cublas)
endif()
gpu_suite_record_status("cuBLAS CPU targets" "${_cpu_enabled}" "${_cpu_reason}")
gpu_suite_record_status("cuBLAS CUDA targets" "${_cuda_enabled}" "${_cuda_reason}")
gpu_suite_record_status("cuBLAS OpenACC targets" "${_openacc_enabled}" "${_openacc_reason}")

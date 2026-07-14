include(GpuSuiteCpuDependencies)
include(GpuSuiteProbe)

set(_cpu_enabled FALSE)
set(_cpu_reason "GPU_SUITE_BUILD_CPU is OFF")
if(GPU_SUITE_BUILD_CPU)
  if(GPU_SUITE_CPU_SPARSE_BACKEND STREQUAL "REFERENCE")
    set(_cpu_enabled TRUE)
    set(_cpu_variant reference-csr)
    set(_cpu_backend cpu-reference-csr)
    set(_cpu_reason "")
  else()
    gpu_suite_find_onemkl_sparse()
    if(GPU_SUITE_MKL_SPARSE_FOUND)
      set(_cpu_enabled TRUE)
      set(_cpu_variant onemkl-sparse)
      set(_cpu_backend cpu-onemkl)
      set(_cpu_reason "")
    else()
      set(_cpu_reason "${GPU_SUITE_MKL_SPARSE_REASON}")
    endif()
  endif()
endif()
set(_cpu_example_enabled FALSE)
set(_cpu_example_reason "oneMKL Sparse is unavailable")
if(_cpu_enabled AND TARGET GpuSuite::MKLSparse)
  set(_cpu_example_enabled TRUE)
  set(_cpu_example_reason "")
elseif(_cpu_enabled AND GPU_SUITE_CPU_SPARSE_BACKEND STREQUAL "REFERENCE")
  set(_cpu_example_reason
      "canonical sparse_cpu teaching example requires oneMKL Sparse")
endif()
if(_cpu_example_enabled AND GPU_SUITE_BUILD_EXAMPLES)
  add_executable(sparse_cpu examples/sparse_cpu.c)
  target_link_libraries(sparse_cpu PRIVATE GpuSuite::MKLSparse)
  gpu_suite_configure_executable(sparse_cpu examples/sparse_cpu.c
    LIBRARY cusparse IMPLEMENTATION cpu ROLE example BUILD_PROFILE cpu-cuda
    BACKEND_VARIANT "${_cpu_variant}" CPU_BACKENDS "${_cpu_backend}")
endif()
if(_cpu_enabled AND GPU_SUITE_BUILD_BENCHMARKS)
  add_executable(sparse_cpu_bench benchmarks/sparse_cpu_bench.c)
  target_link_libraries(sparse_cpu_bench PRIVATE gpu_suite::common)
  if(TARGET GpuSuite::MKLSparse)
    target_link_libraries(sparse_cpu_bench PRIVATE GpuSuite::MKLSparse)
    target_compile_definitions(sparse_cpu_bench PRIVATE GPU_SUITE_USE_MKL_SPARSE=1)
  endif()
  target_compile_definitions(sparse_cpu_bench PRIVATE
    "GPU_SUITE_COMPILED_CPU_BACKEND=\"${_cpu_backend}\"")
  gpu_suite_configure_executable(sparse_cpu_bench benchmarks/sparse_cpu_bench.c
    LIBRARY cusparse IMPLEMENTATION cpu ROLE benchmark BUILD_PROFILE cpu-cuda
    BACKEND_VARIANT "${_cpu_variant}" CPU_BACKENDS "${_cpu_backend}")
endif()

set(_probe [=[
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cstddef>
int main() {
  cusparseHandle_t handle;
  cusparseSpMatDescr_t matrix;
  cusparseDnVecDescr_t vector_x;
  cusparseDnVecDescr_t vector_y;
  double alpha = 1.0;
  double beta = 0.0;
  std::size_t buffer_size = 0;
  if (cudaFree(0) != cudaSuccess) return 1;
  if (cusparseCreate(&handle) != CUSPARSE_STATUS_SUCCESS) return 2;
  (void)cusparseCreateCsr(&matrix,1,1,1,0,0,0,CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
  (void)cusparseCreateDnVec(&vector_x,1,0,CUDA_R_64F);
  (void)cusparseCreateDnVec(&vector_y,1,0,CUDA_R_64F);
  (void)cusparseSpMV_bufferSize(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha,matrix,vector_x,&beta,vector_y,CUDA_R_64F,
      CUSPARSE_SPMV_ALG_DEFAULT,&buffer_size);
  (void)cusparseSpMV(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,matrix,
      vector_x,&beta,vector_y,CUDA_R_64F,CUSPARSE_SPMV_ALG_DEFAULT,0);
  (void)cusparseDestroyDnVec(vector_y);
  (void)cusparseDestroyDnVec(vector_x);
  (void)cusparseDestroySpMat(matrix);
  return cusparseDestroy(handle) == CUSPARSE_STATUS_SUCCESS ? 0 : 3;
}
]=])
set(_cuda_enabled FALSE)
set(_cuda_reason "direct CUDA targets are disabled")
if(GPU_SUITE_CUDA_TARGETS_ENABLED AND TARGET CUDA::cudart AND TARGET CUDA::cusparse)
  gpu_suite_check_cuda_link(GPU_SUITE_CUSPARSE_CUDA_LINKS SOURCE "${_probe}"
    LIBRARIES CUDA::cudart CUDA::cusparse)
  if(GPU_SUITE_CUSPARSE_CUDA_LINKS)
    set(_cuda_enabled TRUE)
    set(_cuda_reason "")
  else()
    set(_cuda_reason
        "CUDA Runtime/cuSPARSE descriptor and SpMV symbols did not link")
  endif()
endif()
if(_cuda_enabled AND GPU_SUITE_BUILD_EXAMPLES)
  add_executable(sparse_gpu examples/sparse_gpu.cu)
  target_link_libraries(sparse_gpu PRIVATE CUDA::cudart CUDA::cusparse)
  gpu_suite_configure_executable(sparse_gpu examples/sparse_gpu.cu LIBRARY cusparse IMPLEMENTATION cuda ROLE example BUILD_PROFILE cpu-cuda BACKEND_VARIANT cuda-cusparse)
endif()
if(_cuda_enabled AND GPU_SUITE_BUILD_BENCHMARKS)
  add_executable(sparse_gpu_bench benchmarks/sparse_gpu_bench.cu)
  target_link_libraries(sparse_gpu_bench PRIVATE gpu_suite::common CUDA::cudart CUDA::cusparse)
  gpu_suite_configure_executable(sparse_gpu_bench benchmarks/sparse_gpu_bench.cu LIBRARY cusparse IMPLEMENTATION cuda ROLE benchmark BUILD_PROFILE cpu-cuda BACKEND_VARIANT cuda-cusparse)
endif()
set(_openacc_enabled FALSE)
set(_openacc_reason "OpenACC targets are disabled")
if(GPU_SUITE_OPENACC_TARGETS_ENABLED AND TARGET OpenACC::OpenACC_CXX AND TARGET CUDA::cudart AND TARGET CUDA::cusparse)
  gpu_suite_check_cxx_link(GPU_SUITE_CUSPARSE_OPENACC_LINKS SOURCE "${_probe}" LIBRARIES OpenACC::OpenACC_CXX CUDA::cudart CUDA::cusparse)
  if(GPU_SUITE_CUSPARSE_OPENACC_LINKS)
    set(_openacc_enabled TRUE)
    set(_openacc_reason "")
  else()
    set(_openacc_reason "nvc++ OpenACC CUDA Runtime/cuSPARSE probe failed")
  endif()
endif()
if(_openacc_enabled AND GPU_SUITE_BUILD_EXAMPLES)
  add_executable(openacc_cusparse examples/openacc_cusparse.cpp)
  target_link_libraries(openacc_cusparse PRIVATE OpenACC::OpenACC_CXX CUDA::cudart CUDA::cusparse)
  gpu_suite_configure_executable(openacc_cusparse examples/openacc_cusparse.cpp LIBRARY cusparse IMPLEMENTATION openacc ROLE example BUILD_PROFILE openacc BACKEND_VARIANT nvhpc-cusparse)
endif()
if(_openacc_enabled AND GPU_SUITE_BUILD_BENCHMARKS)
  add_executable(openacc_cusparse_bench benchmarks/openacc_cusparse_bench.cpp)
  target_link_libraries(openacc_cusparse_bench PRIVATE gpu_suite::common OpenACC::OpenACC_CXX CUDA::cudart CUDA::cusparse)
  gpu_suite_configure_executable(openacc_cusparse_bench benchmarks/openacc_cusparse_bench.cpp LIBRARY cusparse IMPLEMENTATION openacc ROLE benchmark BUILD_PROFILE openacc BACKEND_VARIANT nvhpc-cusparse)
endif()
gpu_suite_record_status("cuSPARSE CPU targets" "${_cpu_enabled}" "${_cpu_reason}")
gpu_suite_record_status("cuSPARSE canonical CPU example"
                        "${_cpu_example_enabled}" "${_cpu_example_reason}")
gpu_suite_record_status("cuSPARSE CUDA targets" "${_cuda_enabled}" "${_cuda_reason}")
gpu_suite_record_status("cuSPARSE OpenACC targets" "${_openacc_enabled}" "${_openacc_reason}")

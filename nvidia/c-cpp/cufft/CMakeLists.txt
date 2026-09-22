include(GpuSuiteProbe)

set(_cufft_cpu_reason "GPU_SUITE_BUILD_CPU is OFF")
set(_cufft_cpu_enabled FALSE)
set(_cufft_threads_reason "base CPU target is disabled")
if(GPU_SUITE_BUILD_CPU)
  find_package(FFTW3f QUIET)
  if(FFTW3f_FOUND)
    set(_cufft_cpu_enabled TRUE)
    set(_cufft_cpu_reason "")
    set(_cufft_threads_reason "${FFTW3f_THREADS_DISABLE_REASON}")
  else()
    set(_cufft_cpu_reason
        "FFTW3f plan/execute compile-and-link probe failed")
  endif()
endif()

if(_cufft_cpu_enabled AND GPU_SUITE_BUILD_EXAMPLES)
  add_executable(fft_cpu examples/fft_cpu.c)
  target_link_libraries(fft_cpu PRIVATE FFTW3f::fftw3f)
  gpu_suite_configure_executable(fft_cpu examples/fft_cpu.c
    LIBRARY cufft IMPLEMENTATION cpu ROLE example
    BUILD_PROFILE cpu-cuda BACKEND_VARIANT fftw3f-serial
    CPU_BACKENDS cpu-fftw-serial)
endif()

if(_cufft_cpu_enabled AND GPU_SUITE_BUILD_BENCHMARKS)
  add_executable(fft_cpu_bench benchmarks/fft_cpu_bench.c)
  target_link_libraries(fft_cpu_bench PRIVATE gpu_suite::common FFTW3f::fftw3f)
  if(FFTW3f_THREADS_FOUND)
    target_compile_definitions(fft_cpu_bench PRIVATE GPU_SUITE_HAVE_FFTW_THREADS=1)
    target_link_libraries(fft_cpu_bench PRIVATE FFTW3f::threads)
    set(_cufft_cpu_variant "fftw3f-serial+threaded")
    set(_cufft_cpu_backends cpu-fftw-serial cpu-fftw-threaded)
  else()
    set(_cufft_cpu_variant "fftw3f-serial")
    set(_cufft_cpu_backends cpu-fftw-serial)
  endif()
  gpu_suite_configure_executable(fft_cpu_bench benchmarks/fft_cpu_bench.c
    LIBRARY cufft IMPLEMENTATION cpu ROLE benchmark
    BUILD_PROFILE cpu-cuda BACKEND_VARIANT "${_cufft_cpu_variant}"
    CPU_BACKENDS ${_cufft_cpu_backends})
endif()

set(_cufft_probe [=[
#include <cuda_runtime_api.h>
#include <cufft.h>
int main() {
  cufftHandle plan;
  int n[1] = {4};
  if (cudaFree(0) != cudaSuccess) return 1;
  if (cufftPlanMany(&plan, 1, n, 0, 1, 4, 0, 1, 4,
                    CUFFT_C2C, 1) != CUFFT_SUCCESS) return 2;
  cufftComplex *data = 0;
  (void)cufftExecC2C(plan, data, data, CUFFT_FORWARD);
  (void)cufftDestroy(plan);
  return 0;
}
]=])

set(_cufft_cuda_enabled FALSE)
set(_cufft_cuda_reason "direct CUDA targets are disabled")
if(GPU_SUITE_CUDA_TARGETS_ENABLED)
  if(NOT (TARGET CUDA::cudart AND TARGET CUDA::cufft))
    set(_cufft_cuda_reason "CUDA::cudart or CUDA::cufft target is unavailable")
  else()
    gpu_suite_check_cuda_link(GPU_SUITE_CUFFT_CUDA_LINKS
      SOURCE "${_cufft_probe}"
      LIBRARIES CUDA::cudart CUDA::cufft)
    if(GPU_SUITE_CUFFT_CUDA_LINKS)
      set(_cufft_cuda_enabled TRUE)
      set(_cufft_cuda_reason "")
    else()
      set(_cufft_cuda_reason
          "CUDA Runtime and cuFFT symbols did not compile and link")
    endif()
  endif()
endif()

if(_cufft_cuda_enabled AND GPU_SUITE_BUILD_EXAMPLES)
  add_executable(fft_gpu examples/fft_gpu.cu)
  target_link_libraries(fft_gpu PRIVATE CUDA::cudart CUDA::cufft)
  gpu_suite_configure_executable(fft_gpu examples/fft_gpu.cu
    LIBRARY cufft IMPLEMENTATION cuda ROLE example
    BUILD_PROFILE cpu-cuda BACKEND_VARIANT cuda-cufft)
endif()
if(_cufft_cuda_enabled AND GPU_SUITE_BUILD_BENCHMARKS)
  add_executable(fft_gpu_bench benchmarks/fft_gpu_bench.cu)
  target_link_libraries(fft_gpu_bench PRIVATE gpu_suite::common
                        CUDA::cudart CUDA::cufft)
  gpu_suite_configure_executable(fft_gpu_bench benchmarks/fft_gpu_bench.cu
    LIBRARY cufft IMPLEMENTATION cuda ROLE benchmark
    BUILD_PROFILE cpu-cuda BACKEND_VARIANT cuda-cufft)
endif()

set(_cufft_openacc_enabled FALSE)
set(_cufft_openacc_reason "OpenACC targets are disabled")
if(GPU_SUITE_OPENACC_TARGETS_ENABLED)
  if(NOT (TARGET OpenACC::OpenACC_CXX AND TARGET CUDA::cudart AND
          TARGET CUDA::cufft))
    set(_cufft_openacc_reason
        "OpenACC::OpenACC_CXX, CUDA::cudart, or CUDA::cufft is unavailable")
  else()
    gpu_suite_check_cxx_link(GPU_SUITE_CUFFT_OPENACC_LINKS
      SOURCE "${_cufft_probe}"
      LIBRARIES OpenACC::OpenACC_CXX CUDA::cudart CUDA::cufft)
    if(GPU_SUITE_CUFFT_OPENACC_LINKS)
      set(_cufft_openacc_enabled TRUE)
      set(_cufft_openacc_reason "")
    else()
      set(_cufft_openacc_reason
          "nvc++ OpenACC CUDA Runtime/cuFFT compile-and-link probe failed")
    endif()
  endif()
endif()

if(_cufft_openacc_enabled AND GPU_SUITE_BUILD_EXAMPLES)
  add_executable(openacc_cufft examples/openacc_cufft.cpp)
  target_link_libraries(openacc_cufft PRIVATE OpenACC::OpenACC_CXX
                        CUDA::cudart CUDA::cufft)
  gpu_suite_configure_executable(openacc_cufft examples/openacc_cufft.cpp
    LIBRARY cufft IMPLEMENTATION openacc ROLE example
    BUILD_PROFILE openacc BACKEND_VARIANT nvhpc-cufft)
endif()
if(_cufft_openacc_enabled AND GPU_SUITE_BUILD_BENCHMARKS)
  add_executable(openacc_cufft_bench benchmarks/openacc_cufft_bench.cpp)
  target_link_libraries(openacc_cufft_bench PRIVATE gpu_suite::common
                        OpenACC::OpenACC_CXX CUDA::cudart CUDA::cufft)
  gpu_suite_configure_executable(openacc_cufft_bench
    benchmarks/openacc_cufft_bench.cpp
    LIBRARY cufft IMPLEMENTATION openacc ROLE benchmark
    BUILD_PROFILE openacc BACKEND_VARIANT nvhpc-cufft)
endif()

gpu_suite_record_status("cuFFT CPU targets" "${_cufft_cpu_enabled}"
                        "${_cufft_cpu_reason}")
gpu_suite_record_status("cuFFT threaded CPU series" "${FFTW3f_THREADS_FOUND}"
                        "${_cufft_threads_reason}")
gpu_suite_record_status("cuFFT CUDA targets" "${_cufft_cuda_enabled}"
                        "${_cufft_cuda_reason}")
gpu_suite_record_status("cuFFT OpenACC targets" "${_cufft_openacc_enabled}"
                        "${_cufft_openacc_reason}")

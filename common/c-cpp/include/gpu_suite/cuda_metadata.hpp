#ifndef GPU_SUITE_CUDA_METADATA_HPP
#define GPU_SUITE_CUDA_METADATA_HPP

#include "gpu_suite/result.h"

#include <cuda_runtime.h>

#if defined(GPU_SUITE_CUDA_LIBRARY_CUBLAS)
#include <cublas_v2.h>
#elif defined(GPU_SUITE_CUDA_LIBRARY_CUFFT)
#include <cufft.h>
#elif defined(GPU_SUITE_CUDA_LIBRARY_CUSPARSE)
#include <cusparse.h>
#elif defined(GPU_SUITE_CUDA_LIBRARY_CUSOLVER)
#include <cusolverDn.h>
#elif defined(GPU_SUITE_CUDA_LIBRARY_CURAND)
#include <curand.h>
#elif defined(GPU_SUITE_CUDA_LIBRARY_THRUST)
#include <thrust/version.h>
#endif

#include <cstddef>
#include <cstdio>
#include <cstring>

namespace gpu_suite {

inline void format_cuda_version(int version, char *output,
                                std::size_t output_size) {
  const int major = version / 1000;
  const int minor = (version % 1000) / 10;
  const int patch = version % 10;
  (void)std::snprintf(output, output_size, "%d.%d.%d", major, minor, patch);
}

inline bool apply_cuda_runtime_metadata(gpu_suite_result &result, int device) {
  cudaDeviceProp properties{};
  int driver_version = 0;
  int runtime_version = 0;
  cudaError_t status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceProperties: %s\n",
                 cudaGetErrorString(status));
    return false;
  }
  status = cudaDriverGetVersion(&driver_version);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaDriverGetVersion: %s\n",
                 cudaGetErrorString(status));
    return false;
  }
  status = cudaRuntimeGetVersion(&runtime_version);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaRuntimeGetVersion: %s\n",
                 cudaGetErrorString(status));
    return false;
  }

  (void)std::snprintf(result.gpu_name_storage,
                      sizeof(result.gpu_name_storage), "%s", properties.name);
  const unsigned char *uuid =
      reinterpret_cast<const unsigned char *>(properties.uuid.bytes);
  (void)std::snprintf(
      result.gpu_uuid_storage, sizeof(result.gpu_uuid_storage),
      "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
      "%02x%02x%02x%02x%02x%02x",
      uuid[0], uuid[1], uuid[2], uuid[3], uuid[4], uuid[5], uuid[6], uuid[7],
      uuid[8], uuid[9], uuid[10], uuid[11], uuid[12], uuid[13], uuid[14],
      uuid[15]);
  if (result.gpu_name != nullptr &&
      std::strcmp(result.gpu_name, result.gpu_name_storage) != 0) {
    std::fprintf(stderr,
                 "CUDA device name differs from node-local GPU metadata\n");
    return false;
  }
  if (result.gpu_uuid != nullptr &&
      std::strcmp(result.gpu_uuid, result.gpu_uuid_storage) != 0) {
    std::fprintf(stderr,
                 "CUDA device UUID differs from node-local GPU metadata\n");
    return false;
  }
  format_cuda_version(driver_version, result.cuda_driver_version_storage,
                      sizeof(result.cuda_driver_version_storage));
  format_cuda_version(runtime_version, result.cuda_runtime_version_storage,
                      sizeof(result.cuda_runtime_version_storage));
  result.gpu_name = result.gpu_name_storage;
  result.gpu_uuid = result.gpu_uuid_storage;
  result.cuda_driver_version = result.cuda_driver_version_storage;
  result.cuda_runtime_version = result.cuda_runtime_version_storage;

  int library_version = 0;
#if defined(GPU_SUITE_CUDA_LIBRARY_CUBLAS)
  cublasHandle_t handle = nullptr;
  cublasStatus_t cublas_status = cublasCreate(&handle);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cublasCreate metadata probe: status %d\n",
                 static_cast<int>(cublas_status));
    return false;
  }
  cublas_status = cublasGetVersion(handle, &library_version);
  const cublasStatus_t cublas_destroy_status = cublasDestroy(handle);
  if (cublas_status != CUBLAS_STATUS_SUCCESS ||
      cublas_destroy_status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr,
                 "cublasGetVersion/cublasDestroy metadata probe: status %d/%d\n",
                 static_cast<int>(cublas_status),
                 static_cast<int>(cublas_destroy_status));
    return false;
  }
#elif defined(GPU_SUITE_CUDA_LIBRARY_CUFFT)
  const cufftResult cufft_status = cufftGetVersion(&library_version);
  if (cufft_status != CUFFT_SUCCESS) {
    std::fprintf(stderr, "cufftGetVersion metadata probe: status %d\n",
                 static_cast<int>(cufft_status));
    return false;
  }
#elif defined(GPU_SUITE_CUDA_LIBRARY_CUSPARSE)
  cusparseHandle_t handle = nullptr;
  cusparseStatus_t cusparse_status = cusparseCreate(&handle);
  if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
    std::fprintf(stderr, "cusparseCreate metadata probe: status %d\n",
                 static_cast<int>(cusparse_status));
    return false;
  }
  cusparse_status = cusparseGetVersion(handle, &library_version);
  const cusparseStatus_t cusparse_destroy_status = cusparseDestroy(handle);
  if (cusparse_status != CUSPARSE_STATUS_SUCCESS ||
      cusparse_destroy_status != CUSPARSE_STATUS_SUCCESS) {
    std::fprintf(
        stderr,
        "cusparseGetVersion/cusparseDestroy metadata probe: status %d/%d\n",
        static_cast<int>(cusparse_status),
        static_cast<int>(cusparse_destroy_status));
    return false;
  }
#elif defined(GPU_SUITE_CUDA_LIBRARY_CUSOLVER)
  int major = 0;
  int minor = 0;
  int patch = 0;
  const cusolverStatus_t major_status =
      cusolverGetProperty(MAJOR_VERSION, &major);
  const cusolverStatus_t minor_status =
      cusolverGetProperty(MINOR_VERSION, &minor);
  const cusolverStatus_t patch_status =
      cusolverGetProperty(PATCH_LEVEL, &patch);
  if (major_status != CUSOLVER_STATUS_SUCCESS ||
      minor_status != CUSOLVER_STATUS_SUCCESS ||
      patch_status != CUSOLVER_STATUS_SUCCESS) {
    std::fprintf(
        stderr,
        "cusolverGetProperty metadata probe: status %d/%d/%d\n",
        static_cast<int>(major_status), static_cast<int>(minor_status),
        static_cast<int>(patch_status));
    return false;
  }
  (void)std::snprintf(result.library_version_storage,
                      sizeof(result.library_version_storage), "%d.%d.%d",
                      major, minor, patch);
  result.library_version = result.library_version_storage;
  return true;
#elif defined(GPU_SUITE_CUDA_LIBRARY_CURAND)
  const curandStatus_t curand_status = curandGetVersion(&library_version);
  if (curand_status != CURAND_STATUS_SUCCESS) {
    std::fprintf(stderr, "curandGetVersion metadata probe: status %d\n",
                 static_cast<int>(curand_status));
    return false;
  }
#elif defined(GPU_SUITE_CUDA_LIBRARY_THRUST)
  library_version = THRUST_VERSION;
#endif
  (void)std::snprintf(result.library_version_storage,
                      sizeof(result.library_version_storage), "%d",
                      library_version);
  result.library_version = result.library_version_storage;
  return true;
}

} // namespace gpu_suite

#endif

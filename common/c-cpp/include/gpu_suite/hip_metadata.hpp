#ifndef GPU_SUITE_HIP_METADATA_HPP
#define GPU_SUITE_HIP_METADATA_HPP

#include "gpu_suite/result.h"
#include <hip/hip_runtime.h>
#if defined(GPU_SUITE_HIP_LIBRARY_ROCBLAS)
#include <rocblas/rocblas.h>
#elif defined(GPU_SUITE_HIP_LIBRARY_ROCSPARSE)
#include <rocsparse/rocsparse.h>
#elif defined(GPU_SUITE_HIP_LIBRARY_ROCSOLVER)
#include <rocsolver/rocsolver.h>
#elif defined(GPU_SUITE_HIP_LIBRARY_ROCRAND)
#include <rocrand/rocrand.h>
#elif defined(GPU_SUITE_HIP_LIBRARY_ROCTHRUST)
#include <thrust/version.h>
#else
// Retain the original rocFFT-only embedding interface.
#include <rocfft/rocfft.h>
#endif
#include <cstdio>
#include <cstring>

namespace gpu_suite {
inline bool hip_library_version(char *buffer, std::size_t capacity) {
#if defined(GPU_SUITE_HIP_LIBRARY_ROCBLAS)
  return rocblas_get_version_string(buffer, capacity) == rocblas_status_success;
#elif defined(GPU_SUITE_HIP_LIBRARY_ROCSPARSE)
  rocsparse_handle handle = nullptr;
  if (rocsparse_create_handle(&handle) != rocsparse_status_success) return false;
  int version = 0;
  const auto status = rocsparse_get_version(handle, &version);
  const auto cleanup = rocsparse_destroy_handle(handle);
  if (status != rocsparse_status_success || cleanup != rocsparse_status_success)
    return false;
  const int length = std::snprintf(buffer, capacity, "%d", version);
  return length > 0 && static_cast<std::size_t>(length) < capacity;
#elif defined(GPU_SUITE_HIP_LIBRARY_ROCSOLVER)
  return rocsolver_get_version_string(buffer, capacity) == rocblas_status_success;
#elif defined(GPU_SUITE_HIP_LIBRARY_ROCRAND)
  int version = 0;
  if (rocrand_get_version(&version) != ROCRAND_STATUS_SUCCESS) return false;
  const int length = std::snprintf(buffer, capacity, "%d", version);
  return length > 0 && static_cast<std::size_t>(length) < capacity;
#elif defined(GPU_SUITE_HIP_LIBRARY_ROCTHRUST)
  const int length = std::snprintf(buffer, capacity, "%d", THRUST_VERSION);
  return length > 0 && static_cast<std::size_t>(length) < capacity;
#else
  const rocfft_status status = rocfft_get_version_string(buffer, capacity);
  if (status != rocfft_status_success) {
    std::fprintf(stderr, "HIP metadata: rocfft_get_version_string failed (status=%d)\n",
                 static_cast<int>(status));
    return false;
  }
  return true;
#endif
}

inline bool hip_json_int(gpu_suite_json_value *object, const char *key, int value) {
  gpu_suite_json_value *number = gpu_suite_json_int(value);
  if (number == nullptr) return false;
  if (gpu_suite_json_object_set(object, key, number) != GPU_SUITE_OK) {
    gpu_suite_json_free(number);
    return false;
  }
  return true;
}

inline bool hip_metadata_status(hipError_t status, const char *operation) {
  if (status == hipSuccess) return true;
  std::fprintf(stderr, "HIP metadata: %s failed (status=%d: %s)\n", operation,
               static_cast<int>(status), hipGetErrorString(status));
  return false;
}

// HIP's encoded version numbers are stored verbatim, never as CUDA versions.
inline bool apply_hip_runtime_metadata(gpu_suite_result &result, int device) {
  // apply_options owns runtime and its build provenance. Augment that object;
  // object_set deliberately rejects duplicate keys rather than replacing them.
  gpu_suite_json_value *runtime_object =
      gpu_suite_json_object_get_object(result.parameters, "runtime");
  if (runtime_object == nullptr ||
      !gpu_suite_json_object_has(runtime_object, "build_metadata_sha256")) {
    std::fprintf(stderr, "HIP metadata: runtime build metadata is missing or invalid\n");
    return false;
  }
  if (gpu_suite_json_object_has(runtime_object, "hip_runtime_version") ||
      gpu_suite_json_object_has(runtime_object, "hip_driver_version")) {
    std::fprintf(stderr, "HIP metadata: runtime versions are already registered\n");
    return false;
  }
  hipDeviceProp_t properties{};
  int runtime = 0;
  int driver = 0;
  char library_version[sizeof(result.library_version_storage)]{};
  if (!hip_metadata_status(hipGetDeviceProperties(&properties, device),
                            "hipGetDeviceProperties") ||
      !hip_metadata_status(hipRuntimeGetVersion(&runtime), "hipRuntimeGetVersion") ||
      !hip_metadata_status(hipDriverGetVersion(&driver), "hipDriverGetVersion")) {
    return false;
  }
  if (!hip_library_version(library_version, sizeof(library_version))) {
    std::fprintf(stderr, "HIP metadata: library version query failed\n");
    return false;
  }
  if (result.gpu_name != nullptr &&
      std::strcmp(result.gpu_name, properties.name) != 0) {
    std::fprintf(stderr, "HIP device name differs from node-local GPU metadata\n");
    return false;
  }
  if (!hip_json_int(runtime_object, "hip_runtime_version", runtime) ||
      !hip_json_int(runtime_object, "hip_driver_version", driver)) {
    std::fprintf(stderr, "HIP metadata: could not append runtime versions\n");
    return false;
  }
  std::snprintf(result.gpu_name_storage, sizeof(result.gpu_name_storage), "%s",
                properties.name);
  result.gpu_name = result.gpu_name_storage;
  // No CUDA UUID or CUDA-version inference is made for a HIP device.
  result.gpu_uuid = nullptr;
  result.cuda_driver_version = nullptr;
  result.cuda_runtime_version = nullptr;
  std::snprintf(result.library_version_storage, sizeof(result.library_version_storage),
                "%s", library_version);
  result.library_version = result.library_version_storage;
  return true;
}
} // namespace gpu_suite
#endif

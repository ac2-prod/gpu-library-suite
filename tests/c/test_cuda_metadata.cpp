#include "gpu_suite/cuda_metadata.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <cstring>

namespace {

bool fail_runtime = false;
bool fail_library = false;

} // namespace

const char *cudaGetErrorString(cudaError_t) { return "fixture CUDA failure"; }

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *properties, int device) {
  if (properties == nullptr || device != 0) {
    return 1;
  }
  (void)std::snprintf(properties->name, sizeof(properties->name), "%s",
                      "Fixture GPU");
  for (int index = 0; index < 16; ++index) {
    properties->uuid.bytes[index] = static_cast<char>(index);
  }
  return cudaSuccess;
}

cudaError_t cudaDriverGetVersion(int *version) {
  if (version == nullptr) {
    return 1;
  }
  *version = 12080;
  return cudaSuccess;
}

cudaError_t cudaRuntimeGetVersion(int *version) {
  if (version == nullptr || fail_runtime) {
    return 1;
  }
  *version = 12070;
  return cudaSuccess;
}

#if defined(GPU_SUITE_CUDA_LIBRARY_CUBLAS)
cublasStatus_t cublasCreate(cublasHandle_t *handle) {
  if (handle == nullptr || fail_library) {
    return 1;
  }
  *handle = reinterpret_cast<void *>(1);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion(cublasHandle_t, int *version) {
  if (version == nullptr || fail_library) {
    return 1;
  }
  *version = 120801;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy(cublasHandle_t) {
  return fail_library ? 1 : CUBLAS_STATUS_SUCCESS;
}
#elif defined(GPU_SUITE_CUDA_LIBRARY_CUFFT)
cufftResult cufftGetVersion(int *version) {
  if (version == nullptr || fail_library) {
    return 1;
  }
  *version = 11203;
  return CUFFT_SUCCESS;
}
#elif defined(GPU_SUITE_CUDA_LIBRARY_CUSPARSE)
cusparseStatus_t cusparseCreate(cusparseHandle_t *handle) {
  if (handle == nullptr || fail_library) {
    return 1;
  }
  *handle = reinterpret_cast<void *>(1);
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseGetVersion(cusparseHandle_t, int *version) {
  if (version == nullptr || fail_library) {
    return 1;
  }
  *version = 12040;
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroy(cusparseHandle_t) {
  return fail_library ? 1 : CUSPARSE_STATUS_SUCCESS;
}
#elif defined(GPU_SUITE_CUDA_LIBRARY_CUSOLVER)
cusolverStatus_t cusolverGetProperty(libraryPropertyType type, int *value) {
  if (value == nullptr || fail_library) {
    return 1;
  }
  if (type == MAJOR_VERSION) {
    *value = 12;
  } else if (type == MINOR_VERSION) {
    *value = 4;
  } else {
    *value = 1;
  }
  return CUSOLVER_STATUS_SUCCESS;
}
#elif defined(GPU_SUITE_CUDA_LIBRARY_CURAND)
curandStatus_t curandGetVersion(int *version) {
  if (version == nullptr || fail_library) {
    return 1;
  }
  *version = 10307;
  return CURAND_STATUS_SUCCESS;
}
#endif

int main() {
  gpu_suite_result result{};
  assert(gpu_suite::apply_cuda_runtime_metadata(result, 0));
  assert(std::strcmp(result.gpu_name, "Fixture GPU") == 0);
  assert(std::strcmp(
             result.gpu_uuid,
             "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f") == 0);
  assert(std::strcmp(result.cuda_driver_version, "12.8.0") == 0);
  assert(std::strcmp(result.cuda_runtime_version, "12.7.0") == 0);
#if defined(GPU_SUITE_CUDA_LIBRARY_CUBLAS)
  assert(std::strcmp(result.library_version, "120801") == 0);
#elif defined(GPU_SUITE_CUDA_LIBRARY_CUFFT)
  assert(std::strcmp(result.library_version, "11203") == 0);
#elif defined(GPU_SUITE_CUDA_LIBRARY_CUSPARSE)
  assert(std::strcmp(result.library_version, "12040") == 0);
#elif defined(GPU_SUITE_CUDA_LIBRARY_CUSOLVER)
  assert(std::strcmp(result.library_version, "12.4.1") == 0);
#elif defined(GPU_SUITE_CUDA_LIBRARY_CURAND)
  assert(std::strcmp(result.library_version, "10307") == 0);
#elif defined(GPU_SUITE_CUDA_LIBRARY_THRUST)
  assert(std::strcmp(result.library_version, "200000") == 0);
#endif

  gpu_suite_result mismatched{};
  mismatched.gpu_name = "Different GPU";
  assert(!gpu_suite::apply_cuda_runtime_metadata(mismatched, 0));

  fail_runtime = true;
  gpu_suite_result missing_runtime{};
  assert(!gpu_suite::apply_cuda_runtime_metadata(missing_runtime, 0));
  fail_runtime = false;

#if !defined(GPU_SUITE_CUDA_LIBRARY_THRUST)
  fail_library = true;
  gpu_suite_result missing_library{};
  assert(!gpu_suite::apply_cuda_runtime_metadata(missing_library, 0));
#endif
  return 0;
}

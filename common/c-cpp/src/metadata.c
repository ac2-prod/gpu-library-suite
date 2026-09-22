#include "gpu_suite/metadata.h"

#include "gpu_suite/build_metadata.h"

#include <string.h>

bool gpu_suite_build_git_metadata_available(void) {
  return GPU_SUITE_BUILD_GIT_METADATA_AVAILABLE != 0;
}

const char *gpu_suite_build_git_commit(void) {
  return gpu_suite_build_git_metadata_available() ? GPU_SUITE_BUILD_GIT_COMMIT
                                                   : NULL;
}

bool gpu_suite_build_git_dirty(void) {
  return gpu_suite_build_git_metadata_available() &&
         GPU_SUITE_BUILD_GIT_DIRTY != 0;
}

const char *gpu_suite_build_type(void) { return GPU_SUITE_BUILD_TYPE; }

const char *gpu_suite_build_profile(void) { return GPU_SUITE_BUILD_PROFILE; }

const char *gpu_suite_build_metadata_sha256(void) {
  return GPU_SUITE_BUILD_METADATA_SHA256;
}

const char *gpu_suite_build_compiler(gpu_suite_implementation implementation) {
  if (implementation == GPU_SUITE_IMPLEMENTATION_CUDA &&
      GPU_SUITE_BUILD_CUDA_COMPILER[0] != '\0') {
    return GPU_SUITE_BUILD_CUDA_COMPILER;
  }
  if (implementation == GPU_SUITE_IMPLEMENTATION_OPENACC) {
    return GPU_SUITE_BUILD_CXX_COMPILER;
  }
  return GPU_SUITE_BUILD_C_COMPILER;
}

const char *
gpu_suite_build_compiler_version(gpu_suite_implementation implementation) {
  if (implementation == GPU_SUITE_IMPLEMENTATION_CUDA &&
      GPU_SUITE_BUILD_CUDA_COMPILER_VERSION[0] != '\0') {
    return GPU_SUITE_BUILD_CUDA_COMPILER_VERSION;
  }
  if (implementation == GPU_SUITE_IMPLEMENTATION_OPENACC) {
    return GPU_SUITE_BUILD_CXX_COMPILER_VERSION;
  }
  return GPU_SUITE_BUILD_C_COMPILER_VERSION;
}

const char *
gpu_suite_build_global_configure_flags(
    gpu_suite_implementation implementation) {
  if (implementation == GPU_SUITE_IMPLEMENTATION_CUDA) {
    return GPU_SUITE_BUILD_CUDA_FLAGS;
  }
  if (implementation == GPU_SUITE_IMPLEMENTATION_OPENACC) {
    return GPU_SUITE_BUILD_CXX_FLAGS;
  }
  return GPU_SUITE_BUILD_C_FLAGS;
}

static bool cpu_benchmark_uses_cxx(const char *benchmark) {
  return benchmark != NULL &&
         (strcmp(benchmark, "curand") == 0 || strcmp(benchmark, "thrust") == 0);
}

const char *gpu_suite_build_benchmark_compiler(
    gpu_suite_implementation implementation, const char *benchmark) {
  if (implementation == GPU_SUITE_IMPLEMENTATION_CPU &&
      cpu_benchmark_uses_cxx(benchmark)) {
    return GPU_SUITE_BUILD_CXX_COMPILER;
  }
  return gpu_suite_build_compiler(implementation);
}

const char *gpu_suite_build_benchmark_compiler_version(
    gpu_suite_implementation implementation, const char *benchmark) {
  if (implementation == GPU_SUITE_IMPLEMENTATION_CPU &&
      cpu_benchmark_uses_cxx(benchmark)) {
    return GPU_SUITE_BUILD_CXX_COMPILER_VERSION;
  }
  return gpu_suite_build_compiler_version(implementation);
}

const char *gpu_suite_build_benchmark_global_configure_flags(
    gpu_suite_implementation implementation, const char *benchmark) {
  if (implementation == GPU_SUITE_IMPLEMENTATION_CPU &&
      cpu_benchmark_uses_cxx(benchmark)) {
    return GPU_SUITE_BUILD_CXX_FLAGS;
  }
  return gpu_suite_build_global_configure_flags(implementation);
}

const char *gpu_suite_build_cuda_toolkit_version(void) {
  return GPU_SUITE_BUILD_CUDA_TOOLKIT_VERSION;
}

const char *gpu_suite_build_cuda_toolkit_root(void) {
  return GPU_SUITE_BUILD_CUDA_TOOLKIT_ROOT;
}

const char *gpu_suite_build_nvhpc_gpu_target(void) {
  return GPU_SUITE_BUILD_NVHPC_GPU_TARGET;
}

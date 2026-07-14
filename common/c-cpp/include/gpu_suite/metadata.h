#ifndef GPU_SUITE_METADATA_H
#define GPU_SUITE_METADATA_H

#include "gpu_suite/cli.h"

#ifdef __cplusplus
extern "C" {
#endif

const char *gpu_suite_build_git_commit(void);
bool gpu_suite_build_git_dirty(void);
const char *gpu_suite_build_type(void);
const char *gpu_suite_build_profile(void);
const char *gpu_suite_build_metadata_sha256(void);
const char *gpu_suite_build_compiler(gpu_suite_implementation implementation);
const char *
gpu_suite_build_compiler_version(gpu_suite_implementation implementation);
const char *
gpu_suite_build_compiler_flags(gpu_suite_implementation implementation);
const char *gpu_suite_build_benchmark_compiler(
    gpu_suite_implementation implementation, const char *benchmark);
const char *gpu_suite_build_benchmark_compiler_version(
    gpu_suite_implementation implementation, const char *benchmark);
const char *gpu_suite_build_benchmark_compiler_flags(
    gpu_suite_implementation implementation, const char *benchmark);
const char *gpu_suite_build_cuda_toolkit_version(void);
const char *gpu_suite_build_cuda_toolkit_root(void);
const char *gpu_suite_build_nvhpc_gpu_target(void);

#ifdef __cplusplus
}
#endif

#endif

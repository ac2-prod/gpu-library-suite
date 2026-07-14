#ifndef GPU_SUITE_TEST_CUDA_RUNTIME_H
#define GPU_SUITE_TEST_CUDA_RUNTIME_H

#include <stddef.h>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

typedef int cudaError_t;

enum {
  cudaSuccess = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  CUDA_R_64F = 3
};

const char *cudaGetErrorString(cudaError_t error);
cudaError_t cudaMalloc(void **pointer, size_t size);
cudaError_t cudaFree(void *pointer);
cudaError_t cudaMemcpy(void *destination, const void *source, size_t size,
                       int kind);
cudaError_t cudaMemset(void *destination, int value, size_t size);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaSetDevice(int device);

#endif

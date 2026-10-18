#ifndef GPU_SUITE_SYNTAX_HIP_RUNTIME_H
#define GPU_SUITE_SYNTAX_HIP_RUNTIME_H
// Declaration-only fixture for host syntax checks. No HIP operations are implemented.
#include <stddef.h>
struct float2 { float x, y; };
enum hipError_t { hipSuccess = 0 };
enum hipMemcpyKind { hipMemcpyHostToDevice, hipMemcpyDeviceToHost };
enum hipMemoryType { hipMemoryTypeHost, hipMemoryTypeDevice };
struct hipPointerAttribute_t { hipMemoryType type; int device; };
struct hipDeviceProp_t { char name[256]; };
hipError_t hipMalloc(void **, size_t);
hipError_t hipFree(void *);
hipError_t hipMemcpy(void *, const void *, size_t, hipMemcpyKind);
hipError_t hipMemset(void *, int, size_t);
hipError_t hipDeviceSynchronize();
hipError_t hipSetDevice(int);
hipError_t hipGetDevice(int *);
hipError_t hipPointerGetAttributes(hipPointerAttribute_t *, const void *);
hipError_t hipGetDeviceProperties(hipDeviceProp_t *, int);
hipError_t hipRuntimeGetVersion(int *);
hipError_t hipDriverGetVersion(int *);
const char *hipGetErrorString(hipError_t);
#endif

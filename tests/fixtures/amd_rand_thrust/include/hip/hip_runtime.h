#ifndef GPU_SUITE_FAKE_RAND_THRUST_HIP_H
#define GPU_SUITE_FAKE_RAND_THRUST_HIP_H
// Host-only test double. It neither loads HIP nor accesses any GPU.
#include <cstdlib>
#include <cstring>
#include <cstdio>
#ifndef __host__
#define __host__
#define __device__
#endif
inline void fixture_event(const char *event) {
  if (std::getenv("GPU_SUITE_FIXTURE_EVENTS")) std::fprintf(stderr, "fixture:%s\n", event);
}
enum hipError_t { hipSuccess = 0, hipErrorInvalidValue = 1 };
enum hipMemcpyKind { hipMemcpyHostToDevice, hipMemcpyDeviceToHost };
enum hipMemoryType { hipMemoryTypeHost, hipMemoryTypeDevice };
struct hipPointerAttribute_t { hipMemoryType type; int device; };
struct hipDeviceProp_t { char name[256]; };
inline hipError_t hipMalloc(void **p, std::size_t n) {
  fixture_event("allocate"); *p = std::malloc(n);
  return *p ? hipSuccess : hipErrorInvalidValue;
}
inline hipError_t hipFree(void *p) { fixture_event("free"); std::free(p); return hipSuccess; }
inline hipError_t hipMemcpy(void *dst, const void *src, std::size_t n, hipMemcpyKind) {
  fixture_event("copy"); std::memcpy(dst, src, n); return hipSuccess;
}
inline hipError_t hipDeviceSynchronize() { fixture_event("sync"); return hipSuccess; }
inline hipError_t hipSetDevice(int device) { return device == 0 ? hipSuccess : hipErrorInvalidValue; }
inline hipError_t hipPointerGetAttributes(hipPointerAttribute_t *p, const void *) {
  p->type = hipMemoryTypeDevice; p->device = std::getenv("GPU_SUITE_FIXTURE_WRONG_DEVICE") ? 1 : 0;
  return hipSuccess;
}
inline hipError_t hipGetDeviceProperties(hipDeviceProp_t *p, int) {
  std::strcpy(p->name, "HOST FIXTURE, NOT A GPU"); return hipSuccess;
}
inline hipError_t hipRuntimeGetVersion(int *p) { *p = 70200000; return hipSuccess; }
inline hipError_t hipDriverGetVersion(int *p) { *p = 70200000; return hipSuccess; }
inline const char *hipGetErrorString(hipError_t) { return "host HIP test double"; }
#endif

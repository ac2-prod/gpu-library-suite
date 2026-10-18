#ifndef GPU_SUITE_TEST_SOLVER_HIP_H
#define GPU_SUITE_TEST_SOLVER_HIP_H
// Host-only fake: validates orchestration and errors, never GPU/offload support.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
inline bool fixture_mode(const char *name) {
  const char *mode = std::getenv("GPU_SUITE_SOLVER_FAULT");
  return mode && std::strcmp(mode, name) == 0;
}
inline void fixture_event(const char *event) {
  if (std::getenv("GPU_SUITE_SOLVER_TRACE")) std::fprintf(stderr, "fixture:%s\n", event);
}
inline std::set<void *> &fixture_allocations() { static std::set<void *> values; return values; }
enum hipError_t { hipSuccess = 0, hipErrorUnknown = 1 };
enum hipMemcpyKind { hipMemcpyHostToDevice, hipMemcpyDeviceToHost };
enum hipMemoryType { hipMemoryTypeHost, hipMemoryTypeDevice };
struct hipPointerAttribute_t { hipMemoryType type; int device; };
struct hipDeviceProp_t { char name[256]; };
inline hipError_t hipMalloc(void **p, size_t bytes) {
  if (fixture_mode("allocation")) return hipErrorUnknown;
  *p = std::malloc(bytes); if (!*p) return hipErrorUnknown;
  fixture_allocations().insert(*p); fixture_event("allocate"); return hipSuccess;
}
inline hipError_t hipFree(void *p) {
  if (!fixture_allocations().erase(p)) std::abort();
  std::free(p); fixture_event("free"); return hipSuccess;
}
inline hipError_t hipMemcpy(void *dst, const void *src, size_t bytes, hipMemcpyKind) {
  if (fixture_mode("copy")) return hipErrorUnknown;
  std::memcpy(dst, src, bytes); return hipSuccess;
}
inline hipError_t hipMemset(void *dst, int value, size_t bytes) { std::memset(dst, value, bytes); return hipSuccess; }
inline hipError_t hipDeviceSynchronize() { return fixture_mode("synchronize") ? hipErrorUnknown : hipSuccess; }
inline hipError_t hipSetDevice(int d) { return d == 0 ? hipSuccess : hipErrorUnknown; }
inline hipError_t hipGetDevice(int *d) { *d = 0; return hipSuccess; }
inline hipError_t hipPointerGetAttributes(hipPointerAttribute_t *p, const void *) {
  p->type = hipMemoryTypeDevice; p->device = fixture_mode("wrong-device") ? 1 : 0; return hipSuccess;
}
inline hipError_t hipGetDeviceProperties(hipDeviceProp_t *p, int) { std::strcpy(p->name, "synthetic-no-gpu"); return hipSuccess; }
inline hipError_t hipRuntimeGetVersion(int *v) { *v = 1; return hipSuccess; }
inline hipError_t hipDriverGetVersion(int *v) { *v = 1; return hipSuccess; }
inline const char *hipGetErrorString(hipError_t) { return "injected HIP error"; }
#endif

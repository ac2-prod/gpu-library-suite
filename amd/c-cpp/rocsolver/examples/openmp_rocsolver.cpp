#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef _OPENMP
#error "OpenMP target support is required"
#endif
static bool hip_ok(hipError_t status, const char *name) {
  if (status == hipSuccess) return true;
  std::fprintf(stderr, "%s: %s\n", name, hipGetErrorString(status));
  return false;
}
static bool solver_ok(rocblas_status status, const char *name) {
  if (status == rocblas_status_success) return true;
  std::fprintf(stderr, "%s: rocBLAS status %d\n", name, (int)status);
  return false;
}
static bool mapped_device(const void *p, int device) {
  hipPointerAttribute_t attr{};
  return hip_ok(hipPointerGetAttributes(&attr, p), "OpenMP allocation") &&
         attr.type == hipMemoryTypeDevice && attr.device == device;
}
int main() {
  const int n = 1024, nrhs = 16;
  const size_t ac = (size_t)n * n, bc = (size_t)n * nrhs;
  const int device = omp_get_default_device();
  if (device < 0 || device >= omp_get_num_devices()) {
    std::fprintf(stderr, "OpenMP target device unavailable\n");
    return EXIT_FAILURE;
  }
  int initial = 1, hip_device = -1;
#pragma omp target device(device) map(from: initial)
  { initial = omp_is_initial_device(); }
  if (initial || !hip_ok(hipGetDevice(&hip_device), "hipGetDevice")) {
    std::fprintf(stderr, "OpenMP host fallback rejected\n");
    return EXIT_FAILURE;
  }
  std::vector<double> a(ac), b(bc, 2.0 * n);
  std::vector<rocblas_int> pivots(n, 0);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      a[i + (size_t)j * n] = i == j ? n + 1.0 : 1.0;
  double *ap = a.data(), *bp = b.data();
  rocblas_int info_value = 0, *ip = &info_value, *pp = pivots.data();
  rocblas_handle handle = nullptr;
  bool ok = solver_ok(rocblas_create_handle(&handle), "rocblas_create_handle");
  if (ok) {
#pragma omp target data device(device) map(to: ap[0:ac], pp[0:n], ip[0:1]) map(tofrom: bp[0:bc]) use_device_ptr(ap, bp, pp, ip)
    {
      ok = mapped_device(ap, hip_device) && mapped_device(bp, hip_device) &&
           mapped_device(pp, hip_device) && mapped_device(ip, hip_device) &&
           solver_ok(rocsolver_dgetrf(handle, n, n, ap, n, pp, ip), "rocsolver_dgetrf") &&
           hip_ok(hipDeviceSynchronize(), "getrf completion") &&
           hip_ok(hipMemcpy(&info_value, ip, sizeof(info_value), hipMemcpyDeviceToHost), "getrf info");
      if (ok && info_value == 0)
        ok = solver_ok(rocsolver_dgetrs(handle, rocblas_operation_none, n, nrhs,
                                        ap, n, pp, bp, n), "rocsolver_dgetrs") &&
             hip_ok(hipDeviceSynchronize(), "getrs completion");
      else
        ok = false;
      ok = hip_ok(hipDeviceSynchronize(), "mapped cleanup completion") && ok;
    } // Only OpenMP owns these mappings; completion precedes copy-back/unmap.
  }
  double max_error = 0.0;
  for (double x : b) {
    if (!std::isfinite(x)) { ok = false; max_error = INFINITY; }
    else max_error = std::fmax(max_error, std::fabs(x - 1.0));
  }
  std::printf("OpenMP rocSOLVER getrf info=%d, max solution error=%.17g\n", (int)info_value, max_error);
  if (handle) {
    ok = hip_ok(hipDeviceSynchronize(), "cleanup completion") && ok;
    ok = solver_ok(rocblas_destroy_handle(handle), "rocblas_destroy_handle") && ok;
  }
  return ok && max_error <= 1e-12 ? EXIT_SUCCESS : EXIT_FAILURE;
}

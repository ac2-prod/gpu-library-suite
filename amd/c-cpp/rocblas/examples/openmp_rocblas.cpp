#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <omp.h>
#ifndef _OPENMP
#error "OpenMP target compilation is required"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool mapped_device(const void *pointer) {
  hipPointerAttribute_t attributes{};
  int device = -1;
  return hipGetDevice(&device) == hipSuccess &&
         hipPointerGetAttributes(&attributes, pointer) == hipSuccess &&
         attributes.type == hipMemoryTypeDevice && attributes.device == device;
}

static bool prove_offload() {
  const int device = omp_get_default_device();
  if (device < 0 || device >= omp_get_num_devices()) {
    std::fprintf(stderr, "OpenMP target device unavailable\n");
    return false;
  }
  int initial = 1;
#pragma omp target device(device) map(from: initial)
  { initial = omp_is_initial_device(); }
  if (initial) std::fprintf(stderr, "OpenMP host fallback rejected\n");
  return initial == 0;
}

int main() {
  if (hipSetDevice(0) != hipSuccess || !prove_offload()) return EXIT_FAILURE;
  const int m = 1024, n = 1024, k = 1024;
  const double alpha = 1.0, beta = 1.0;
  std::vector<double> a(static_cast<std::size_t>(m) * k, 1.0);
  std::vector<double> b(static_cast<std::size_t>(k) * n, 1.0);
  std::vector<double> c(static_cast<std::size_t>(m) * n, 1.0);
  double *ap = a.data(), *bp = b.data(), *cp = c.data();
  const std::size_t ac = a.size(), bc = b.size(), cc = c.size();
  rocblas_handle handle = nullptr;
  if (rocblas_create_handle(&handle) != rocblas_status_success) {
    std::fprintf(stderr, "rocblas_create_handle failed\n");
    return EXIT_FAILURE;
  }
  bool ok = true;
#pragma omp target data map(to: ap[0 : ac], bp[0 : bc]) map(tofrom: cp[0 : cc])
  {
#pragma omp target data use_device_ptr(ap, bp, cp)
    {
      ok = ok && (mapped_device(ap) && mapped_device(bp) && mapped_device(cp));
      ok = ok && rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k, &alpha, ap, m,
                       bp, k, &beta, cp, m) == rocblas_status_success;
    }
    ok = ok && hipDeviceSynchronize() == hipSuccess;
  }
  ok = rocblas_destroy_handle(handle) == rocblas_status_success && ok;
  double max_error = 0.0;
  for (double value : c)
    max_error = std::isfinite(value) ? std::fmax(max_error, std::fabs(value - (k + 1.0))) : INFINITY;
  std::printf("OpenMP-managed rocBLAS DGEMM complete; max error = %.17g\n",
              max_error);
  return ok && max_error <= 1.0e-10 ? EXIT_SUCCESS : EXIT_FAILURE;
}

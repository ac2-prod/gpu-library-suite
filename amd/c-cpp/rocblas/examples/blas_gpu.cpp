#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  const int m = 1024, n = 1024, k = 1024;
  const double alpha = 1.0, beta = 1.0;
  std::vector<double> a(static_cast<std::size_t>(m) * k, 1.0);
  std::vector<double> b(static_cast<std::size_t>(k) * n, 1.0);
  std::vector<double> c(static_cast<std::size_t>(m) * n, 1.0);
  double *device_a = nullptr, *device_b = nullptr, *device_c = nullptr;
  rocblas_handle handle = nullptr;
  int status = EXIT_FAILURE;
  if (hipMalloc(reinterpret_cast<void **>(&device_a),
                 a.size() * sizeof(double)) != hipSuccess ||
      hipMalloc(reinterpret_cast<void **>(&device_b),
                 b.size() * sizeof(double)) != hipSuccess ||
      hipMalloc(reinterpret_cast<void **>(&device_c),
                 c.size() * sizeof(double)) != hipSuccess) {
    std::fprintf(stderr, "hipMalloc failed\n");
    goto cleanup;
  }
  if (hipMemcpy(device_a, a.data(), a.size() * sizeof(double),
                 hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(device_b, b.data(), b.size() * sizeof(double),
                 hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(device_c, c.data(), c.size() * sizeof(double),
                 hipMemcpyHostToDevice) != hipSuccess) {
    std::fprintf(stderr, "H2D copy failed\n");
    goto cleanup;
  }
  if (rocblas_create_handle(&handle) != rocblas_status_success ||
      rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k, &alpha, device_a,
                  m, device_b, k, &beta, device_c,
                  m) != rocblas_status_success ||
      hipMemcpy(c.data(), device_c, c.size() * sizeof(double),
                 hipMemcpyDeviceToHost) != hipSuccess) {
    std::fprintf(stderr, "rocBLAS DGEMM pipeline failed\n");
    goto cleanup;
  }
  {
    double max_error = 0.0;
    for (double value : c)
      max_error = std::isfinite(value) ? std::fmax(max_error, std::fabs(value - (k + 1.0))) : INFINITY;
    std::printf("rocBLAS DGEMM complete; max error = %.17g\n", max_error);
    status = max_error <= 1.0e-10 ? EXIT_SUCCESS : EXIT_FAILURE;
  }
cleanup:
  if (handle != nullptr)
    if (rocblas_destroy_handle(handle) != rocblas_status_success) status = EXIT_FAILURE;
  if (device_c != nullptr)
    if (hipFree(device_c) != hipSuccess) status = EXIT_FAILURE;
  if (device_b != nullptr)
    if (hipFree(device_b) != hipSuccess) status = EXIT_FAILURE;
  if (device_a != nullptr)
    if (hipFree(device_a) != hipSuccess) status = EXIT_FAILURE;
  return status;
}

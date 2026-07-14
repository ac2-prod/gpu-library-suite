#include <cublas_v2.h>
#include <cuda_runtime.h>

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
  cublasHandle_t handle = nullptr;
  int status = EXIT_FAILURE;
  if (cudaMalloc(reinterpret_cast<void **>(&device_a),
                 a.size() * sizeof(double)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void **>(&device_b),
                 b.size() * sizeof(double)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void **>(&device_c),
                 c.size() * sizeof(double)) != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc failed\n");
    goto cleanup;
  }
  if (cudaMemcpy(device_a, a.data(), a.size() * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(device_b, b.data(), b.size() * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(device_c, c.data(), c.size() * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    std::fprintf(stderr, "H2D copy failed\n");
    goto cleanup;
  }
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS ||
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, device_a,
                  m, device_b, k, &beta, device_c,
                  m) != CUBLAS_STATUS_SUCCESS ||
      cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemcpy(c.data(), device_c, c.size() * sizeof(double),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::fprintf(stderr, "cuBLAS DGEMM pipeline failed\n");
    goto cleanup;
  }
  {
    double max_error = 0.0;
    for (double value : c)
      max_error = std::fmax(max_error, std::fabs(value - (k + 1.0)));
    std::printf("cuBLAS DGEMM complete; max error = %.17g\n", max_error);
    status = max_error <= 1.0e-10 ? EXIT_SUCCESS : EXIT_FAILURE;
  }
cleanup:
  if (handle != nullptr)
    (void)cublasDestroy(handle);
  if (device_c != nullptr)
    (void)cudaFree(device_c);
  if (device_b != nullptr)
    (void)cudaFree(device_b);
  if (device_a != nullptr)
    (void)cudaFree(device_a);
  return status;
}

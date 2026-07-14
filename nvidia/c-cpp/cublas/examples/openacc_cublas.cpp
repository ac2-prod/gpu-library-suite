#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
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
  double *ap = a.data(), *bp = b.data(), *cp = c.data();
  const std::size_t ac = a.size(), bc = b.size(), cc = c.size();
  cublasHandle_t handle = nullptr;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cublasCreate failed\n");
    return EXIT_FAILURE;
  }
  bool ok = true;
#pragma acc data copyin(ap[0 : ac], bp[0 : bc]) copy(cp[0 : cc])
  {
#pragma acc host_data use_device(ap, bp, cp)
    {
      ok = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, ap, m,
                       bp, k, &beta, cp, m) == CUBLAS_STATUS_SUCCESS;
    }
    ok = ok && cudaDeviceSynchronize() == cudaSuccess;
  }
  ok = ok && cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS;
  double max_error = 0.0;
  for (double value : c)
    max_error = std::max(max_error, std::fabs(value - (k + 1.0)));
  std::printf("OpenACC-managed cuBLAS DGEMM complete; max error = %.17g\n",
              max_error);
  return ok && max_error <= 1.0e-10 ? EXIT_SUCCESS : EXIT_FAILURE;
}

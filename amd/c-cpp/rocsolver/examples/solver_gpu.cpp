#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

int main() {
  const int n = 1024, nrhs = 16;
  std::vector<double> a((size_t)n * n), b((size_t)n * nrhs, 2.0 * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      a[i + (size_t)j * n] = i == j ? n + 1.0 : 1.0;
  double *da = nullptr, *db = nullptr;
  rocblas_int *piv = nullptr, *info = nullptr, host_info = 0;
  rocblas_handle handle = nullptr;
  bool ok = solver_ok(rocblas_create_handle(&handle), "rocblas_create_handle") &&
      hip_ok(hipMalloc((void **)&da, a.size() * sizeof(double)), "hipMalloc A") &&
      hip_ok(hipMalloc((void **)&db, b.size() * sizeof(double)), "hipMalloc B") &&
      hip_ok(hipMalloc((void **)&piv, n * sizeof(*piv)), "hipMalloc pivots") &&
      hip_ok(hipMalloc((void **)&info, sizeof(*info)), "hipMalloc info") &&
      hip_ok(hipMemcpy(da, a.data(), a.size() * sizeof(double), hipMemcpyHostToDevice), "copy A") &&
      hip_ok(hipMemcpy(db, b.data(), b.size() * sizeof(double), hipMemcpyHostToDevice), "copy B") &&
      solver_ok(rocsolver_dgetrf(handle, n, n, da, n, piv, info), "rocsolver_dgetrf") &&
      hip_ok(hipDeviceSynchronize(), "getrf completion") &&
      hip_ok(hipMemcpy(&host_info, info, sizeof(host_info), hipMemcpyDeviceToHost), "getrf info");
  // A failed factorization must never be passed to getrs. getrs has no info output.
  if (ok && host_info == 0)
    ok = solver_ok(rocsolver_dgetrs(handle, rocblas_operation_none, n, nrhs,
                                    da, n, piv, db, n), "rocsolver_dgetrs") &&
         hip_ok(hipDeviceSynchronize(), "getrs completion") &&
         hip_ok(hipMemcpy(b.data(), db, b.size() * sizeof(double), hipMemcpyDeviceToHost), "copy solution");
  else
    ok = false;
  double max_error = 0.0;
  for (double x : b) {
    if (!std::isfinite(x)) { ok = false; max_error = INFINITY; }
    else max_error = std::fmax(max_error, std::fabs(x - 1.0));
  }
  std::printf("rocSOLVER getrf info=%d, max solution error=%.17g\n", (int)host_info, max_error);
  if (handle) {
    ok = hip_ok(hipDeviceSynchronize(), "cleanup completion") && ok;
    ok = solver_ok(rocblas_destroy_handle(handle), "rocblas_destroy_handle") && ok;
  }
  if (info) ok = hip_ok(hipFree(info), "hipFree info") && ok;
  if (piv) ok = hip_ok(hipFree(piv), "hipFree pivots") && ok;
  if (db) ok = hip_ok(hipFree(db), "hipFree B") && ok;
  if (da) ok = hip_ok(hipFree(da), "hipFree A") && ok;
  return ok && max_error <= 1e-12 ? EXIT_SUCCESS : EXIT_FAILURE;
}

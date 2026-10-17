#ifndef GPU_SUITE_TEST_SOLVER_API_H
#define GPU_SUITE_TEST_SOLVER_API_H
#include <hip/hip_runtime.h>
#include <cmath>
typedef int rocblas_int;
enum rocblas_status { rocblas_status_success, rocblas_status_internal_error,
  rocblas_status_size_increased, rocblas_status_size_unchanged };
enum rocblas_operation { rocblas_operation_none };
struct fixture_handle { bool query = false; bool factored = false; void *workspace = nullptr; };
typedef fixture_handle *rocblas_handle;
inline rocblas_status rocblas_create_handle(rocblas_handle *h) {
  *h = new fixture_handle; fixture_event("create"); return rocblas_status_success;
}
inline rocblas_status rocblas_destroy_handle(rocblas_handle h) {
  delete h; fixture_event("destroy"); return rocblas_status_success;
}
inline rocblas_status rocblas_start_device_memory_size_query(rocblas_handle h) { h->query = true; return rocblas_status_success; }
inline rocblas_status rocblas_stop_device_memory_size_query(rocblas_handle h, size_t *s) { h->query = false; *s = 64; return rocblas_status_success; }
inline rocblas_status rocblas_set_workspace(rocblas_handle h, void *p, size_t) { h->workspace = p; return rocblas_status_success; }
inline rocblas_status rocsolver_dgetrf(rocblas_handle h, int m, int n, double *a, int lda, int *p, int *info) {
  if (h->query) return fixture_mode("query") ? rocblas_status_internal_error : rocblas_status_size_increased;
  fixture_event("getrf");
  if (fixture_mode("getrf")) return rocblas_status_internal_error;
  if (m != n || lda != n) return rocblas_status_internal_error;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      if (a[i + (size_t)j * n] != (i == j ? n + 1.0 : 1.0)) return rocblas_status_internal_error;
  for (int i = 0; i < n; ++i) p[i] = i + 1;
  a[0] = -123; // Destructive factorization: every warm-up/trial must restore A.
  *info = fixture_mode("singular") ? 2 : 0;
  h->factored = true; return rocblas_status_success;
}
inline rocblas_status rocsolver_dgetrs(rocblas_handle h, rocblas_operation, int n, int nrhs,
                                      double *a, int, const int *, double *b, int) {
  if (h->query) return rocblas_status_size_unchanged;
  fixture_event("getrs");
  if (!h->factored || a[0] != -123 || fixture_mode("getrs")) return rocblas_status_internal_error;
  for (size_t i = 0; i < (size_t)n * nrhs; ++i) {
    if (b[i] != 2.0 * n) return rocblas_status_internal_error;
    b[i] = 1.0;
  }
  if (fixture_mode("nonfinite")) b[(size_t)n * nrhs - 1] = NAN;
  if (fixture_mode("wrong-solution")) b[(size_t)n * nrhs - 1] = 1.5;
  h->factored = false; return rocblas_status_success;
}
inline rocblas_status rocsolver_get_version_string(char *p, size_t n) {
  std::snprintf(p, n, "synthetic-rocsolver"); return rocblas_status_success;
}
#endif

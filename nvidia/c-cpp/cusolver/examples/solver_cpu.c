#include GPU_SUITE_LAPACKE_HEADER

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void make_dense_system(int n, int nrhs, double *a, double *b) {
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      a[i + (size_t)j * n] = i == j ? (double)(n + 1) : 1.0;
  for (int rhs = 0; rhs < nrhs; ++rhs)
    for (int i = 0; i < n; ++i)
      b[i + (size_t)rhs * n] = 2.0 * n;
}

int main(void) {
  const int n = 1024, nrhs = 16;
  double *a = (double *)malloc((size_t)n * n * sizeof(*a));
  double *b = (double *)malloc((size_t)n * nrhs * sizeof(*b));
  lapack_int *ipiv = (lapack_int *)malloc((size_t)n * sizeof(*ipiv));
  if (!a || !b || !ipiv) {
    fprintf(stderr, "dense-system allocation failed\n");
    free(ipiv);
    free(b);
    free(a);
    return EXIT_FAILURE;
  }
  make_dense_system(n, nrhs, a, b);
  lapack_int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, a, n, ipiv, b, n);
  double max_error = 0;
  for (size_t i = 0; i < (size_t)n * nrhs; ++i)
    max_error = fmax(max_error, fabs(b[i] - 1.0));
  printf("LAPACKE_dgesv info=%d, max solution error=%.17g\n", (int)info,
         max_error);
  free(ipiv);
  free(b);
  free(a);
  return info == 0 && max_error <= 1e-12 ? EXIT_SUCCESS : EXIT_FAILURE;
}

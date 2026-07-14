#include GPU_SUITE_CBLAS_HEADER

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  const int m = 1024, n = 1024, k = 1024;
  const double alpha = 1.0, beta = 1.0;
  const size_t a_count = (size_t)m * (size_t)k;
  const size_t b_count = (size_t)k * (size_t)n;
  const size_t c_count = (size_t)m * (size_t)n;
  double *a = (double *)malloc(a_count * sizeof(*a));
  double *b = (double *)malloc(b_count * sizeof(*b));
  double *c = (double *)malloc(c_count * sizeof(*c));
  double max_error = 0.0;
  size_t index;
  if (a == NULL || b == NULL || c == NULL) {
    fprintf(stderr, "matrix allocation failed\n");
    free(c);
    free(b);
    free(a);
    return EXIT_FAILURE;
  }
  for (index = 0; index < a_count; ++index)
    a[index] = 1.0;
  for (index = 0; index < b_count; ++index)
    b[index] = 1.0;
  for (index = 0; index < c_count; ++index)
    c[index] = 1.0;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, m,
              b, k, beta, c, m);
  for (index = 0; index < c_count; ++index)
    max_error = fmax(max_error, fabs(c[index] - (double)(k + 1)));
  printf("CBLAS DGEMM complete; max error = %.17g\n", max_error);
  free(c);
  free(b);
  free(a);
  return max_error <= 1.0e-10 ? EXIT_SUCCESS : EXIT_FAILURE;
}

/* Numerical/lifecycle fixture, deliberately distinct from the production library. */
#include "aoclsparse.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef FIXTURE_EXTERNAL_DEPENDENCY
extern int fixture_dependency(void);
#endif
struct fixture_matrix { int n; const int *row, *col; const double *val; int hinted, optimized; };
struct fixture_descriptor { int base, type, diag; };
static int event(const char *name, double y) {
  const char *trace = getenv("GPU_SUITE_AOCL_FIXTURE_TRACE");
  if (trace) {
    FILE *file = fopen(trace, "a");
    if (!file) abort();
    fprintf(file, "%s %.17g\n", name, y);
    fclose(file);
  }
  const char *fail = getenv("GPU_SUITE_AOCL_FIXTURE_FAIL");
  return fail && strcmp(fail, name) == 0;
}
const char *aoclsparse_get_version(void) {
#ifdef FIXTURE_EXTERNAL_DEPENDENCY
  if (fixture_dependency() != 7) abort();
#endif
  const char *threads = getenv("GPU_SUITE_AOCL_FIXTURE_EXPECT_THREADS");
  if (threads && (!getenv("AOCLSPARSE_NUM_THREADS") || !getenv("OMP_NUM_THREADS") ||
      !getenv("OMP_DYNAMIC") || strcmp(threads, getenv("AOCLSPARSE_NUM_THREADS")) ||
      strcmp(threads, getenv("OMP_NUM_THREADS")) || strcmp("FALSE", getenv("OMP_DYNAMIC"))))
    abort();
  return event("version", 0) ? "" : "AOCL-Sparse 5.1-fixture";
}
aoclsparse_status aoclsparse_create_mat_descr(aoclsparse_mat_descr *descr) {
  if (event("descriptor", 0)) return aoclsparse_status_internal_error;
  *descr = calloc(1, sizeof(**descr));
  return *descr ? aoclsparse_status_success : aoclsparse_status_internal_error;
}
aoclsparse_status aoclsparse_set_mat_index_base(aoclsparse_mat_descr descr, aoclsparse_index_base base) {
  if (!descr || base || event("base", 0)) return aoclsparse_status_internal_error;
  descr->base = 1; return aoclsparse_status_success;
}
aoclsparse_status aoclsparse_set_mat_type(aoclsparse_mat_descr descr, aoclsparse_matrix_type type) {
  if (!descr || type || event("type", 0)) return aoclsparse_status_internal_error;
  descr->type = 1; return aoclsparse_status_success;
}
aoclsparse_status aoclsparse_set_mat_diag_type(aoclsparse_mat_descr descr, aoclsparse_diag_type diag) {
  if (!descr || diag || event("diag", 0)) return aoclsparse_status_internal_error;
  descr->diag = 1; return aoclsparse_status_success;
}
aoclsparse_status aoclsparse_create_dcsr(aoclsparse_matrix *matrix, aoclsparse_index_base base,
    aoclsparse_int m, aoclsparse_int n, aoclsparse_int nnz,
    aoclsparse_int *row, aoclsparse_int *col, double *val) {
  if (event("create", 0) || base || m != n || n < 1 || row[0] || row[n] != nnz)
    return aoclsparse_status_internal_error;
  for (int i=0; i<n; ++i) for (int j=row[i]; j<row[i+1]; ++j)
    if (col[j] < 0 || col[j] >= n || (j>row[i] && col[j-1]>=col[j]))
      return aoclsparse_status_internal_error;
  *matrix = malloc(sizeof(**matrix));
  if (!*matrix) return aoclsparse_status_internal_error;
  **matrix = (struct fixture_matrix){n, row, col, val, 0, 0};
  return aoclsparse_status_success;
}
aoclsparse_status aoclsparse_set_mv_hint(aoclsparse_matrix matrix,
    aoclsparse_operation op, const aoclsparse_mat_descr descr,
    aoclsparse_int expected_no_of_calls) {
  if (!matrix || !descr || op || expected_no_of_calls != 1 || event("hint", 0))
    return aoclsparse_status_internal_error;
  matrix->hinted = 1;
  return aoclsparse_status_success;
}
aoclsparse_status aoclsparse_optimize(aoclsparse_matrix matrix) {
  if (!matrix || !matrix->hinted || event("optimize", 0))
    return aoclsparse_status_internal_error;
  matrix->optimized = 1;
  return aoclsparse_status_success;
}
aoclsparse_status aoclsparse_dmv(aoclsparse_operation op, const double *alpha,
    aoclsparse_matrix matrix, const aoclsparse_mat_descr descr,
    const double *x, const double *beta, double *y) {
  if (!matrix || !matrix->optimized || !descr || !descr->base || !descr->type || !descr->diag || op || event("mv", y[0]))
    return aoclsparse_status_internal_error;
  for (int i=0; i<matrix->n; ++i) {
    double sum=0;
    for (int j=matrix->row[i]; j<matrix->row[i+1]; ++j) sum += matrix->val[j] * x[matrix->col[j]];
    y[i] = *alpha * sum + *beta * y[i];
  }
  if (getenv("GPU_SUITE_TEST_NONFINITE")) y[0] = NAN;
  return aoclsparse_status_success;
}
aoclsparse_status aoclsparse_destroy(aoclsparse_matrix *matrix) {
  int fail = event("destroy", 0);
  free(*matrix); *matrix = NULL;
  return fail ? aoclsparse_status_internal_error : aoclsparse_status_success;
}
aoclsparse_status aoclsparse_destroy_mat_descr(aoclsparse_mat_descr descr) {
  int fail = event("destroy_descriptor", 0);
  free(descr);
  return fail ? aoclsparse_status_internal_error : aoclsparse_status_success;
}

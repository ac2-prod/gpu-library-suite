/* Test-only GNU Fortran ABI adapters to the existing controlled CPU providers.
 * This tests caller data/order/error handling, not the oneMKL binary interface. */
#include "cblas.h"
#include "lapacke.h"
#include <stdio.h>
#include <stdlib.h>

void mkl_set_num_threads(int threads) { (void)threads; }
void mkl_get_version_string(char *text, int length) {
  (void)snprintf(text, (size_t)length, "SYNTHETIC Fortran CPU test provider");
}
void dgemm_(const char *ta, const char *tb, const int *m, const int *n,
            const int *k, const double *alpha, const double *a, const int *lda,
            const double *b, const int *ldb, const double *beta, double *c,
            const int *ldc) {
  if (*ta != 'N' || *tb != 'N') abort();
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, *m, *n, *k,
              *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}
void dgetrf_(const int *m, const int *n, double *a, const int *lda,
             int *piv, int *info) {
  const char *injected = getenv("GPU_SUITE_TEST_GETRF_INFO");
  *info = injected ? atoi(injected) : LAPACKE_dgetrf(LAPACK_COL_MAJOR,*m,*n,a,*lda,piv);
}
void dgetrs_(const char *trans, const int *n, const int *nrhs, const double *a,
             const int *lda, const int *piv, double *b, const int *ldb, int *info) {
  const char *injected = getenv("GPU_SUITE_TEST_GETRS_INFO");
  *info = injected ? atoi(injected) : LAPACKE_dgetrs(LAPACK_COL_MAJOR,*trans,*n,*nrhs,a,*lda,piv,b,*ldb);
}
void dgesv_(const int *n, const int *nrhs, double *a, const int *lda,
            int *piv, double *b, const int *ldb, int *info) {
  dgetrf_(n,n,a,lda,piv,info);
  if (*info == 0) dgetrs_("N",n,nrhs,a,lda,piv,b,ldb,info);
}

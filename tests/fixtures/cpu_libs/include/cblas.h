#ifndef GPU_SUITE_TEST_CBLAS_H
#define GPU_SUITE_TEST_CBLAS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { CblasRowMajor = 101, CblasColMajor = 102 } CBLAS_ORDER;
typedef enum {
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasConjTrans = 113
} CBLAS_TRANSPOSE;

void cblas_dgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE trans_a,
                 CBLAS_TRANSPOSE trans_b, int m, int n, int k, double alpha,
                 const double *a, int lda, const double *b, int ldb,
                 double beta, double *c, int ldc);

#ifdef __cplusplus
}
#endif

#endif

#ifndef GPU_SUITE_TEST_CUFFT_H
#define GPU_SUITE_TEST_CUFFT_H

typedef int cufftHandle;
typedef int cufftResult;

typedef struct {
  float x;
  float y;
} cufftComplex;
cufftResult cufftGetVersion(int *version);

enum { CUFFT_SUCCESS = 0, CUFFT_C2C = 0x29, CUFFT_FORWARD = -1 };

cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed,
                          int istride, int idist, int *onembed, int ostride,
                          int odist, int type, int batch);
cufftResult cufftExecC2C(cufftHandle plan, cufftComplex *input,
                         cufftComplex *output, int direction);
cufftResult cufftDestroy(cufftHandle plan);

#endif

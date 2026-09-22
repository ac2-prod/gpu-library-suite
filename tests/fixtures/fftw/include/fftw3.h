#ifndef GPU_SUITE_TEST_FFTW3_H
#define GPU_SUITE_TEST_FFTW3_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef float fftwf_complex[2];
typedef struct gpu_suite_fake_fftw_plan *fftwf_plan;

#define FFTW_FORWARD (-1)
#define FFTW_ESTIMATE (1U << 6)

void *fftwf_malloc(size_t size);
void fftwf_free(void *pointer);
fftwf_plan fftwf_plan_many_dft(int rank, const int *n, int howmany,
                               fftwf_complex *input, const int *inembed,
                               int istride, int idist, fftwf_complex *output,
                               const int *onembed, int ostride, int odist,
                               int sign, unsigned flags);
void fftwf_execute(const fftwf_plan plan);
void fftwf_destroy_plan(fftwf_plan plan);
int fftwf_init_threads(void);
void fftwf_plan_with_nthreads(int threads);
void fftwf_cleanup_threads(void);

#ifdef __cplusplus
}
#endif

#endif

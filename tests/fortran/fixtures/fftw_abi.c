/* Minimal test-only additions for the Fortran interface's C bindings. */
#include "fftw3.h"
#include <stddef.h>
const char fftwf_version[] = "SYNTHETIC Fortran FFT test provider";
void *fftwf_alloc_complex(size_t n) { return fftwf_malloc(n * sizeof(fftwf_complex)); }
void fftwf_execute_dft(const fftwf_plan plan, fftwf_complex *input, fftwf_complex *output) {
  (void)input; (void)output; /* Tests reuse the arrays bound into this plan. */
  fftwf_execute(plan);
}

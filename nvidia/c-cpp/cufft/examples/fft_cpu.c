#include <fftw3.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  const int nfft = 1024;
  const int batch = 4096;
  const size_t count = (size_t)nfft * (size_t)batch;
  fftwf_complex *input = NULL;
  fftwf_complex *output = NULL;
  fftwf_plan plan = NULL;
  int length[1] = {nfft};
  size_t index;
  float max_error = 0.0F;
  int status = EXIT_FAILURE;

  input = (fftwf_complex *)fftwf_malloc(sizeof(*input) * count);
  output = (fftwf_complex *)fftwf_malloc(sizeof(*output) * count);
  if (input == NULL || output == NULL) {
    fprintf(stderr, "FFTW allocation failed\n");
    goto cleanup;
  }
  for (index = 0U; index < count; ++index) {
    input[index][0] = 1.0F;
    input[index][1] = 0.0F;
  }

  plan = fftwf_plan_many_dft(1, length, batch, input, NULL, 1, nfft, output,
                             NULL, 1, nfft, FFTW_FORWARD, FFTW_ESTIMATE);
  if (plan == NULL) {
    fprintf(stderr, "fftwf_plan_many_dft failed\n");
    goto cleanup;
  }
  fftwf_execute(plan);

  for (index = 0U; index < count; ++index) {
    const int frequency = (int)(index % (size_t)nfft);
    const float expected = frequency == 0 ? (float)nfft : 0.0F;
    const float real_error = fabsf(output[index][0] - expected);
    const float imag_error = fabsf(output[index][1]);
    if (real_error > max_error) {
      max_error = real_error;
    }
    if (imag_error > max_error) {
      max_error = imag_error;
    }
  }
  printf("FFTW batched C2C forward transform complete; max error = %.9g\n",
         (double)max_error);
  status = max_error <= 1.0e-4F ? EXIT_SUCCESS : EXIT_FAILURE;

cleanup:
  if (plan != NULL) {
    fftwf_destroy_plan(plan);
  }
  fftwf_free(output);
  fftwf_free(input);
  return status;
}

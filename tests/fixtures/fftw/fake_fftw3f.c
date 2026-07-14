#include "fftw3.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int fixture_nonfinite(float *value) {
  const char *requested = getenv("GPU_SUITE_TEST_NONFINITE");
  if (requested == NULL)
    return 0;
  if (strcmp(requested, "nan") == 0)
    *value = NAN;
  else if (strcmp(requested, "inf") == 0)
    *value = INFINITY;
  else if (strcmp(requested, "-inf") == 0)
    *value = -INFINITY;
  else
    return 0;
  return 1;
}

struct gpu_suite_fake_fftw_plan {
  int n;
  int batch;
  int input_stride;
  int input_distance;
  int output_stride;
  int output_distance;
  int sign;
  fftwf_complex *input;
  fftwf_complex *output;
};

void *fftwf_malloc(size_t size) { return malloc(size); }

void fftwf_free(void *pointer) { free(pointer); }

fftwf_plan fftwf_plan_many_dft(int rank, const int *n, int howmany,
                               fftwf_complex *input, const int *inembed,
                               int istride, int idist, fftwf_complex *output,
                               const int *onembed, int ostride, int odist,
                               int sign, unsigned flags) {
  struct gpu_suite_fake_fftw_plan *plan;
  (void)inembed;
  (void)onembed;
  (void)flags;
  if (rank != 1 || n == NULL || n[0] <= 0 || howmany <= 0 || input == NULL ||
      output == NULL || sign != FFTW_FORWARD) {
    return NULL;
  }
  plan = (struct gpu_suite_fake_fftw_plan *)malloc(sizeof(*plan));
  if (plan == NULL) {
    return NULL;
  }
  plan->n = n[0];
  plan->batch = howmany;
  plan->input_stride = istride;
  plan->input_distance = idist;
  plan->output_stride = ostride;
  plan->output_distance = odist;
  plan->sign = sign;
  plan->input = input;
  plan->output = output;
  return plan;
}

void fftwf_execute(const fftwf_plan plan) {
  int transform;
  if (plan == NULL) {
    return;
  }
  for (transform = 0; transform < plan->batch; ++transform) {
    double real_sum = 0.0;
    double imag_sum = 0.0;
    int sample;
    int frequency;
    for (sample = 0; sample < plan->n; ++sample) {
      const int input_index =
          transform * plan->input_distance + sample * plan->input_stride;
      real_sum += (double)plan->input[input_index][0];
      imag_sum += (double)plan->input[input_index][1];
    }
    for (frequency = 0; frequency < plan->n; ++frequency) {
      const int output_index =
          transform * plan->output_distance + frequency * plan->output_stride;
      plan->output[output_index][0] = frequency == 0 ? (float)real_sum : 0.0F;
      plan->output[output_index][1] = frequency == 0 ? (float)imag_sum : 0.0F;
    }
  }
  if (plan->batch > 0 && plan->n > 1)
    (void)fixture_nonfinite(&plan->output[plan->output_stride][0]);
}

void fftwf_destroy_plan(fftwf_plan plan) { free(plan); }

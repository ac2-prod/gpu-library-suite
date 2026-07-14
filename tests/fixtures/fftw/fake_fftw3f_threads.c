#include "fftw3.h"

static int selected_threads = 1;

int fftwf_init_threads(void) {
  selected_threads = 1;
  return 1;
}

void fftwf_plan_with_nthreads(int threads) { selected_threads = threads; }

void fftwf_cleanup_threads(void) { selected_threads = 1; }

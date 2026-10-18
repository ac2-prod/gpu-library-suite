#ifndef GPU_SUITE_FAKE_ROCRAND_H
#define GPU_SUITE_FAKE_ROCRAND_H
#include <hip/hip_runtime.h>
#include <limits>
#include <random>
enum rocrand_status { ROCRAND_STATUS_SUCCESS = 0, ROCRAND_STATUS_INTERNAL_ERROR = 1 };
enum rocrand_rng_type { ROCRAND_RNG_PSEUDO_DEFAULT };
enum rocrand_ordering { ROCRAND_ORDERING_PSEUDO_DEFAULT };
struct fixture_generator {
  std::mt19937_64 engine;
  unsigned long long seed = 0, offset = 0;
  bool initialized = false;
};
using rocrand_generator = fixture_generator *;
inline rocrand_status rocrand_create_generator(rocrand_generator *p, rocrand_rng_type) {
  fixture_event("create_generator"); *p = new fixture_generator; return ROCRAND_STATUS_SUCCESS;
}
inline rocrand_status rocrand_destroy_generator(rocrand_generator p) {
  fixture_event("destroy_generator"); delete p; return ROCRAND_STATUS_SUCCESS;
}
inline rocrand_status rocrand_set_seed(rocrand_generator p, unsigned long long seed) {
  fixture_event("seed"); p->seed = seed; p->initialized = false; return ROCRAND_STATUS_SUCCESS;
}
inline rocrand_status rocrand_set_offset(rocrand_generator p, unsigned long long offset) {
  fixture_event("offset"); p->offset = offset; p->initialized = false; return ROCRAND_STATUS_SUCCESS;
}
inline rocrand_status rocrand_set_ordering(rocrand_generator, rocrand_ordering) {
  return ROCRAND_STATUS_SUCCESS;
}
inline rocrand_status rocrand_initialize_generator(rocrand_generator p) {
  fixture_event("initialize"); p->engine.seed(p->seed); p->engine.discard(p->offset);
  p->initialized = true; return ROCRAND_STATUS_SUCCESS;
}
inline rocrand_status rocrand_generate_uniform_double(rocrand_generator p, double *out, std::size_t n) {
  if (!p->initialized) { fixture_event("lazy_initialize"); rocrand_initialize_generator(p); }
  fixture_event("generate");
  if (std::getenv("GPU_SUITE_FIXTURE_GENERATION_FAIL")) return ROCRAND_STATUS_INTERNAL_ERROR;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (std::size_t i = 0; i < n; ++i) out[i] = 1.0 - distribution(p->engine);
  if (std::getenv("GPU_SUITE_FIXTURE_NONFINITE")) out[0] = std::numeric_limits<double>::quiet_NaN();
  if (std::getenv("GPU_SUITE_FIXTURE_BAD_RANGE")) out[0] = 2.0;
  return ROCRAND_STATUS_SUCCESS;
}
inline rocrand_status rocrand_get_version(int *p) { *p = 400100; return ROCRAND_STATUS_SUCCESS; }
#endif

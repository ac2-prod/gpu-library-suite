#include "gpu_suite/clock.h"

#include "test_support.h"
#include <math.h>
#include <string.h>

static int is_digit(char value) { return value >= '0' && value <= '9'; }

int main(void) {
  char error[128] = {0};
  char timestamp[GPU_SUITE_TIMESTAMP_CAPACITY] = {0};
  struct timespec start = {10, 900000000L};
  struct timespec end = {12, 100000000L};
  struct timespec now;
  double resolution = 0.0;
  size_t index;
  const size_t digits[] = {0U,  1U,  2U,  3U,  5U,  6U,  8U,  9U, 11U,
                           12U, 14U, 15U, 17U, 18U, 20U, 21U, 22U};

  CHECK(fabs(gpu_suite_clock_elapsed(&start, &end) - 1.2) < 1e-12);
  CHECK(gpu_suite_clock_elapsed(NULL, &end) < 0.0);
  CHECK(gpu_suite_clock_resolution(&resolution, error, sizeof(error)) ==
        GPU_SUITE_OK);
  CHECK(isfinite(resolution));
  CHECK(resolution > 0.0);
  CHECK(gpu_suite_clock_now(&now, error, sizeof(error)) == GPU_SUITE_OK);
  CHECK(gpu_suite_utc_timestamp(timestamp, error, sizeof(error)) ==
        GPU_SUITE_OK);
  CHECK(strlen(timestamp) == 24U);
  CHECK(timestamp[4] == '-' && timestamp[7] == '-' && timestamp[10] == 'T');
  CHECK(timestamp[13] == ':' && timestamp[16] == ':' && timestamp[19] == '.' &&
        timestamp[23] == 'Z');
  for (index = 0U; index < sizeof(digits) / sizeof(digits[0]); ++index) {
    CHECK(is_digit(timestamp[digits[index]]));
  }
  CHECK(gpu_suite_measurement_start(timestamp, &start, error,
                                    sizeof(error)) == GPU_SUITE_OK);
  CHECK(gpu_suite_measurement_end(&end, timestamp, error, sizeof(error)) ==
        GPU_SUITE_OK);
  CHECK(gpu_suite_clock_elapsed(&start, &end) >= 0.0);
  return 0;
}

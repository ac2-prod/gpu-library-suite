#include "gpu_suite/clock.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

static void set_error(char *error, size_t error_size, const char *message) {
  if (error != NULL && error_size > 0U) {
    (void)snprintf(error, error_size, "%s", message);
  }
}

int gpu_suite_clock_resolution(double *resolution_sec, char *error,
                               size_t error_size) {
  struct timespec resolution;

  if (resolution_sec == NULL) {
    set_error(error, error_size, "resolution output is null");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (clock_getres(CLOCK_MONOTONIC, &resolution) != 0) {
    set_error(error, error_size, strerror(errno));
    return GPU_SUITE_ERROR_IO;
  }
  *resolution_sec =
      (double)resolution.tv_sec + (double)resolution.tv_nsec / 1000000000.0;
  return GPU_SUITE_OK;
}

int gpu_suite_clock_now(struct timespec *value, char *error,
                        size_t error_size) {
  if (value == NULL) {
    set_error(error, error_size, "clock output is null");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (clock_gettime(CLOCK_MONOTONIC, value) != 0) {
    set_error(error, error_size, strerror(errno));
    return GPU_SUITE_ERROR_IO;
  }
  return GPU_SUITE_OK;
}

double gpu_suite_clock_elapsed(const struct timespec *start,
                               const struct timespec *end) {
  time_t seconds;
  long nanoseconds;

  if (start == NULL || end == NULL) {
    return -1.0;
  }
  seconds = end->tv_sec - start->tv_sec;
  nanoseconds = end->tv_nsec - start->tv_nsec;
  if (nanoseconds < 0L) {
    --seconds;
    nanoseconds += 1000000000L;
  }
  return (double)seconds + (double)nanoseconds / 1000000000.0;
}

int gpu_suite_utc_timestamp(char output[GPU_SUITE_TIMESTAMP_CAPACITY],
                            char *error, size_t error_size) {
  struct timespec now;
  struct tm utc;
  int written;

  if (output == NULL) {
    set_error(error, error_size, "timestamp output is null");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (clock_gettime(CLOCK_REALTIME, &now) != 0) {
    set_error(error, error_size, strerror(errno));
    return GPU_SUITE_ERROR_IO;
  }
  if (gmtime_r(&now.tv_sec, &utc) == NULL) {
    set_error(error, error_size, "gmtime_r failed");
    return GPU_SUITE_ERROR_IO;
  }
  written = snprintf(output, GPU_SUITE_TIMESTAMP_CAPACITY,
                     "%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ", utc.tm_year + 1900,
                     utc.tm_mon + 1, utc.tm_mday, utc.tm_hour, utc.tm_min,
                     utc.tm_sec, now.tv_nsec / 1000000L);
  if (written != GPU_SUITE_TIMESTAMP_CAPACITY - 1) {
    set_error(error, error_size, "timestamp formatting failed");
    return GPU_SUITE_ERROR_FORMAT;
  }
  return GPU_SUITE_OK;
}

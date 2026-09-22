#include "gpu_suite/cli.h"

#include "gpu_suite/json.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
  OPTION_SIZE,
  OPTION_WARMUP,
  OPTION_REPEAT,
  OPTION_TRIALS,
  OPTION_SCOPE,
  OPTION_VERIFY,
  OPTION_OUTPUT,
  OPTION_FORMAT,
  OPTION_DEVICE,
  OPTION_RUN_ID,
  OPTION_SYSTEM_LABEL,
  OPTION_NODE_INDEX,
  OPTION_WAVE,
  OPTION_SEED,
  OPTION_CPU_THREADS,
  OPTION_CPU_THREADS_EFFECTIVE,
  OPTION_CPU_BACKEND_ROLE,
  OPTION_SERIES_ROLE,
  OPTION_CPU_PARALLELISM,
  OPTION_IMPLEMENTATION_ORDER,
  OPTION_ABS_TOLERANCE,
  OPTION_REL_TOLERANCE,
  OPTION_SIGMA_MULTIPLIER,
  OPTION_EXPECTED_MEAN,
  OPTION_EXPECTED_SECOND_CENTRAL_MOMENT,
  OPTION_BATCH,
  OPTION_TRANSFORM,
  OPTION_M,
  OPTION_N,
  OPTION_K,
  OPTION_ALPHA,
  OPTION_BETA,
  OPTION_NX,
  OPTION_NY,
  OPTION_NRHS,
  OPTION_CPU_BACKEND,
  OPTION_GENERATOR,
  OPTION_DISTRIBUTION,
  OPTION_OFFSET,
  OPTION_ORDER,
  OPTION_OPERATION,
  OPTION_COUNT,
  OPTION_UNKNOWN = -1
} option_id;

typedef struct {
  const char *name;
  option_id id;
} option_definition;

static const option_definition OPTION_DEFINITIONS[] = {
    {"--size", OPTION_SIZE},
    {"--warmup", OPTION_WARMUP},
    {"--repeat", OPTION_REPEAT},
    {"--trials", OPTION_TRIALS},
    {"--scope", OPTION_SCOPE},
    {"--verify", OPTION_VERIFY},
    {"--output", OPTION_OUTPUT},
    {"--format", OPTION_FORMAT},
    {"--device", OPTION_DEVICE},
    {"--run-id", OPTION_RUN_ID},
    {"--system-label", OPTION_SYSTEM_LABEL},
    {"--node-index", OPTION_NODE_INDEX},
    {"--wave", OPTION_WAVE},
    {"--seed", OPTION_SEED},
    {"--cpu-threads", OPTION_CPU_THREADS},
    {"--cpu-threads-effective", OPTION_CPU_THREADS_EFFECTIVE},
    {"--cpu-backend-role", OPTION_CPU_BACKEND_ROLE},
    {"--series-role", OPTION_SERIES_ROLE},
    {"--cpu-parallelism", OPTION_CPU_PARALLELISM},
    {"--implementation-order", OPTION_IMPLEMENTATION_ORDER},
    {"--abs-tolerance", OPTION_ABS_TOLERANCE},
    {"--rel-tolerance", OPTION_REL_TOLERANCE},
    {"--sigma-multiplier", OPTION_SIGMA_MULTIPLIER},
    {"--expected-mean", OPTION_EXPECTED_MEAN},
    {"--expected-second-central-moment", OPTION_EXPECTED_SECOND_CENTRAL_MOMENT},
    {"--batch", OPTION_BATCH},
    {"--transform", OPTION_TRANSFORM},
    {"--m", OPTION_M},
    {"--n", OPTION_N},
    {"--k", OPTION_K},
    {"--alpha", OPTION_ALPHA},
    {"--beta", OPTION_BETA},
    {"--nx", OPTION_NX},
    {"--ny", OPTION_NY},
    {"--nrhs", OPTION_NRHS},
    {"--cpu-backend", OPTION_CPU_BACKEND},
    {"--generator", OPTION_GENERATOR},
    {"--distribution", OPTION_DISTRIBUTION},
    {"--offset", OPTION_OFFSET},
    {"--order", OPTION_ORDER},
    {"--operation", OPTION_OPERATION},
};

static const char *const IMPLEMENTATION_ORDERS[] = {
    "cpu,cuda,openacc", "cpu,openacc,cuda", "cuda,cpu,openacc",
    "cuda,openacc,cpu", "openacc,cpu,cuda", "openacc,cuda,cpu"};

static void set_error(char *error, size_t error_size, const char *message) {
  if (error != NULL && error_size > 0U) {
    (void)snprintf(error, error_size, "%s", message);
  }
}

static int copy_value(char *destination, size_t capacity, const char *value,
                      char *error, size_t error_size) {
  size_t length;
  if (value == NULL || !gpu_suite_utf8_validate(value)) {
    set_error(error, error_size, "option value is not valid UTF-8");
    return GPU_SUITE_ERROR_INVALID;
  }
  length = strlen(value);
  if (length >= capacity) {
    set_error(error, error_size, "option value is too long");
    return GPU_SUITE_ERROR_INVALID;
  }
  memcpy(destination, value, length + 1U);
  return GPU_SUITE_OK;
}

static int parse_u64(const char *text, uint64_t *result, bool allow_zero,
                     char *error, size_t error_size) {
  unsigned long long parsed;
  char *end = NULL;

  if (text == NULL || text[0] == '\0' || text[0] == '-' || text[0] == '+') {
    set_error(error, error_size, "expected an unsigned decimal integer");
    return GPU_SUITE_ERROR_INVALID;
  }
  errno = 0;
  parsed = strtoull(text, &end, 10);
  if (errno == ERANGE || end == text || *end != '\0' ||
      (!allow_zero && parsed == 0ULL)) {
    set_error(error, error_size, "invalid unsigned decimal integer");
    return GPU_SUITE_ERROR_INVALID;
  }
  *result = (uint64_t)parsed;
  return GPU_SUITE_OK;
}

static int parse_int(const char *text, int *result, bool allow_zero,
                     char *error, size_t error_size) {
  long parsed;
  char *end = NULL;

  if (text == NULL || text[0] == '\0' || text[0] == '+') {
    set_error(error, error_size, "expected a decimal integer");
    return GPU_SUITE_ERROR_INVALID;
  }
  errno = 0;
  parsed = strtol(text, &end, 10);
  if (errno == ERANGE || end == text || *end != '\0' || parsed < 0L ||
      parsed > INT_MAX || (!allow_zero && parsed == 0L)) {
    set_error(error, error_size, "invalid decimal integer");
    return GPU_SUITE_ERROR_INVALID;
  }
  *result = (int)parsed;
  return GPU_SUITE_OK;
}

static int parse_double(const char *text, double *result, char *error,
                        size_t error_size) {
  double parsed;
  char *end = NULL;
  int status = gpu_suite_numeric_locale_initialize(error, error_size);
  if (status != GPU_SUITE_OK) {
    return status;
  }
  if (text == NULL || text[0] == '\0') {
    set_error(error, error_size, "expected a finite floating-point value");
    return GPU_SUITE_ERROR_INVALID;
  }
  errno = 0;
  parsed = strtod(text, &end);
  if (errno == ERANGE || end == text || *end != '\0' || !isfinite(parsed)) {
    set_error(error, error_size, "invalid finite floating-point value");
    return GPU_SUITE_ERROR_INVALID;
  }
  *result = parsed;
  return GPU_SUITE_OK;
}

static option_id find_option(const char *name) {
  size_t index;
  for (index = 0U;
       index < sizeof(OPTION_DEFINITIONS) / sizeof(OPTION_DEFINITIONS[0]);
       ++index) {
    if (strcmp(name, OPTION_DEFINITIONS[index].name) == 0) {
      return OPTION_DEFINITIONS[index].id;
    }
  }
  return OPTION_UNKNOWN;
}

const char *gpu_suite_benchmark_name(gpu_suite_benchmark_kind value) {
  switch (value) {
  case GPU_SUITE_BENCHMARK_CUFFT:
    return "cufft";
  case GPU_SUITE_BENCHMARK_CUBLAS:
    return "cublas";
  case GPU_SUITE_BENCHMARK_CUSPARSE:
    return "cusparse";
  case GPU_SUITE_BENCHMARK_CUSOLVER:
    return "cusolver";
  case GPU_SUITE_BENCHMARK_CURAND:
    return "curand";
  case GPU_SUITE_BENCHMARK_THRUST:
    return "thrust";
  }
  return "unknown";
}

const char *gpu_suite_implementation_name(gpu_suite_implementation value) {
  switch (value) {
  case GPU_SUITE_IMPLEMENTATION_CPU:
    return "cpu";
  case GPU_SUITE_IMPLEMENTATION_CUDA:
    return "cuda";
  case GPU_SUITE_IMPLEMENTATION_OPENACC:
    return "openacc";
  }
  return "unknown";
}

const char *gpu_suite_scope_name(gpu_suite_scope value) {
  switch (value) {
  case GPU_SUITE_SCOPE_COMPUTE:
    return "compute";
  case GPU_SUITE_SCOPE_END_TO_END:
    return "end-to-end";
  }
  return "unknown";
}

const char *gpu_suite_format_name(gpu_suite_output_format value) {
  switch (value) {
  case GPU_SUITE_FORMAT_CSV:
    return "csv";
  case GPU_SUITE_FORMAT_JSONL:
    return "jsonl";
  }
  return "unknown";
}

bool gpu_suite_run_id_validate(const char *value) {
  size_t index;
  size_t length;

  if (value == NULL) {
    return false;
  }
  length = strlen(value);
  if (length == 0U || length > 128U || value[0] == '.' ||
      strstr(value, "..") != NULL) {
    return false;
  }
  for (index = 0U; index < length; ++index) {
    unsigned char character = (unsigned char)value[index];
    bool allowed = (character >= 'A' && character <= 'Z') ||
                   (character >= 'a' && character <= 'z') ||
                   (character >= '0' && character <= '9') || character == '.' ||
                   character == '_' || character == '-';
    if (!allowed || (index == 0U && !(character >= 'A' && character <= 'Z') &&
                     !(character >= 'a' && character <= 'z') &&
                     !(character >= '0' && character <= '9'))) {
      return false;
    }
  }
  return true;
}

bool gpu_suite_implementation_order_validate(const char *value) {
  size_t index;
  if (value == NULL) {
    return false;
  }
  for (index = 0U;
       index < sizeof(IMPLEMENTATION_ORDERS) / sizeof(IMPLEMENTATION_ORDERS[0]);
       ++index) {
    if (strcmp(value, IMPLEMENTATION_ORDERS[index]) == 0) {
      return true;
    }
  }
  return false;
}

void gpu_suite_options_init(gpu_suite_options *options,
                            gpu_suite_benchmark_kind benchmark,
                            gpu_suite_implementation implementation) {
  if (options == NULL) {
    return;
  }
  memset(options, 0, sizeof(*options));
  options->benchmark = benchmark;
  options->implementation = implementation;
  options->warmup = 1;
  options->repeat = 1;
  options->trials = 1;
  options->scope = GPU_SUITE_SCOPE_COMPUTE;
  options->verify = true;
  (void)snprintf(options->output, sizeof(options->output), "-");
  options->format = GPU_SUITE_FORMAT_JSONL;
  options->device = 0;
  (void)snprintf(options->run_id, sizeof(options->run_id), "standalone");
  (void)snprintf(options->system_label, sizeof(options->system_label), "local");
  options->node_index = 0;
  options->wave = 0;
  options->seed = 1234U;
  options->cpu_threads = 1;
  options->cpu_threads_effective = 0;
  (void)snprintf(options->cpu_backend_role,
                 sizeof(options->cpu_backend_role), "production");
  (void)snprintf(options->series_role, sizeof(options->series_role),
                 "primary");
  (void)snprintf(options->cpu_parallelism,
                 sizeof(options->cpu_parallelism), "unknown");
  options->abs_tolerance = 1.0e-12;
  options->rel_tolerance = 1.0e-10;
  options->sigma_multiplier = 6.0;
  options->expected_mean = 0.5;
  options->expected_second_central_moment = 1.0 / 12.0;
  (void)snprintf(options->implementation_order,
                 sizeof(options->implementation_order), "cpu,cuda,openacc");
  options->batch = 1U;
  (void)snprintf(options->transform, sizeof(options->transform), "c2c-forward");
  options->alpha = 1.0;
  options->beta = 0.0;
  options->nrhs = 1U;
  (void)snprintf(options->generator, sizeof(options->generator),
                 "pseudo-default");
  (void)snprintf(options->distribution, sizeof(options->distribution),
                 "uniform-double");
  (void)snprintf(options->order, sizeof(options->order), "default");
  (void)snprintf(options->operation, sizeof(options->operation),
                 "transform-reduce-square-sum");

  switch (benchmark) {
  case GPU_SUITE_BENCHMARK_CUFFT:
    options->abs_tolerance = 1.0e-4;
    options->rel_tolerance = 1.0e-5;
    (void)snprintf(options->cpu_backend, sizeof(options->cpu_backend),
                   "cpu-fftw-threaded");
    break;
  case GPU_SUITE_BENCHMARK_CUBLAS:
  case GPU_SUITE_BENCHMARK_CUSPARSE:
    options->beta = 1.0;
    (void)snprintf(options->cpu_backend, sizeof(options->cpu_backend),
                   "cpu-onemkl");
    break;
  case GPU_SUITE_BENCHMARK_CUSOLVER:
    (void)snprintf(options->cpu_backend, sizeof(options->cpu_backend),
                   "cpu-onemkl");
    break;
  case GPU_SUITE_BENCHMARK_CURAND:
    (void)snprintf(options->cpu_backend, sizeof(options->cpu_backend),
                   "cpu-std-random-serial");
    break;
  case GPU_SUITE_BENCHMARK_THRUST:
    (void)snprintf(options->cpu_backend, sizeof(options->cpu_backend),
                   "cpu-stl-serial");
    break;
  }
}

static int parse_option_value(gpu_suite_options *options, option_id option,
                              const char *value, char *error,
                              size_t error_size) {
  switch (option) {
  case OPTION_SIZE:
    if (parse_u64(value, &options->size, false, error, error_size) !=
        GPU_SUITE_OK) {
      return GPU_SUITE_ERROR_INVALID;
    }
    options->size_set = true;
    return GPU_SUITE_OK;
  case OPTION_WARMUP:
    return parse_int(value, &options->warmup, true, error, error_size);
  case OPTION_REPEAT:
    return parse_int(value, &options->repeat, false, error, error_size);
  case OPTION_TRIALS:
    return parse_int(value, &options->trials, false, error, error_size);
  case OPTION_SCOPE:
    if (strcmp(value, "compute") == 0) {
      options->scope = GPU_SUITE_SCOPE_COMPUTE;
    } else if (strcmp(value, "end-to-end") == 0) {
      options->scope = GPU_SUITE_SCOPE_END_TO_END;
    } else {
      set_error(error, error_size, "--scope must be compute or end-to-end");
      return GPU_SUITE_ERROR_INVALID;
    }
    return GPU_SUITE_OK;
  case OPTION_VERIFY:
    if (strcmp(value, "true") == 0) {
      options->verify = true;
    } else if (strcmp(value, "false") == 0) {
      options->verify = false;
    } else {
      set_error(error, error_size, "--verify must be true or false");
      return GPU_SUITE_ERROR_INVALID;
    }
    return GPU_SUITE_OK;
  case OPTION_OUTPUT:
    return copy_value(options->output, sizeof(options->output), value, error,
                      error_size);
  case OPTION_FORMAT:
    if (strcmp(value, "csv") == 0) {
      options->format = GPU_SUITE_FORMAT_CSV;
    } else if (strcmp(value, "jsonl") == 0) {
      options->format = GPU_SUITE_FORMAT_JSONL;
    } else {
      set_error(error, error_size, "--format must be csv or jsonl");
      return GPU_SUITE_ERROR_INVALID;
    }
    return GPU_SUITE_OK;
  case OPTION_DEVICE:
    return parse_int(value, &options->device, true, error, error_size);
  case OPTION_RUN_ID:
    return copy_value(options->run_id, sizeof(options->run_id), value, error,
                      error_size);
  case OPTION_SYSTEM_LABEL:
    return copy_value(options->system_label, sizeof(options->system_label),
                      value, error, error_size);
  case OPTION_NODE_INDEX:
    return parse_int(value, &options->node_index, true, error, error_size);
  case OPTION_WAVE:
    return parse_int(value, &options->wave, true, error, error_size);
  case OPTION_SEED:
    return parse_u64(value, &options->seed, true, error, error_size);
  case OPTION_CPU_THREADS:
    return parse_int(value, &options->cpu_threads, false, error, error_size);
  case OPTION_CPU_THREADS_EFFECTIVE:
    return parse_int(value, &options->cpu_threads_effective, false, error,
                     error_size);
  case OPTION_CPU_BACKEND_ROLE:
    return copy_value(options->cpu_backend_role,
                      sizeof(options->cpu_backend_role), value, error,
                      error_size);
  case OPTION_SERIES_ROLE:
    return copy_value(options->series_role, sizeof(options->series_role),
                      value, error, error_size);
  case OPTION_CPU_PARALLELISM:
    return copy_value(options->cpu_parallelism,
                      sizeof(options->cpu_parallelism), value, error,
                      error_size);
  case OPTION_IMPLEMENTATION_ORDER:
    return copy_value(options->implementation_order,
                      sizeof(options->implementation_order), value, error,
                      error_size);
  case OPTION_ABS_TOLERANCE:
    return parse_double(value, &options->abs_tolerance, error, error_size);
  case OPTION_REL_TOLERANCE:
    return parse_double(value, &options->rel_tolerance, error, error_size);
  case OPTION_SIGMA_MULTIPLIER:
    return parse_double(value, &options->sigma_multiplier, error, error_size);
  case OPTION_EXPECTED_MEAN:
    return parse_double(value, &options->expected_mean, error, error_size);
  case OPTION_EXPECTED_SECOND_CENTRAL_MOMENT:
    return parse_double(value, &options->expected_second_central_moment, error,
                        error_size);
  case OPTION_BATCH:
    return parse_u64(value, &options->batch, false, error, error_size);
  case OPTION_TRANSFORM:
    return copy_value(options->transform, sizeof(options->transform), value,
                      error, error_size);
  case OPTION_M:
    if (parse_u64(value, &options->m, false, error, error_size) !=
        GPU_SUITE_OK) {
      return GPU_SUITE_ERROR_INVALID;
    }
    options->m_set = true;
    return GPU_SUITE_OK;
  case OPTION_N:
    if (parse_u64(value, &options->n, false, error, error_size) !=
        GPU_SUITE_OK) {
      return GPU_SUITE_ERROR_INVALID;
    }
    options->n_set = true;
    return GPU_SUITE_OK;
  case OPTION_K:
    if (parse_u64(value, &options->k, false, error, error_size) !=
        GPU_SUITE_OK) {
      return GPU_SUITE_ERROR_INVALID;
    }
    options->k_set = true;
    return GPU_SUITE_OK;
  case OPTION_ALPHA:
    return parse_double(value, &options->alpha, error, error_size);
  case OPTION_BETA:
    return parse_double(value, &options->beta, error, error_size);
  case OPTION_NX:
    if (parse_u64(value, &options->nx, false, error, error_size) !=
        GPU_SUITE_OK) {
      return GPU_SUITE_ERROR_INVALID;
    }
    options->nx_set = true;
    return GPU_SUITE_OK;
  case OPTION_NY:
    if (parse_u64(value, &options->ny, false, error, error_size) !=
        GPU_SUITE_OK) {
      return GPU_SUITE_ERROR_INVALID;
    }
    options->ny_set = true;
    return GPU_SUITE_OK;
  case OPTION_NRHS:
    return parse_u64(value, &options->nrhs, false, error, error_size);
  case OPTION_CPU_BACKEND:
    return copy_value(options->cpu_backend, sizeof(options->cpu_backend), value,
                      error, error_size);
  case OPTION_GENERATOR:
    return copy_value(options->generator, sizeof(options->generator), value,
                      error, error_size);
  case OPTION_DISTRIBUTION:
    return copy_value(options->distribution, sizeof(options->distribution),
                      value, error, error_size);
  case OPTION_OFFSET:
    return parse_u64(value, &options->offset, true, error, error_size);
  case OPTION_ORDER:
    return copy_value(options->order, sizeof(options->order), value, error,
                      error_size);
  case OPTION_OPERATION:
    return copy_value(options->operation, sizeof(options->operation), value,
                      error, error_size);
  case OPTION_COUNT:
  case OPTION_UNKNOWN:
    break;
  }
  set_error(error, error_size, "unknown option");
  return GPU_SUITE_ERROR_INVALID;
}

gpu_suite_parse_result gpu_suite_options_parse(gpu_suite_options *options,
                                               int argc, char **argv,
                                               char *error, size_t error_size) {
  uint64_t seen = 0U;
  int index;

  if (options == NULL || argc < 0 || (argc > 0 && argv == NULL)) {
    set_error(error, error_size, "invalid command-line arguments");
    return GPU_SUITE_PARSE_ERROR;
  }
  for (index = 1; index < argc; ++index) {
    option_id option;
    uint64_t bit;
    if (strcmp(argv[index], "--help") == 0) {
      return GPU_SUITE_PARSE_HELP;
    }
    option = find_option(argv[index]);
    if (option == OPTION_UNKNOWN) {
      set_error(error, error_size, "unknown command-line option");
      return GPU_SUITE_PARSE_ERROR;
    }
    bit = UINT64_C(1) << (unsigned)option;
    if ((seen & bit) != 0U) {
      set_error(error, error_size, "duplicate command-line option");
      return GPU_SUITE_PARSE_ERROR;
    }
    seen |= bit;
    if (index + 1 >= argc) {
      set_error(error, error_size, "missing command-line option value");
      return GPU_SUITE_PARSE_ERROR;
    }
    ++index;
    if (parse_option_value(options, option, argv[index], error, error_size) !=
        GPU_SUITE_OK) {
      return GPU_SUITE_PARSE_ERROR;
    }
  }
  if (gpu_suite_options_validate(options, error, error_size) != GPU_SUITE_OK) {
    return GPU_SUITE_PARSE_ERROR;
  }
  return GPU_SUITE_PARSE_OK;
}

static uint64_t integer_square_root(uint64_t value) {
  uint64_t low = 0U;
  uint64_t high = value < UINT64_C(4294967295) ? value : UINT64_C(4294967295);
  uint64_t answer = 0U;
  while (low <= high) {
    uint64_t middle = low + (high - low) / 2U;
    if (middle == 0U || middle <= value / middle) {
      answer = middle;
      low = middle + 1U;
    } else {
      high = middle - 1U;
    }
  }
  return answer;
}

int gpu_suite_options_validate(const gpu_suite_options *options, char *error,
                               size_t error_size) {
  if (options == NULL) {
    set_error(error, error_size, "options are null");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (options->warmup < 0 || options->repeat <= 0 || options->trials <= 0 ||
      options->device < 0 || options->node_index < 0 || options->wave < 0 ||
      options->cpu_threads <= 0 || options->cpu_threads_effective < 0) {
    set_error(error, error_size, "invalid common numeric option");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (!gpu_suite_run_id_validate(options->run_id)) {
    set_error(error, error_size, "invalid --run-id");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (options->system_label[0] == '\0' ||
      !gpu_suite_utf8_validate(options->system_label)) {
    set_error(error, error_size, "invalid --system-label");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (options->output[0] == '\0' || !gpu_suite_utf8_validate(options->output)) {
    set_error(error, error_size, "invalid --output");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (!gpu_suite_implementation_order_validate(options->implementation_order)) {
    set_error(error, error_size, "invalid --implementation-order");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (!isfinite(options->abs_tolerance) || options->abs_tolerance < 0.0 ||
      !isfinite(options->rel_tolerance) || options->rel_tolerance < 0.0 ||
      !isfinite(options->sigma_multiplier) ||
      options->sigma_multiplier <= 0.0 || !isfinite(options->expected_mean) ||
      !isfinite(options->expected_second_central_moment) ||
      options->expected_second_central_moment < 0.0) {
    set_error(error, error_size, "invalid verification operand");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (options->implementation == GPU_SUITE_IMPLEMENTATION_CPU &&
      options->cpu_backend[0] == '\0') {
    set_error(error, error_size, "CPU implementation requires --cpu-backend");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (strcmp(options->series_role, "primary") != 0 &&
      strcmp(options->series_role, "auxiliary") != 0) {
    set_error(error, error_size, "invalid --series-role");
    return GPU_SUITE_ERROR_INVALID;
  }
  if (options->implementation == GPU_SUITE_IMPLEMENTATION_CPU) {
    if (strcmp(options->cpu_backend_role, "production") != 0 &&
        strcmp(options->cpu_backend_role, "reference") != 0) {
      set_error(error, error_size, "invalid --cpu-backend-role");
      return GPU_SUITE_ERROR_INVALID;
    }
    if (strcmp(options->cpu_parallelism, "serial") != 0 &&
        strcmp(options->cpu_parallelism, "threaded") != 0 &&
        strcmp(options->cpu_parallelism, "unknown") != 0) {
      set_error(error, error_size, "invalid --cpu-parallelism");
      return GPU_SUITE_ERROR_INVALID;
    }
  }

  switch (options->benchmark) {
  case GPU_SUITE_BENCHMARK_CUFFT:
    if (!options->size_set || options->size == 0U || options->batch == 0U ||
        strcmp(options->transform, "c2c-forward") != 0) {
      set_error(error, error_size, "invalid cuFFT options");
      return GPU_SUITE_ERROR_INVALID;
    }
    break;
  case GPU_SUITE_BENCHMARK_CUBLAS:
    if (options->size_set &&
        (options->m_set || options->n_set || options->k_set)) {
      set_error(error, error_size,
                "--size is mutually exclusive with --m, --n, and --k");
      return GPU_SUITE_ERROR_INVALID;
    }
    if (!options->size_set &&
        !(options->m_set && options->n_set && options->k_set)) {
      set_error(error, error_size,
                "cuBLAS requires --size or all of --m, --n, and --k");
      return GPU_SUITE_ERROR_INVALID;
    }
    break;
  case GPU_SUITE_BENCHMARK_CUSPARSE:
    if (options->size_set && (options->nx_set || options->ny_set)) {
      set_error(error, error_size,
                "--size is mutually exclusive with --nx and --ny");
      return GPU_SUITE_ERROR_INVALID;
    }
    if (!options->size_set && !(options->nx_set && options->ny_set)) {
      set_error(error, error_size,
                "cuSPARSE requires --size or both --nx and --ny");
      return GPU_SUITE_ERROR_INVALID;
    }
    if (options->size_set) {
      uint64_t root = integer_square_root(options->size);
      if (root == 0U || root * root != options->size) {
        set_error(error, error_size,
                  "cuSPARSE --size must be a positive perfect square");
        return GPU_SUITE_ERROR_INVALID;
      }
    }
    break;
  case GPU_SUITE_BENCHMARK_CUSOLVER:
    if (!options->size_set || options->size == 0U || options->nrhs == 0U) {
      set_error(error, error_size, "invalid cuSOLVER dimensions");
      return GPU_SUITE_ERROR_INVALID;
    }
    if (options->repeat != 1) {
      set_error(error, error_size, "cuSOLVER requires --repeat 1");
      return GPU_SUITE_ERROR_INVALID;
    }
    break;
  case GPU_SUITE_BENCHMARK_CURAND:
    if (!options->size_set || options->size == 0U ||
        strcmp(options->generator, "pseudo-default") != 0 ||
        strcmp(options->distribution, "uniform-double") != 0 ||
        strcmp(options->order, "default") != 0) {
      set_error(error, error_size, "invalid cuRAND options");
      return GPU_SUITE_ERROR_INVALID;
    }
    break;
  case GPU_SUITE_BENCHMARK_THRUST:
    if (!options->size_set || options->size == 0U ||
        strcmp(options->operation, "transform-reduce-square-sum") != 0) {
      set_error(error, error_size, "invalid Thrust options");
      return GPU_SUITE_ERROR_INVALID;
    }
    break;
  }
  return GPU_SUITE_OK;
}

void gpu_suite_options_usage(FILE *stream, const char *program,
                             gpu_suite_benchmark_kind benchmark) {
  if (stream == NULL) {
    return;
  }
  (void)fprintf(stream,
                "Usage: %s --size N [common options] [library options]\n"
                "Benchmark: %s\n"
                "Common: --warmup N --repeat N --trials N "
                "--scope compute|end-to-end\n"
                "        --verify true|false --output PATH|- "
                "--format csv|jsonl --device N\n"
                "        --run-id ID --system-label LABEL --node-index N "
                "--wave N --seed N\n"
                "        --cpu-threads N [--cpu-threads-effective N] "
                "--cpu-backend-role ROLE --series-role ROLE "
                "--cpu-parallelism KIND --implementation-order LIST "
                "--abs-tolerance X --rel-tolerance X --help\n",
                program == NULL ? "benchmark" : program,
                gpu_suite_benchmark_name(benchmark));
}

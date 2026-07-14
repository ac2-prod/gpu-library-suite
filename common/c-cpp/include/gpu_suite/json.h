#ifndef GPU_SUITE_JSON_H
#define GPU_SUITE_JSON_H

#include "gpu_suite/common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpu_suite_json_value gpu_suite_json_value;

gpu_suite_json_value *gpu_suite_json_null(void);
gpu_suite_json_value *gpu_suite_json_bool(bool value);
gpu_suite_json_value *gpu_suite_json_int(int64_t value);
gpu_suite_json_value *gpu_suite_json_double(double value);
gpu_suite_json_value *gpu_suite_json_string(const char *value);
gpu_suite_json_value *gpu_suite_json_array(void);
gpu_suite_json_value *gpu_suite_json_object(void);

int gpu_suite_json_array_append(gpu_suite_json_value *array,
                                gpu_suite_json_value *value);
int gpu_suite_json_object_set(gpu_suite_json_value *object, const char *key,
                              gpu_suite_json_value *value);
bool gpu_suite_json_is_array(const gpu_suite_json_value *value);
bool gpu_suite_json_is_object(const gpu_suite_json_value *value);
bool gpu_suite_json_object_has(const gpu_suite_json_value *object,
                               const char *key);
bool gpu_suite_json_object_value_is_null(const gpu_suite_json_value *object,
                                         const char *key);

int gpu_suite_json_serialize(const gpu_suite_json_value *value, char **output,
                             size_t *length, char *error, size_t error_size);
gpu_suite_json_value *gpu_suite_json_clone(const gpu_suite_json_value *value);
void gpu_suite_json_free(gpu_suite_json_value *value);

int gpu_suite_numeric_locale_initialize(char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif

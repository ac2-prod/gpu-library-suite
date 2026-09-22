#include "gpu_suite/json.h"

#include "gpu_suite/checked.h"

#include <locale.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
  JSON_NULL,
  JSON_BOOL,
  JSON_INT,
  JSON_DOUBLE,
  JSON_STRING,
  JSON_ARRAY,
  JSON_OBJECT
} json_type;

typedef struct {
  char *key;
  struct gpu_suite_json_value *value;
} json_member;

struct gpu_suite_json_value {
  json_type type;
  union {
    bool boolean;
    int64_t integer;
    double number;
    char *string;
    struct {
      struct gpu_suite_json_value **items;
      size_t count;
      size_t capacity;
    } array;
    struct {
      json_member *members;
      size_t count;
      size_t capacity;
    } object;
  } data;
};

typedef struct {
  char *data;
  size_t length;
  size_t capacity;
} json_buffer;

static void set_error(char *error, size_t error_size, const char *message) {
  if (error != NULL && error_size > 0U) {
    (void)snprintf(error, error_size, "%s", message);
  }
}

static char *duplicate_string(const char *value) {
  size_t length;
  char *copy;

  if (value == NULL) {
    return NULL;
  }
  length = strlen(value);
  copy = (char *)malloc(length + 1U);
  if (copy != NULL) {
    memcpy(copy, value, length + 1U);
  }
  return copy;
}

bool gpu_suite_utf8_validate(const char *text) {
  const unsigned char *cursor = (const unsigned char *)text;

  if (text == NULL) {
    return false;
  }
  while (*cursor != 0U) {
    if (*cursor <= 0x7fU) {
      ++cursor;
    } else if (*cursor >= 0xc2U && *cursor <= 0xdfU) {
      if ((cursor[1] & 0xc0U) != 0x80U) {
        return false;
      }
      cursor += 2;
    } else if (*cursor == 0xe0U) {
      if (cursor[1] < 0xa0U || cursor[1] > 0xbfU ||
          (cursor[2] & 0xc0U) != 0x80U) {
        return false;
      }
      cursor += 3;
    } else if ((*cursor >= 0xe1U && *cursor <= 0xecU) ||
               (*cursor >= 0xeeU && *cursor <= 0xefU)) {
      if ((cursor[1] & 0xc0U) != 0x80U || (cursor[2] & 0xc0U) != 0x80U) {
        return false;
      }
      cursor += 3;
    } else if (*cursor == 0xedU) {
      if (cursor[1] < 0x80U || cursor[1] > 0x9fU ||
          (cursor[2] & 0xc0U) != 0x80U) {
        return false;
      }
      cursor += 3;
    } else if (*cursor == 0xf0U) {
      if (cursor[1] < 0x90U || cursor[1] > 0xbfU ||
          (cursor[2] & 0xc0U) != 0x80U || (cursor[3] & 0xc0U) != 0x80U) {
        return false;
      }
      cursor += 4;
    } else if (*cursor >= 0xf1U && *cursor <= 0xf3U) {
      if ((cursor[1] & 0xc0U) != 0x80U || (cursor[2] & 0xc0U) != 0x80U ||
          (cursor[3] & 0xc0U) != 0x80U) {
        return false;
      }
      cursor += 4;
    } else if (*cursor == 0xf4U) {
      if (cursor[1] < 0x80U || cursor[1] > 0x8fU ||
          (cursor[2] & 0xc0U) != 0x80U || (cursor[3] & 0xc0U) != 0x80U) {
        return false;
      }
      cursor += 4;
    } else {
      return false;
    }
  }
  return true;
}

static gpu_suite_json_value *new_value(json_type type) {
  gpu_suite_json_value *value =
      (gpu_suite_json_value *)calloc(1U, sizeof(*value));
  if (value != NULL) {
    value->type = type;
  }
  return value;
}

gpu_suite_json_value *gpu_suite_json_null(void) { return new_value(JSON_NULL); }

gpu_suite_json_value *gpu_suite_json_bool(bool value) {
  gpu_suite_json_value *result = new_value(JSON_BOOL);
  if (result != NULL) {
    result->data.boolean = value;
  }
  return result;
}

gpu_suite_json_value *gpu_suite_json_int(int64_t value) {
  gpu_suite_json_value *result = new_value(JSON_INT);
  if (result != NULL) {
    result->data.integer = value;
  }
  return result;
}

gpu_suite_json_value *gpu_suite_json_double(double value) {
  gpu_suite_json_value *result;
  if (!isfinite(value)) {
    return NULL;
  }
  result = new_value(JSON_DOUBLE);
  if (result != NULL) {
    result->data.number = value;
  }
  return result;
}

gpu_suite_json_value *gpu_suite_json_string(const char *value) {
  gpu_suite_json_value *result;
  if (!gpu_suite_utf8_validate(value)) {
    return NULL;
  }
  result = new_value(JSON_STRING);
  if (result == NULL) {
    return NULL;
  }
  result->data.string = duplicate_string(value);
  if (result->data.string == NULL) {
    free(result);
    return NULL;
  }
  return result;
}

gpu_suite_json_value *gpu_suite_json_array(void) {
  return new_value(JSON_ARRAY);
}

gpu_suite_json_value *gpu_suite_json_object(void) {
  return new_value(JSON_OBJECT);
}

static int grow_array(void **items, size_t item_size, size_t *capacity,
                      size_t required) {
  size_t new_capacity = *capacity == 0U ? 8U : *capacity;
  size_t bytes;
  void *replacement;

  while (new_capacity < required) {
    if (!gpu_suite_checked_mul_size(new_capacity, 2U, &new_capacity)) {
      return GPU_SUITE_ERROR_OVERFLOW;
    }
  }
  if (!gpu_suite_checked_mul_size(new_capacity, item_size, &bytes)) {
    return GPU_SUITE_ERROR_OVERFLOW;
  }
  replacement = realloc(*items, bytes);
  if (replacement == NULL) {
    return GPU_SUITE_ERROR_NOMEM;
  }
  *items = replacement;
  *capacity = new_capacity;
  return GPU_SUITE_OK;
}

int gpu_suite_json_array_append(gpu_suite_json_value *array,
                                gpu_suite_json_value *value) {
  int status;
  if (array == NULL || value == NULL || array->type != JSON_ARRAY) {
    return GPU_SUITE_ERROR_INVALID;
  }
  if (array->data.array.count == array->data.array.capacity) {
    status = grow_array(
        (void **)&array->data.array.items, sizeof(*array->data.array.items),
        &array->data.array.capacity, array->data.array.count + 1U);
    if (status != GPU_SUITE_OK) {
      return status;
    }
  }
  array->data.array.items[array->data.array.count++] = value;
  return GPU_SUITE_OK;
}

int gpu_suite_json_object_set(gpu_suite_json_value *object, const char *key,
                              gpu_suite_json_value *value) {
  size_t index;
  int status;
  char *key_copy;

  if (object == NULL || key == NULL || value == NULL ||
      object->type != JSON_OBJECT || !gpu_suite_utf8_validate(key)) {
    return GPU_SUITE_ERROR_INVALID;
  }
  for (index = 0U; index < object->data.object.count; ++index) {
    if (strcmp(object->data.object.members[index].key, key) == 0) {
      return GPU_SUITE_ERROR_EXISTS;
    }
  }
  key_copy = duplicate_string(key);
  if (key_copy == NULL) {
    return GPU_SUITE_ERROR_NOMEM;
  }
  if (object->data.object.count == object->data.object.capacity) {
    status = grow_array((void **)&object->data.object.members,
                        sizeof(*object->data.object.members),
                        &object->data.object.capacity,
                        object->data.object.count + 1U);
    if (status != GPU_SUITE_OK) {
      free(key_copy);
      return status;
    }
  }
  object->data.object.members[object->data.object.count].key = key_copy;
  object->data.object.members[object->data.object.count].value = value;
  ++object->data.object.count;
  return GPU_SUITE_OK;
}

bool gpu_suite_json_is_array(const gpu_suite_json_value *value) {
  return value != NULL && value->type == JSON_ARRAY;
}

bool gpu_suite_json_is_object(const gpu_suite_json_value *value) {
  return value != NULL && value->type == JSON_OBJECT;
}

bool gpu_suite_json_object_has(const gpu_suite_json_value *object,
                               const char *key) {
  size_t index;
  if (!gpu_suite_json_is_object(object) || key == NULL) {
    return false;
  }
  for (index = 0U; index < object->data.object.count; ++index) {
    if (strcmp(object->data.object.members[index].key, key) == 0) {
      return true;
    }
  }
  return false;
}

bool gpu_suite_json_object_value_is_null(const gpu_suite_json_value *object,
                                         const char *key) {
  size_t index;
  if (!gpu_suite_json_is_object(object) || key == NULL) {
    return false;
  }
  for (index = 0U; index < object->data.object.count; ++index) {
    if (strcmp(object->data.object.members[index].key, key) == 0) {
      return object->data.object.members[index].value->type == JSON_NULL;
    }
  }
  return false;
}

static int buffer_reserve(json_buffer *buffer, size_t additional) {
  size_t required;
  size_t capacity;
  char *replacement;

  if (!gpu_suite_checked_add_size(buffer->length, additional, &required) ||
      !gpu_suite_checked_add_size(required, 1U, &required)) {
    return GPU_SUITE_ERROR_OVERFLOW;
  }
  if (required <= buffer->capacity) {
    return GPU_SUITE_OK;
  }
  capacity = buffer->capacity == 0U ? 128U : buffer->capacity;
  while (capacity < required) {
    if (!gpu_suite_checked_mul_size(capacity, 2U, &capacity)) {
      capacity = required;
      break;
    }
  }
  replacement = (char *)realloc(buffer->data, capacity);
  if (replacement == NULL) {
    return GPU_SUITE_ERROR_NOMEM;
  }
  buffer->data = replacement;
  buffer->capacity = capacity;
  return GPU_SUITE_OK;
}

static int buffer_append(json_buffer *buffer, const char *text, size_t length) {
  int status = buffer_reserve(buffer, length);
  if (status != GPU_SUITE_OK) {
    return status;
  }
  memcpy(buffer->data + buffer->length, text, length);
  buffer->length += length;
  buffer->data[buffer->length] = '\0';
  return GPU_SUITE_OK;
}

static int buffer_character(json_buffer *buffer, char value) {
  return buffer_append(buffer, &value, 1U);
}

static int serialize_string(json_buffer *buffer, const char *value) {
  const unsigned char *cursor = (const unsigned char *)value;
  int status = buffer_character(buffer, '"');
  if (status != GPU_SUITE_OK) {
    return status;
  }
  while (*cursor != 0U) {
    char escaped[7];
    const char *replacement = NULL;
    switch (*cursor) {
    case '"':
      replacement = "\\\"";
      break;
    case '\\':
      replacement = "\\\\";
      break;
    case '\b':
      replacement = "\\b";
      break;
    case '\f':
      replacement = "\\f";
      break;
    case '\n':
      replacement = "\\n";
      break;
    case '\r':
      replacement = "\\r";
      break;
    case '\t':
      replacement = "\\t";
      break;
    default:
      break;
    }
    if (replacement != NULL) {
      status = buffer_append(buffer, replacement, strlen(replacement));
    } else if (*cursor < 0x20U) {
      (void)snprintf(escaped, sizeof(escaped), "\\u%04x", (unsigned)*cursor);
      status = buffer_append(buffer, escaped, 6U);
    } else {
      status = buffer_append(buffer, (const char *)cursor, 1U);
    }
    if (status != GPU_SUITE_OK) {
      return status;
    }
    ++cursor;
  }
  return buffer_character(buffer, '"');
}

static int member_compare(const void *left, const void *right) {
  const json_member *const *left_member = (const json_member *const *)left;
  const json_member *const *right_member = (const json_member *const *)right;
  return strcmp((*left_member)->key, (*right_member)->key);
}

static int serialize_value(json_buffer *buffer,
                           const gpu_suite_json_value *value);

static int serialize_object(json_buffer *buffer,
                            const gpu_suite_json_value *value) {
  json_member **members = NULL;
  size_t index;
  size_t bytes;
  int status = buffer_character(buffer, '{');

  if (status != GPU_SUITE_OK) {
    return status;
  }
  if (value->data.object.count > 0U) {
    if (!gpu_suite_checked_mul_size(value->data.object.count, sizeof(*members),
                                    &bytes)) {
      return GPU_SUITE_ERROR_OVERFLOW;
    }
    members = (json_member **)malloc(bytes);
    if (members == NULL) {
      return GPU_SUITE_ERROR_NOMEM;
    }
    for (index = 0U; index < value->data.object.count; ++index) {
      members[index] = &value->data.object.members[index];
    }
    qsort(members, value->data.object.count, sizeof(*members), member_compare);
  }
  for (index = 0U; index < value->data.object.count; ++index) {
    if (index > 0U &&
        (status = buffer_character(buffer, ',')) != GPU_SUITE_OK) {
      break;
    }
    status = serialize_string(buffer, members[index]->key);
    if (status != GPU_SUITE_OK) {
      break;
    }
    status = buffer_character(buffer, ':');
    if (status != GPU_SUITE_OK) {
      break;
    }
    status = serialize_value(buffer, members[index]->value);
    if (status != GPU_SUITE_OK) {
      break;
    }
  }
  free(members);
  if (status != GPU_SUITE_OK) {
    return status;
  }
  return buffer_character(buffer, '}');
}

static int format_double(double value, char output[32]) {
  char *exponent;
  char *digits;
  size_t prefix_length;
  long exponent_value;
  char *end = NULL;

  if (!isfinite(value)) {
    return GPU_SUITE_ERROR_FORMAT;
  }
  if (value == 0.0) {
    (void)snprintf(output, 32U, "0");
    return GPU_SUITE_OK;
  }
  if (snprintf(output, 32U, "%.17g", value) < 1) {
    return GPU_SUITE_ERROR_FORMAT;
  }
  exponent = strchr(output, 'e');
  if (exponent == NULL) {
    exponent = strchr(output, 'E');
  }
  if (exponent == NULL) {
    return GPU_SUITE_OK;
  }
  *exponent = '\0';
  digits = exponent + 1;
  exponent_value = strtol(digits, &end, 10);
  if (end == digits || *end != '\0') {
    return GPU_SUITE_ERROR_FORMAT;
  }
  prefix_length = strlen(output);
  if (snprintf(output + prefix_length, 32U - prefix_length, "e%ld",
               exponent_value) < 2) {
    return GPU_SUITE_ERROR_FORMAT;
  }
  return GPU_SUITE_OK;
}

static int serialize_value(json_buffer *buffer,
                           const gpu_suite_json_value *value) {
  size_t index;
  int status;
  char number[32];
  int written;

  if (value == NULL) {
    return GPU_SUITE_ERROR_INVALID;
  }
  switch (value->type) {
  case JSON_NULL:
    return buffer_append(buffer, "null", 4U);
  case JSON_BOOL:
    return value->data.boolean ? buffer_append(buffer, "true", 4U)
                               : buffer_append(buffer, "false", 5U);
  case JSON_INT:
    written = snprintf(number, sizeof(number), "%lld",
                       (long long)value->data.integer);
    if (written < 1 || (size_t)written >= sizeof(number)) {
      return GPU_SUITE_ERROR_FORMAT;
    }
    return buffer_append(buffer, number, (size_t)written);
  case JSON_DOUBLE:
    status = format_double(value->data.number, number);
    if (status != GPU_SUITE_OK) {
      return status;
    }
    return buffer_append(buffer, number, strlen(number));
  case JSON_STRING:
    return serialize_string(buffer, value->data.string);
  case JSON_ARRAY:
    status = buffer_character(buffer, '[');
    if (status != GPU_SUITE_OK) {
      return status;
    }
    for (index = 0U; index < value->data.array.count; ++index) {
      if (index > 0U &&
          (status = buffer_character(buffer, ',')) != GPU_SUITE_OK) {
        return status;
      }
      status = serialize_value(buffer, value->data.array.items[index]);
      if (status != GPU_SUITE_OK) {
        return status;
      }
    }
    return buffer_character(buffer, ']');
  case JSON_OBJECT:
    return serialize_object(buffer, value);
  }
  return GPU_SUITE_ERROR_FORMAT;
}

int gpu_suite_numeric_locale_initialize(char *error, size_t error_size) {
  const char *selected = setlocale(LC_NUMERIC, "C");
  if (selected == NULL || strcmp(selected, "C") != 0) {
    set_error(error, error_size, "could not select the C numeric locale");
    return GPU_SUITE_ERROR_FORMAT;
  }
  return GPU_SUITE_OK;
}

int gpu_suite_json_serialize(const gpu_suite_json_value *value, char **output,
                             size_t *length, char *error, size_t error_size) {
  json_buffer buffer = {0};
  int status;

  if (value == NULL || output == NULL || length == NULL) {
    set_error(error, error_size, "invalid JSON serialization argument");
    return GPU_SUITE_ERROR_INVALID;
  }
  *output = NULL;
  *length = 0U;
  status = gpu_suite_numeric_locale_initialize(error, error_size);
  if (status != GPU_SUITE_OK) {
    return status;
  }
  status = serialize_value(&buffer, value);
  if (status != GPU_SUITE_OK) {
    free(buffer.data);
    set_error(error, error_size, "JSON serialization failed");
    return status;
  }
  if (buffer.data == NULL) {
    buffer.data = duplicate_string("");
    if (buffer.data == NULL) {
      set_error(error, error_size, "out of memory");
      return GPU_SUITE_ERROR_NOMEM;
    }
  }
  *output = buffer.data;
  *length = buffer.length;
  return GPU_SUITE_OK;
}

gpu_suite_json_value *gpu_suite_json_clone(const gpu_suite_json_value *value) {
  gpu_suite_json_value *copy;
  gpu_suite_json_value *child;
  size_t index;
  int status;

  if (value == NULL) {
    return NULL;
  }
  switch (value->type) {
  case JSON_NULL:
    return gpu_suite_json_null();
  case JSON_BOOL:
    return gpu_suite_json_bool(value->data.boolean);
  case JSON_INT:
    return gpu_suite_json_int(value->data.integer);
  case JSON_DOUBLE:
    return gpu_suite_json_double(value->data.number);
  case JSON_STRING:
    return gpu_suite_json_string(value->data.string);
  case JSON_ARRAY:
    copy = gpu_suite_json_array();
    if (copy == NULL) {
      return NULL;
    }
    for (index = 0U; index < value->data.array.count; ++index) {
      child = gpu_suite_json_clone(value->data.array.items[index]);
      if (child == NULL) {
        gpu_suite_json_free(copy);
        return NULL;
      }
      status = gpu_suite_json_array_append(copy, child);
      if (status != GPU_SUITE_OK) {
        gpu_suite_json_free(child);
        gpu_suite_json_free(copy);
        return NULL;
      }
    }
    return copy;
  case JSON_OBJECT:
    copy = gpu_suite_json_object();
    if (copy == NULL) {
      return NULL;
    }
    for (index = 0U; index < value->data.object.count; ++index) {
      child = gpu_suite_json_clone(value->data.object.members[index].value);
      if (child == NULL) {
        gpu_suite_json_free(copy);
        return NULL;
      }
      status = gpu_suite_json_object_set(
          copy, value->data.object.members[index].key, child);
      if (status != GPU_SUITE_OK) {
        gpu_suite_json_free(child);
        gpu_suite_json_free(copy);
        return NULL;
      }
    }
    return copy;
  }
  return NULL;
}

void gpu_suite_json_free(gpu_suite_json_value *value) {
  size_t index;
  if (value == NULL) {
    return;
  }
  switch (value->type) {
  case JSON_STRING:
    free(value->data.string);
    break;
  case JSON_ARRAY:
    for (index = 0U; index < value->data.array.count; ++index) {
      gpu_suite_json_free(value->data.array.items[index]);
    }
    free(value->data.array.items);
    break;
  case JSON_OBJECT:
    for (index = 0U; index < value->data.object.count; ++index) {
      free(value->data.object.members[index].key);
      gpu_suite_json_free(value->data.object.members[index].value);
    }
    free(value->data.object.members);
    break;
  case JSON_NULL:
  case JSON_BOOL:
  case JSON_INT:
  case JSON_DOUBLE:
    break;
  }
  free(value);
}

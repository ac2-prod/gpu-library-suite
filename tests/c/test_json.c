#include "gpu_suite/json.h"

#include "test_support.h"
#include <locale.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static char *serialize(gpu_suite_json_value *value) {
  char error[128] = {0};
  char *output = NULL;
  size_t length = 0U;
  CHECK(gpu_suite_json_serialize(value, &output, &length, error,
                                 sizeof(error)) == GPU_SUITE_OK);
  CHECK(output != NULL);
  CHECK(strlen(output) == length);
  return output;
}

int main(void) {
  gpu_suite_json_value *object = gpu_suite_json_object();
  gpu_suite_json_value *array = gpu_suite_json_array();
  gpu_suite_json_value *duplicate = gpu_suite_json_int(2);
  gpu_suite_json_value *clone;
  char *text;
  const char invalid_utf8[] = {(char)0xc0, (char)0x80, '\0'};
  const char *locale_candidates[] = {"de_DE.UTF-8", "fr_FR.UTF-8", "de_DE",
                                     "fr_FR"};
  size_t index;

  CHECK(object != NULL && array != NULL && duplicate != NULL);
  CHECK(gpu_suite_json_array_append(array, gpu_suite_json_string("x\n\"y")) ==
        GPU_SUITE_OK);
  CHECK(gpu_suite_json_array_append(array, gpu_suite_json_double(-0.0)) ==
        GPU_SUITE_OK);
  CHECK(gpu_suite_json_object_set(object, "z", array) == GPU_SUITE_OK);
  CHECK(gpu_suite_json_object_set(object, "a", gpu_suite_json_double(1e+20)) ==
        GPU_SUITE_OK);
  CHECK(gpu_suite_json_object_set(object, "a", duplicate) ==
        GPU_SUITE_ERROR_EXISTS);
  gpu_suite_json_free(duplicate);

  text = serialize(object);
  CHECK(strcmp(text, "{\"a\":1e20,\"z\":[\"x\\n\\\"y\",0]}") == 0);
  free(text);

  clone = gpu_suite_json_clone(object);
  CHECK(clone != NULL);
  text = serialize(clone);
  CHECK(strcmp(text, "{\"a\":1e20,\"z\":[\"x\\n\\\"y\",0]}") == 0);
  free(text);
  gpu_suite_json_free(clone);

  CHECK(gpu_suite_json_double(NAN) == NULL);
  CHECK(gpu_suite_json_double(INFINITY) == NULL);
  CHECK(gpu_suite_json_string(invalid_utf8) == NULL);
  CHECK(gpu_suite_utf8_validate("日本語"));
  CHECK(!gpu_suite_utf8_validate(invalid_utf8));

  for (index = 0U;
       index < sizeof(locale_candidates) / sizeof(locale_candidates[0]);
       ++index) {
    if (setlocale(LC_NUMERIC, locale_candidates[index]) != NULL) {
      gpu_suite_json_value *number = gpu_suite_json_double(1.5);
      CHECK(number != NULL);
      text = serialize(number);
      CHECK(strcmp(text, "1.5") == 0);
      free(text);
      gpu_suite_json_free(number);
      break;
    }
  }
  CHECK(strcmp(setlocale(LC_NUMERIC, NULL), "C") == 0);
  gpu_suite_json_free(object);
  return 0;
}

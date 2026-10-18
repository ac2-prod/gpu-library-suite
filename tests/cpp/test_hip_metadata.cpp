// Host-only API fixtures. Real common result/JSON code is linked below this
// boundary; these synthetic records do not validate any GPU operation.
#include "gpu_suite/hip_metadata.hpp"
#include "gpu_suite/metadata.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdlib>
#include <string>

namespace {
const char *failure = "";
int api_calls = 0;
std::string serialized(const gpu_suite_json_value *value) {
  char *text = nullptr;
  char error[256]{};
  size_t length = 0;
  assert(gpu_suite_json_serialize(value, &text, &length, error, sizeof(error)) ==
         GPU_SUITE_OK);
  const std::string result(text, length);
  std::free(text);
  return result;
}

void test_borrowed_object() {
  gpu_suite_json_value *parent = gpu_suite_json_object();
  gpu_suite_json_value *child = gpu_suite_json_object();
  gpu_suite_json_value *scalar = gpu_suite_json_int(3);
  assert(gpu_suite_json_object_set(parent, "child", child) == GPU_SUITE_OK);
  assert(gpu_suite_json_object_set(parent, "scalar", scalar) == GPU_SUITE_OK);
  assert(gpu_suite_json_object_get_object(nullptr, "child") == nullptr);
  assert(gpu_suite_json_object_get_object(scalar, "child") == nullptr);
  assert(gpu_suite_json_object_get_object(parent, nullptr) == nullptr);
  assert(gpu_suite_json_object_get_object(parent, "missing") == nullptr);
  assert(gpu_suite_json_object_get_object(parent, "scalar") == nullptr);
  gpu_suite_json_value *borrowed = gpu_suite_json_object_get_object(parent, "child");
  assert(borrowed == child);
  assert(gpu_suite_json_object_set(borrowed, "added", gpu_suite_json_int(1)) ==
         GPU_SUITE_OK);
  assert(serialized(parent).find("\"child\":{\"added\":1}") != std::string::npos);
  gpu_suite_json_value *replacement = gpu_suite_json_object();
  assert(gpu_suite_json_object_set(parent, "child", replacement) != GPU_SUITE_OK);
  gpu_suite_json_free(replacement); // Rejected insertion retains caller ownership.
  assert(gpu_suite_json_object_get_object(parent, "child") == child);
  gpu_suite_json_free(parent); // Parent alone owns and destroys child/scalar.
}
} // namespace

const char *hipGetErrorString(hipError_t) { return "fixture HIP error"; }
hipError_t hipGetDeviceProperties(hipDeviceProp_t *properties, int device) {
  ++api_calls;
  assert(device == 0);
  if (std::strcmp(failure, "hipGetDeviceProperties") == 0)
    return static_cast<hipError_t>(1);
  std::snprintf(properties->name, sizeof(properties->name), "%s", "Fixture GPU");
  return hipSuccess;
}
hipError_t hipRuntimeGetVersion(int *version) {
  ++api_calls;
  if (std::strcmp(failure, "hipRuntimeGetVersion") == 0)
    return static_cast<hipError_t>(1);
  *version = 70200000;
  return hipSuccess;
}
hipError_t hipDriverGetVersion(int *version) {
  ++api_calls;
  if (std::strcmp(failure, "hipDriverGetVersion") == 0)
    return static_cast<hipError_t>(1);
  *version = 70100000;
  return hipSuccess;
}
rocfft_status rocfft_get_version_string(char *version, size_t length) {
  ++api_calls;
  if (std::strcmp(failure, "rocfft_get_version_string") == 0)
    return static_cast<rocfft_status>(1);
  std::snprintf(version, length, "%s", "fixture-rocfft-1.0");
  return rocfft_status_success;
}

int main(int argc, char **argv) {
  assert(argc == 4);
  const std::string scenario(argv[1]);
  if (scenario == "getter") {
    test_borrowed_object();
    return 0;
  }
  const gpu_suite_implementation implementation = std::strcmp(argv[3], "hip") == 0
      ? GPU_SUITE_IMPLEMENTATION_HIP : GPU_SUITE_IMPLEMENTATION_OPENMP;
  const gpu_suite_output_format format = std::strcmp(argv[2], "jsonl") == 0
      ? GPU_SUITE_FORMAT_JSONL : GPU_SUITE_FORMAT_CSV;
  gpu_suite_result result{};
  gpu_suite_options options{};
  char error[256]{};
  gpu_suite_options_init(&options, GPU_SUITE_BENCHMARK_ROCFFT, implementation);
  options.size = 8;
  options.size_set = true;
  options.batch = 3;
  options.repeat = 2;
  assert(gpu_suite_result_init(&result) == GPU_SUITE_OK);
  assert(gpu_suite_result_apply_options(&result, &options, error, sizeof(error)) ==
         GPU_SUITE_OK);
  gpu_suite_json_value *runtime =
      gpu_suite_json_object_get_object(result.parameters, "runtime");
  assert(runtime != nullptr);
  const std::string build_member = std::string("\"build_metadata_sha256\":\"") +
      gpu_suite_build_metadata_sha256() + "\"";
  assert(serialized(runtime) == "{" + build_member + "}");
  if (scenario == "missing-runtime" || scenario == "nonobject-runtime" ||
      scenario == "missing-build") {
    gpu_suite_json_free(result.parameters);
    result.parameters = gpu_suite_json_object();
    if (scenario != "missing-runtime") {
      assert(gpu_suite_json_object_set(result.parameters, "runtime",
          scenario == "nonobject-runtime" ? gpu_suite_json_int(1)
                                           : gpu_suite_json_object()) == GPU_SUITE_OK);
    }
  } else if (scenario == "duplicate") {
    assert(gpu_suite::apply_hip_runtime_metadata(result, 0));
    api_calls = 0;
  } else if (scenario == "device-mismatch") {
    result.gpu_name = "Other GPU";
  } else if (scenario != "success") {
    failure = argv[1];
  }
  const std::string before = serialized(result.parameters);
  const bool success = gpu_suite::apply_hip_runtime_metadata(result, 0);
  if (!success) {
    assert(serialized(result.parameters) == before);
    if (scenario == "duplicate" || scenario == "missing-runtime" ||
        scenario == "nonobject-runtime" || scenario == "missing-build")
      assert(api_calls == 0);
    std::printf("%s\n", before.c_str());
    gpu_suite_result_destroy(&result);
    return 1;
  }
  assert(scenario == "success");
  assert(api_calls == 4);
  assert(gpu_suite_json_object_get_object(result.parameters, "runtime") == runtime);
  assert(serialized(runtime).find(build_member) != std::string::npos);
  assert(std::strcmp(result.library_version, "fixture-rocfft-1.0") == 0);
  result.measurement_start_timestamp = "2026-07-14T00:00:00.000Z";
  result.measurement_end_timestamp = "2026-07-14T00:00:00.010Z";
  result.elapsed_total_sec = gpu_suite_optional_double_value(0.01);
  result.elapsed_sec = gpu_suite_optional_double_value(0.005);
  result.verification_status = "skipped";
  result.failure_origin = nullptr;
  result.exit_code = gpu_suite_optional_int_value(0);
  result.status = "success";
  result.message = "";
  assert(gpu_suite_result_write(stdout, format, true, &result, error, sizeof(error)) ==
         GPU_SUITE_OK);
  gpu_suite_result_destroy(&result);
  return 0;
}

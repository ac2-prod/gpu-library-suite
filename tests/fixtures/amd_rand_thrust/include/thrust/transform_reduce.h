#pragma once
#include <hip/hip_runtime.h>
#include <limits>
#include <stdexcept>
namespace thrust {
template<class It, class Unary, class T, class Binary>
T transform_reduce(int, It first, It last, Unary unary, T value, Binary binary) {
  fixture_event("reduce");
  if (std::getenv("GPU_SUITE_FIXTURE_THROW")) throw std::runtime_error("injected rocThrust failure");
  if (std::getenv("GPU_SUITE_FIXTURE_THROW_UNKNOWN")) throw 42;
  for (; first != last; ++first) value = binary(value, unary(*first));
  if (std::getenv("GPU_SUITE_FIXTURE_NONFINITE")) return std::numeric_limits<T>::infinity();
  return value;
}
}

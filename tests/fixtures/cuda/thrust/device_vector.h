#ifndef GPU_SUITE_TEST_THRUST_DEVICE_VECTOR_H
#define GPU_SUITE_TEST_THRUST_DEVICE_VECTOR_H

#include "device_ptr.h"

#include <cstddef>

namespace thrust {

template <typename T> class device_vector {
public:
  device_vector(std::size_t count, const T &value) : count_(count) {
    (void)value;
  }
  template <typename Container>
  explicit device_vector(const Container &container)
      : count_(container.size()) {}
  device_ptr<T> begin() { return device_ptr<T>(); }
  device_ptr<T> end() { return begin() + count_; }
  device_ptr<const T> begin() const { return device_ptr<const T>(); }
  device_ptr<const T> end() const { return begin() + count_; }

private:
  std::size_t count_;
};

} // namespace thrust

#endif

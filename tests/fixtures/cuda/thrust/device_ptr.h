#ifndef GPU_SUITE_TEST_THRUST_DEVICE_PTR_H
#define GPU_SUITE_TEST_THRUST_DEVICE_PTR_H

#include <cstddef>

namespace thrust {

template <typename T> class device_ptr {
public:
  explicit device_ptr(T *pointer = nullptr) : pointer_(pointer) {}
  device_ptr operator+(std::size_t offset) const {
    return device_ptr(pointer_ + offset);
  }

private:
  T *pointer_;
};

template <typename T> device_ptr<T> device_pointer_cast(T *pointer) {
  return device_ptr<T>(pointer);
}

} // namespace thrust

#endif

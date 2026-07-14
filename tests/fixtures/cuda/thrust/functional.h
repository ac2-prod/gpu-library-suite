#ifndef GPU_SUITE_TEST_THRUST_FUNCTIONAL_H
#define GPU_SUITE_TEST_THRUST_FUNCTIONAL_H

namespace thrust {

template <typename T> struct plus {
  T operator()(const T &left, const T &right) const { return left + right; }
};

} // namespace thrust

#endif

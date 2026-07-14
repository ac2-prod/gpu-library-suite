#ifndef GPU_SUITE_TEST_THRUST_TRANSFORM_REDUCE_H
#define GPU_SUITE_TEST_THRUST_TRANSFORM_REDUCE_H

namespace thrust {

template <typename Iterator, typename Unary, typename T, typename Binary>
T transform_reduce(Iterator first, Iterator last, Unary unary, T initial,
                   Binary binary) {
  (void)first;
  (void)last;
  (void)unary;
  (void)binary;
  return initial;
}

} // namespace thrust

#endif

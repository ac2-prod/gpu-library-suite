#ifndef GPU_SUITE_TEST_SUPPORT_H
#define GPU_SUITE_TEST_SUPPORT_H

#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      (void)fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, \
                    #condition);                                               \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#endif

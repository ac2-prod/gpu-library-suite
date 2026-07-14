#ifndef GPU_SUITE_TEST_CURAND_H
#define GPU_SUITE_TEST_CURAND_H

#include <stddef.h>

typedef void *curandGenerator_t;
typedef int curandStatus_t;
typedef int curandRngType_t;

enum { CURAND_STATUS_SUCCESS = 0, CURAND_RNG_PSEUDO_DEFAULT = 100 };

curandStatus_t curandCreateGenerator(curandGenerator_t *generator,
                                     curandRngType_t type);
curandStatus_t curandDestroyGenerator(curandGenerator_t generator);
curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator,
                                                  unsigned long long seed);
curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator,
                                        unsigned long long offset);
curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator,
                                           double *output, size_t count);

#endif

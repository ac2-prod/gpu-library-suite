#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <exception>
#include <cstdio>
#include <limits>

struct square
{
    __host__ __device__
    double operator()(double x) const
    {
        return x * x;
    }
};

extern "C"
double thrust_square_sum(const double *d_values, int num_elem)
{
    thrust::device_ptr<const double> first =
        thrust::device_pointer_cast(d_values);

    try {
        return thrust::transform_reduce(first, first + num_elem,
                                       square{}, 0.0, thrust::plus<double>());
    } catch (const std::exception &error) {
        std::fprintf(stderr, "Thrust transform_reduce: %s\n", error.what());
        return std::numeric_limits<double>::quiet_NaN();
    }
}

program openmp_rocthrust
    use iso_c_binding
    use ieee_arithmetic
    use hipfort
    use omp_lib
    implicit none
    integer, parameter :: num_elem=2**24
    real(c_double), allocatable, target :: values(:)
    real(c_double) :: result
    integer :: status
    logical :: on_host
    interface
        integer(c_int) function rocthrust_square_sum(device_values,count,result) bind(C)
            import c_int,c_ptr,c_size_t,c_double
            type(c_ptr), value :: device_values
            integer(c_size_t), value :: count
            real(c_double), intent(out) :: result
        end function
    end interface
    if (omp_get_num_devices()<1) error stop 'OpenMP device unavailable'
    call omp_set_default_device(0)
    if (hipSetDevice(0_c_int)/=0) error stop 'hipSetDevice failed'
    on_host=.true.
    !$omp target map(from:on_host)
    on_host=omp_is_initial_device()
    !$omp end target
    if (on_host) error stop 'OpenMP target fell back to host'
    allocate(values(num_elem),stat=status)
    if (status/=0) error stop 'host allocation failed'
    values=1.0_c_double
    !$omp target data map(to:values) use_device_addr(values)
    status=rocthrust_square_sum(c_loc(values(1)),int(num_elem,c_size_t),result)
    !$omp end target data
    deallocate(values)
    if (status/=0) error stop 'rocThrust transform_reduce failed'
    print '(a,f20.1)', 'Sum of squares = ',result
    if (.not.ieee_is_finite(result)) error stop 'nonfinite reduction result'
    if (result/=real(num_elem,c_double)) error stop 'reduction verification failed'
end program openmp_rocthrust

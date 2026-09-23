program openacc_thrust
    use iso_c_binding, only: c_int, c_double
    use cudafor
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    integer :: allocation_status

    interface
        function thrust_square_sum(d_values, num_elem) &
                bind(C, name="thrust_square_sum") result(res)
            import :: c_int, c_double
            real(c_double), device, intent(in) :: d_values(*)
            integer(c_int), value :: num_elem
            real(c_double) :: res
        end function thrust_square_sum
    end interface

    integer(c_int), parameter :: num_elem = 2_c_int**24
    integer :: istat
    real(c_double), allocatable :: values(:)
    real(c_double) :: result

    allocate(values(num_elem), stat=allocation_status)

    if (allocation_status /= 0) error stop 'allocation failed'
    values = 1.0_c_double

    !$acc data copyin(values)
        !$acc host_data use_device(values)
            result = thrust_square_sum(values, num_elem)
        !$acc end host_data
        ! Synchronous Thrust already completes; retained as a teaching check.
        istat = cudaDeviceSynchronize()
    if (istat /= 0) error stop 'cudaDeviceSynchronize failed'
    !$acc end data

    if (.not. ieee_is_finite(result)) error stop 'nonfinite reduction'
    if (abs(result-real(num_elem,8)) > 1.0d-10*real(num_elem,8)) error stop 'reduction verification failed'
    print *, 'verification PASS'
    print *, "result = ", result
    deallocate(values)
end program openacc_thrust

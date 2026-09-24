program reduce_cpu
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    integer :: allocation_status

    integer(c_int), parameter :: num_elem = 2_c_int**24
    real(c_double), allocatable :: values(:)
    real(c_double) :: result

    allocate(values(num_elem), stat=allocation_status)

    if (allocation_status /= 0) error stop 'allocation failed'
    values = 1.0_c_double

    result = sum(values * values)
    if (.not. ieee_is_finite(result)) error stop 'nonfinite reduction'
    if (abs(result-real(num_elem,8)) > 1.0d-10*real(num_elem,8)) error stop 'reduction verification failed'
    print *, 'verification PASS'
    print *, "result = ", result

    deallocate(values)
end program reduce_cpu

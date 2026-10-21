program reduce_cpu
    use iso_c_binding
    use ieee_arithmetic
    implicit none
    integer, parameter :: num_elem=2**24
    real(c_double), allocatable :: values(:)
    real(c_double) :: result
    integer :: status
    allocate(values(num_elem),stat=status)
    if (status/=0) error stop 'host allocation failed'
    values=1.0_c_double
    result=sum(values*values)
    deallocate(values)
    print '(a,f20.1)', 'Sum of squares = ',result
    if (.not.ieee_is_finite(result)) error stop 'nonfinite reduction result'
    if (result/=real(num_elem,c_double)) error stop 'reduction verification failed'
end program reduce_cpu

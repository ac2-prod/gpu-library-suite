program reduce_gpu
    use iso_c_binding
    use ieee_arithmetic
    use hipfort
    implicit none
    integer, parameter :: num_elem=2**24
    real(c_double), allocatable, target :: values(:)
    real(c_double) :: result
    type(c_ptr) :: d_values=c_null_ptr
    integer :: status
    interface
        integer(c_int) function rocthrust_square_sum(device_values,count,result) bind(C)
            import c_int,c_ptr,c_size_t,c_double
            type(c_ptr), value :: device_values
            integer(c_size_t), value :: count
            real(c_double), intent(out) :: result
        end function
    end interface
    allocate(values(num_elem),stat=status)
    call check(status,'host allocation')
    values=1.0_c_double
    call check(hipMalloc(d_values,size(values,kind=c_size_t)*c_sizeof(values(1))),'hipMalloc')
    call check(hipMemcpy(d_values,c_loc(values(1)),size(values,kind=c_size_t)*c_sizeof(values(1)), &
               hipMemcpyHostToDevice),'hipMemcpy')
    call check(rocthrust_square_sum(d_values,int(num_elem,c_size_t),result),'rocThrust transform_reduce')
    ! transform_reduce returns a host scalar after its device work completes.
    status=hipFree(d_values)
    d_values=c_null_ptr
    call check(status,'hipFree')
    deallocate(values)
    print '(a,f20.1)', 'Sum of squares = ',result
    if (.not.ieee_is_finite(result)) error stop 'nonfinite reduction result'
    if (result/=real(num_elem,c_double)) error stop 'reduction verification failed'
contains
    subroutine check(code,operation)
        integer, intent(in) :: code
        character(*), intent(in) :: operation
        integer :: cleanup_status
        if (code/=0) then
            print *,trim(operation),' failed: ',code
            if (c_associated(d_values)) cleanup_status=hipFree(d_values)
            error stop 1
        end if
    end subroutine
end program reduce_gpu

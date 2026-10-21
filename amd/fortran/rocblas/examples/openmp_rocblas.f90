program openmp_rocblas
    use iso_c_binding
    use ieee_arithmetic, only: ieee_is_finite
    use omp_lib
    use hipfort, only: hipSetDevice,hipMalloc,hipFree,hipMemcpy,hipDeviceSynchronize, &
                       hipMemcpyHostToDevice,hipMemcpyDeviceToHost
    use hipfort_rocblas, only: rocblas_create_handle,rocblas_destroy_handle,rocblas_operation_none, &
                              rocblas_dgemm
    implicit none
    integer(c_int), parameter :: m=1024,n=1024,k=1024
    real(c_double), target :: alpha=1.0_c_double,beta=1.0_c_double
    real(c_double), allocatable, target :: mat_A(:,:),mat_B(:,:),mat_C(:,:)
    type(c_ptr) :: handle=c_null_ptr
    integer :: device,allocation_status
    logical :: initial,success=.true.
    real(c_double) :: error

    ! Regions below use this unchanged OpenMP default device.
    device=omp_get_default_device()
    if (device < 0 .or. device >= omp_get_num_devices()) error stop 'OpenMP target unavailable'
    initial=.true.
    !$omp target map(from: initial)
    initial=omp_is_initial_device()
    !$omp end target
    if (initial) error stop 'OpenMP host fallback rejected'
    call record(hipSetDevice(int(device,c_int)),'hipSetDevice')
    if (.not. success) error stop 'HIP device selection failed'
    allocate(mat_A(m,k),mat_B(k,n),mat_C(m,n),stat=allocation_status)
    if (allocation_status /= 0) error stop 'host allocation failed'
    mat_A=1; mat_B=1; mat_C=1
    call record(rocblas_create_handle(handle),'rocblas_create_handle')
    if (success) then
        !$omp target data map(to: mat_A,mat_B) map(tofrom: mat_C)
        !$omp target data use_device_addr(mat_A,mat_B,mat_C)
        call record(rocblas_dgemm(handle,rocblas_operation_none,rocblas_operation_none,m,n,k, &
                                 alpha,c_loc(mat_A),m,c_loc(mat_B),k,beta,c_loc(mat_C),m),'rocblas_dgemm')
        !$omp end target data
        ! Library work must finish before OpenMP copies back and releases its mapping.
        call record(hipDeviceSynchronize(),'DGEMM completion')
        !$omp end target data
    end if
    if (success) then
        success=all(ieee_is_finite(mat_C))
        if (success) then
            error=maxval(abs(mat_C-real(k+1,c_double)))
            print *, 'OpenMP rocBLAS DGEMM complete; max error = ',error
            success=error <= 1.0e-10_c_double
        end if
    end if
    if (c_associated(handle)) call record(rocblas_destroy_handle(handle),'destroy handle')
    deallocate(mat_A,mat_B,mat_C)
    if (.not. success) error stop 'OpenMP rocBLAS pipeline or verification failed'
contains
    subroutine record(status,label)
        integer(c_int), intent(in) :: status
        character(*), intent(in) :: label
        if (status /= 0) then
            print *, trim(label),': status ',status
            success=.false.
        end if
    end subroutine
end program openmp_rocblas

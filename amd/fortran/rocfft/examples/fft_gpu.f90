program fft_gpu
    use, intrinsic :: iso_c_binding
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use hipfort, only: hipSetDevice, hipMalloc, hipFree, hipMemcpy, hipDeviceSynchronize, &
                       hipMemcpyHostToDevice, hipMemcpyDeviceToHost
    use hipfort_rocfft, only: rocfft_setup, rocfft_cleanup, rocfft_plan_create, rocfft_plan_destroy, &
        rocfft_execution_info_create, rocfft_execution_info_destroy, rocfft_plan_get_work_buffer_size, &
        rocfft_execution_info_set_work_buffer, rocfft_execute, rocfft_placement_notinplace, &
        rocfft_transform_type_complex_forward, rocfft_precision_single
    implicit none
    integer(c_size_t), parameter :: nfft=1024, batch=4096, count=nfft*batch
    integer(c_size_t), target :: lengths(1)=[nfft]
    integer(c_size_t) :: bytes, work_bytes
    complex(c_float_complex), allocatable, target :: h_in(:), h_out(:)
    type(c_ptr) :: plan=c_null_ptr, info=c_null_ptr, workspace=c_null_ptr
    type(c_ptr) :: d_in=c_null_ptr, d_out=c_null_ptr
    integer :: allocation_status, j
    real(c_float) :: max_error

    call check(hipSetDevice(0), 'hipSetDevice')
    allocate(h_in(count),h_out(count),stat=allocation_status)
    if (allocation_status /= 0) error stop 'host allocation failed'
    h_in=cmplx(1,0,kind=c_float)
    h_out=cmplx(0,0,kind=c_float)
    bytes=count*c_sizeof(h_in(1))
    call check(rocfft_setup(), 'rocfft_setup')
    call check(rocfft_plan_create(plan,rocfft_placement_notinplace, &
        rocfft_transform_type_complex_forward,rocfft_precision_single, &
        1_c_size_t,c_loc(lengths),batch,c_null_ptr), 'rocfft_plan_create')
    call check(rocfft_execution_info_create(info), 'rocfft_execution_info_create')
    call check(rocfft_plan_get_work_buffer_size(plan,work_bytes), 'rocfft_plan_get_work_buffer_size')
    if (work_bytes > 0) then
        call check(hipMalloc(workspace,work_bytes), 'hipMalloc workspace')
        call check(rocfft_execution_info_set_work_buffer(info,workspace,work_bytes), 'set workspace')
    end if

    call check(hipMalloc(d_in,bytes), 'hipMalloc input')
    call check(hipMalloc(d_out,bytes), 'hipMalloc output')
    call check(hipMemcpy(d_in,c_loc(h_in(1)),bytes,hipMemcpyHostToDevice), 'copy input')
    ! hipFORT passes these pointer slots by reference, as rocFFT's void** requires.
    call check(rocfft_execute(plan,d_in,d_out,info), 'rocfft_execute')
    call check(hipDeviceSynchronize(), 'hipDeviceSynchronize')
    call check(hipMemcpy(c_loc(h_out(1)),d_out,bytes,hipMemcpyDeviceToHost), 'copy result')

    if (.not. all(ieee_is_finite(real(h_out))) .or. &
        .not. all(ieee_is_finite(aimag(h_out)))) error stop 'nonfinite FFT output'
    max_error=0
    do j=1,int(count)
        if (mod(j-1,int(nfft)) == 0) then
            max_error=max(max_error,abs(h_out(j)-cmplx(nfft,0,kind=c_float)))
        else
            max_error=max(max_error,abs(h_out(j)))
        end if
    end do
    if (max_error > 1.0e-4_c_float) error stop 'FFT verification failed'
    print *, 'rocFFT batched C2C forward complete; max error = ',max_error
    call check(rocfft_execution_info_destroy(info), 'rocfft_execution_info_destroy')
    call check(rocfft_plan_destroy(plan), 'rocfft_plan_destroy')
    if (c_associated(workspace)) call check(hipFree(workspace), 'hipFree workspace')
    call check(hipFree(d_in), 'hipFree input')
    call check(hipFree(d_out), 'hipFree output')
    call check(rocfft_cleanup(), 'rocfft_cleanup')
    deallocate(h_in,h_out)
contains
    subroutine check(status,operation)
        integer(c_int), intent(in) :: status
        character(*), intent(in) :: operation
        if (status /= 0) then
            print *, trim(operation), ': status ',status
            error stop 1
        end if
    end subroutine
end program fft_gpu

program openacc_cufft
    use, intrinsic :: iso_c_binding
    use cudafor
    use cufft
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    integer :: allocation_status

    integer, parameter :: nfft = 1024
    integer, parameter :: batch = 4096
    integer, parameter :: num_elem = nfft * batch
    integer :: plan, istat
    complex(C_FLOAT_COMPLEX), allocatable :: &
        fft_in(:), fft_out(:)

    allocate(fft_in(num_elem), fft_out(num_elem), stat=allocation_status)

    if (allocation_status /= 0) error stop 'allocation failed'
    fft_in = (1.0_C_FLOAT, 0.0_C_FLOAT)

    istat = cufftPlan1d(plan, nfft, CUFFT_C2C, batch)
    if (istat /= 0) error stop 'cufftPlan1d failed'

    !$acc data copyin(fft_in(1:num_elem)) &
    !$acc      copyout(fft_out(1:num_elem))
        !$acc host_data use_device(fft_in, fft_out)
            istat = cufftExecC2C(plan, fft_in, fft_out, &
                                 CUFFT_FORWARD)
    if (istat /= 0) error stop 'cufftExecC2C failed'
        !$acc end host_data
        istat = cudaDeviceSynchronize()
    if (istat /= 0) error stop 'cudaDeviceSynchronize failed'
    !$acc end data

    print *, "fft_out(1) = ", fft_out(1)

    block
        integer :: b, j
        do b = 0, batch-1
            if (abs(fft_out(b*nfft+1)-cmplx(nfft,0,kind=4)) > 1.0e-5*real(nfft)) &
                error stop 'FFT DC verification failed'
            do j = 1, nfft
                if (.not. ieee_is_finite(real(fft_out(b*nfft+j))) .or. &
                    .not. ieee_is_finite(aimag(fft_out(b*nfft+j)))) error stop 'nonfinite FFT'
                if (j > 1) then
                    if (abs(fft_out(b*nfft+j)) > 1.0e-4) error stop 'FFT non-DC verification failed'
                end if
            end do
        end do
    end block
    print *, 'verification PASS'
    istat = cufftDestroy(plan)
    if (istat /= 0) error stop 'cufftDestroy failed'
    deallocate(fft_in, fft_out)
end program openacc_cufft

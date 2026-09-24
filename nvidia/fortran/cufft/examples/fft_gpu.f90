program fft_gpu
    use, intrinsic :: iso_c_binding
    use cudafor
    use cufft
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    integer :: allocation_status

    integer, parameter :: nfft = 1024   ! FFT長
    integer, parameter :: batch = 4096  ! FFTの本数
    integer, parameter :: num_elem = nfft * batch
    integer :: plan, istat

    complex(C_FLOAT_COMPLEX), allocatable :: h_in(:), h_out(:)
    complex(C_FLOAT_COMPLEX), device, allocatable :: d_in(:), d_out(:)

    allocate(h_in(num_elem), h_out(num_elem), stat=allocation_status)

    if (allocation_status /= 0) error stop 'allocation failed'
    h_in = (1.0_C_FLOAT, 0.0_C_FLOAT)

    allocate(d_in(num_elem), d_out(num_elem), stat=allocation_status)

    if (allocation_status /= 0) error stop 'allocation failed'
    istat = cudaMemcpy(d_in, h_in, num_elem, &
                       cudaMemcpyHostToDevice)
    if (istat /= 0) error stop 'cudaMemcpy failed'

    istat = cufftPlan1d(plan, nfft, CUFFT_C2C, batch)
    if (istat /= 0) error stop 'cufftPlan1d failed'
    istat = cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD)
    if (istat /= 0) error stop 'cufftExecC2C failed'

    istat = cudaMemcpy(h_out, d_out, num_elem, &
                       cudaMemcpyDeviceToHost)
    if (istat /= 0) error stop 'cudaMemcpy failed'

    block
        integer :: b, j
        do b = 0, batch-1
            if (abs(h_out(b*nfft+1)-cmplx(nfft,0,kind=4)) > 1.0e-5*real(nfft)) &
                error stop 'FFT DC verification failed'
            do j = 1, nfft
                if (.not. ieee_is_finite(real(h_out(b*nfft+j))) .or. &
                    .not. ieee_is_finite(aimag(h_out(b*nfft+j)))) error stop 'nonfinite FFT'
                if (j > 1) then
                    if (abs(h_out(b*nfft+j)) > 1.0e-4) error stop 'FFT non-DC verification failed'
                end if
            end do
        end do
    end block
    print *, 'verification PASS'
    istat = cufftDestroy(plan)
    if (istat /= 0) error stop 'cufftDestroy failed'

    deallocate(d_in, d_out)
    deallocate(h_in, h_out)
end program fft_gpu

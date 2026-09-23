program fft_cpu
    use, intrinsic :: iso_c_binding
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    integer :: allocation_status
    include 'fftw3.f03'

    integer(C_INT), parameter :: nfft = 1024  ! FFT長
    integer(C_INT), parameter :: batch = 4096 ! FFTの本数
    integer(C_INT), parameter :: rank = 1
    integer, parameter :: num_elem = nfft * batch
    integer(C_INT) :: dims(1)
    type(C_PTR) :: plan, p_in, p_out
    complex(C_FLOAT_COMPLEX), pointer :: in(:), out(:)

    dims = [nfft]
    p_in  = fftwf_alloc_complex(int(num_elem, C_SIZE_T))
    p_out = fftwf_alloc_complex(int(num_elem, C_SIZE_T))
    if (.not. c_associated(p_in) .or. .not. c_associated(p_out)) error stop 'FFTW allocation failed'
    call c_f_pointer(p_in,  in,  [num_elem])
    call c_f_pointer(p_out, out, [num_elem])

    in = (1.0_C_FLOAT, 0.0_C_FLOAT)

    plan = fftwf_plan_many_dft(rank, dims, batch, &
        in,  dims, 1, nfft, &
        out, dims, 1, nfft, &
        FFTW_FORWARD, FFTW_ESTIMATE)

    if (.not. c_associated(plan)) error stop 'FFTW plan failed'
    call fftwf_execute_dft(plan, in, out)

    block
        integer :: b, j
        do b = 0, batch-1
            if (abs(out(b*nfft+1)-cmplx(nfft,0,kind=4)) > 1.0e-5*real(nfft)) &
                error stop 'FFT DC verification failed'
            do j = 1, nfft
                if (.not. ieee_is_finite(real(out(b*nfft+j))) .or. &
                    .not. ieee_is_finite(aimag(out(b*nfft+j)))) error stop 'nonfinite FFT'
                if (j > 1) then
                    if (abs(out(b*nfft+j)) > 1.0e-4) error stop 'FFT non-DC verification failed'
                end if
            end do
        end do
    end block
    print *, 'verification PASS'
    call fftwf_destroy_plan(plan)
    call fftwf_free(p_in)
    call fftwf_free(p_out)
end program fft_cpu

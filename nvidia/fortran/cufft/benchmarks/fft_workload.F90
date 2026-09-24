module gpu_suite_workload
    use gpu_suite_fortran
    use iso_c_binding
    use ieee_arithmetic
#if GPU_SUITE_IMPL != 0
    use cudafor
    use cufft
#endif
#if GPU_SUITE_IMPL == 2
    use openacc
#endif
    implicit none
#if GPU_SUITE_IMPL == 0
    include 'fftw3.f03'
    type(c_ptr) :: plan=c_null_ptr, p_in=c_null_ptr, p_out=c_null_ptr
    complex(c_float_complex), pointer :: fft_in(:),fft_out(:)
    logical :: threads_initialized=.false.
#else
    integer :: plan
    complex(c_float_complex), allocatable :: fft_in(:),fft_out(:)
#endif
#if GPU_SUITE_IMPL == 1
    complex(c_float_complex), device, allocatable :: d_in(:),d_out(:)
#endif
    integer(c_int64_t) :: count
    logical :: data_present=.false.
contains
    subroutine host_setup()
        integer :: status
        count=checked_product(o%n,o%batch)
#if GPU_SUITE_IMPL == 0
        p_in=fftwf_alloc_complex(int(count,c_size_t)); p_out=fftwf_alloc_complex(int(count,c_size_t))
        call require(c_associated(p_in).and.c_associated(p_out),'FFTW aligned allocation failed')
        call c_f_pointer(p_in,fft_in,[count]); call c_f_pointer(p_out,fft_out,[count])
#ifdef GPU_SUITE_HAVE_FFTW_THREADS
        if (o%fftw_threaded) then
            call require(fftwf_init_threads()/=0,'fftwf_init_threads failed')
            threads_initialized=.true.
            call fftwf_plan_with_nthreads(o%threads)
        end if
#else
        call require(.not. o%fftw_threaded,'FFTW threads backend unavailable')
#endif
#else
        call check(cudaSetDevice(o%device),'cudaSetDevice')
        allocate(fft_in(count),fft_out(count),stat=status)
        call check(status,'host allocation')
#endif
#if GPU_SUITE_IMPL == 2
        call acc_set_device_num(o%device,acc_device_nvidia)
#endif
    end subroutine
    subroutine reset()
        fft_in=cmplx(1,0,kind=c_float); fft_out=cmplx(0,0,kind=c_float)
    end subroutine
    subroutine setup()
#if GPU_SUITE_IMPL == 0
        integer(c_int) :: dims(1)
        dims=[o%n]
        plan=fftwf_plan_many_dft(1,dims,o%batch,fft_in,dims,1,o%n,fft_out,dims,1,o%n, &
                                FFTW_FORWARD,FFTW_ESTIMATE)
        call require(c_associated(plan),'FFTW plan failed')
#elif GPU_SUITE_IMPL == 1
        integer :: status
        allocate(d_in(count),d_out(count),stat=status)
        call check(status,'device allocation')
        call restore()
#else
        !$acc enter data copyin(fft_in) create(fft_out)
        data_present=.true.
#endif
#if GPU_SUITE_IMPL != 0
        call check(cufftPlan1d(plan,o%n,CUFFT_C2C,o%batch),'cufftPlan1d')
#endif
    end subroutine
    subroutine restore()
#if GPU_SUITE_IMPL == 1
        call check(cudaMemcpy(d_in,fft_in,count,cudaMemcpyHostToDevice),'copy FFT input')
#elif GPU_SUITE_IMPL == 2
        !$acc update device(fft_in)
#endif
    end subroutine
    subroutine operation()
#if GPU_SUITE_IMPL == 0
        call fftwf_execute_dft(plan,fft_in,fft_out)
#elif GPU_SUITE_IMPL == 1
        call check(cufftExecC2C(plan,d_in,d_out,CUFFT_FORWARD),'cufftExecC2C')
#else
        integer :: status
        !$acc host_data use_device(fft_in,fft_out)
        status=cufftExecC2C(plan,fft_in,fft_out,CUFFT_FORWARD)
        !$acc end host_data
        call check(status,'cufftExecC2C')
#endif
    end subroutine
    subroutine synchronize()
#if GPU_SUITE_IMPL != 0
        call check(cudaDeviceSynchronize(),'cudaDeviceSynchronize')
#endif
    end subroutine
    subroutine download(end_to_end)
        logical, intent(in) :: end_to_end
#if GPU_SUITE_IMPL == 1
        call check(cudaMemcpy(fft_out,d_out,count,cudaMemcpyDeviceToHost),'copy FFT output')
#elif GPU_SUITE_IMPL == 2
        if (end_to_end) then
            !$acc exit data copyout(fft_out) delete(fft_in)
            data_present=.false.
        else
            !$acc update self(fft_out)
        end if
#endif
    end subroutine
    subroutine cleanup()
#if GPU_SUITE_IMPL == 0
        call fftwf_destroy_plan(plan)
        plan=c_null_ptr
#else
        call check(cufftDestroy(plan),'cufftDestroy')
#endif
#if GPU_SUITE_IMPL == 1
        deallocate(d_in,d_out)
#elif GPU_SUITE_IMPL == 2
        if (data_present) then
            !$acc exit data delete(fft_in,fft_out)
            data_present=.false.
        end if
#endif
    end subroutine
    subroutine verify(metrics,scale,getrf,getrs)
        real(c_double), intent(out) :: metrics(4),scale
        integer, intent(out) :: getrf,getrs
        integer(c_int64_t) :: j
        complex(c_double_complex) :: value
        getrf=0; getrs=0; metrics=0; scale=1
        do j=1,count
            if (.not. ieee_is_finite(real(fft_out(j))) .or. .not. ieee_is_finite(aimag(fft_out(j)))) then
                metrics(1:2)=nan_value()
                return
            end if
            value=cmplx(fft_out(j),kind=c_double)
            if (mod(j-1,int(o%n,c_int64_t))==0) then
                metrics(1)=max(metrics(1),abs(value-real(o%n,c_double))/real(o%n,c_double))
            else
                metrics(2)=max(metrics(2),abs(value))
            end if
        end do
    end subroutine
    subroutine host_cleanup()
#if GPU_SUITE_IMPL == 0
        call fftwf_free(p_in); call fftwf_free(p_out)
        nullify(fft_in,fft_out)
#ifdef GPU_SUITE_HAVE_FFTW_THREADS
        if (threads_initialized) call fftwf_cleanup_threads()
#endif
#else
        deallocate(fft_in,fft_out)
#endif
    end subroutine
end module gpu_suite_workload

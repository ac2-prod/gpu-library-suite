module gpu_suite_workload
    use gpu_suite_fortran
    use iso_c_binding
    use ieee_arithmetic
#if GPU_SUITE_IMPL != 0
    use hipfort, only: hipSetDevice, hipMalloc, hipFree, hipMemcpy, hipMemset, hipDeviceSynchronize, &
                       hipMemcpyHostToDevice, hipMemcpyDeviceToHost
    use hipfort_rocfft, only: rocfft_setup, rocfft_cleanup, rocfft_plan_create, rocfft_plan_destroy, &
        rocfft_execution_info_create, rocfft_execution_info_destroy, rocfft_plan_get_work_buffer_size, &
        rocfft_execution_info_set_work_buffer, rocfft_execute, rocfft_placement_notinplace, &
        rocfft_transform_type_complex_forward, rocfft_precision_single
#endif
#if GPU_SUITE_IMPL == 2
    use omp_lib
#endif
    implicit none
#if GPU_SUITE_IMPL == 2
    integer :: omp_device
    interface
        function check_mapped_device(pointer,device) bind(C,name='gpu_suite_f_check_mapped_device') result(status)
            import c_ptr,c_int
            type(c_ptr), value :: pointer
            integer(c_int), value :: device
            integer(c_int) :: status
        end function
    end interface
#endif
#if GPU_SUITE_IMPL == 0
    include 'fftw3.f03'
    type(c_ptr) :: plan=c_null_ptr, p_in=c_null_ptr, p_out=c_null_ptr
    complex(c_float_complex), pointer :: fft_in(:),fft_out(:)
    logical :: threads_initialized=.false.
#else
    complex(c_float_complex), allocatable, target :: fft_in(:),fft_out(:)
    type(c_ptr) :: plan=c_null_ptr, info=c_null_ptr, workspace=c_null_ptr
    type(c_ptr) :: d_in=c_null_ptr, d_out=c_null_ptr
    integer(c_size_t) :: bytes,work_bytes
#endif
    integer(c_int64_t) :: count
    logical :: data_present=.false.
contains
    subroutine host_setup()
        integer :: status
#if GPU_SUITE_IMPL == 2
        logical :: on_host
#endif
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
        call require(.not. o%fftw_threaded,'FFTW threaded backend unavailable')
#endif
#else
        call check(hipSetDevice(o%device),'hipSetDevice')
        call check(rocfft_setup(),'rocfft_setup')
        allocate(fft_in(count),fft_out(count),stat=status)
        call check(status,'host allocation')
        bytes=int(count,c_size_t)*c_sizeof(fft_in(1))
#endif
#if GPU_SUITE_IMPL == 2
        ! Regions below use this unchanged OpenMP default device.
        omp_device=omp_get_default_device()
        call require(omp_device >= 0 .and. omp_device < omp_get_num_devices(),'OpenMP GPU unavailable')
        on_host=.true.
        !$omp target map(from:on_host)
        on_host=omp_is_initial_device()
        !$omp end target
        call require(.not. on_host,'OpenMP host fallback rejected')
#endif
        fft_in=cmplx(1,0,kind=c_float); fft_out=cmplx(0,0,kind=c_float)
    end subroutine
    subroutine reset()
#if GPU_SUITE_IMPL == 1
        ! The direct HIP reference retains its immutable host input; restore()
        ! copies it and clears device output before each library segment.
        if (o%scope == 0) return
#endif
        fft_in=cmplx(1,0,kind=c_float); fft_out=cmplx(0,0,kind=c_float)
    end subroutine
    subroutine setup()
#if GPU_SUITE_IMPL == 0
        integer(c_int) :: dims(1)
        dims=[o%n]
        plan=fftwf_plan_many_dft(1,dims,o%batch,fft_in,dims,1,o%n,fft_out,dims,1,o%n, &
                                FFTW_FORWARD,FFTW_ESTIMATE)
        call require(c_associated(plan),'FFTW plan failed')
#else
        integer(c_size_t), target :: lengths(1)
        lengths=[int(o%n,c_size_t)]
        call check(rocfft_plan_create(plan,rocfft_placement_notinplace, &
            rocfft_transform_type_complex_forward,rocfft_precision_single,1_c_size_t, &
            c_loc(lengths),int(o%batch,c_size_t),c_null_ptr),'rocfft_plan_create')
        call check(rocfft_execution_info_create(info),'rocfft_execution_info_create')
        call check(rocfft_plan_get_work_buffer_size(plan,work_bytes),'rocfft_plan_get_work_buffer_size')
        if (work_bytes > 0) then
            call check(hipMalloc(workspace,work_bytes),'hipMalloc workspace')
            call check(rocfft_execution_info_set_work_buffer(info,workspace,work_bytes),'set workspace')
        end if
#if GPU_SUITE_IMPL == 1
        call check(hipMalloc(d_in,bytes),'hipMalloc input')
        call check(hipMalloc(d_out,bytes),'hipMalloc output')
        if (o%scope /= 0) call restore()
#else
        !$omp target enter data map(to:fft_in) map(alloc:fft_out)
        data_present=.true.
#endif
#endif
    end subroutine
    subroutine restore()
        ! rocFFT may overwrite out-of-place input. The AMD driver calls this
        ! before EVERY GPU FFT, outside each library timing interval.
#if GPU_SUITE_IMPL == 1
        call check(hipMemcpy(d_in,c_loc(fft_in(1)),bytes,hipMemcpyHostToDevice),'restore FFT input')
        call check(hipMemset(d_out,0_c_int,bytes),'restore FFT output')
#elif GPU_SUITE_IMPL == 2
        !$omp target update to(fft_in,fft_out)
#endif
    end subroutine
    subroutine operation(repetitions,timed)
        integer, intent(in) :: repetitions
        logical, intent(in) :: timed
        call require(repetitions == 1,'FFT requires separately restored/timed transforms')
#if GPU_SUITE_IMPL == 2
        ! C/C++ enters and validates use_device_ptr before each library timer,
        ! and leaves it after the timer. E2E already has an outer timer.
        !$omp target data use_device_addr(fft_in,fft_out)
        d_in=c_loc(fft_in(1)); d_out=c_loc(fft_out(1))
        call check(check_mapped_device(d_in,int(o%device,c_int)),'mapped FFT input device')
        call check(check_mapped_device(d_out,int(o%device,c_int)),'mapped FFT output device')
#endif
        if (timed) call start_timer()
#if GPU_SUITE_IMPL == 0
        ! Retain the established array-explicit FFTW Fortran binding.
        call fftwf_execute_dft(plan,fft_in,fft_out)
#else
        call check(rocfft_execute(plan,d_in,d_out,info),'rocfft_execute')
        call synchronize()
#endif
        if (timed) call end_timer()
#if GPU_SUITE_IMPL == 2
        !$omp end target data
#endif
    end subroutine
    subroutine synchronize()
#if GPU_SUITE_IMPL != 0
        call check(hipDeviceSynchronize(),'hipDeviceSynchronize')
#endif
    end subroutine
    subroutine download(end_to_end)
        logical, intent(in) :: end_to_end
#if GPU_SUITE_IMPL == 1
        call check(hipMemcpy(c_loc(fft_out(1)),d_out,bytes,hipMemcpyDeviceToHost),'copy FFT output')
#elif GPU_SUITE_IMPL == 2
        if (end_to_end) then
            !$omp target exit data map(from:fft_out) map(delete:fft_in)
            data_present=.false.
        else
            !$omp target update from(fft_out)
        end if
#endif
    end subroutine
    subroutine cleanup()
#if GPU_SUITE_IMPL == 0
        call fftwf_destroy_plan(plan)
        plan=c_null_ptr
#else
        call check(rocfft_execution_info_destroy(info),'rocfft_execution_info_destroy')
        call check(rocfft_plan_destroy(plan),'rocfft_plan_destroy')
        if (c_associated(workspace)) call check(hipFree(workspace),'hipFree workspace')
        info=c_null_ptr; plan=c_null_ptr; workspace=c_null_ptr
#endif
#if GPU_SUITE_IMPL == 1
        call check(hipFree(d_in),'hipFree input')
        call check(hipFree(d_out),'hipFree output')
        d_in=c_null_ptr; d_out=c_null_ptr
#elif GPU_SUITE_IMPL == 2
        if (data_present) then
            !$omp target exit data map(delete:fft_in,fft_out)
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
                metrics(1)=max(metrics(1),max(abs(real(value)-real(o%n,c_double)), &
                    abs(aimag(value)))/real(o%n,c_double))
            else
                metrics(2)=max(metrics(2),abs(real(value)),abs(aimag(value)))
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
        call check(rocfft_cleanup(),'rocfft_cleanup')
#endif
    end subroutine
end module gpu_suite_workload

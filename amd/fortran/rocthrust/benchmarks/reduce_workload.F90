module gpu_suite_workload
    use gpu_suite_fortran
    use iso_c_binding
    use ieee_arithmetic
#if GPU_SUITE_IMPL != 0
    use hipfort, only: hipMalloc,hipFree,hipMemcpy,hipSetDevice,hipMemcpyHostToDevice
#endif
#if GPU_SUITE_IMPL == 2
    use omp_lib
#endif
    implicit none
    real(c_double), allocatable, target :: values(:)
    real(c_double) :: result
    type(c_ptr) :: d_values=c_null_ptr
    logical :: data_present=.false.
#if GPU_SUITE_IMPL != 0
    interface
        integer(c_int) function rocthrust_square_sum(device_values,count,result) bind(C)
            import c_int,c_ptr,c_size_t,c_double
            type(c_ptr), value :: device_values
            integer(c_size_t), value :: count
            real(c_double), intent(out) :: result
        end function
    end interface
#endif
#if GPU_SUITE_IMPL == 2
    interface
        integer(c_int) function gpu_suite_f_check_mapped_device(pointer,device) bind(C)
            import c_int,c_ptr
            type(c_ptr), value :: pointer
            integer(c_int), value :: device
        end function
    end interface
#endif
contains
    subroutine host_setup()
        integer :: status
        integer(c_int64_t) :: count
#if GPU_SUITE_IMPL == 2
        logical :: on_host
#endif
        count=checked_product(o%n,1)
#if GPU_SUITE_IMPL != 0
        call check(hipSetDevice(int(o%device,c_int)),'hipSetDevice')
#endif
#if GPU_SUITE_IMPL == 2
        call require(omp_get_default_device()>=0 .and. omp_get_default_device()<omp_get_num_devices(), &
                     'OpenMP default device unavailable')
        on_host=.true.
        !$omp target map(from:on_host)
        on_host=omp_is_initial_device()
        !$omp end target
        call require(.not.on_host,'OpenMP target fell back to host')
#endif
        allocate(values(o%n),stat=status)
        call check(status,'host allocation')
        values=1
    end subroutine
    subroutine reset()
        result=0
    end subroutine
    subroutine setup()
#if GPU_SUITE_IMPL == 1
        call check(hipMalloc(d_values,size(values,kind=c_size_t)*c_sizeof(values(1))),'allocate values')
        call check(hipMemcpy(d_values,c_loc(values(1)),size(values,kind=c_size_t)*c_sizeof(values(1)), &
                   hipMemcpyHostToDevice),'copy values')
#elif GPU_SUITE_IMPL == 2
        !$omp target enter data map(to:values)
        data_present=.true.
        !$omp target data use_device_addr(values)
        call check(gpu_suite_f_check_mapped_device(c_loc(values(1)),int(o%device,c_int)),'values device mapping')
        !$omp end target data
#endif
    end subroutine
    subroutine restore()
        ! Immutable input survives warm-up and trials, as in C/C++.
    end subroutine
    subroutine operation(repetitions,timed)
        integer, intent(in) :: repetitions
        logical, intent(in) :: timed
        integer :: repetition
        if (timed) call start_timer()
        do repetition=1,repetitions
#if GPU_SUITE_IMPL == 0
        result=sum(values*values)
#elif GPU_SUITE_IMPL == 1
        call check(rocthrust_square_sum(d_values,int(o%n,c_size_t),result),'rocThrust transform_reduce')
#else
        !$omp target data use_device_addr(values)
        call check(rocthrust_square_sum(c_loc(values(1)),int(o%n,c_size_t),result),'rocThrust transform_reduce')
        !$omp end target data
#endif
        end do
        ! The synchronous host scalar is the completion boundary, including
        ! the last repeat. AMD C/C++ adds no hipDeviceSynchronize here.
        if (timed) call end_timer()
    end subroutine
    subroutine synchronize()
        ! transform_reduce already completed and returned the host scalar.
    end subroutine
    subroutine download(end_to_end)
        logical, intent(in) :: end_to_end
        ! The wrapper already returned the scalar on the host.
    end subroutine
    subroutine cleanup()
#if GPU_SUITE_IMPL == 1
        call check(hipFree(d_values),'free values')
        d_values=c_null_ptr
#elif GPU_SUITE_IMPL == 2
        if (data_present) then
            !$omp target exit data map(delete:values)
            data_present=.false.
        end if
#endif
    end subroutine
    subroutine verify(metrics,scale,getrf,getrs)
        real(c_double), intent(out) :: metrics(4),scale
        integer, intent(out) :: getrf,getrs
        getrf=0; getrs=0; metrics=0; scale=real(o%n,c_double)
        if (.not.ieee_is_finite(result)) then
            metrics(1)=nan_value()
        else
            metrics(1)=abs(result-scale)
        end if
    end subroutine
    subroutine host_cleanup()
        deallocate(values)
    end subroutine
end module gpu_suite_workload

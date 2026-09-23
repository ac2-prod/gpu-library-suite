module gpu_suite_workload
    use gpu_suite_fortran
    use iso_c_binding
    use ieee_arithmetic
#if GPU_SUITE_IMPL != 0
    use cudafor
#endif
#if GPU_SUITE_IMPL == 2
    use openacc
#endif
    implicit none
    real(c_double), allocatable :: values(:)
    real(c_double) :: result
#if GPU_SUITE_IMPL == 1
    real(c_double), device, allocatable :: d_values(:)
#endif
#if GPU_SUITE_IMPL != 0
    interface
        function thrust_square_sum(d_values,num_elem) bind(C,name='thrust_square_sum') result(res)
            import c_int,c_double
            real(c_double), device, intent(in) :: d_values(*)
            integer(c_int), value :: num_elem
            real(c_double) :: res
        end function
    end interface
#endif
    logical :: data_present=.false.
contains
    subroutine host_setup()
        integer :: status
        integer(c_int64_t) :: count
        count=checked_product(o%n,1)
#if GPU_SUITE_IMPL != 0
        call check(cudaSetDevice(o%device),'cudaSetDevice')
#endif
#if GPU_SUITE_IMPL == 2
        call acc_set_device_num(o%device,acc_device_nvidia)
#endif
        allocate(values(o%n),stat=status)
        call check(status,'host allocation')
    end subroutine
    subroutine reset()
        values=1; result=0
    end subroutine
    subroutine setup()
#if GPU_SUITE_IMPL == 1
        integer :: status
        allocate(d_values(o%n),stat=status)
        call check(status,'device allocation')
        call restore()
#elif GPU_SUITE_IMPL == 2
        !$acc enter data copyin(values)
        data_present=.true.
#endif
    end subroutine
    subroutine restore()
#if GPU_SUITE_IMPL == 1
        call check(cudaMemcpy(d_values,values,o%n,cudaMemcpyHostToDevice),'copy values')
#elif GPU_SUITE_IMPL == 2
        !$acc update device(values)
#endif
    end subroutine
    subroutine operation()
#if GPU_SUITE_IMPL == 0
        result=sum(values*values)
#elif GPU_SUITE_IMPL == 1
        result=thrust_square_sum(d_values,o%n)
#else
        !$acc host_data use_device(values)
        result=thrust_square_sum(values,o%n)
        !$acc end host_data
#endif
        ! Synchronous transform_reduce returns a host scalar. No per-repeat sync.
    end subroutine
    subroutine synchronize()
#if GPU_SUITE_IMPL != 0
        call check(cudaDeviceSynchronize(),'cudaDeviceSynchronize')
#endif
    end subroutine
    subroutine download(end_to_end)
        logical, intent(in) :: end_to_end
#if GPU_SUITE_IMPL == 2
        if (end_to_end) then
            !$acc exit data delete(values)
            data_present=.false.
        end if
#endif
    end subroutine
    subroutine cleanup()
#if GPU_SUITE_IMPL == 1
        deallocate(d_values)
#elif GPU_SUITE_IMPL == 2
        if (data_present) then
            !$acc exit data delete(values)
            data_present=.false.
        end if
#endif
    end subroutine
    subroutine verify(metrics,scale,getrf,getrs)
        real(c_double), intent(out) :: metrics(4),scale
        integer, intent(out) :: getrf,getrs
        getrf=0; getrs=0; metrics=0; scale=real(o%n,c_double)
        if (.not. ieee_is_finite(result)) then
            metrics(1)=nan_value()
        else
            metrics(1)=abs(result-scale)
        end if
    end subroutine
    subroutine host_cleanup()
        deallocate(values)
    end subroutine
end module gpu_suite_workload

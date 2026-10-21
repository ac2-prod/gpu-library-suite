module gpu_suite_workload
    use gpu_suite_fortran
    use iso_c_binding
    use ieee_arithmetic
#if GPU_SUITE_IMPL != 0
    use hipfort, only: hipSetDevice, hipMalloc, hipFree, hipMemcpy, hipDeviceSynchronize, &
                       hipMemcpyHostToDevice, hipMemcpyDeviceToHost
    use hipfort_rocrand, only: rocrand_create_generator, rocrand_destroy_generator, rocrand_set_seed, &
        rocrand_set_offset, rocrand_initialize_generator, rocrand_generate_uniform_double
    use hipfort_rocrand_enums, only: rocrand_rng_pseudo_default
#endif
#if GPU_SUITE_IMPL == 2
    use omp_lib
#endif
    implicit none
#if GPU_SUITE_IMPL != 0
    ! This rocRAND C API is not exported by all hipFORT module versions.
    ! Keep the same explicit DEFAULT ordering, including its state reset.
    integer(c_int), parameter :: rocrand_ordering_pseudo_default=101_c_int
    interface
        function rocrand_set_ordering(generator,ordering) bind(C,name='rocrand_set_ordering') result(status)
            import c_ptr,c_int
            type(c_ptr), value :: generator
            integer(c_int), value :: ordering
            integer(c_int) :: status
        end function
    end interface
#endif
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
    real(c_double), allocatable, target :: rand_out(:)
#if GPU_SUITE_IMPL == 0
    integer, allocatable :: seed_values(:)
    real(c_double), allocatable :: discard(:)
#else
    type(c_ptr) :: gen=c_null_ptr, d_rand=c_null_ptr
    integer(c_size_t) :: bytes
#endif
    logical :: data_present=.false.
contains
    subroutine host_setup()
        integer :: status,seed_size
        integer(c_int64_t) :: count
#if GPU_SUITE_IMPL == 2
        logical :: on_host
#endif
        count=checked_product(o%n,1)
#if GPU_SUITE_IMPL != 0
        call check(hipSetDevice(o%device),'hipSetDevice')
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
        allocate(rand_out(count),stat=status)
        call check(status,'host allocation')
        rand_out=0
#if GPU_SUITE_IMPL == 0
        call require(o%seed <= int(huge(0),c_int64_t),'Fortran seed exceeds default integer')
        call random_seed(size=seed_size)
        allocate(seed_values(seed_size),discard(4096),stat=status)
        call check(status,'seed/discard allocation')
        seed_values=int(o%seed)
#else
        bytes=int(count,c_size_t)*c_sizeof(rand_out(1))
#endif
    end subroutine
    subroutine reset()
        ! Generation overwrites all output; only generator state is restored.
    end subroutine
    subroutine setup()
#if GPU_SUITE_IMPL != 0
        call check(rocrand_create_generator(gen,rocrand_rng_pseudo_default),'rocrand_create_generator')
#endif
#if GPU_SUITE_IMPL == 1
        call check(hipMalloc(d_rand,bytes),'hipMalloc output')
#elif GPU_SUITE_IMPL == 2
        !$omp target enter data map(alloc:rand_out)
        data_present=.true.
        !$omp target data use_device_addr(rand_out)
        call check(check_mapped_device(c_loc(rand_out(1)),int(o%device,c_int)),'mapped random output device')
        !$omp end target data
#endif
        call restore()
    end subroutine
    subroutine restore()
#if GPU_SUITE_IMPL == 0
        integer(c_int64_t) :: remaining
        integer :: count
        call random_seed(put=seed_values)
        remaining=o%offset
        do while (remaining > 0)
            count=int(min(remaining,int(size(discard),c_int64_t)))
            call random_number(discard(1:count))
            remaining=remaining-count
        end do
#else
        call check(rocrand_set_seed(gen,o%seed),'rocrand_set_seed')
        call check(rocrand_set_offset(gen,o%offset),'rocrand_set_offset')
        call check(rocrand_set_ordering(gen,rocrand_ordering_pseudo_default),'rocrand_set_ordering')
        ! Seed/offset changes invalidate state: initialize before library timing.
        call check(rocrand_initialize_generator(gen),'rocrand_initialize_generator')
        call check(hipDeviceSynchronize(),'synchronize initialized generator')
#endif
    end subroutine
    subroutine operation(repetitions,timed)
        integer, intent(in) :: repetitions
        logical, intent(in) :: timed
        integer :: repetition
        if (timed) call start_timer()
#if GPU_SUITE_IMPL == 2
        ! C/C++ times one address region around the whole generation sequence.
        !$omp target data use_device_addr(rand_out)
        d_rand=c_loc(rand_out(1))
#endif
        do repetition=1,repetitions
#if GPU_SUITE_IMPL == 0
        call random_number(rand_out)
#else
        call check(rocrand_generate_uniform_double(gen,d_rand,int(o%n,c_size_t)), &
                   'rocrand_generate_uniform_double')
#endif
        end do
#if GPU_SUITE_IMPL == 2
        !$omp end target data
#endif
#if GPU_SUITE_IMPL != 0
        call check(hipDeviceSynchronize(),'generation completion')
#endif
        if (timed) call end_timer()
    end subroutine
    subroutine synchronize()
        ! restore() already completes generator initialization before timing.
    end subroutine
    subroutine download(end_to_end)
        logical, intent(in) :: end_to_end
#if GPU_SUITE_IMPL == 1
        call check(hipMemcpy(c_loc(rand_out(1)),d_rand,bytes,hipMemcpyDeviceToHost),'copy random output')
#elif GPU_SUITE_IMPL == 2
        ! Like C/C++, retrieve inside E2E, release in untimed cleanup.
        !$omp target update from(rand_out)
#endif
    end subroutine
    subroutine cleanup()
#if GPU_SUITE_IMPL != 0
        call check(rocrand_destroy_generator(gen),'rocrand_destroy_generator')
        gen=c_null_ptr
#endif
#if GPU_SUITE_IMPL == 1
        call check(hipFree(d_rand),'hipFree output')
        d_rand=c_null_ptr
#elif GPU_SUITE_IMPL == 2
        if (data_present) then
            !$omp target exit data map(delete:rand_out)
            data_present=.false.
        end if
#endif
    end subroutine
    subroutine verify(metrics,scale,getrf,getrs)
        real(c_double), intent(out) :: metrics(4),scale
        integer, intent(out) :: getrf,getrs
        getrf=0; getrs=0; scale=1
        if (.not. all(ieee_is_finite(rand_out))) then
            metrics=nan_value()
        else
            metrics(1)=minval(rand_out); metrics(2)=maxval(rand_out)
            metrics(3)=sum(rand_out)/real(o%n,c_double)
            metrics(4)=sum((rand_out-0.5_c_double)**2)/real(o%n,c_double)
        end if
    end subroutine
    subroutine host_cleanup()
        deallocate(rand_out)
#if GPU_SUITE_IMPL == 0
        deallocate(seed_values,discard)
#endif
    end subroutine
end module gpu_suite_workload

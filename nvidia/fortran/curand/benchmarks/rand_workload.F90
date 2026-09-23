module gpu_suite_workload
    use gpu_suite_fortran
    use iso_c_binding
    use ieee_arithmetic
#if GPU_SUITE_IMPL != 0
    use cudafor
    use curand
#endif
#if GPU_SUITE_IMPL == 2
    use openacc
#endif
    implicit none
    real(c_double), allocatable :: rand_out(:)
#if GPU_SUITE_IMPL == 0
    integer, allocatable :: seed_values(:)
    real(c_double), allocatable :: discard(:)
#elif GPU_SUITE_IMPL == 1
    real(c_double), device, allocatable :: d_rand(:)
#endif
#if GPU_SUITE_IMPL != 0
    type(curandGenerator) :: gen
#endif
    logical :: data_present=.false.
contains
    subroutine host_setup()
        integer :: status, seed_size
        integer(c_int64_t) :: count
        count=checked_product(o%n,1)
#if GPU_SUITE_IMPL != 0
        call check(cudaSetDevice(o%device),'cudaSetDevice')
#endif
#if GPU_SUITE_IMPL == 2
        call acc_set_device_num(o%device,acc_device_nvidia)
#endif
        allocate(rand_out(o%n),stat=status)
        call check(status,'host allocation')
#if GPU_SUITE_IMPL == 0
        call require(o%seed <= int(huge(0),c_int64_t),'Fortran seed exceeds default integer')
        call random_seed(size=seed_size)
        allocate(seed_values(seed_size),discard(4096),stat=status)
        call check(status,'seed/discard allocation')
        seed_values=int(o%seed)
#endif
    end subroutine
    subroutine reset()
        rand_out=0
    end subroutine
    subroutine setup()
#if GPU_SUITE_IMPL == 1
        integer :: status
        allocate(d_rand(o%n),stat=status)
        call check(status,'device allocation')
#elif GPU_SUITE_IMPL == 2
        !$acc enter data create(rand_out)
        data_present=.true.
#endif
#if GPU_SUITE_IMPL != 0
        call check(curandCreateGenerator(gen,CURAND_RNG_PSEUDO_DEFAULT),'curandCreateGenerator')
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
        call check(curandSetPseudoRandomGeneratorSeed(gen,o%seed),'curandSetPseudoRandomGeneratorSeed')
        call check(curandSetGeneratorOffset(gen,o%offset),'curandSetGeneratorOffset')
        call check(curandSetGeneratorOrdering(gen,CURAND_ORDERING_PSEUDO_DEFAULT),'curandSetGeneratorOrdering')
#endif
    end subroutine
    subroutine operation()
#if GPU_SUITE_IMPL == 0
        call random_number(rand_out)
#elif GPU_SUITE_IMPL == 1
        call check(curandGenerateUniformDouble(gen,d_rand,int(o%n,c_intptr_t)),'curandGenerateUniformDouble')
#else
        integer :: status
        !$acc host_data use_device(rand_out)
        status=curandGenerateUniformDouble(gen,rand_out,int(o%n,c_intptr_t))
        !$acc end host_data
        call check(status,'curandGenerateUniformDouble')
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
        call check(cudaMemcpy(rand_out,d_rand,o%n,cudaMemcpyDeviceToHost),'copy random result')
#elif GPU_SUITE_IMPL == 2
        if (end_to_end) then
            !$acc exit data copyout(rand_out)
            data_present=.false.
        else
            !$acc update self(rand_out)
        end if
#endif
    end subroutine
    subroutine cleanup()
#if GPU_SUITE_IMPL != 0
        call check(curandDestroyGenerator(gen),'curandDestroyGenerator')
#endif
#if GPU_SUITE_IMPL == 1
        deallocate(d_rand)
#elif GPU_SUITE_IMPL == 2
        if (data_present) then
            !$acc exit data delete(rand_out)
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

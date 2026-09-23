program openacc_curand
    use cudafor
    use curand
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    integer :: allocation_status

    integer(8), parameter :: num_rand = 2_8**24, seed = 1234_8
    integer :: istat
    type(curandGenerator) :: gen
    real(8), allocatable :: rand_out(:)

    allocate(rand_out(num_rand), stat=allocation_status)

    if (allocation_status /= 0) error stop 'allocation failed'
    istat = curandCreateGenerator(gen, CURAND_RNG_PSEUDO_DEFAULT)
    if (istat /= 0) error stop 'curandCreateGenerator failed'
    istat = curandSetPseudoRandomGeneratorSeed(gen, seed)
    if (istat /= 0) error stop 'curandSetPseudoRandomGeneratorSeed failed'


    !$acc data copyout(rand_out)
        !$acc host_data use_device(rand_out)
            istat = curandGenerateUniformDouble(gen, rand_out, num_rand)
    if (istat /= 0) error stop 'curandGenerateUniformDouble failed'
        !$acc end host_data
        istat = cudaDeviceSynchronize()
    if (istat /= 0) error stop 'cudaDeviceSynchronize failed'
    !$acc end data

    print *, "rand_out(1) = ", rand_out(1)

    block
        real(8) :: mean, moment, bound
        if (.not. all(ieee_is_finite(rand_out))) error stop 'nonfinite random value'
        if (minval(rand_out) < 0.0d0 .or. maxval(rand_out) > 1.0d0) error stop 'random range failed'
        mean = sum(rand_out)/real(num_rand,8)
        moment = sum((rand_out-0.5d0)**2)/real(num_rand,8)
        bound = 6.0d0/sqrt(12.0d0*real(num_rand,8))
        if (abs(mean-0.5d0) > bound) error stop 'random mean failed'
        bound = 6.0d0/sqrt(180.0d0*real(num_rand,8))
        if (abs(moment-1.0d0/12.0d0) > bound) error stop 'random moment failed'
    end block
    print *, 'verification PASS'
    istat = curandDestroyGenerator(gen)
    if (istat /= 0) error stop 'curandDestroyGenerator failed'
    deallocate(rand_out)
end program openacc_curand

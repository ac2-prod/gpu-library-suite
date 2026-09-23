program rand_gpu
    use cudafor
    use curand
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    integer :: allocation_status

    integer(8), parameter :: num_rand = 2_8**24, seed = 1234_8
    integer :: istat
    type(curandGenerator) :: gen
    real(8), allocatable :: h_rand(:)
    real(8), device, allocatable :: d_rand(:)

    allocate(h_rand(num_rand), stat=allocation_status)

    if (allocation_status /= 0) error stop 'allocation failed'
    allocate(d_rand(num_rand), stat=allocation_status)
    if (allocation_status /= 0) error stop 'allocation failed'
    istat = curandCreateGenerator(gen, CURAND_RNG_PSEUDO_DEFAULT)
    if (istat /= 0) error stop 'curandCreateGenerator failed'
    istat = curandSetPseudoRandomGeneratorSeed(gen, seed)
    if (istat /= 0) error stop 'curandSetPseudoRandomGeneratorSeed failed'

    istat = curandGenerateUniformDouble(gen, d_rand, num_rand)
    if (istat /= 0) error stop 'curandGenerateUniformDouble failed'


    istat = cudaMemcpy(h_rand, d_rand, num_rand, &
                       cudaMemcpyDeviceToHost)
    if (istat /= 0) error stop 'cudaMemcpy failed'

    block
        real(8) :: mean, moment, bound
        if (.not. all(ieee_is_finite(h_rand))) error stop 'nonfinite random value'
        if (minval(h_rand) < 0.0d0 .or. maxval(h_rand) > 1.0d0) error stop 'random range failed'
        mean = sum(h_rand)/real(num_rand,8)
        moment = sum((h_rand-0.5d0)**2)/real(num_rand,8)
        bound = 6.0d0/sqrt(12.0d0*real(num_rand,8))
        if (abs(mean-0.5d0) > bound) error stop 'random mean failed'
        bound = 6.0d0/sqrt(180.0d0*real(num_rand,8))
        if (abs(moment-1.0d0/12.0d0) > bound) error stop 'random moment failed'
    end block
    print *, 'verification PASS'
    istat = curandDestroyGenerator(gen)
    if (istat /= 0) error stop 'curandDestroyGenerator failed'
    deallocate(d_rand)
    deallocate(h_rand)
end program rand_gpu

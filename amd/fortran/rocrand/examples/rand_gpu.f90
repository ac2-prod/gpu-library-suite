program rand_gpu
    use, intrinsic :: iso_c_binding
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use hipfort, only: hipSetDevice, hipMalloc, hipFree, hipMemcpy, hipDeviceSynchronize, &
                       hipMemcpyHostToDevice, hipMemcpyDeviceToHost
    use hipfort_rocrand, only: rocrand_create_generator, rocrand_destroy_generator, rocrand_set_seed, &
        rocrand_set_offset, rocrand_initialize_generator, rocrand_generate_uniform_double
    use hipfort_rocrand_enums, only: rocrand_rng_pseudo_default
    implicit none
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
    integer(c_size_t), parameter :: num_rand=2_c_size_t**24
    integer(c_int64_t), parameter :: seed=1234_c_int64_t
    real(c_double), allocatable, target :: rand_out(:)
    type(c_ptr) :: gen=c_null_ptr, d_rand=c_null_ptr
    integer :: allocation_status

    call check(hipSetDevice(0), 'hipSetDevice')
    allocate(rand_out(num_rand),stat=allocation_status)
    if (allocation_status /= 0) error stop 'host allocation failed'
    call check(rocrand_create_generator(gen,rocrand_rng_pseudo_default), 'rocrand_create_generator')
    call check(rocrand_set_seed(gen,seed), 'rocrand_set_seed')
    call check(rocrand_set_offset(gen,0_c_int64_t), 'rocrand_set_offset')
    call check(rocrand_set_ordering(gen,rocrand_ordering_pseudo_default), 'rocrand_set_ordering')
    call check(hipMalloc(d_rand,num_rand*c_sizeof(rand_out(1))), 'hipMalloc')
    call check(rocrand_generate_uniform_double(gen,d_rand,num_rand), 'rocrand_generate_uniform_double')
    call check(hipDeviceSynchronize(), 'hipDeviceSynchronize')
    call check(hipMemcpy(c_loc(rand_out(1)),d_rand,num_rand*c_sizeof(rand_out(1)), &
        hipMemcpyDeviceToHost), 'copy result')
    if (.not. all(ieee_is_finite(rand_out))) error stop 'nonfinite random value'
    if (minval(rand_out) <= 0.0_c_double .or. maxval(rand_out) > 1.0_c_double) &
        error stop 'rocRAND range verification failed'
    print *, 'rocRAND generated ',num_rand,' uniform doubles in (0,1]; range ', &
             minval(rand_out),' to ',maxval(rand_out)
    call check(rocrand_destroy_generator(gen), 'rocrand_destroy_generator')
    call check(hipFree(d_rand), 'hipFree')
    deallocate(rand_out)
contains
    subroutine check(status,operation)
        integer(c_int), intent(in) :: status
        character(*), intent(in) :: operation
        if (status /= 0) then
            print *, trim(operation), ': status ',status
            error stop 1
        end if
    end subroutine
end program rand_gpu

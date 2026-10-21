module gpu_suite_fortran
    use, intrinsic :: iso_c_binding
    use, intrinsic :: ieee_arithmetic
    implicit none
    private
    public :: options, o, run_benchmark, require, check, checked_int, checked_product, nan_value
    public :: cuda_malloc_bytes, cuda_free_bytes
    type :: options
        integer :: library, implementation, n, m, k, batch, nx, ny, nrhs
        integer :: warmup, repeat, trials, scope, device, threads
        integer(c_int64_t) :: seed, offset
        logical :: verify, fftw_threaded
        real(c_double) :: alpha, beta
    end type
    type(options), save :: o
    interface
        ! Explicit byte-sized workspace allocation, including a genuine NULL
        ! workspace when the CUDA library requests zero bytes. Interfaces alone
        ! do not introduce a CUDA link dependency into CPU benchmarks.
        integer(c_int) function cuda_malloc_bytes(pointer, bytes) bind(C,name='cudaMalloc')
            import c_int, c_ptr, c_size_t
            type(c_ptr), intent(out) :: pointer
            integer(c_size_t), value :: bytes
        end function
        integer(c_int) function cuda_free_bytes(pointer) bind(C,name='cudaFree')
            import c_int, c_ptr
            type(c_ptr), value :: pointer
        end function
        subroutine initialize(library, implementation) bind(C,name='gpu_suite_f_initialize')
            import c_int
            integer(c_int), value :: library, implementation
        end subroutine
        subroutine argument(value) bind(C,name='gpu_suite_f_argument')
            import c_char
            character(kind=c_char), intent(in) :: value(*)
        end subroutine
        function parse() bind(C,name='gpu_suite_f_parse') result(status)
            import c_int
            integer(c_int) :: status
        end function
        subroutine fail(message) bind(C,name='gpu_suite_f_fail')
            import c_char
            character(kind=c_char), intent(in) :: message(*)
        end subroutine
        function integer_option(key) bind(C,name='gpu_suite_f_integer') result(value)
            import c_int, c_int64_t
            integer(c_int), value :: key
            integer(c_int64_t) :: value
        end function
        function real_option(key) bind(C,name='gpu_suite_f_real') result(value)
            import c_int, c_double
            integer(c_int), value :: key
            real(c_double) :: value
        end function
        subroutine trial(index) bind(C,name='gpu_suite_f_trial')
            import c_int
            integer(c_int), value :: index
        end subroutine
        subroutine start_timer() bind(C,name='gpu_suite_f_start')
        end subroutine
        subroutine end_timer() bind(C,name='gpu_suite_f_end')
        end subroutine
        function finish(metrics, scale, getrf, getrs) bind(C,name='gpu_suite_f_finish') result(status)
            import c_int, c_double
            real(c_double), intent(in) :: metrics(4)
            real(c_double), value :: scale
            integer(c_int), value :: getrf, getrs
            integer(c_int) :: status
        end function
        subroutine close_writer(status) bind(C,name='gpu_suite_f_close')
            import c_int
            integer(c_int), value :: status
        end subroutine
    end interface
    abstract interface
        subroutine action()
        end subroutine
        subroutine repeated_action(repetitions)
            integer, intent(in) :: repetitions
        end subroutine
        subroutine download_action(end_to_end)
            logical, intent(in) :: end_to_end
        end subroutine
        subroutine verification_action(metrics, scale, getrf, getrs)
            import c_double
            real(c_double), intent(out) :: metrics(4), scale
            integer, intent(out) :: getrf, getrs
        end subroutine
    end interface
contains
    subroutine require(condition, message)
        logical, intent(in) :: condition
        character(*), intent(in) :: message
        if (.not. condition) call fail(trim(message)//c_null_char)
    end subroutine
    subroutine check(status, operation)
        integer, intent(in) :: status
        character(*), intent(in) :: operation
        character(32) :: number
        if (status == 0) return
        write(number,'(I0)') status
        call require(status == 0, operation//': status '//trim(number))
    end subroutine
    function checked_int(value) result(converted)
        integer(c_int64_t), intent(in) :: value
        integer :: converted
        call require(value >= 0 .and. value <= int(huge(converted),c_int64_t), 'dimension exceeds LP64 integer')
        converted = int(value)
    end function
    function checked_product(a,b) result(product)
        integer, intent(in) :: a,b
        integer(c_int64_t) :: product
        call require(a > 0 .and. b > 0, 'nonpositive array dimensions')
        product = int(a,c_int64_t)*int(b,c_int64_t)
        call require(product <= huge(product)/16, 'array byte count overflows int64')
    end function
    function nan_value() result(value)
        real(c_double) :: value
        value = ieee_value(0.0_c_double,ieee_quiet_nan)
    end function
    subroutine read_options(library,implementation)
        integer, intent(in) :: library, implementation
        character(:), allocatable :: arg
        integer :: i, length, status
        call initialize(library, implementation)
        do i = 0, command_argument_count()
            call get_command_argument(i,length=length,status=status)
            call require(status == 0, 'command argument length failed')
            allocate(character(length) :: arg)
            call get_command_argument(i,value=arg,status=status)
            call require(status == 0, 'command argument read failed')
            call argument(arg//c_null_char)
            deallocate(arg)
        end do
        status = parse()
        if (status == 1) call close_writer(0)
        if (status /= 0) call close_writer(2)
        o%library=library; o%implementation=implementation
        o%n=checked_int(integer_option(1)); o%batch=checked_int(integer_option(2))
        if (library == 1) then
            o%m=checked_int(integer_option(3)); o%n=checked_int(integer_option(4))
            o%k=checked_int(integer_option(5))
        end if
        if (library == 2) then
            o%nx=checked_int(integer_option(6)); o%ny=checked_int(integer_option(7))
        end if
        o%nrhs=checked_int(integer_option(8)); o%seed=integer_option(9); o%offset=integer_option(10)
        o%warmup=int(integer_option(11)); o%repeat=int(integer_option(12)); o%trials=int(integer_option(13))
        o%scope=int(integer_option(14)); o%verify=integer_option(15)/=0
        o%device=int(integer_option(16)); o%threads=int(integer_option(17)); o%fftw_threaded=integer_option(18)/=0
        o%alpha=real_option(1); o%beta=real_option(2)
    end subroutine
    subroutine run_benchmark(library, implementation, host_setup, reset, setup, restore, &
                             operation, synchronize, download, cleanup, verify, host_cleanup)
        integer, intent(in) :: library, implementation
        procedure(action) :: host_setup, reset, setup, restore, synchronize, cleanup, host_cleanup
        procedure(repeated_action) :: operation
        procedure(download_action) :: download
        procedure(verification_action) :: verify
        integer :: warm, index, repetition, failures, info_getrf, info_getrs, status
        real(c_double) :: metrics(4), scale
        call read_options(library,implementation)
        call host_setup()
        call reset()
        if (o%scope == 0) then
            call setup()
            select case (library)
            case (0,2,5) ! FFT/SpMV/reduction: retain the context and complete once.
                if (o%warmup > 0) call operation(o%warmup)
                if (.not. (library == 0 .and. implementation == 2 .and. o%warmup == 0)) call synchronize()
            case (1) ! Direct CUDA DGEMM restores C on each warm-up; CPU/ACC do not.
                do warm=1,o%warmup
                    if (implementation == 1) call restore()
                    call operation(1)
                end do
                call synchronize()
            case (3) ! LU destroys A/B. No warm-up result download in C/C++.
                do warm=1,o%warmup
                    call reset()
                    call restore()
                    call operation(1)
                    call synchronize()
                end do
            case (4) ! Advance the generator across warm-up calls.
                do warm=1,o%warmup
                    call operation(1)
                    call synchronize()
                end do
            end select
        else if (library == 1 .and. implementation == 0) then
            ! CPU DGEMM warms the same C repeatedly in both reference scopes.
            if (o%warmup > 0) call operation(o%warmup)
        else
            do warm=1,o%warmup
                call reset()
                call setup()
                call operation(1)
                call synchronize()
                call download(.true.)
                call cleanup()
            end do
        end if
        failures=0
        do index = 0, o%trials-1
            call trial(index)
            if (o%scope == 0) then
                call reset()
                call restore()
                call synchronize()
                call start_timer()
                ! The workload encloses its entire loop in the corresponding
                ! C/C++ device-address region; synchronization stays outside it.
                call operation(o%repeat)
                call synchronize()
                call end_timer()
                call download(.false.)
            else
                do repetition=1,o%repeat
                    ! Host allocation/input restoration are outside every one-shot interval.
                    call reset()
                    call start_timer()
                    call setup()
                    call operation(1)
                    call synchronize()
                    call download(.true.)
                    call end_timer()
                    call cleanup()
                end do
            end if
            metrics=0; scale=1; info_getrf=0; info_getrs=0
            call verify(metrics,scale,info_getrf,info_getrs)
            status=finish(metrics,scale,info_getrf,info_getrs)
            failures=max(failures,status)
        end do
        if (o%scope == 0) call cleanup()
        call host_cleanup()
        call close_writer(failures)
    end subroutine
end module gpu_suite_fortran

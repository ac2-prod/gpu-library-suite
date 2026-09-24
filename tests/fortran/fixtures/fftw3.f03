! Minimal test-only C interop declarations, not a copy of FFTW's distribution.
integer(c_int), parameter :: FFTW_FORWARD=-1, FFTW_ESTIMATE=64
interface
    type(c_ptr) function fftwf_alloc_complex(n) bind(C)
        import c_ptr,c_size_t
        integer(c_size_t), value :: n
    end function
    subroutine fftwf_free(p) bind(C)
        import c_ptr
        type(c_ptr), value :: p
    end subroutine
    type(c_ptr) function fftwf_plan_many_dft(rank,n,batch,input,inembed,istride,idist, &
                                           output,onembed,ostride,odist,sign,flags) bind(C)
        import c_ptr,c_int,c_float_complex
        integer(c_int), value :: rank,batch,istride,idist,ostride,odist,sign,flags
        integer(c_int) :: n(*),inembed(*),onembed(*)
        complex(c_float_complex) :: input(*),output(*)
    end function
    subroutine fftwf_execute_dft(plan,input,output) bind(C)
        import c_ptr,c_float_complex
        type(c_ptr), value :: plan
        complex(c_float_complex) :: input(*),output(*)
    end subroutine
    subroutine fftwf_destroy_plan(plan) bind(C)
        import c_ptr
        type(c_ptr), value :: plan
    end subroutine
    integer(c_int) function fftwf_init_threads() bind(C)
        import c_int
    end function
    subroutine fftwf_plan_with_nthreads(n) bind(C)
        import c_int
        integer(c_int), value :: n
    end subroutine
    subroutine fftwf_cleanup_threads() bind(C)
    end subroutine
end interface

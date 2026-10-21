! Explicit bindings for the rocSPARSE generic descriptor and v2 SpMV C ABI.
! Kept separate because installed hipFORT releases may lack the v2 bindings.
! API: https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/generic.html
! Enum ABI: ROCm/rocm-libraries projects/rocsparse/library/include/rocsparse-types.h
module rocsparse_v2_c
    use iso_c_binding
    implicit none
    private
    public :: rocsparse_operation_none,rocsparse_index_base_zero,rocsparse_indextype_i32,rocsparse_datatype_f64_r
    public :: rocsparse_spmv_alg_csr_rowsplit,rocsparse_spmv_input_alg,rocsparse_spmv_input_operation
    public :: rocsparse_spmv_input_scalar_datatype,rocsparse_spmv_input_compute_datatype
    public :: rocsparse_v2_spmv_stage_analysis,rocsparse_v2_spmv_stage_compute
    public :: rocsparse_create_handle,rocsparse_destroy_handle,rocsparse_create_csr_descr,rocsparse_destroy_spmat_descr
    public :: rocsparse_create_dnvec_descr,rocsparse_destroy_dnvec_descr,rocsparse_create_spmv_descr,rocsparse_destroy_spmv_descr
    public :: rocsparse_spmv_set_input,rocsparse_v2_spmv_buffer_size,rocsparse_v2_spmv
    integer(c_int), parameter :: rocsparse_operation_none=111_c_int
    integer(c_int), parameter :: rocsparse_index_base_zero=0_c_int
    integer(c_int), parameter :: rocsparse_indextype_i32=2_c_int
    integer(c_int), parameter :: rocsparse_datatype_f64_r=152_c_int
    integer(c_int), parameter :: rocsparse_spmv_alg_csr_rowsplit=3_c_int
    integer(c_int), parameter :: rocsparse_spmv_input_alg=0_c_int
    integer(c_int), parameter :: rocsparse_spmv_input_operation=1_c_int
    integer(c_int), parameter :: rocsparse_spmv_input_scalar_datatype=2_c_int
    integer(c_int), parameter :: rocsparse_spmv_input_compute_datatype=3_c_int
    integer(c_int), parameter :: rocsparse_v2_spmv_stage_analysis=0_c_int
    integer(c_int), parameter :: rocsparse_v2_spmv_stage_compute=1_c_int
    interface
        integer(c_int) function rocsparse_create_handle(handle) bind(C)
            import c_int,c_ptr
            type(c_ptr), intent(out) :: handle
        end function
        integer(c_int) function rocsparse_destroy_handle(handle) bind(C)
            import c_int,c_ptr
            type(c_ptr), value :: handle
        end function
        integer(c_int) function rocsparse_create_csr_descr(matrix,m,n,nnz,row,col,val,rt,ct,base,dt) bind(C)
            import c_int,c_int64_t,c_ptr
            type(c_ptr), intent(out) :: matrix
            integer(c_int64_t), value :: m,n,nnz
            type(c_ptr), value :: row,col,val
            integer(c_int), value :: rt,ct,base,dt
        end function
        integer(c_int) function rocsparse_destroy_spmat_descr(matrix) bind(C)
            import c_int,c_ptr
            type(c_ptr), value :: matrix
        end function
        integer(c_int) function rocsparse_create_dnvec_descr(vec,n,data,dt) bind(C)
            import c_int,c_int64_t,c_ptr
            type(c_ptr), intent(out) :: vec
            integer(c_int64_t), value :: n
            type(c_ptr), value :: data
            integer(c_int), value :: dt
        end function
        integer(c_int) function rocsparse_destroy_dnvec_descr(vec) bind(C)
            import c_int,c_ptr
            type(c_ptr), value :: vec
        end function
        integer(c_int) function rocsparse_create_spmv_descr(descr) bind(C)
            import c_int,c_ptr
            type(c_ptr), intent(out) :: descr
        end function
        integer(c_int) function rocsparse_destroy_spmv_descr(descr) bind(C)
            import c_int,c_ptr
            type(c_ptr), value :: descr
        end function
        integer(c_int) function rocsparse_spmv_set_input(handle,descr,field,data,bytes,error) bind(C)
            import c_int,c_size_t,c_ptr
            type(c_ptr), value :: handle,descr,data,error
            integer(c_int), value :: field
            integer(c_size_t), value :: bytes
        end function
        integer(c_int) function rocsparse_v2_spmv_buffer_size(handle,descr,matrix,x,y,stage,bytes,error) bind(C)
            import c_int,c_size_t,c_ptr
            type(c_ptr), value :: handle,descr,matrix,x,y,error
            integer(c_int), value :: stage
            integer(c_size_t), intent(out) :: bytes
        end function
        integer(c_int) function rocsparse_v2_spmv(handle,descr,alpha,matrix,x,beta,y,stage,bytes,buffer,error) bind(C)
            import c_int,c_size_t,c_ptr
            type(c_ptr), value :: handle,descr,alpha,matrix,x,beta,y,buffer,error
            integer(c_int), value :: stage
            integer(c_size_t), value :: bytes
        end function
    end interface
end module rocsparse_v2_c

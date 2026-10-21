program sparse_gpu
    use iso_c_binding
    use ieee_arithmetic
    use amd_poisson_csr
    use hipfort, only: hipMalloc,hipFree,hipMemcpy,hipSetDevice,hipDeviceSynchronize, &
                       hipMemcpyHostToDevice,hipMemcpyDeviceToHost
    use rocsparse_v2_c
    implicit none
    integer, parameter :: nx=1024,ny=1024
    integer :: status,ix,iy,p,neighbors
    real(c_double) :: max_error,expected
    logical :: failed=.false.
    integer(c_int), allocatable, target :: row_offsets(:),col_indices(:)
    real(c_double), allocatable, target :: values(:),vec_x(:),vec_y(:)
    real(c_double), target :: alpha,beta
    integer(c_int) :: nrow,ncol,nnz
    type(c_ptr) :: mat_A=c_null_ptr
    type(c_ptr) :: handle=c_null_ptr,x_descr=c_null_ptr,y_descr=c_null_ptr,spmv=c_null_ptr
    type(c_ptr) :: d_row=c_null_ptr,d_col=c_null_ptr,d_values=c_null_ptr,d_x=c_null_ptr,d_y=c_null_ptr
    type(c_ptr) :: workspace=c_null_ptr
    integer(c_size_t) :: buffer_size=0
    integer(c_int), target :: alg=rocsparse_spmv_alg_csr_rowsplit
    integer(c_int), target :: op=rocsparse_operation_none,datatype=rocsparse_datatype_f64_r
    logical :: data_present=.false.

    call make_poisson2d_csr(nx,ny,nrow,ncol,nnz,row_offsets,col_indices,values)
    allocate(vec_x(ncol),vec_y(nrow),stat=status)
    call check(status,'host allocation')
    vec_x=1; vec_y=1; alpha=1; beta=1
    call check(hipSetDevice(0_c_int),'hipSetDevice')
    call check(hipMalloc(d_row,size(row_offsets,kind=c_size_t)*c_sizeof(row_offsets(1))),'allocate row offsets')
    call check(hipMalloc(d_col,size(col_indices,kind=c_size_t)*c_sizeof(col_indices(1))),'allocate columns')
    call check(hipMalloc(d_values,size(values,kind=c_size_t)*c_sizeof(values(1))),'allocate values')
    call check(hipMalloc(d_x,size(vec_x,kind=c_size_t)*c_sizeof(vec_x(1))),'allocate x')
    call check(hipMalloc(d_y,size(vec_y,kind=c_size_t)*c_sizeof(vec_y(1))),'allocate y')
    call check(hipMemcpy(d_row,c_loc(row_offsets(1)),size(row_offsets,kind=c_size_t)*c_sizeof(row_offsets(1)), &
               hipMemcpyHostToDevice),'copy CSR offsets')
    call check(hipMemcpy(d_col,c_loc(col_indices(1)),size(col_indices,kind=c_size_t)*c_sizeof(col_indices(1)), &
               hipMemcpyHostToDevice),'copy CSR columns')
    call check(hipMemcpy(d_values,c_loc(values(1)),size(values,kind=c_size_t)*c_sizeof(values(1)), &
               hipMemcpyHostToDevice),'copy CSR values')
    call check(hipMemcpy(d_x,c_loc(vec_x(1)),size(vec_x,kind=c_size_t)*c_sizeof(vec_x(1)), &
               hipMemcpyHostToDevice),'restore x')
    call check(hipMemcpy(d_y,c_loc(vec_y(1)),size(vec_y,kind=c_size_t)*c_sizeof(vec_y(1)), &
               hipMemcpyHostToDevice),'restore initial y')
    call check(rocsparse_create_handle(handle),'rocsparse_create_handle')
    call check(rocsparse_create_csr_descr(mat_A,int(nrow,c_int64_t),int(ncol,c_int64_t),int(nnz,c_int64_t), &
               d_row,d_col,d_values,rocsparse_indextype_i32,rocsparse_indextype_i32, &
               rocsparse_index_base_zero,rocsparse_datatype_f64_r),'rocsparse_create_csr_descr')
    call check(rocsparse_create_dnvec_descr(x_descr,int(ncol,c_int64_t),d_x,rocsparse_datatype_f64_r), &
               'rocsparse_create_dnvec_descr x')
    call check(rocsparse_create_dnvec_descr(y_descr,int(nrow,c_int64_t),d_y,rocsparse_datatype_f64_r), &
               'rocsparse_create_dnvec_descr y')
    call check(rocsparse_create_spmv_descr(spmv),'rocsparse_create_spmv_descr')
    call check(rocsparse_spmv_set_input(handle,spmv,rocsparse_spmv_input_alg,c_loc(alg),c_sizeof(alg),c_null_ptr), &
               'SpMV algorithm')
    call check(rocsparse_spmv_set_input(handle,spmv,rocsparse_spmv_input_operation,c_loc(op),c_sizeof(op),c_null_ptr), &
               'SpMV operation')
    call check(rocsparse_spmv_set_input(handle,spmv,rocsparse_spmv_input_scalar_datatype, &
               c_loc(datatype),c_sizeof(datatype),c_null_ptr),'SpMV scalar type')
    call check(rocsparse_spmv_set_input(handle,spmv,rocsparse_spmv_input_compute_datatype, &
               c_loc(datatype),c_sizeof(datatype),c_null_ptr),'SpMV compute type')
    buffer_size=0
    call check(rocsparse_v2_spmv_buffer_size(handle,spmv,mat_A,x_descr,y_descr, &
               rocsparse_v2_spmv_stage_analysis,buffer_size,c_null_ptr),'SpMV analysis workspace size')
    call require(buffer_size>=0,'SpMV workspace size overflow')
    if (buffer_size>0) call check(hipMalloc(workspace,buffer_size),'allocate analysis workspace')
    call check(rocsparse_v2_spmv(handle,spmv,c_loc(alpha),mat_A,x_descr,c_loc(beta),y_descr, &
               rocsparse_v2_spmv_stage_analysis,buffer_size,workspace,c_null_ptr),'SpMV analysis')
    call check(hipDeviceSynchronize(),'synchronize SpMV analysis')
    if (c_associated(workspace)) call check(hipFree(workspace),'free analysis workspace')
    workspace=c_null_ptr
    call check(rocsparse_v2_spmv_buffer_size(handle,spmv,mat_A,x_descr,y_descr, &
               rocsparse_v2_spmv_stage_compute,buffer_size,c_null_ptr),'SpMV compute workspace size')
    call require(buffer_size>=0,'SpMV workspace size overflow')
    if (buffer_size>0) call check(hipMalloc(workspace,buffer_size),'allocate compute workspace')
    call check(rocsparse_v2_spmv(handle,spmv,c_loc(alpha),mat_A,x_descr,c_loc(beta),y_descr, &
               rocsparse_v2_spmv_stage_compute,buffer_size,workspace,c_null_ptr),'SpMV compute')
    call check(hipDeviceSynchronize(),'hipDeviceSynchronize')
    call check(hipMemcpy(c_loc(vec_y(1)),d_y,size(vec_y,kind=c_size_t)*c_sizeof(vec_y(1)), &
               hipMemcpyDeviceToHost),'copy y result')
    max_error=0
    call require(all(ieee_is_finite(vec_y)),'nonfinite SpMV result')
    do iy=1,ny
        do ix=1,nx
            p=(iy-1)*nx+ix
            neighbors=0
            if (ix>1) neighbors=neighbors+1
            if (ix<nx) neighbors=neighbors+1
            if (iy>1) neighbors=neighbors+1
            if (iy<ny) neighbors=neighbors+1
            expected=real(5-neighbors,c_double)
            max_error=max(max_error,abs(vec_y(p)-expected))
        end do
    end do
    call release()
    if (failed) error stop 'resource cleanup failed'
    deallocate(row_offsets,col_indices,values,vec_x,vec_y)
    print '(a,es16.8)', 'Poisson SpMV complete; max error = ',max_error
    if (max_error>1.0e-12_c_double) error stop 'SpMV verification failed'
contains
    subroutine check(code,operation)
        integer, intent(in) :: code
        character(*), intent(in) :: operation
        if (code/=0) then
            print *,trim(operation),' failed: ',code
            call release()
            error stop 1
        end if
    end subroutine
    subroutine require(condition,message)
        logical, intent(in) :: condition
        character(*), intent(in) :: message
        if (.not.condition) then
            print *,message
            call release()
            error stop 1
        end if
    end subroutine
    subroutine record_cleanup(code)
        integer, intent(in) :: code
        if (code/=0) failed=.true.
    end subroutine
    subroutine release()
        if (c_associated(spmv)) call record_cleanup(rocsparse_destroy_spmv_descr(spmv))
        spmv=c_null_ptr
        if (c_associated(y_descr)) call record_cleanup(rocsparse_destroy_dnvec_descr(y_descr))
        y_descr=c_null_ptr
        if (c_associated(x_descr)) call record_cleanup(rocsparse_destroy_dnvec_descr(x_descr))
        x_descr=c_null_ptr
        if (c_associated(mat_A)) call record_cleanup(rocsparse_destroy_spmat_descr(mat_A))
        mat_A=c_null_ptr
        if (c_associated(workspace)) call record_cleanup(hipFree(workspace))
        workspace=c_null_ptr
        if (c_associated(handle)) call record_cleanup(rocsparse_destroy_handle(handle))
        handle=c_null_ptr
        if (c_associated(d_y)) call record_cleanup(hipFree(d_y))
        d_y=c_null_ptr
        if (c_associated(d_x)) call record_cleanup(hipFree(d_x))
        d_x=c_null_ptr
        if (c_associated(d_values)) call record_cleanup(hipFree(d_values))
        d_values=c_null_ptr
        if (c_associated(d_col)) call record_cleanup(hipFree(d_col))
        d_col=c_null_ptr
        if (c_associated(d_row)) call record_cleanup(hipFree(d_row))
        d_row=c_null_ptr
    end subroutine
end program sparse_gpu

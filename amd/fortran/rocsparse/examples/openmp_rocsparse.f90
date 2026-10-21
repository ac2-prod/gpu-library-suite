program openmp_rocsparse
    use iso_c_binding
    use ieee_arithmetic
    use amd_poisson_csr
    use hipfort, only: hipMalloc,hipFree,hipMemcpy,hipSetDevice,hipDeviceSynchronize, &
                       hipMemcpyHostToDevice,hipMemcpyDeviceToHost
    use rocsparse_v2_c
    use omp_lib
    implicit none
    integer, parameter :: nx=1024,ny=1024
    integer :: status,ix,iy,p,neighbors
    real(c_double) :: max_error,expected
    logical :: failed=.false.
    logical :: on_host
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
    call require(omp_get_num_devices()>0,'OpenMP device unavailable')
    call omp_set_default_device(0)
    on_host=.true.
    !$omp target map(from:on_host)
    on_host=omp_is_initial_device()
    !$omp end target
    call require(.not.on_host,'OpenMP target fell back to host')
    !$omp target enter data map(to:row_offsets,col_indices,values,vec_x,vec_y)
    data_present=.true.
    !$omp target data use_device_addr(row_offsets,col_indices,values,vec_x,vec_y)
    d_row=c_loc(row_offsets(1)); d_col=c_loc(col_indices(1)); d_values=c_loc(values(1))
    d_x=c_loc(vec_x(1)); d_y=c_loc(vec_y(1))
    !$omp end target data
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
    !$omp target update from(vec_y)
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
        if (data_present) then
            !$omp target exit data map(delete:row_offsets,col_indices,values,vec_x,vec_y)
            data_present=.false.
        end if
    end subroutine
end program openmp_rocsparse

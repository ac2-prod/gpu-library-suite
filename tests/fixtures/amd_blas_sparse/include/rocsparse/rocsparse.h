#ifndef GPU_SUITE_FAKE_ROCSPARSE_H
#define GPU_SUITE_FAKE_ROCSPARSE_H
// Host-only API fixture; signatures follow the ROCm 7.2 generic API.
#include <stddef.h>
#include <stdint.h>
typedef struct fake_sparse_handle *rocsparse_handle;
typedef struct fake_matrix *rocsparse_spmat_descr;
typedef const struct fake_matrix *rocsparse_const_spmat_descr;
typedef struct fake_vector *rocsparse_dnvec_descr;
typedef const struct fake_vector *rocsparse_const_dnvec_descr;
typedef struct fake_spmv *rocsparse_spmv_descr;
typedef struct fake_error *rocsparse_error;
enum rocsparse_status { rocsparse_status_success, rocsparse_status_internal_error };
enum rocsparse_operation { rocsparse_operation_none };
enum rocsparse_indextype { rocsparse_indextype_i32 };
enum rocsparse_index_base { rocsparse_index_base_zero };
enum rocsparse_datatype { rocsparse_datatype_f64_r };
enum rocsparse_spmv_alg { rocsparse_spmv_alg_csr_rowsplit };
enum rocsparse_v2_spmv_stage { rocsparse_v2_spmv_stage_analysis,
    rocsparse_v2_spmv_stage_compute };
enum rocsparse_spmv_input { rocsparse_spmv_input_alg, rocsparse_spmv_input_operation,
    rocsparse_spmv_input_scalar_datatype, rocsparse_spmv_input_compute_datatype };
rocsparse_status rocsparse_create_handle(rocsparse_handle *);
rocsparse_status rocsparse_destroy_handle(rocsparse_handle);
rocsparse_status rocsparse_create_csr_descr(rocsparse_spmat_descr *, int64_t,
    int64_t, int64_t, void *, void *, void *, rocsparse_indextype,
    rocsparse_indextype, rocsparse_index_base, rocsparse_datatype);
rocsparse_status rocsparse_create_dnvec_descr(rocsparse_dnvec_descr *, int64_t,
    void *, rocsparse_datatype);
rocsparse_status rocsparse_destroy_spmat_descr(rocsparse_spmat_descr);
rocsparse_status rocsparse_destroy_dnvec_descr(rocsparse_dnvec_descr);
rocsparse_status rocsparse_create_spmv_descr(rocsparse_spmv_descr *);
rocsparse_status rocsparse_destroy_spmv_descr(rocsparse_spmv_descr);
rocsparse_status rocsparse_spmv_set_input(rocsparse_handle, rocsparse_spmv_descr,
    rocsparse_spmv_input, const void *, size_t, rocsparse_error *);
rocsparse_status rocsparse_v2_spmv_buffer_size(rocsparse_handle, rocsparse_spmv_descr,
    rocsparse_const_spmat_descr, rocsparse_const_dnvec_descr,
    rocsparse_const_dnvec_descr, rocsparse_v2_spmv_stage, size_t *, rocsparse_error *);
rocsparse_status rocsparse_v2_spmv(rocsparse_handle, rocsparse_spmv_descr,
    const void *, rocsparse_const_spmat_descr, rocsparse_const_dnvec_descr,
    const void *, rocsparse_dnvec_descr, rocsparse_v2_spmv_stage,
    size_t, void *, rocsparse_error *);
rocsparse_status rocsparse_get_version(rocsparse_handle, int *);
rocsparse_status rocsparse_get_git_rev(rocsparse_handle, char *);
#endif

#ifndef GPU_SUITE_TEST_CUSPARSE_H
#define GPU_SUITE_TEST_CUSPARSE_H

#include "cuda_runtime.h"

#include <stddef.h>
#include <stdint.h>

typedef void *cusparseHandle_t;
typedef void *cusparseSpMatDescr_t;
typedef void *cusparseDnVecDescr_t;
typedef int cusparseStatus_t;

enum {
  CUSPARSE_STATUS_SUCCESS = 0,
  CUSPARSE_INDEX_32I = 1,
  CUSPARSE_INDEX_BASE_ZERO = 2,
  CUSPARSE_OPERATION_NON_TRANSPOSE = 3,
  CUSPARSE_SPMV_ALG_DEFAULT = 4
};

cusparseStatus_t cusparseCreate(cusparseHandle_t *handle);
cusparseStatus_t cusparseDestroy(cusparseHandle_t handle);
cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t *matrix, int64_t rows,
                                   int64_t columns, int64_t nnz,
                                   void *row_offsets, void *column_indices,
                                   void *values, int row_offset_type,
                                   int column_index_type, int index_base,
                                   int value_type);
cusparseStatus_t cusparseDestroySpMat(cusparseSpMatDescr_t matrix);
cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t *vector, int64_t size,
                                     void *values, int value_type);
cusparseStatus_t cusparseDestroyDnVec(cusparseDnVecDescr_t vector);
cusparseStatus_t
cusparseSpMV_bufferSize(cusparseHandle_t handle, int operation,
                        const void *alpha, cusparseSpMatDescr_t matrix,
                        cusparseDnVecDescr_t x, const void *beta,
                        cusparseDnVecDescr_t y, int compute_type, int algorithm,
                        size_t *buffer_size);
cusparseStatus_t cusparseSpMV(cusparseHandle_t handle, int operation,
                              const void *alpha, cusparseSpMatDescr_t matrix,
                              cusparseDnVecDescr_t x, const void *beta,
                              cusparseDnVecDescr_t y, int compute_type,
                              int algorithm, void *buffer);

#endif

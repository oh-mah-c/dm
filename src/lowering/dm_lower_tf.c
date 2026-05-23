#include "lowering/dm_lower_tf.h"
#include "core/dm_dtype.h"
#include <tensorflow/c/c_api.h>
#include <tensorflow/c/eager/c_api.h>
#include <stdlib.h>
#include <string.h>

static TF_DataType dm_to_tf_dtype(DM_DType dtype) {
    switch(dtype) {
        case DM_DTYPE_F32: return TF_FLOAT;
        case DM_DTYPE_F64: return TF_DOUBLE;
        case DM_DTYPE_I32: return TF_INT32;
        case DM_DTYPE_I64: return TF_INT64;
        case DM_DTYPE_U8:  return TF_UINT8;
        case DM_DTYPE_BOOL:return TF_BOOL;
        case DM_DTYPE_STRING: return TF_STRING;
        default:           return TF_FLOAT;
    }
}

static DM_DType tf_to_dm_dtype(TF_DataType dtype) {
    switch(dtype) {
        case TF_FLOAT:  return DM_DTYPE_F32;
        case TF_DOUBLE: return DM_DTYPE_F64;
        case TF_INT32:  return DM_DTYPE_I32;
        case TF_INT64:  return DM_DTYPE_I64;
        case TF_UINT8:  return DM_DTYPE_U8;
        case TF_BOOL:   return DM_DTYPE_BOOL;
        case TF_STRING: return DM_DTYPE_STRING;
        default:        return DM_DTYPE_F32;
    }
}

static void noop_dealloc(void *data, size_t len, void *arg) {
    (void)data; (void)len; (void)arg;
}

int dm_block_to_tf_tensor(const DM_Block *block, DM_TF_Tensor **out_tensor) {
    if (!block || !out_tensor) return -1;
    
    TF_Status *s = TF_NewStatus();
    TF_Tensor *tf_t = TF_NewTensor(
        dm_to_tf_dtype(block->dtype),
        block->shape,
        block->ndim,
        block->data,
        block->bytes,
        noop_dealloc,
        NULL
    );
    
    if (!tf_t) {
        TF_DeleteStatus(s);
        return -1;
    }
    
    TFE_TensorHandle *h = TFE_NewTensorHandle(tf_t, s);
    TF_DeleteTensor(tf_t);
    
    if (TF_GetCode(s) != TF_OK) {
        TF_DeleteStatus(s);
        return -1;
    }
    
    TF_DeleteStatus(s);
    *out_tensor = (DM_TF_Tensor*)h;
    return 0;
}

int dm_tf_tensor_to_block(const DM_TF_Tensor *tensor, DM_Block *out, DM_BlockKind kind, DM_Role role) {
    if (!tensor || !out) return -1;
    
    TFE_TensorHandle *h = (TFE_TensorHandle*)tensor;
    TF_Status *s = TF_NewStatus();
    TF_Tensor *tf_t = TFE_TensorHandleResolve(h, s);
    
    if (TF_GetCode(s) != TF_OK) {
        TF_DeleteStatus(s);
        return -1;
    }
    
    int ndim = TF_NumDims(tf_t);
    int64_t shape[DM_MAX_DIMS];
    for (int i = 0; i < ndim && i < DM_MAX_DIMS; ++i) {
        shape[i] = TF_Dim(tf_t, i);
    }
    
    DM_DType dtype = tf_to_dm_dtype(TF_TensorType(tf_t));
    
    dm_block_create(out, kind, dtype, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, ndim, shape);
    out->role = role;
    
    if (out->bytes > 0 && out->data) {
        memcpy(out->data, TF_TensorData(tf_t), out->bytes);
    }
    
    TF_DeleteTensor(tf_t);
    TF_DeleteStatus(s);
    return 0;
}

void dm_tf_tensor_free(void *tensor) {
    if (!tensor) return;
    TFE_TensorHandle *h = (TFE_TensorHandle*)tensor;
    // We assume the handle wasn't passed into an op that consumed it.
    // TFE_DeleteTensorHandle is not exported directly in some TF versions without Eager API header,
    // but we have <tensorflow/c/eager/c_api.h>.
    TFE_DeleteTensorHandle(h);
}

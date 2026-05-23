#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "models/tensor.h"
#include <stdio.h>

static TFE_Context* tf_ctx = NULL;

static void dm_tf_init() {
    if (tf_ctx != NULL) return;
    TF_Status* status = TF_NewStatus();
    TFE_ContextOptions* opts = TFE_NewContextOptions();
    tf_ctx = TFE_NewContext(opts, status);
    TFE_DeleteContextOptions(opts);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Failed to initialize TensorFlow Eager Context: %s\n", TF_Message(status));
    }
    TF_DeleteStatus(status);
}

static void dummy_deallocator(void* data, size_t length, void* arg) {}

static TFE_TensorHandle* dm_to_tf_nocopy(const DM_Tensor* t) {
    if (!t) return NULL;
    dm_tf_init();
    int64_t dims[4] = {t->n, t->c, t->h, t->w};
    size_t size = t->n * t->c * t->h * t->w * sizeof(float);
    TF_Tensor* tf_t = TF_NewTensor(TF_FLOAT, dims, 4, t->data, size, dummy_deallocator, NULL);
    
    TF_Status* status = TF_NewStatus();
    TFE_TensorHandle* h = TFE_NewTensorHandle(tf_t, status);
    TF_DeleteStatus(status);
    TF_DeleteTensor(tf_t);
    return h;
}

static void tf_to_dm_nocopy(TFE_TensorHandle* h, DM_Tensor* out) {
    if (!h || !out) return;
    TF_Status* status = TF_NewStatus();
    TF_Tensor* tf_t = TFE_TensorHandleResolve(h, status);
    if (TF_GetCode(status) == TF_OK) {
        size_t size = TF_TensorByteSize(tf_t);
        memcpy(out->data, TF_TensorData(tf_t), size);
    } else {
        fprintf(stderr, "Failed to resolve TF handle to DM_Tensor: %s\n", TF_Message(status));
    }
    TF_DeleteTensor(tf_t);
    TF_DeleteStatus(status);
}

static TFE_TensorHandle* execute_tf_op1(const char* op_name, TFE_TensorHandle* input) {
    dm_tf_init();
    TF_Status* status = TF_NewStatus();
    TFE_Op* op = TFE_NewOp(tf_ctx, op_name, status);
    TFE_OpAddInput(op, input, status);
    
    TFE_TensorHandle* retvals[1] = {NULL};
    int num_retvals = 1;
    TFE_Execute(op, retvals, &num_retvals, status);
    
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "TF Op %s failed: %s\n", op_name, TF_Message(status));
    }
    TFE_DeleteOp(op);
    TF_DeleteStatus(status);
    return retvals[0];
}

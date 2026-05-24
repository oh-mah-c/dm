import re
import os

filepath = 'src/core/dm_engine.c'
with open(filepath, 'r') as f:
    text = f.read()

# Replace execute_conv2d signature and body
def replace_execute_conv2d(m):
    return '''static int execute_conv2d(const DM_Block *input, const DM_Block *filter, int stride, DM_Block *out) {
    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "Conv2D", s);
    TFE_OpAddInput(op, (TFE_TensorHandle*)input->handle, s);
    TFE_OpAddInput(op, (TFE_TensorHandle*)filter->handle, s);
    int64_t strides[4]   = {1, 1, stride, stride};
    int64_t dilations[4] = {1, 1, 1, 1};
    TFE_OpSetAttrIntList(op, "strides",   strides,   4);
    TFE_OpSetAttrIntList(op, "dilations", dilations, 4);
    TFE_OpSetAttrString(op, "padding",     "SAME",  4);
    TFE_OpSetAttrString(op, "data_format", "NCHW",  4);
    TFE_TensorHandle *ret[1] = {NULL};
    int nret = 1;
    TFE_Execute(op, ret, &nret, s);
    if (TF_GetCode(s) != TF_OK) {
        fprintf(stderr, "[dm_engine] Conv2D failed: %s\\n", TF_Message(s));
        TFE_DeleteOp(op);
        TF_DeleteStatus(s);
        return -1;
    }
    TFE_DeleteOp(op);
    TF_DeleteStatus(s);
    
    dm_block_create(out, DM_KIND_EXTERNAL, DM_DTYPE_F32, DM_LAYOUT_TF_HANDLE, DM_BACKEND_TENSORFLOW, 0, NULL);
    out->handle = ret[0];
    out->owns_handle = 1;
    out->handle_destructor = dm_tf_tensor_free;
    return 0;
}'''

text = re.sub(r'static TFE_TensorHandle \*execute_conv2d.*?return ret\[0\];\s*\}', replace_execute_conv2d, text, flags=re.DOTALL)

def replace_execute_bias_add(m):
    return '''static int execute_bias_add(const DM_Block *input, const DM_Block *bias, DM_Block *out) {
    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "BiasAdd", s);
    TFE_OpAddInput(op, (TFE_TensorHandle*)input->handle, s);
    TFE_OpAddInput(op, (TFE_TensorHandle*)bias->handle, s);
    TFE_OpSetAttrString(op, "data_format", "NCHW", 4);
    TFE_TensorHandle *ret[1] = {NULL};
    int nret = 1;
    TFE_Execute(op, ret, &nret, s);
    if (TF_GetCode(s) != TF_OK) {
        fprintf(stderr, "[dm_engine] BiasAdd failed: %s\\n", TF_Message(s));
        TFE_DeleteOp(op);
        TF_DeleteStatus(s);
        return -1;
    }
    TFE_DeleteOp(op);
    TF_DeleteStatus(s);
    
    dm_block_create(out, DM_KIND_EXTERNAL, DM_DTYPE_F32, DM_LAYOUT_TF_HANDLE, DM_BACKEND_TENSORFLOW, 0, NULL);
    out->handle = ret[0];
    out->owns_handle = 1;
    out->handle_destructor = dm_tf_tensor_free;
    return 0;
}'''
text = re.sub(r'static TFE_TensorHandle \*execute_bias_add.*?return ret\[0\];\s*\}', replace_execute_bias_add, text, flags=re.DOTALL)

def replace_execute_depthwise(m):
    return '''static int execute_depthwise_conv2d(const DM_Block *input, const DM_Block *filter, int stride, DM_Block *out) {
    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "DepthwiseConv2dNative", s);
    TFE_OpAddInput(op, (TFE_TensorHandle*)input->handle, s);
    TFE_OpAddInput(op, (TFE_TensorHandle*)filter->handle, s);
    int64_t strides[4]   = {1, 1, stride, stride};
    int64_t dilations[4] = {1, 1, 1, 1};
    TFE_OpSetAttrIntList(op, "strides",   strides,   4);
    TFE_OpSetAttrIntList(op, "dilations", dilations, 4);
    TFE_OpSetAttrString(op, "padding",     "SAME",  4);
    TFE_OpSetAttrString(op, "data_format", "NCHW",  4);
    TFE_TensorHandle *ret[1] = {NULL};
    int nret = 1;
    TFE_Execute(op, ret, &nret, s);
    if (TF_GetCode(s) != TF_OK) {
        fprintf(stderr, "[dm_engine] DepthwiseConv2dNative failed: %s\\n", TF_Message(s));
        TFE_DeleteOp(op);
        TF_DeleteStatus(s);
        return -1;
    }
    TFE_DeleteOp(op);
    TF_DeleteStatus(s);
    
    dm_block_create(out, DM_KIND_EXTERNAL, DM_DTYPE_F32, DM_LAYOUT_TF_HANDLE, DM_BACKEND_TENSORFLOW, 0, NULL);
    out->handle = ret[0];
    out->owns_handle = 1;
    out->handle_destructor = dm_tf_tensor_free;
    return 0;
}'''
text = re.sub(r'static TFE_TensorHandle \*execute_depthwise_conv2d.*?return ret\[0\];\s*\}', replace_execute_depthwise, text, flags=re.DOTALL)

def replace_execute_matmul(m):
    return '''static int execute_tf_matmul(const DM_Block *a, const DM_Block *b, bool ta, bool tb, DM_Block *out) {
    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "MatMul", s);
    TFE_OpAddInput(op, (TFE_TensorHandle*)a->handle, s);
    TFE_OpAddInput(op, (TFE_TensorHandle*)b->handle, s);
    TFE_OpSetAttrBool(op, "transpose_a", (unsigned char)(ta ? 1 : 0));
    TFE_OpSetAttrBool(op, "transpose_b", (unsigned char)(tb ? 1 : 0));
    TFE_TensorHandle *ret[1] = {NULL};
    int nret = 1;
    TFE_Execute(op, ret, &nret, s);
    if (TF_GetCode(s) != TF_OK) {
        fprintf(stderr, "[dm_engine] MatMul failed: %s\\n", TF_Message(s));
        TFE_DeleteOp(op);
        TF_DeleteStatus(s);
        return -1;
    }
    TFE_DeleteOp(op);
    TF_DeleteStatus(s);
    
    dm_block_create(out, DM_KIND_EXTERNAL, DM_DTYPE_F32, DM_LAYOUT_TF_HANDLE, DM_BACKEND_TENSORFLOW, 0, NULL);
    out->handle = ret[0];
    out->owns_handle = 1;
    out->handle_destructor = dm_tf_tensor_free;
    return 0;
}'''
text = re.sub(r'static TFE_TensorHandle \*execute_tf_matmul.*?return ret\[0\];\s*\}', replace_execute_matmul, text, flags=re.DOTALL)

# And dm_conv2d_same
def replace_conv2d_same(m):
    return '''int dm_conv2d_same(const DM_Block *in, DM_Block *out,
                   const float *w, const float *b,
                   int out_c, int kernel, int stride) {
    if (!dm_block_is_nchw4(in) || !dm_block_is_nchw4(out)) return -1;
    if (!in || !out || !w || out_c <= 0 || kernel <= 0 || stride <= 0) return -1;
    int oh = out_size_same(DM_NCHW_H(in), stride);
    int ow = out_size_same(DM_NCHW_W(in), stride);
    if (dm_block_create(out, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){DM_NCHW_N(in), out_c, oh, ow}) != 0) return -1;

    DM_Block tf_in, tf_w, tf_out;
    if (dm_lower_block(in, &tf_in, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) return -1;

    DM_Block w_cpu;
    dm_block_view(&w_cpu, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){out_c, DM_NCHW_C(in), kernel, kernel}, (void*)w);
    if (dm_lower_block(&w_cpu, &tf_w, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) {
        dm_block_free(&tf_in);
        return -1;
    }

    if (execute_conv2d(&tf_in, &tf_w, stride, &tf_out) != 0) {
        dm_block_free(&tf_in); dm_block_free(&tf_w); return -1;
    }

    if (b) {
        DM_Block tf_b, tf_out2;
        DM_Block b_cpu;
        dm_block_view(&b_cpu, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 1, (int64_t[]){out_c}, (void*)b);
        if (dm_lower_block(&b_cpu, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) == 0) {
            if (execute_bias_add(&tf_out, &tf_b, &tf_out2) == 0) {
                dm_block_free(&tf_out);
                tf_out = tf_out2;
            }
            dm_block_free(&tf_b);
        }
    }

    dm_lower_block(&tf_out, out, DM_BACKEND_CPU, DM_LOWER_COPY);

    dm_block_free(&tf_in);
    dm_block_free(&tf_w);
    dm_block_free(&tf_out);
    return 0;
}'''
text = re.sub(r'int dm_conv2d_same\([^\{]+\{.*?conv2d_err:.*?return -1;\s*\}', replace_conv2d_same, text, flags=re.DOTALL)

with open('scratch/refactor_engine2.c', 'w') as f:
    f.write(text)

print("Generated refactored engine code.")

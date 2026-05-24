import re

with open("src/core/dm_engine.c", "r") as f: text = f.read()

old_pool = """int dm_max_pool2d_same(const DM_Block *in, DM_Block *out, int kernel, int stride) {
    if (!in || !out || kernel <= 0 || stride <= 0) return -1;
    int oh = out_size_same(DM_NCHW_H(in), stride);
    int ow = out_size_same(DM_NCHW_W(in), stride);
    if (dm_block_create(out, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){DM_NCHW_N(in), DM_NCHW_C(in), oh, ow}) != 0) return -1;

    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "MaxPool", s);

    TFE_TensorHandle *h_in = dm_to_tf(in);
    TFE_OpAddInput(op, h_in, s);

    int64_t ksize[4]   = {1, 1, kernel, kernel};
    int64_t strides[4] = {1, 1, stride, stride};
    TFE_OpSetAttrIntList(op, "ksize",       ksize,   4);
    TFE_OpSetAttrIntList(op, "strides",     strides, 4);
    TFE_OpSetAttrString(op,  "padding",     "SAME",  4);
    TFE_OpSetAttrString(op,  "data_format", "NCHW",  4);

    TFE_TensorHandle *ret[1] = {NULL};
    int nret = 1;
    TFE_Execute(op, ret, &nret, s);
    if (TF_GetCode(s) != TF_OK)
        fprintf(stderr, "[dm_engine] MaxPool failed: %s\\n", TF_Message(s));

    if (ret[0]) { tf_to_raw(ret[0], out->data); TFE_DeleteTensorHandle(ret[0]); }
    TFE_DeleteTensorHandle(h_in);
    TFE_DeleteOp(op);
    TF_DeleteStatus(s);
    return 0;
}"""

new_pool = """int dm_max_pool2d_same(const DM_Block *in, DM_Block *out, int kernel, int stride) {
    if (!in || !out || kernel <= 0 || stride <= 0) return -1;
    int oh = out_size_same(DM_NCHW_H(in), stride);
    int ow = out_size_same(DM_NCHW_W(in), stride);
    if (dm_block_create(out, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){DM_NCHW_N(in), DM_NCHW_C(in), oh, ow}) != 0) return -1;

    dm_tf_init();
    TF_Status *s = TF_NewStatus();

    TFE_TensorHandle *h_in = dm_to_tf(in);
    if (!h_in) { TF_DeleteStatus(s); return -1; }

    TFE_TensorHandle *h_out = NULL;
    if (dm_get_layout_policy() == DM_LAYOUT_POLICY_AUTO_TRANSPOSE) {
        int perm_to_nhwc[4] = {0, 2, 3, 1};
        TFE_TensorHandle *h_nhwc = execute_tf_transpose(h_in, perm_to_nhwc, 4);
        if (!h_nhwc) { TFE_DeleteTensorHandle(h_in); TF_DeleteStatus(s); return -1; }

        TFE_Op *op = TFE_NewOp(tf_ctx, "MaxPool", s);
        TFE_OpAddInput(op, h_nhwc, s);
        int64_t ksize[4]   = {1, kernel, kernel, 1};
        int64_t strides[4] = {1, stride, stride, 1};
        TFE_OpSetAttrIntList(op, "ksize",       ksize,   4);
        TFE_OpSetAttrIntList(op, "strides",     strides, 4);
        TFE_OpSetAttrString(op,  "padding",     "SAME",  4);
        TFE_OpSetAttrString(op,  "data_format", "NHWC",  4);

        TFE_TensorHandle *ret[1] = {NULL};
        int nret = 1;
        TFE_Execute(op, ret, &nret, s);
        TFE_DeleteOp(op);
        TFE_DeleteTensorHandle(h_nhwc);

        if (TF_GetCode(s) != TF_OK) {
            fprintf(stderr, "[dm_engine] MaxPool failed: %s\\n", TF_Message(s));
            TFE_DeleteTensorHandle(h_in); TF_DeleteStatus(s); return -1;
        }

        int perm_to_nchw[4] = {0, 3, 1, 2};
        h_out = execute_tf_transpose(ret[0], perm_to_nchw, 4);
        TFE_DeleteTensorHandle(ret[0]);
    } else {
        TFE_Op *op = TFE_NewOp(tf_ctx, "MaxPool", s);
        TFE_OpAddInput(op, h_in, s);
        int64_t ksize[4]   = {1, 1, kernel, kernel};
        int64_t strides[4] = {1, 1, stride, stride};
        TFE_OpSetAttrIntList(op, "ksize",       ksize,   4);
        TFE_OpSetAttrIntList(op, "strides",     strides, 4);
        TFE_OpSetAttrString(op,  "padding",     "SAME",  4);
        TFE_OpSetAttrString(op,  "data_format", "NCHW",  4);

        TFE_TensorHandle *ret[1] = {NULL};
        int nret = 1;
        TFE_Execute(op, ret, &nret, s);
        TFE_DeleteOp(op);

        if (TF_GetCode(s) != TF_OK) {
            fprintf(stderr, "[dm_engine] MaxPool failed: %s\\n", TF_Message(s));
            TFE_DeleteTensorHandle(h_in); TF_DeleteStatus(s); return -1;
        }
        h_out = ret[0];
    }

    if (h_out) { tf_to_raw(h_out, out->data); TFE_DeleteTensorHandle(h_out); }
    TFE_DeleteTensorHandle(h_in);
    TF_DeleteStatus(s);
    return 0;
}"""

# Do regex replacement to handle slight whitespace differences
text = re.sub(r'int dm_max_pool2d_same\(const DM_Block \*in, DM_Block \*out, int kernel, int stride\) \{.*?return 0;\n\}', new_pool, text, flags=re.DOTALL)
with open("src/core/dm_engine.c", "w") as f: f.write(text)


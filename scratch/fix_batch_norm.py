import re

with open("src/core/dm_engine.c", "r") as f: text = f.read()

old_bn = """int dm_batch_norm(DM_Block *t,
                  const float *gamma, const float *beta,
                  const float *mean,  const float *var,
                  float eps) {
    if (!t || !gamma || !beta || !mean || !var) return -1;
    int C = DM_NCHW_C(t);

    dm_tf_init();
    TF_Status *s = TF_NewStatus();

    TFE_TensorHandle *h_x     = dm_to_tf(t);
    int64_t cdims[1] = {C};
    TFE_TensorHandle *h_scale = raw_to_tf(gamma, cdims, 1);
    TFE_TensorHandle *h_offset= raw_to_tf(beta,  cdims, 1);
    TFE_TensorHandle *h_mean  = raw_to_tf(mean,  cdims, 1);
    TFE_TensorHandle *h_var   = raw_to_tf(var,   cdims, 1);

    TFE_TensorHandle *ret[5] = {NULL, NULL, NULL, NULL, NULL};
    int nret = 5;

    if (dm_get_layout_policy() == DM_LAYOUT_POLICY_AUTO_TRANSPOSE) {
        int perm_to_nhwc[4] = {0, 2, 3, 1};
        TFE_TensorHandle *h_nhwc = execute_tf_transpose(h_x, perm_to_nhwc, 4);
        
        TFE_Op *op = TFE_NewOp(tf_ctx, "FusedBatchNorm", s);
        TFE_OpAddInput(op, h_nhwc,      s);
        TFE_OpAddInput(op, h_scale,  s);
        TFE_OpAddInput(op, h_offset, s);
        TFE_OpAddInput(op, h_mean,   s);
        TFE_OpAddInput(op, h_var,    s);

        TFE_OpSetAttrFloat(op,  "epsilon",     eps);
        TFE_OpSetAttrBool(op,   "is_training", 0);
        TFE_OpSetAttrString(op, "data_format", "NHWC", 4);

        TFE_Execute(op, ret, &nret, s);
        TFE_DeleteOp(op);
        TFE_DeleteTensorHandle(h_nhwc);
        
        if (TF_GetCode(s) != TF_OK) {
            fprintf(stderr, "[dm_engine] FusedBatchNorm failed: %s\\n", TF_Message(s));
        } else {
            int perm_to_nchw[4] = {0, 3, 1, 2};
            TFE_TensorHandle *h_out = execute_tf_transpose(ret[0], perm_to_nchw, 4);
            TFE_DeleteTensorHandle(ret[0]);
            ret[0] = h_out;
        }
    } else {
        TFE_Op *op = TFE_NewOp(tf_ctx, "FusedBatchNorm", s);
        TFE_OpAddInput(op, h_x,      s);
        TFE_OpAddInput(op, h_scale,  s);
        TFE_OpAddInput(op, h_offset, s);
        TFE_OpAddInput(op, h_mean,   s);
        TFE_OpAddInput(op, h_var,    s);

        TFE_OpSetAttrFloat(op,  "epsilon",     eps);
        TFE_OpSetAttrBool(op,   "is_training", 0);
        TFE_OpSetAttrString(op, "data_format", "NCHW", 4);

        TFE_Execute(op, ret, &nret, s);
        if (TF_GetCode(s) != TF_OK)
            fprintf(stderr, "[dm_engine] FusedBatchNorm failed: %s\\n", TF_Message(s));
        TFE_DeleteOp(op);
    }

    TF_DeleteStatus(s);

    if (ret[0]) { tf_to_dm(ret[0], t); }
    for (int i = 0; i < 5; i++) if (ret[i]) TFE_DeleteTensorHandle(ret[i]);
    TFE_DeleteTensorHandle(h_x);
    TFE_DeleteTensorHandle(h_scale);
    TFE_DeleteTensorHandle(h_offset);
    TFE_DeleteTensorHandle(h_mean);
    TFE_DeleteTensorHandle(h_var);
    return 0;
}"""

text = re.sub(r'int dm_batch_norm\(DM_Block \*t,.*?return 0;\n\}', old_bn, text, flags=re.DOTALL)
with open("src/core/dm_engine.c", "w") as f: f.write(text)

